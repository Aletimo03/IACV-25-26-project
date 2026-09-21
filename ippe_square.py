"""A small, from-scratch IPPE solver for OpenCV's canonical square layout.

The implementation follows the planar homography formulation of IPPE.  It
does not call OpenCV: OpenCV is used only by :mod:`pipeline` as a reference
for the synthetic validation experiment.
"""

from __future__ import annotations

import numpy as np

from geometry.camera import Camera


def _validate_ippe_square_inputs(
    object_points: np.ndarray,
    image_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the four canonical, zero-centred IPPE-square correspondences.

    ``SOLVEPNP_IPPE_SQUARE`` is not a generic coplanar solver.  It requires
    the square points to be ordered ``top-left, top-right, bottom-right,
    bottom-left`` when viewed from the marker's positive Z side.  Rejecting
    other layouts early avoids a plausible but geometrically meaningless pose.
    """
    obj = np.asarray(object_points, dtype=np.float64)
    img = np.asarray(image_points, dtype=np.float64)

    if obj.shape != (4, 3):
        raise ValueError("IPPE_SQUARE requires object_points with shape (4, 3)")
    if img.shape != (4, 2):
        raise ValueError("IPPE_SQUARE requires image_points with shape (4, 2)")
    if not np.all(np.isfinite(obj)) or not np.all(np.isfinite(img)):
        raise ValueError("Object and image points must be finite")
    if not np.allclose(obj[:, 2], 0.0, atol=1e-12):
        raise ValueError("IPPE_SQUARE requires all object points on Z=0")

    side = np.linalg.norm(obj[1] - obj[0])
    if side <= np.finfo(np.float64).eps:
        raise ValueError("The square side length must be positive")
    half_side = side / 2.0
    expected = np.array(
        [
            [-half_side, half_side, 0.0],
            [half_side, half_side, 0.0],
            [half_side, -half_side, 0.0],
            [-half_side, -half_side, 0.0],
        ],
        dtype=np.float64,
    )
    if not np.allclose(obj, expected, rtol=1e-9, atol=1e-12):
        raise ValueError(
            "Object points must use OpenCV's canonical centred IPPE_SQUARE "
            "ordering"
        )
    return obj, img


def rotation_aligning_e3_to(vector: np.ndarray) -> np.ndarray:
    """Return the minimum rotation that maps ``e3`` onto ``vector``.

    This is the closed-form Rodrigues expression used by IPPE's first
    algorithm.  The anti-parallel case has an arbitrary, valid 180-degree
    solution and is included for numerical completeness.
    """
    e3 = np.array([0.0, 0.0, 1.0])
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("Cannot align a rotation to the zero vector")
    target = vector / norm
    cross = np.cross(e3, target)
    cosine = float(e3 @ target)
    cross_matrix = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    if 1.0 + cosine < 1e-12:
        return np.diag([1.0, -1.0, -1.0])
    return np.eye(3) + cross_matrix + cross_matrix @ cross_matrix / (1.0 + cosine)


def _rank_one_factor(matrix: np.ndarray) -> np.ndarray:
    """Recover ``b`` from a nearly rank-one positive-semidefinite ``b bᵀ``."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalue = max(float(eigenvalues[-1]), 0.0)
    return np.sqrt(eigenvalue) * eigenvectors[:, -1]


def ippe_core(viewpoint: np.ndarray, homography_jacobian: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute the two IPPE rotation candidates from a local homography.

    Args:
        viewpoint: Normalized projection of the square centre, ``(2,)``.
        homography_jacobian: Derivative of the normalized homography at the
            square centre, with shape ``(2, 2)``.
    """
    v = np.asarray(viewpoint, dtype=np.float64)
    J = np.asarray(homography_jacobian, dtype=np.float64)
    if v.shape != (2,) or J.shape != (2, 2):
        raise ValueError("Expected viewpoint (2,) and homography Jacobian (2, 2)")

    rotation_to_viewpoint = rotation_aligning_e3_to(np.array([v[0], v[1], 1.0]))
    B = np.array([[1.0, 0.0, -v[0]], [0.0, 1.0, -v[1]]]) @ rotation_to_viewpoint[:, :2]
    A = np.linalg.solve(B, J)

    singular_values = np.linalg.svd(A, compute_uv=False)
    gamma = float(singular_values[0])
    if gamma <= np.finfo(np.float64).eps:
        raise ValueError("Degenerate homography Jacobian")
    rotation_2x2 = A / gamma
    b = _rank_one_factor(np.eye(2) - rotation_2x2.T @ rotation_2x2)

    first_column = np.array([rotation_2x2[0, 0], rotation_2x2[1, 0], b[0]])
    second_column = np.array([rotation_2x2[0, 1], rotation_2x2[1, 1], b[1]])
    normal = np.cross(first_column, second_column)
    c, a = normal[:2], normal[2]

    def assemble(b_sign: float, c_sign: float) -> np.ndarray:
        local_rotation = np.empty((3, 3))
        local_rotation[:2, :2] = rotation_2x2
        local_rotation[:2, 2] = c_sign * c
        local_rotation[2, :2] = b_sign * b
        local_rotation[2, 2] = a
        return rotation_to_viewpoint @ local_rotation

    return gamma, assemble(+1.0, +1.0), assemble(-1.0, -1.0)


def dlt_homography(object_xy: np.ndarray, image_normalized: np.ndarray) -> np.ndarray:
    """Estimate a plane-to-normalized-image homography with DLT/SVD.

    The square is centred at the origin, so ``H[2, 2]`` is non-zero for the
    valid front-of-camera cases used by this project and can safely fix the
    arbitrary homogeneous scale.
    """
    design_rows: list[list[float]] = []
    for (X, Y), (u, v) in zip(object_xy, image_normalized, strict=True):
        design_rows.append([-X, -Y, -1.0, 0.0, 0.0, 0.0, u * X, u * Y, u])
        design_rows.append([0.0, 0.0, 0.0, -X, -Y, -1.0, v * X, v * Y, v])
    _, _, right_singular_vectors = np.linalg.svd(np.asarray(design_rows))
    homography = right_singular_vectors[-1].reshape(3, 3)
    if abs(homography[2, 2]) <= np.finfo(np.float64).eps:
        raise ValueError("Degenerate homography scale")
    return homography / homography[2, 2]


def homography_jacobian_at_origin(homography: np.ndarray) -> np.ndarray:
    """Return the normalized-homography derivative at the square centre.

    For ``x(X,Y) = (h11 X + h12 Y + h13)/(h31 X + h32 Y + 1)``, the first
    row is ``[h11 - h13*h31, h12 - h13*h32]``.  Keeping this expression in a
    dedicated function makes the coordinate convention and the quotient rule
    explicit.
    """
    H = np.asarray(homography, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError("Homography must have shape (3, 3)")
    if not np.isclose(H[2, 2], 1.0):
        H = H / H[2, 2]
    return np.array(
        [
            [H[0, 0] - H[0, 2] * H[2, 0], H[0, 1] - H[0, 2] * H[2, 1]],
            [H[1, 0] - H[1, 2] * H[2, 0], H[1, 1] - H[1, 2] * H[2, 1]],
        ]
    )


def solve_translation(
    rotation: np.ndarray,
    object_points: np.ndarray,
    image_normalized: np.ndarray,
) -> np.ndarray:
    """Estimate translation for a fixed rotation by linear least squares.

    Each correspondence imposes ``q × (R X + t) = 0``.  Only the first two of
    the three cross-product equations are stacked, which uses every corner
    without explicitly forming a normal-equation inverse.
    """
    rows: list[np.ndarray] = []
    right_hand_sides: list[np.ndarray] = []
    for point, normalized_pixel in zip(object_points, image_normalized, strict=True):
        ray = np.array([normalized_pixel[0], normalized_pixel[1], 1.0])
        ray_cross = np.array(
            [
                [0.0, -ray[2], ray[1]],
                [ray[2], 0.0, -ray[0]],
                [-ray[1], ray[0], 0.0],
            ]
        )
        # Row 3 is a combination of rows 1-2 but reweights the fit; dropping it
        # gives OpenCV's 8x3 system, so both candidates match OpenCV exactly.
        rows.append(ray_cross[:2])
        right_hand_sides.append((-ray_cross @ (rotation @ point))[:2])
    translation, *_ = np.linalg.lstsq(
        np.vstack(rows), np.concatenate(right_hand_sides), rcond=None
    )
    return translation


def project_points(
    rotation: np.ndarray,
    translation: np.ndarray,
    camera_matrix: np.ndarray,
    object_points: np.ndarray,
) -> np.ndarray:
    """Project marker-frame points using a marker-to-camera pose."""
    points_camera = (rotation @ object_points.T).T + translation
    homogeneous_pixels = (camera_matrix @ points_camera.T).T
    return homogeneous_pixels[:, :2] / homogeneous_pixels[:, 2:3]


def reprojection_rmse(projected_points: np.ndarray, image_points: np.ndarray) -> float:
    """Return OpenCV-compatible RMSE over all scalar image residuals.

    OpenCV's value divides the sum of squared x/y residuals by ``2N``.  This
    is a factor of ``sqrt(2)`` smaller than an RMSE of Euclidean point errors.
    The definition is used here so the custom and OpenCV diagnostics compare
    directly.
    """
    residuals = np.asarray(projected_points) - np.asarray(image_points)
    return float(np.sqrt(np.mean(residuals**2)))


def ippe_square(
    camera: Camera,
    object_points: np.ndarray,
    image_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, object]]:
    """Estimate the two IPPE-square poses from four known correspondences.

    Returns the lowest-reprojection-error candidate first together with an
    ``info`` dictionary whose ``solutions`` entry contains *both* unpolished
    candidates as ``(rotation, translation, rmse_px)`` tuples.  Preserving the
    raw pair is essential: an unrestricted nonlinear optimization can make
    both initializations converge to the same solution and hide IPPE's
    two-solution ambiguity.
    """
    object_points, image_points = _validate_ippe_square_inputs(object_points, image_points)
    normalized_homogeneous = (
        camera.K_inv @ np.column_stack([image_points, np.ones(4)]).T
    ).T
    image_normalized = normalized_homogeneous[:, :2]

    homography = dlt_homography(object_points[:, :2], image_normalized)
    homography_jacobian = homography_jacobian_at_origin(homography)
    viewpoint = homography[:2, 2]
    gamma, first_rotation, second_rotation = ippe_core(viewpoint, homography_jacobian)

    solutions: list[tuple[np.ndarray, np.ndarray, float]] = []
    for rotation in (first_rotation, second_rotation):
        translation = solve_translation(rotation, object_points, image_normalized)
        projected = project_points(rotation, translation, camera.K, object_points)
        solutions.append((rotation, translation, reprojection_rmse(projected, image_points)))

    solutions.sort(key=lambda candidate: candidate[2])
    best_rotation, best_translation, best_error = solutions[0]
    return best_rotation, best_translation, best_error, {
        "solutions": solutions,
        "gamma": gamma,
        "homography": homography,
    }
