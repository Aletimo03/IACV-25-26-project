"""Residual Jacobians and local-pose uncertainty diagnostics.

The six parameters are deliberately ordered as
``[t_x, t_y, t_z, omega_x, omega_y, omega_z]``.  Translation is in metres and
the local rotation increment is in radians, so reported singular values and
condition numbers are tied to those stated units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geometry.camera import Camera


def skew(vector: np.ndarray) -> np.ndarray:
    """Return the skew-symmetric matrix ``[vector]_x`` for a 3-vector."""
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(rotation_vector: np.ndarray) -> np.ndarray:
    """Map a rotation vector (radians) to a 3×3 rotation matrix."""
    rotation_vector = np.asarray(rotation_vector, dtype=np.float64)
    angle = np.linalg.norm(rotation_vector)
    cross = skew(rotation_vector)
    if angle < 1e-10:
        # The first two terms of Rodrigues' series avoid division by angle.
        return np.eye(3) + cross + 0.5 * cross @ cross
    return (
        np.eye(3)
        + (np.sin(angle) / angle) * cross
        + ((1.0 - np.cos(angle)) / angle**2) * cross @ cross
    )


def so3_log(rotation: np.ndarray) -> np.ndarray:
    """Map a near-identity rotation matrix to its rotation vector.

    Monte Carlo errors are small, therefore the near-identity branch is the
    relevant one here.  The general branch is retained for normal use away
    from zero rotation.
    """
    R = np.asarray(rotation, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError("Rotation must have shape (3, 3)")
    cosine = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if angle < 1e-8:
        return 0.5 * vee
    return angle * vee / (2.0 * np.sin(angle))


def perturb_pose(
    rotation: np.ndarray,
    translation: np.ndarray,
    delta_theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a local state increment ``[delta_t, delta_omega]`` to a pose.

    The rotation update is right-multiplied: ``R_new = R exp([delta_omega]_x)``.
    Consequently ``delta_omega`` is expressed in the marker frame and the
    analytic Jacobian contains ``-R [X]_x``.
    """
    delta = np.asarray(delta_theta, dtype=np.float64)
    if delta.shape != (6,):
        raise ValueError("delta_theta must have shape (6,)")
    return rotation @ so3_exp(delta[3:]), np.asarray(translation) + delta[:3]


def reprojection_residuals(
    camera: Camera,
    object_points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    image_points: np.ndarray,
) -> np.ndarray:
    """Return stacked pixel residuals ``[du0, dv0, ..., du3, dv3]``."""
    object_points = np.asarray(object_points, dtype=np.float64)
    image_points = np.asarray(image_points, dtype=np.float64)
    points_camera = (rotation @ object_points.T).T + translation
    projected = camera.project(points_camera)
    return (projected - image_points).reshape(-1)


def reprojection_jacobian(
    camera: Camera,
    object_points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Compute the analytic 8×6 Jacobian of reprojection residuals.

    For each marker point ``X``, ``P_c = R X + t`` and a local increment gives
    ``dP_c/d(delta_theta) = [I, -R[X]_x]``.  Multiplying this by the pinhole
    projection derivative yields two residual rows per corner.
    """
    object_points = np.asarray(object_points, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    if object_points.shape != (4, 3):
        raise ValueError("This project expects exactly four object points")
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("Expected rotation (3, 3) and translation (3,)")

    jacobian = np.empty((8, 6), dtype=np.float64)
    for index, marker_point in enumerate(object_points):
        x, y, z = rotation @ marker_point + translation
        if z <= 0.0:
            raise ValueError("Jacobian is undefined for points at or behind the camera")
        projection_derivative = np.array(
            [
                [camera.fx / z, 0.0, -camera.fx * x / z**2],
                [0.0, camera.fy / z, -camera.fy * y / z**2],
            ]
        )
        point_derivative = np.column_stack([np.eye(3), -rotation @ skew(marker_point)])
        jacobian[2 * index : 2 * index + 2] = projection_derivative @ point_derivative
    return jacobian


def jacobian_svd(jacobian: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Singular values, sign-normalised right singular vectors (rows), condition number."""
    _, singular_values, right_singular_vectors = np.linalg.svd(jacobian)
    # A singular vector is only defined up to sign; make its largest entry
    # positive so the same direction reads the same from one pose to the next.
    largest = np.argmax(np.abs(right_singular_vectors), axis=1)
    right_singular_vectors *= np.sign(right_singular_vectors[np.arange(6), largest])[:, None]
    smallest = float(singular_values[-1])
    condition = float(np.inf if smallest <= np.finfo(np.float64).eps else singular_values[0] / smallest)
    return singular_values, right_singular_vectors, condition


@dataclass(frozen=True)
class JacobianDiagnostics:
    """SVD and first-order covariance metrics for one nominal pose."""

    jacobian: np.ndarray
    singular_values: np.ndarray
    smallest_singular_value: float
    condition_number: float
    predicted_covariance: np.ndarray


def analyze_jacobian(
    camera: Camera,
    object_points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    noise_std_px: float,
) -> JacobianDiagnostics:
    """Compute SVD diagnostics and ``sigma² (JᵀJ)^-1`` covariance prediction."""
    if noise_std_px < 0.0:
        raise ValueError("noise_std_px must be non-negative")
    jacobian = reprojection_jacobian(camera, object_points, rotation, translation)
    singular_values, _, condition = jacobian_svd(jacobian)
    smallest = float(singular_values[-1])
    information = jacobian.T @ jacobian
    covariance = noise_std_px**2 * np.linalg.pinv(information, rcond=1e-12)
    return JacobianDiagnostics(
        jacobian=jacobian,
        singular_values=singular_values,
        smallest_singular_value=smallest,
        condition_number=condition,
        predicted_covariance=covariance,
    )
