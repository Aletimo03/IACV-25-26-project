import numpy as np


def square_object_points(side: float) -> np.ndarray:
    """
    Return the 4 corners of a square marker in its OWN coordinate frame.

    Convention:
        - Marker lies flat on the Z=0 plane (planarity → enables IPPE)
        - Origin at the center of the square
        - Corners listed counter-clockwise starting from top-left,
          matching OpenCV's SOLVEPNP_IPPE_SQUARE expectation:

              0 ──────── 1
              │          │
              │  center  │      (viewed from +Z, i.e. front of marker)
              │          │
              3 ──────── 2

    The Z=0 constraint is the key planarity assumption that IPPE exploits.
    A general 3D point projects as p = K(RP + t); when Z=0, the third
    column of R disappears and the projection collapses to a homography:
        p = K · [r1 | r2 | t] · (X, Y, 1)ᵀ

    Args:
        side : marker side length L (in meters, or any consistent unit)

    Returns:
        (4, 3) array of 3D object points
    """
    L = side / 2.0
    return np.array([
        [-L,  L, 0.0],   # 0: top-left
        [ L,  L, 0.0],   # 1: top-right
        [ L, -L, 0.0],   # 2: bottom-right
        [-L, -L, 0.0],   # 3: bottom-left
    ], dtype=np.float64)


def transform_points(
    points_3d: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """
    Apply a rigid body transform (R, t) to a set of 3D points.

    Math:
        P_cam = R · P_marker + t

    Geometrically:
        - R reorients the marker's axes to align with the camera's view
        - t moves the marker's origin to its position in camera space

    A valid rotation R must lie in SO(3):
        Rᵀ R = I         (orthonormal columns)
        det(R) = +1      (right-handed, not a reflection)

    Implementation note:
        We use the form (R @ P.T).T to apply R column-wise to each point,
        because numpy arrays are (N, 3) row vectors. The matrix product
        R @ P.T gives (3, N), and .T returns it to (N, 3).

    Args:
        points_3d : (N, 3) points in the source frame (e.g. marker frame)
        R         : (3, 3) rotation matrix
        t         : (3,)   translation vector

    Returns:
        (N, 3) points expressed in the target frame (e.g. camera frame)
    """
    assert R.shape == (3, 3),  "R must be 3×3"
    assert t.shape == (3,),    "t must be a 3-vector"

    return (R @ points_3d.T).T + t
