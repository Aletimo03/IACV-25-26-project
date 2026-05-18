import numpy as np


class Camera:
    """
    Pinhole camera model.

    Coordinate convention:
        - Camera sits at the origin, looking down the +Z axis.
        - X points right, Y points down, Z points into the scene.

    Projection (3D camera-frame point → 2D pixel):
        u = fx * (X/Z) + cx
        v = fy * (Y/Z) + cy

    In matrix form (homogeneous):
        [u*Z]   [fx   0  cx] [X]
        [v*Z] = [ 0  fy  cy] [Y]  =  K · P_cam
        [  Z]   [ 0   0   1] [Z]

    Then divide by Z to get the actual pixel (u, v).
    """

    def __init__(self, fx: float, fy: float, cx: float, cy: float):
        """
        Args:
            fx, fy : focal lengths in pixels (usually equal for square pixels)
            cx, cy : principal point in pixels (optical axis intersection)
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        # 3x3 intrinsic matrix
        self.K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1],
        ], dtype=np.float64)

        # K inverse — used to "undo" the intrinsics and go to normalized coords
        self.K_inv = np.linalg.inv(self.K)

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Project 3D points (in camera frame) → 2D pixel coordinates.

        Math for each point (X, Y, Z):
            u = fx * (X/Z) + cx      ← perspective divide + shift by principal point
            v = fy * (Y/Z) + cy

        The division by Z is the key: it encodes perspective foreshortening.
        A point twice as far (Z→2Z) produces half the displacement from center.

        Args:
            points_3d : (N, 3) array of 3D points in camera frame

        Returns:
            (N, 2) array of 2D pixel coordinates [u, v]
        """
        assert points_3d.ndim == 2 and points_3d.shape[1] == 3, \
            "Expected (N, 3) array"

        X = points_3d[:, 0]
        Y = points_3d[:, 1]
        Z = points_3d[:, 2]

        u = self.fx * (X / Z) + self.cx
        v = self.fy * (Y / Z) + self.cy

        return np.stack([u, v], axis=1)  # (N, 2)

    def normalize(self, points_2d: np.ndarray) -> np.ndarray:
        """
        Apply K⁻¹ to pixel coordinates → normalized image coordinates.

        Normalized coords are what you get if you imagine a camera with
        K = I (identity): no focal length scaling, principal point at origin.
        They live on the Z=1 plane in camera space.

        Math:
            x_n = (u - cx) / fx
            y_n = (v - cy) / fy

        Equivalently in matrix form:
            [x_n]           [u]
            [y_n] = K⁻¹  · [v]
            [ 1 ]           [1]

        Why this matters for IPPE:
            The homography H maps marker plane → image. If we work in
            normalized coords, H absorbs only the geometry (R, t) and NOT K.
            This makes decomposing H into (R, t) straightforward.

        Args:
            points_2d : (N, 2) array of pixel coordinates [u, v]

        Returns:
            (N, 2) array of normalized coordinates [x_n, y_n]
        """
        assert points_2d.ndim == 2 and points_2d.shape[1] == 2, \
            "Expected (N, 2) array"

        # Lift to homogeneous: (N, 3) with last coord = 1
        ones = np.ones((len(points_2d), 1), dtype=np.float64)
        pts_h = np.hstack([points_2d, ones])   # (N, 3)

        # Apply K⁻¹:  each row is K⁻¹ · [u, v, 1]ᵀ
        pts_n = (self.K_inv @ pts_h.T).T        # (N, 3)

        # Drop the homogeneous coordinate (always 1 after K⁻¹)
        return pts_n[:, :2]                     # (N, 2)


def make_default_camera(width: int = 640, height: int = 480) -> Camera:
    """
    A reasonable synthetic camera for a 640×480 image.

    fx = fy = 800 px corresponds roughly to a ~43° horizontal FOV —
    typical for a moderate telephoto or close-range inspection setup.
    Principal point placed at the image center.
    """
    return Camera(fx=800.0, fy=800.0, cx=width / 2.0, cy=height / 2.0)
