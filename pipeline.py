"""
Synthetic scene generator — Steps 1-5 of the project pipeline.

We synthesize the image a calibrated pinhole camera would observe of a
known planar square marker placed at a known pose.

Steps implemented here:
    1. Generate a 3D plane (the marker plane, Z=0 in its own frame).
    2. Define a square on that plane of side L.
    3. Define a camera with known intrinsics K.
    4. Choose a known marker pose (R_gt, t_gt) relative to the camera.
    5. Project the four 3D corners into the image.

NOT yet implemented (later tasks):
    6. Re-estimate the pose with our IPPE_SQUARE and compare to OpenCV.
    7. Compare estimated pose to ground truth.

Output:
    A dictionary holding every quantity in the scene — ready to feed
    into the pose-estimation stage and the Jacobian analysis later on.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

import config
from camera import Camera, make_default_camera
from marker import square_object_points, transform_points


# ──────────────────────────────────────────────────────────────────────────
# Step 4 — Ground-truth pose generator
# ──────────────────────────────────────────────────────────────────────────

def make_ground_truth_pose(
    euler_xyz_deg: tuple[float, float, float] = (
        config.GT_ROLL_DEG, config.GT_PITCH_DEG, config.GT_YAW_DEG),
    translation_m: tuple[float, float, float] = (
        config.GT_TX_M,   config.GT_TY_M,    config.GT_TZ_M),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a known camera-from-marker pose (R_gt, t_gt).

    The pose maps a point from the marker frame to the camera frame:
        P_cam = R_gt · P_marker + t_gt

    Rotation representation:
        We accept Euler angles (degrees, intrinsic xyz convention) because
        they are intuitive — humans can picture "15° tilt around X, 10°
        around Y, 5° around Z".  Internally we immediately convert to a
        3×3 rotation matrix for use in the projection chain.

        We ALSO return the axis-angle vector ω ∈ ℝ³ (a.k.a. rotation
        vector, a.k.a. so(3) Lie-algebra element). This is the
        parameterization Luca asked us to use later for the Jacobian
        analysis:
            ω = θ · n̂,   ‖ω‖ = θ (rotation angle),   ω/‖ω‖ = n̂ (axis)
            R = exp_so3(ω)   (Rodrigues formula)

        Having ω alongside R from the very first step gets us used to
        seeing the pose in its minimal 6-parameter form (t, ω).

    Translation:
        A reasonable inspection-distance scenario:
            X = 5  cm  (slightly to the right of optical axis)
            Y = 2  cm  (slightly below)
            Z = 50 cm  (half a meter in front of the camera)

    Args:
        euler_xyz_deg : (roll_X, pitch_Y, yaw_Z) in degrees, intrinsic xyz
        translation_m : (X, Y, Z) translation of marker origin in camera frame, meters

    Returns:
        R_gt : (3, 3) ground-truth rotation matrix
        t_gt : (3,)   ground-truth translation vector (meters)
        w_gt : (3,)   ground-truth axis-angle vector ω (radians)
    """
    rotation = Rotation.from_euler('xyz', euler_xyz_deg, degrees=True)
    R_gt = rotation.as_matrix()                  # (3, 3) rotation matrix
    w_gt = rotation.as_rotvec()                  # (3,)   axis-angle (radians)
    t_gt = np.array(translation_m, dtype=np.float64)
    return R_gt, t_gt, w_gt


# ──────────────────────────────────────────────────────────────────────────
# Steps 1-5 orchestrator
# ──────────────────────────────────────────────────────────────────────────

def generate_scene(
    marker_side_m: float = config.MARKER_SIDE_M,
    image_width:   int   = config.IMAGE_WIDTH,
    image_height:  int   = config.IMAGE_HEIGHT,
    camera_fx:     float = config.CAMERA_FX,
    camera_fy:     float = config.CAMERA_FY,
    camera_cx:     float = config.CAMERA_CX,
    camera_cy:     float = config.CAMERA_CY,
    pose_euler_deg: tuple[float, float, float] = (
        config.GT_ROLL_DEG, config.GT_PITCH_DEG, config.GT_YAW_DEG),
    pose_translation_m: tuple[float, float, float] = (
        config.GT_TX_M,   config.GT_TY_M,    config.GT_TZ_M),
) -> dict:
    """
    Execute the full forward simulation (steps 1-5) and return everything.

    Returns a dictionary so teammates downstream can pick out exactly what
    they need without unpacking long tuples.

    Args:
        marker_side_m       : side length L of the square marker (meters)
        image_width/height  : sensor resolution (pixels)
        pose_euler_deg      : ground-truth orientation, intrinsic xyz (degrees)
        pose_translation_m  : ground-truth translation (meters)

    Returns:
        dict with keys:
            'marker_side'    : float, L in meters
            'object_pts'     : (4, 3) corners in marker frame  (Z=0)
            'camera'         : Camera instance (holds K)
            'R_gt'           : (3, 3) ground-truth rotation matrix
            't_gt'           : (3,)   ground-truth translation (meters)
            'w_gt'           : (3,)   ground-truth axis-angle ω (radians)
            'pts_cam'        : (4, 3) corners in camera frame
            'image_pts'      : (4, 2) projected 2D image coordinates (pixels)
    """
    # ── Steps 1 + 2 : marker plane and square on it
    # The marker frame has Z=0 for every corner (planarity).
    # The square sits centered at the origin with side L.
    object_pts = square_object_points(side=marker_side_m)        # (4, 3)

    # ── Step 3 : camera with known intrinsics
    # Pinhole model. All four parameters come from config.py by default;
    # they can still be overridden per-call (useful for sweeps later).
    # Image size is not stored in the Camera object itself (it only carries K),
    # but the principal point typically tracks the image center.
    camera = Camera(fx=camera_fx, fy=camera_fy, cx=camera_cx, cy=camera_cy)

    # ── Step 4 : known ground-truth pose
    # R_gt + t_gt define the rigid transform marker → camera frame.
    # w_gt is the same rotation expressed as a 3-vector (for future Jacobian work).
    R_gt, t_gt, w_gt = make_ground_truth_pose(
        euler_xyz_deg=pose_euler_deg,
        translation_m=pose_translation_m,
    )

    # ── Step 5 : project 3D corners into the image
    # Two sub-steps:
    #   (a) marker frame   →   camera frame    via the rigid transform
    #   (b) camera frame   →   pixel coords    via the pinhole projection
    pts_cam   = transform_points(object_pts, R_gt, t_gt)         # (4, 3) in camera frame
    image_pts = camera.project(pts_cam)                          # (4, 2) in pixels

    return {
        'marker_side':  marker_side_m,
        'object_pts':   object_pts,
        'camera':       camera,
        'image_width':  image_width,
        'image_height': image_height,
        'R_gt':         R_gt,
        't_gt':         t_gt,
        'w_gt':         w_gt,
        'pts_cam':      pts_cam,
        'image_pts':    image_pts,
    }


# ──────────────────────────────────────────────────────────────────────────
# Verbose walkthrough — run directly to see every intermediate quantity
# ──────────────────────────────────────────────────────────────────────────

def _print_scene(scene: dict) -> None:
    """Print every quantity in the synthetic scene for visual inspection."""
    L         = scene['marker_side']
    obj_pts   = scene['object_pts']
    cam       = scene['camera']
    W         = scene['image_width']
    H         = scene['image_height']
    R_gt      = scene['R_gt']
    t_gt      = scene['t_gt']
    w_gt      = scene['w_gt']
    pts_cam   = scene['pts_cam']
    image_pts = scene['image_pts']

    print("=" * 72)
    print("  SYNTHETIC SCENE GENERATION — steps 1 to 5")
    print("=" * 72)

    print(f"\n[Steps 1+2]  Square marker on the Z=0 plane, side L = {L} m")
    print(f"             (origin at marker center, corners labeled 0..3)")
    for i, p in enumerate(obj_pts):
        print(f"               corner {i}: ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})")

    print(f"\n[Step 3]     Pinhole camera intrinsics (K):")
    for row in cam.K:
        print("               " + " ".join(f"{v:8.2f}" for v in row))
    print(f"               image size: {W} × {H} px")

    print(f"\n[Step 4]     Ground-truth pose  (marker → camera frame)")
    print(f"             Translation t_gt (m):")
    print(f"               {t_gt}")
    print(f"             Rotation matrix R_gt:")
    for row in R_gt:
        print("               " + " ".join(f"{v:+.6f}" for v in row))
    angle_deg = np.degrees(np.linalg.norm(w_gt))
    axis      = w_gt / np.linalg.norm(w_gt)
    print(f"             Axis-angle vector ω_gt (rad):")
    print(f"               {w_gt}")
    print(f"               ‖ω‖ = {angle_deg:.4f}°   axis = {axis}")

    print(f"\n[Step 5a]    Marker corners transformed into CAMERA frame (m):")
    for i, p in enumerate(pts_cam):
        print(f"               corner {i}: ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})")
    assert np.all(pts_cam[:, 2] > 0), "All marker corners must be in front of the camera (Z > 0)"
    print(f"             ✓ all Z > 0 — marker is in front of the camera")

    print(f"\n[Step 5b]    Projected 2D image points (pixels):")
    for i, p in enumerate(image_pts):
        print(f"               corner {i}: ({p[0]:8.3f}, {p[1]:8.3f})")
    in_bounds = np.all((image_pts[:, 0] >= 0) & (image_pts[:, 0] < W) &
                       (image_pts[:, 1] >= 0) & (image_pts[:, 1] < H))
    print(f"             ✓ all corners inside the image" if in_bounds
          else "             ✗ WARNING: some corners fall outside the image")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    scene = generate_scene()
    _print_scene(scene)
