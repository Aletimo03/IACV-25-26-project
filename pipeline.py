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

Output:
    A dictionary holding every quantity in the scene — ready to feed
    into the pose-estimation stage and the Jacobian analysis later on.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

import config
from camera import make_camera
from marker import make_square_marker, transform_points

from ippe_square import ippe_square
import cv2


# ──────────────────────────────────────────────────────────────────────────
# Step 4 — Ground-truth pose generator
# ──────────────────────────────────────────────────────────────────────────

def make_ground_truth_pose(
    euler_xyz_deg: tuple[float, float, float] = (
        config.GT_ROTATION_AROUND_X_DEG,
        config.GT_ROTATION_AROUND_Y_DEG,
        config.GT_ROTATION_AROUND_Z_DEG),
    translation_m: tuple[float, float, float] = (
        config.GT_TRANSLATION_X_M,
        config.GT_TRANSLATION_Y_M,
        config.GT_TRANSLATION_Z_M),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a known camera-from-marker pose (R_gt, t_gt).

    The pose maps a point from the marker frame to the camera frame:
        P_cam = R_gt · P_marker + t_gt

    Rotation representation:
        We accept Euler angles (degrees, intrinsic xyz convention) because
        they are intuitive — humans can picture "15° tilt around X, 10°
        around Y, 5° around Z".  Internally we immediately convert to a
        3×3 rotation matrix for use in the projection chain.

    Translation:
        A reasonable inspection-distance scenario:
            X = 5  cm  (slightly to the right of optical axis)
            Y = 2  cm  (slightly below)
            Z = 50 cm  (half a meter in front of the camera)

    Args:
        euler_xyz_deg : (rotation_around_X, rotation_around_Y, rotation_around_Z) in degrees, intrinsic xyz
        translation_m : (X, Y, Z) translation of marker origin in camera frame, meters

    Returns:
        R_gt : (3, 3) ground-truth rotation matrix
        t_gt : (3,)   ground-truth translation vector (meters)
    """
    rotation = Rotation.from_euler('xyz', euler_xyz_deg, degrees=True)
    R_gt = rotation.as_matrix()                  # (3, 3) rotation matrix
    t_gt = np.array(translation_m, dtype=np.float64)
    return R_gt, t_gt


# ──────────────────────────────────────────────────────────────────────────
# Steps 1-5 orchestrator
# ──────────────────────────────────────────────────────────────────────────

def generate_scene() -> dict:
    """
    Execute the full forward simulation (steps 1-5) and return everything.

    This function only orchestrates. Every scene parameter lives in
    config.py and is pulled in by the individual factories:
        - make_square_marker()    → marker geometry
        - make_camera()           → camera intrinsics
        - make_ground_truth_pose()→ ground-truth pose
    To run a different scene, edit config.py (or call the factories directly
    with overrides)

    Returns a dictionary so downstream code can pick out exactly what it
    needs without unpacking long tuples.

    Returns:
        dict with keys:
            'marker_side'    : float, L in meters
            'object_pts'     : (4, 3) corners in marker frame  (Z=0)
            'camera'         : Camera instance (holds K)
            'image_width'    : int, image width in pixels
            'image_height'   : int, image height in pixels
            'R_gt'           : (3, 3) ground-truth rotation matrix
            't_gt'           : (3,)   ground-truth translation (meters)
            'pts_cam'        : (4, 3) corners in camera frame
            'image_pts'      : (4, 2) projected 2D image coordinates (pixels)
    """
    # ── Steps 1 + 2 : marker plane and square on it
    object_pts = make_square_marker()                            # (4, 3)

    # ── Step 3 : camera with known intrinsics
    camera = make_camera()

    # ── Step 4 : known ground-truth pose (rigid transform marker → camera)
    R_gt, t_gt = make_ground_truth_pose()

    # ── Step 5 : project 3D corners into the image
    #   (a) marker frame → camera frame  via the rigid transform
    #   (b) camera frame → pixel coords  via the pinhole projection
    pts_cam   = transform_points(object_pts, R_gt, t_gt)         # (4, 3) in camera frame
    image_pts = camera.project(pts_cam)                          # (4, 2) in pixels

    # ── Step 6 : estimate camera pose with implemented IPPE_SQUARE and validate with opencv algorithm
    dist = np.zeros(5)

    R_ours, t_ours, err_ours, info = ippe_square(camera, object_pts, image_pts)

    _, rvecs_cv, tvecs_cv, errs_cv = cv2.solvePnPGeneric(
        object_pts, image_pts, camera.K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    R_cv = cv2.Rodrigues(rvecs_cv[0])[0]
    t_cv = tvecs_cv[0].ravel()

    # ── Step 7 : compare estimated poses against the ground truth
    return {
        'marker_side':  config.MARKER_SIDE_M,
        'object_pts':   object_pts,
        'camera':       camera,
        'image_width':  config.IMAGE_WIDTH,
        'image_height': config.IMAGE_HEIGHT,
        'R_gt':         R_gt,
        't_gt':         t_gt,
        'pts_cam':      pts_cam,
        'image_pts':    image_pts,
        # step 6 — our IPPE_SQUARE
        'R_ours':        R_ours,
        't_ours':        t_ours,
        'reproj_err_ours': err_ours,
        'solutions':    info['solutions'],     # both candidates, sorted by error
        'gamma':        info['gamma'],
        # step 6 — OpenCV reference
        'R_cv':         R_cv,
        't_cv':         t_cv,
        'reproj_err_cv': float(errs_cv[0][0]),
        'solutions_cv': [(cv2.Rodrigues(r)[0], t.ravel(), float(e[0]))
                         for r, t, e in zip(rvecs_cv, tvecs_cv, errs_cv)],
    }


# ──────────────────────────────────────────────────────────────────────────
# Step 7 — pose comparison metrics
# ──────────────────────────────────────────────────────────────────────────

def pose_error(R: np.ndarray, t: np.ndarray,
               R_ref: np.ndarray, t_ref: np.ndarray) -> tuple[float, float]:
    """
    Compare a pose (R, t) against a reference pose (R_ref, t_ref).

    Returns:
        rot_err_deg   : geodesic angle of R_ref^T · R, in degrees
        trans_err_mm  : Euclidean norm of the translation difference, in mm
    """
    R_delta = R_ref.T @ R
    cos_angle = np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)
    rot_err_deg = np.degrees(np.arccos(cos_angle))
    trans_err_mm = np.linalg.norm(np.asarray(t) - np.asarray(t_ref)) * 1000.0
    return float(rot_err_deg), float(trans_err_mm)


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
    pts_cam   = scene['pts_cam']
    image_pts = scene['image_pts']

    print("=" * 72)
    print("  SYNTHETIC SCENE GENERATION + POSE ESTIMATION — steps 1 to 7")
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

    # ── Step 6 : pose estimation, ours vs OpenCV
    print(f"\n[Step 6]     Pose re-estimation from the 2D-3D correspondences")

    LBL = 12          # width of the left-hand row-label column
    COL = 45          # width of one candidate column

    def _print_candidates(title: str, sols: list) -> None:
        """Print the two pose candidates of one solver side by side."""
        def cell(k: int, kind: str, row: int = 0) -> str:
            if k >= len(sols):
                return "—".ljust(COL)
            R_c, t_c, _ = sols[k]
            if kind == 't':
                s = f"{t_c[0]:+.10f} {t_c[1]:+.10f} {t_c[2]:+.10f}"
            else:
                s = " ".join(f"{v:+.10f}" for v in R_c[row])
            return s.ljust(COL)

        def line(label: str, kind: str, row: int = 0) -> None:
            print(("               " + label.ljust(LBL)
                   + cell(0, kind, row) + " │ " + cell(1, kind, row)).rstrip())

        print(f"\n             {title}")
        print("               " + "".ljust(LBL)
              + "candidate 1 (best)".ljust(COL) + " │ " + "candidate 2")
        print("               " + "─" * LBL + "─" * COL + "─┼─" + "─" * COL)
        line("t (m)",  't')
        line("R",      'R', 0)
        line("",       'R', 1)
        line("",       'R', 2)

    _print_candidates("Our IPPE_SQUARE:", scene['solutions'])
    _print_candidates("OpenCV SOLVEPNP_IPPE_SQUARE:", scene['solutions_cv'])

    d_rot, d_trans = pose_error(scene['R_ours'], scene['t_ours'],
                                scene['R_cv'],  scene['t_cv'])
    print(f"\n             ours vs OpenCV agreement: "
          f"Δrot = {d_rot:.10f}°, Δtrans = {d_trans:.10f} mm")

    print(f"\n             Two-fold planar ambiguity — errors of both candidates:")
    print(f"               {'source':<10}"
          f"{'│ candidate 1 (best)':<58}{'│ candidate 2':<58}".rstrip())
    print(f"               {'':<10}"
          f"│ {'reproj RMSE (px)':>20}{'rot err (°)':>17}{'trans err (mm)':>17} "
          f"│ {'reproj RMSE (px)':>20}{'rot err (°)':>17}{'trans err (mm)':>17}")
    print("               " + "─" * 10 + "┼" + "─" * 57 + "┼" + "─" * 57)

    def _cand_cells(sols):
        cells = ""
        for k in range(2):
            if k < len(sols):
                R_c, t_c, e_c = sols[k]
                r_e, t_e = pose_error(R_c, t_c, R_gt, t_gt)
                cells += f"│ {e_c:>20.10e}{r_e:>17.10f}{t_e:>17.10f} "
            else:
                cells += f"│ {'—':>20}{'—':>17}{'—':>17} "
        return cells

    print((f"               {'ours':<10}" + _cand_cells(scene['solutions'])).rstrip())
    print((f"               {'OpenCV':<10}" + _cand_cells(scene['solutions_cv'])).rstrip())

    e0 = scene['solutions'][0][2]
    e1 = scene['solutions'][1][2] if len(scene['solutions']) > 1 else float('inf')
    ratio = e1 / e0 if e0 > 0 else float('inf')
    print(f"               error ratio second/first = {ratio:.10f}  "
          f"({'well separated' if ratio > 3 else 'AMBIGUOUS — candidates hard to tell apart'})")

    # ── Step 7 : estimated pose vs ground truth
    print(f"\n[Step 7]     Estimated pose vs ground truth")
    rot_err, trans_err = pose_error(scene['R_ours'], scene['t_ours'], R_gt, t_gt)
    rot_err_cv, trans_err_cv = pose_error(scene['R_cv'], scene['t_cv'], R_gt, t_gt)

    poses = [("ground truth", R_gt, t_gt),
             ("ours",         scene['R_ours'], scene['t_ours']),
             ("OpenCV",       scene['R_cv'],   scene['t_cv'])]

    def gt_cell(kind: str, R_p, t_p, row: int = 0) -> str:
        s = (f"{t_p[0]:+.10f} {t_p[1]:+.10f} {t_p[2]:+.10f}" if kind == 't'
             else " ".join(f"{v:+.10f}" for v in R_p[row]))
        return s.ljust(COL)

    print("\n               " + "".ljust(LBL)
          + " │ ".join(name.ljust(COL) for name, _, _ in poses).rstrip())
    print("               " + "─" * LBL + ("─" * COL + "─┼─") * 2 + "─" * COL)
    print(("               " + "t (m)".ljust(LBL)
           + " │ ".join(gt_cell('t', R_p, t_p) for _, R_p, t_p in poses)).rstrip())
    for r in range(3):
        print(("               " + ("R" if r == 0 else "").ljust(LBL)
               + " │ ".join(gt_cell('R', R_p, t_p, r)
                            for _, R_p, t_p in poses)).rstrip())

    print(f"\n               {'method':<22}{'rot err (°)':>17}{'trans err (mm)':>18}")
    print(f"               {'ours (IPPE_SQUARE)':<22}{rot_err:>17.10f}{trans_err:>18.10f}")
    print(f"               {'OpenCV IPPE_SQUARE':<22}{rot_err_cv:>17.10f}{trans_err_cv:>18.10f}")
    ok = rot_err < 1e-3 and trans_err < 1e-3
    print("             ✓ noiseless recovery of the ground-truth pose" if ok
          else "             ✗ estimated pose deviates from ground truth "
               "(expected only with noise, or wrong candidate selected)")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    scene = generate_scene()
    _print_scene(scene)
