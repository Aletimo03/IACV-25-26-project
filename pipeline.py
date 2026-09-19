"""Steps 1-7: the synthetic pipeline, end to end, on one scene.

Build the marker and the camera, pick a known pose, project the four corners,
recover both IPPE candidates with our own solver, and check them against
OpenCV and against the ground truth. Running this file prints every
intermediate quantity. The two viewpoint studies live in experiment1.py and
experiment2.py.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import config
from geometry import make_camera, make_square_marker, transform_points
from ippe_square import ippe_square, project_points


def make_ground_truth_pose(
    euler_xyz_deg: tuple[float, float, float] = (
        config.GT_ROTATION_AROUND_X_DEG,
        config.GT_ROTATION_AROUND_Y_DEG,
        config.GT_ROTATION_AROUND_Z_DEG,
    ),
    translation_m: tuple[float, float, float] = (
        config.GT_TRANSLATION_X_M,
        config.GT_TRANSLATION_Y_M,
        config.GT_TRANSLATION_Z_M,
    ),
) -> tuple[np.ndarray, np.ndarray]:
    """The ground-truth pose from the angles and offsets in config.py.

    SciPy's lower-case "xyz" means extrinsic (fixed-axis) Euler angles. They
    are here only to describe the scene in readable terms -- nothing in the
    estimation ever touches Euler angles.

    The flip matters. With R = I the marker's +Z normal points the same way as
    the camera's, i.e. away from us, so small Euler angles would be tilting the
    *back* of the marker. Composing with a 180-degree rotation about X puts the
    marker face towards the camera, like a real detected ArUco, and lets the
    config angles mean what their comments say: tilt away from fronto-parallel.
    """
    face_camera = np.diag([1.0, -1.0, -1.0])
    rotation = Rotation.from_euler("xyz", euler_xyz_deg, degrees=True).as_matrix() @ face_camera
    return rotation, np.asarray(translation_m, dtype=np.float64)


def _opencv_ippe_square_solutions(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """OpenCV's answer, used purely as an external check on ours."""
    _, rotation_vectors, translation_vectors, errors = cv2.solvePnPGeneric(
        object_points,
        image_points,
        camera_matrix,
        np.zeros(5),
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    solutions = [
        (cv2.Rodrigues(rotation_vector)[0], translation_vector.ravel(), float(error[0]))
        for rotation_vector, translation_vector, error in zip(
            rotation_vectors, translation_vectors, errors, strict=True
        )
    ]
    solutions.sort(key=lambda candidate: candidate[2])
    if len(solutions) != 2:
        raise RuntimeError("OpenCV IPPE_SQUARE did not return two candidates")
    return solutions


def generate_scene() -> dict[str, object]:
    """Build the default scene and solve it with both solvers.

    The pose always maps marker coordinates into the camera frame,
    P_cam = R @ P_marker + t. Image points come straight out of the pinhole
    model: no detector, no image processing, no lens distortion.
    """
    object_points = make_square_marker()
    camera = make_camera()
    rotation_gt, translation_gt = make_ground_truth_pose()
    points_camera = transform_points(object_points, rotation_gt, translation_gt)
    if np.any(points_camera[:, 2] <= 0.0):
        raise ValueError("The configured marker must lie entirely in front of the camera")
    image_points = camera.project(points_camera)

    rotation_custom, translation_custom, custom_error, info = ippe_square(
        camera, object_points, image_points
    )
    opencv_solutions = _opencv_ippe_square_solutions(object_points, image_points, camera.K)
    rotation_cv, translation_cv, cv_error = opencv_solutions[0]
    return {
        "marker_side": config.MARKER_SIDE_M,
        "object_points": object_points,
        "camera": camera,
        "image_width": config.IMAGE_WIDTH,
        "image_height": config.IMAGE_HEIGHT,
        "R_gt": rotation_gt,
        "t_gt": translation_gt,
        "points_camera": points_camera,
        "image_points": image_points,
        "R_custom": rotation_custom,
        "t_custom": translation_custom,
        "reprojection_error_custom": custom_error,
        "solutions_custom": info["solutions"],
        "gamma": info["gamma"],
        "R_cv": rotation_cv,
        "t_cv": translation_cv,
        "reprojection_error_cv": cv_error,
        "solutions_cv": opencv_solutions,
    }


def pose_error(
    rotation: np.ndarray,
    translation: np.ndarray,
    rotation_reference: np.ndarray,
    translation_reference: np.ndarray,
) -> tuple[float, float]:
    """Rotation error in degrees and translation error in millimetres."""
    relative_rotation = rotation_reference.T @ rotation
    cosine = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
    rotation_error_deg = float(np.rad2deg(np.arccos(cosine)))
    translation_error_mm = float(np.linalg.norm(translation - translation_reference) * 1000.0)
    return rotation_error_deg, translation_error_mm


def validate_noiseless_scene(scene: dict[str, object]) -> None:
    """Complain loudly if our primary pose drifts away from OpenCV's."""
    rotation_error_deg, translation_error_mm = pose_error(
        scene["R_custom"], scene["t_custom"], scene["R_cv"], scene["t_cv"]
    )
    if rotation_error_deg > 1e-5 or translation_error_mm > 0.05:
        raise AssertionError(
            "Custom IPPE-square does not agree with OpenCV in the noiseless scene: "
            f"{rotation_error_deg:.3e} deg, {translation_error_mm:.3e} mm"
        )


def viewing_angle_deg(rotation: np.ndarray, translation: np.ndarray) -> float:
    """Angle between the marker normal and the line back to the camera.

    Same quantity the experiments sweep over: 0 is fronto-parallel, and
    anything above 90 means we are looking at the back of the marker.
    """
    marker_normal_in_camera = rotation @ np.array([0.0, 0.0, 1.0])
    marker_to_camera = -translation / np.linalg.norm(translation)
    cosine = np.clip(marker_normal_in_camera @ marker_to_camera, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def _matrix_lines(matrix: np.ndarray, indent: str = "      ") -> str:
    """Print a matrix with aligned columns and signs."""
    return "\n".join(
        indent + "[" + "  ".join(f"{value:+9.6f}" for value in row) + "]" for row in matrix
    )


def _print_baseline(scene: dict[str, object]) -> None:
    """Walk the whole pipeline on the default scene, printing as we go."""
    rotation_gt = scene["R_gt"]
    translation_gt = scene["t_gt"]
    object_points = scene["object_points"]
    camera = scene["camera"]
    image_points = scene["image_points"]

    print("=" * 78)
    print("SYNTHETIC IPPE-SQUARE PIPELINE --- STEPS 1-7")
    print("=" * 78)
    print("Convention: P_cam = R @ P_marker + t; camera +X right, +Y down, +Z forward.")

    print("\n[Steps 1-2] Marker plane and square")
    print(f"  side L = {scene['marker_side']} m, corners centred on the marker Z=0 plane")
    print("  corner   X (m)      Y (m)      Z (m)")
    for index, point in enumerate(object_points):
        print(f"  {index:<8} {point[0]:+.4f}    {point[1]:+.4f}    {point[2]:+.4f}")

    print("\n[Step 3] Camera intrinsics")
    print(f"  image {scene['image_width']} x {scene['image_height']} px, "
          f"fx={camera.fx:g}, fy={camera.fy:g}, cx={camera.cx:g}, cy={camera.cy:g}")

    print("\n[Step 4] Ground-truth pose (marker -> camera)")
    print("    R_gt =")
    print(_matrix_lines(rotation_gt))
    print(f"    t_gt = [{translation_gt[0]:+.4f}, {translation_gt[1]:+.4f}, "
          f"{translation_gt[2]:+.4f}] m")
    print(f"    marker normal in camera frame = "
          f"{np.array2string(rotation_gt @ np.array([0.0, 0.0, 1.0]), precision=4)}")
    print(f"    viewing angle from fronto-parallel = "
          f"{viewing_angle_deg(rotation_gt, translation_gt):.2f} deg "
          f"(< 90 means the marker faces the camera)")

    print("\n[Step 5] Forward projection")
    print("  corner   X_cam      Y_cam      Z_cam        u (px)     v (px)")
    for index, (camera_point, pixel) in enumerate(zip(scene["points_camera"], image_points)):
        print(
            f"  {index:<8} {camera_point[0]:+.4f}    {camera_point[1]:+.4f}    "
            f"{camera_point[2]:+.4f}     {pixel[0]:8.3f}   {pixel[1]:8.3f}"
        )

    print("\n[Step 6] Pose recovery: both IPPE-square candidates")
    for source, solutions in (
        ("custom", scene["solutions_custom"]),
        ("OpenCV", scene["solutions_cv"]),
    ):
        for index, (rotation, translation, error) in enumerate(solutions, start=1):
            rotation_error, translation_error = pose_error(
                rotation, translation, rotation_gt, translation_gt
            )
            print(f"\n  {source} candidate {index}:  RMSE = {error:.6g} px")
            print("    R =")
            print(_matrix_lines(rotation))
            print(f"    t = [{translation[0]:+.4f}, {translation[1]:+.4f}, "
                  f"{translation[2]:+.4f}] m")
            print(f"    error vs ground truth: {rotation_error:.6g} deg, "
                  f"{translation_error:.6g} mm")

    print("\n[Step 7] Reprojection residuals of the best custom candidate")
    projected = project_points(
        scene["R_custom"], scene["t_custom"], camera.K, object_points
    )
    print("  corner   u_obs      v_obs      u_proj     v_proj      du (px)      dv (px)")
    for index, (observed, estimated) in enumerate(zip(image_points, projected)):
        print(
            f"  {index:<8} {observed[0]:8.3f}   {observed[1]:8.3f}   "
            f"{estimated[0]:8.3f}   {estimated[1]:8.3f}   "
            f"{estimated[0] - observed[0]:+10.3e}   {estimated[1] - observed[1]:+10.3e}"
        )
    print(f"  RMSE over the 8 scalar residuals = {scene['reprojection_error_custom']:.6g} px")

    print("\n[Validation] custom vs OpenCV primary solution")
    rotation_error, translation_error = pose_error(
        scene["R_custom"], scene["t_custom"], scene["R_cv"], scene["t_cv"]
    )
    print(f"  difference: {rotation_error:.6g} deg, {translation_error:.6g} mm")
    validate_noiseless_scene(scene)
    print("  OK: the custom solver matches OpenCV within tolerance.")


def main() -> None:
    """Run the pipeline on the scene described by config.py."""
    _print_baseline(generate_scene())


if __name__ == "__main__":
    main()
