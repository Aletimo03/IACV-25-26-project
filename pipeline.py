"""Command-line orchestration for synthetic IPPE-square validation experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import config
from camera import make_camera
from experiments import (
    plot_monte_carlo_comparison,
    plot_viewpoint_sweep,
    run_monte_carlo_experiment,
    run_viewpoint_sweep,
    write_monte_carlo_csv,
    write_viewpoint_sweep_csv,
)
from ippe_square import ippe_square
from jacobian import analyze_jacobian
from marker import make_square_marker, transform_points


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
    """Build the configured marker-to-camera ground-truth pose.

    SciPy's lower-case ``xyz`` sequence represents fixed-axis (extrinsic)
    Euler rotations.  Euler angles are used only to specify a readable scene;
    all estimation and uncertainty calculations use rotation matrices and
    local rotation vectors.
    """
    rotation = Rotation.from_euler("xyz", euler_xyz_deg, degrees=True).as_matrix()
    return rotation, np.asarray(translation_m, dtype=np.float64)


def _opencv_ippe_square_solutions(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Run OpenCV only as the black-box validation reference."""
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
    """Generate the default scene and validate custom IPPE against OpenCV.

    The pose always maps marker-frame coordinates to the camera frame:
    ``P_cam = R @ P_marker + t``.  Image observations are generated directly
    from the pinhole model, so no detector, image processing, or distortion is
    involved.
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
    diagnostics = analyze_jacobian(
        camera,
        object_points,
        rotation_gt,
        translation_gt,
        config.CORNER_NOISE_STD_PX,
    )
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
        "jacobian_diagnostics": diagnostics,
    }


def pose_error(
    rotation: np.ndarray,
    translation: np.ndarray,
    rotation_reference: np.ndarray,
    translation_reference: np.ndarray,
) -> tuple[float, float]:
    """Return geodesic rotation error (degrees) and translation error (mm)."""
    relative_rotation = rotation_reference.T @ rotation
    cosine = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
    rotation_error_deg = float(np.rad2deg(np.arccos(cosine)))
    translation_error_mm = float(np.linalg.norm(translation - translation_reference) * 1000.0)
    return rotation_error_deg, translation_error_mm


def validate_noiseless_scene(scene: dict[str, object]) -> None:
    """Raise if the synthetic custom/OpenCV primary solutions disagree."""
    rotation_error_deg, translation_error_mm = pose_error(
        scene["R_custom"], scene["t_custom"], scene["R_cv"], scene["t_cv"]
    )
    if rotation_error_deg > 1e-5 or translation_error_mm > 0.05:
        raise AssertionError(
            "Custom IPPE-square does not agree with OpenCV in the noiseless scene: "
            f"{rotation_error_deg:.3e} deg, {translation_error_mm:.3e} mm"
        )


def _print_baseline(scene: dict[str, object]) -> None:
    """Print a concise, reproducible validation summary for the default scene."""
    validate_noiseless_scene(scene)
    rotation_gt = scene["R_gt"]
    translation_gt = scene["t_gt"]
    print("Synthetic IPPE-square baseline")
    print("  Convention: P_cam = R @ P_marker + t; camera +Z is forward.")
    print("  Object corners follow OpenCV's canonical IPPE_SQUARE order.")
    print("\n  candidate     source       RMSE (px)     rot. error (deg)    trans. error (mm)")
    for source, solutions in (
        ("custom", scene["solutions_custom"]),
        ("OpenCV", scene["solutions_cv"]),
    ):
        for index, (rotation, translation, error) in enumerate(solutions, start=1):
            rotation_error, translation_error = pose_error(
                rotation, translation, rotation_gt, translation_gt
            )
            print(
                f"  {index:<13} {source:<10} {error:>11.6g}"
                f" {rotation_error:>19.9g} {translation_error:>20.9g}"
            )

    diagnostics = scene["jacobian_diagnostics"]
    print("\n  Jacobian diagnostics at the ground truth")
    print(f"  shape: {diagnostics.jacobian.shape}")
    print("  singular values:", np.array2string(diagnostics.singular_values, precision=5))
    print(f"  smallest singular value: {diagnostics.smallest_singular_value:.6g}")
    print(f"  condition number: {diagnostics.condition_number:.6g}")


def _run_sweep(output_directory: Path) -> None:
    """Run and persist the configured two-candidate viewpoint sweep."""
    camera = make_camera()
    object_points = make_square_marker()
    results = run_viewpoint_sweep(
        camera,
        object_points,
        config.SWEEP_RADIUS_M,
        config.SWEEP_MIN_ANGLE_DEG,
        config.SWEEP_MAX_ANGLE_DEG,
        config.SWEEP_SAMPLES,
        config.CORNER_NOISE_STD_PX,
        config.SWEEP_AZIMUTH_DEG,
    )
    plot_path = plot_viewpoint_sweep(results, output_directory / "viewpoint_sweep.png")
    csv_path = write_viewpoint_sweep_csv(results, output_directory / "viewpoint_sweep.csv")
    print("\nViewpoint sweep")
    print(f"  wrote {plot_path}")
    print(f"  wrote {csv_path}")
    print(
        "  candidate-2 RMSE: "
        f"{results[0].candidate_rmse_px[1]:.6g} px at {results[0].viewing_angle_deg:g}° "
        f"→ {results[-1].candidate_rmse_px[1]:.6g} px at {results[-1].viewing_angle_deg:g}°"
    )


def _run_monte_carlo(output_directory: Path) -> None:
    """Run and persist the configured empirical-versus-Jacobian noise study."""
    summaries = run_monte_carlo_experiment(
        make_camera(),
        make_square_marker(),
        config.SWEEP_RADIUS_M,
        config.MONTE_CARLO_VIEWING_ANGLES_DEG,
        config.CORNER_NOISE_STD_PX,
        config.MONTE_CARLO_TRIALS,
        config.MONTE_CARLO_SEED,
        config.SWEEP_AZIMUTH_DEG,
    )
    plot_path = plot_monte_carlo_comparison(summaries, output_directory / "monte_carlo_variance.png")
    csv_path = write_monte_carlo_csv(summaries, output_directory / "monte_carlo_summary.csv")
    print("\nMonte Carlo (independent Gaussian noise per image coordinate)")
    print(f"  wrote {plot_path}")
    print(f"  wrote {csv_path}")
    for summary in summaries:
        print(
            f"  {summary.viewing_angle_deg:g}°: sigma_min={summary.smallest_singular_value:.6g}, "
            f"cond(J)={summary.condition_number:.6g}"
        )
        print("    empirical variance:", np.array2string(summary.empirical_variance, precision=4))
        print("    predicted variance:", np.array2string(summary.predicted_variance, precision=4))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true", help="run the noiseless custom/OpenCV validation")
    parser.add_argument("--sweep", action="store_true", help="run the viewpoint sweep and save its plot")
    parser.add_argument("--monte-carlo", action="store_true", help="run the Gaussian-noise study and save its plot")
    parser.add_argument("--all", action="store_true", help="run the baseline and both experiments")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(config.OUTPUT_DIRECTORY),
        help=f"directory for generated plots and CSV files (default: {config.OUTPUT_DIRECTORY})",
    )
    return parser.parse_args()


def main() -> None:
    """Run the requested synthetic validation workflow."""
    args = _parse_args()
    run_baseline = args.baseline or args.all or not (args.sweep or args.monte_carlo)
    run_sweep = args.sweep or args.all
    run_monte_carlo = args.monte_carlo or args.all
    if run_baseline:
        _print_baseline(generate_scene())
    if run_sweep or run_monte_carlo:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if run_sweep:
        _run_sweep(args.output_dir)
    if run_monte_carlo:
        _run_monte_carlo(args.output_dir)


if __name__ == "__main__":
    main()
