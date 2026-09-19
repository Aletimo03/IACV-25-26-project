"""Experiment 1: how the two IPPE poses separate as the view gets oblique.

We walk the camera along an arc around the marker, keeping it aimed at the
centre, and at every viewpoint we keep *both* candidates and record their
reprojection errors. Fronto-parallel the two poses are the same; the question
is how much tilt it takes before the image can tell them apart.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config
from geometry.camera import Camera, make_camera
from geometry.marker import make_square_marker
from ippe_square import ippe_square
from jacobian import analyze_jacobian, so3_log
from viewpoint import make_front_arc_pose, project_scene


def _pose_errors(
    rotation: np.ndarray,
    translation: np.ndarray,
    reference_rotation: np.ndarray,
    reference_translation: np.ndarray,
) -> tuple[float, float]:
    """Rotation error in degrees (geodesic) and translation error in metres."""
    rotation_error_rad = np.linalg.norm(so3_log(reference_rotation.T @ rotation))
    translation_error_m = np.linalg.norm(translation - reference_translation)
    return float(np.rad2deg(rotation_error_rad)), float(translation_error_m)


@dataclass(frozen=True)
class ViewpointResult:
    """What we measured at a single viewpoint."""

    viewing_angle_deg: float
    candidate_rmse_px: tuple[float, float]
    candidate_rotation_error_deg: tuple[float, float]
    candidate_translation_error_m: tuple[float, float]
    singular_values: np.ndarray
    smallest_singular_value: float
    condition_number: float


def run_viewpoint_sweep(
    camera: Camera,
    object_points: np.ndarray,
    radius_m: float,
    min_angle_deg: float,
    max_angle_deg: float,
    samples: int,
    noise_std_px: float,
    azimuth_deg: float = 0.0,
) -> list[ViewpointResult]:
    """Run the solver at every angle of the arc and keep both candidates.

    ``noise_std_px`` is not added to the image points -- this sweep is exact.
    It only sets the scale of the covariance in the conditioning columns.
    """
    if samples < 2:
        raise ValueError("samples must be at least two")
    angles = np.linspace(min_angle_deg, max_angle_deg, samples)
    results: list[ViewpointResult] = []
    for angle in angles:
        rotation_gt, translation_gt = make_front_arc_pose(radius_m, float(angle), azimuth_deg)
        image_points = project_scene(camera, object_points, rotation_gt, translation_gt)
        _, _, _, info = ippe_square(camera, object_points, image_points)
        candidates = info["solutions"]
        if len(candidates) != 2:
            raise RuntimeError("IPPE-square must produce exactly two candidates")

        errors = tuple(
            _pose_errors(rotation, translation, rotation_gt, translation_gt)
            for rotation, translation, _ in candidates
        )
        diagnostics = analyze_jacobian(
            camera, object_points, rotation_gt, translation_gt, noise_std_px
        )
        results.append(
            ViewpointResult(
                viewing_angle_deg=float(angle),
                candidate_rmse_px=(float(candidates[0][2]), float(candidates[1][2])),
                candidate_rotation_error_deg=(errors[0][0], errors[1][0]),
                candidate_translation_error_m=(errors[0][1], errors[1][1]),
                singular_values=diagnostics.singular_values,
                smallest_singular_value=diagnostics.smallest_singular_value,
                condition_number=diagnostics.condition_number,
            )
        )
    return results


def plot_viewpoint_sweep(results: list[ViewpointResult], output_path: str | Path) -> Path:
    """The plot of the experiment: both candidates' RMSE against the angle."""
    if not results:
        raise ValueError("Cannot plot an empty viewpoint sweep")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    angles = np.array([result.viewing_angle_deg for result in results])
    errors = np.array([result.candidate_rmse_px for result in results])

    figure, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    axis.plot(angles, errors[:, 0], marker="o", markersize=3, label="candidate 1 (best)")
    axis.plot(angles, errors[:, 1], marker="o", markersize=3, label="candidate 2")
    axis.set_xlabel("Viewing angle from fronto-parallel (degrees)")
    axis.set_ylabel("Reprojection RMSE (px, OpenCV convention)")
    axis.set_title("IPPE-square candidate reprojection errors")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def write_viewpoint_sweep_csv(results: list[ViewpointResult], output_path: str | Path) -> Path:
    """Dump the same numbers to CSV, for the report or for a quick look."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "viewing_angle_deg",
                "candidate_1_rmse_px",
                "candidate_2_rmse_px",
                "candidate_1_rotation_error_deg",
                "candidate_2_rotation_error_deg",
                "candidate_1_translation_error_m",
                "candidate_2_translation_error_m",
                "smallest_singular_value",
                "condition_number",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.viewing_angle_deg,
                    *result.candidate_rmse_px,
                    *result.candidate_rotation_error_deg,
                    *result.candidate_translation_error_m,
                    result.smallest_singular_value,
                    result.condition_number,
                ]
            )
    return output


def main() -> None:
    """Run the sweep with the settings from config.py and save the outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(config.OUTPUT_DIRECTORY),
        help=f"directory for generated plots and CSV files (default: {config.OUTPUT_DIRECTORY})",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = run_viewpoint_sweep(
        make_camera(),
        make_square_marker(),
        config.SWEEP_RADIUS_M,
        config.SWEEP_MIN_ANGLE_DEG,
        config.SWEEP_MAX_ANGLE_DEG,
        config.SWEEP_SAMPLES,
        config.CORNER_NOISE_STD_PX,
        config.SWEEP_AZIMUTH_DEG,
    )
    plot_path = plot_viewpoint_sweep(results, args.output_dir / "viewpoint_sweep.png")
    csv_path = write_viewpoint_sweep_csv(results, args.output_dir / "viewpoint_sweep.csv")
    print("Experiment 1 --- viewpoint sweep")
    print(f"  wrote {plot_path}")
    print(f"  wrote {csv_path}")
    print(
        "  candidate-2 RMSE: "
        f"{results[0].candidate_rmse_px[1]:.6g} px at {results[0].viewing_angle_deg:g}° "
        f"→ {results[-1].candidate_rmse_px[1]:.6g} px at {results[-1].viewing_angle_deg:g}°"
    )


if __name__ == "__main__":
    main()
