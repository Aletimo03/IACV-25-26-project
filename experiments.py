"""Synthetic viewpoint-sweep and Monte Carlo experiments for IPPE-square."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from camera import Camera
from ippe_square import ippe_square
from jacobian import JacobianDiagnostics, analyze_jacobian, so3_log


def make_front_arc_pose(
    radius_m: float,
    viewing_angle_deg: float,
    azimuth_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a marker-to-camera pose for a camera looking at the marker centre.

    ``viewing_angle_deg`` is the angle between the marker's positive normal and
    the marker-to-camera line: 0° is fronto-parallel and increasing values are
    more oblique.  The camera remains on the front hemisphere, with its +Z
    axis pointing from the camera to the marker and its image +Y axis down.
    """
    if radius_m <= 0.0:
        raise ValueError("radius_m must be positive")
    if not 0.0 <= viewing_angle_deg < 90.0:
        raise ValueError("viewing_angle_deg must lie in [0, 90)")

    elevation = np.deg2rad(viewing_angle_deg)
    azimuth = np.deg2rad(azimuth_deg)
    camera_center_marker = radius_m * np.array(
        [
            np.sin(elevation) * np.cos(azimuth),
            np.sin(elevation) * np.sin(azimuth),
            np.cos(elevation),
        ]
    )

    forward_marker = -camera_center_marker / radius_m
    marker_up = np.array([0.0, 1.0, 0.0])
    right_marker = np.cross(forward_marker, marker_up)
    if np.linalg.norm(right_marker) < 1e-12:
        # This fallback only matters for unsupported near-pole azimuths.
        marker_up = np.array([1.0, 0.0, 0.0])
        right_marker = np.cross(forward_marker, marker_up)
    right_marker /= np.linalg.norm(right_marker)
    down_marker = np.cross(forward_marker, right_marker)

    # Columns are camera axes expressed in marker coordinates, hence transpose
    # maps a marker-frame point into the camera frame.
    camera_to_marker = np.column_stack([right_marker, down_marker, forward_marker])
    rotation_marker_to_camera = camera_to_marker.T
    translation_marker_to_camera = -rotation_marker_to_camera @ camera_center_marker
    return rotation_marker_to_camera, translation_marker_to_camera


def _project_scene(
    camera: Camera,
    object_points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Project a valid synthetic pose and reject a marker behind the camera."""
    points_camera = (rotation @ object_points.T).T + translation
    if np.any(points_camera[:, 2] <= 0.0):
        raise ValueError("The synthetic marker must remain in front of the camera")
    return camera.project(points_camera)


def _pose_errors(
    rotation: np.ndarray,
    translation: np.ndarray,
    reference_rotation: np.ndarray,
    reference_translation: np.ndarray,
) -> tuple[float, float]:
    """Return geodesic rotation error in degrees and translation error in metres."""
    rotation_error_rad = np.linalg.norm(so3_log(reference_rotation.T @ rotation))
    translation_error_m = np.linalg.norm(translation - reference_translation)
    return float(np.rad2deg(rotation_error_rad)), float(translation_error_m)


@dataclass(frozen=True)
class ViewpointResult:
    """Solver and conditioning measurements for one camera viewpoint."""

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
    """Solve both IPPE candidates along a front-facing camera arc."""
    if samples < 2:
        raise ValueError("samples must be at least two")
    angles = np.linspace(min_angle_deg, max_angle_deg, samples)
    results: list[ViewpointResult] = []
    for angle in angles:
        rotation_gt, translation_gt = make_front_arc_pose(radius_m, float(angle), azimuth_deg)
        image_points = _project_scene(camera, object_points, rotation_gt, translation_gt)
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
    """Save the required two-candidate reprojection-error-versus-angle plot."""
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
    """Write sweep measurements for plotting or inspection outside Python."""
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


@dataclass(frozen=True)
class MonteCarloSummary:
    """Empirical and first-order local-pose uncertainty for one viewpoint."""

    viewing_angle_deg: float
    trials: int
    noise_std_px: float
    empirical_covariance: np.ndarray
    predicted_covariance: np.ndarray
    singular_values: np.ndarray
    smallest_singular_value: float
    condition_number: float

    @property
    def empirical_variance(self) -> np.ndarray:
        """Diagonal of the sampled local-pose covariance."""
        return np.diag(self.empirical_covariance)

    @property
    def predicted_variance(self) -> np.ndarray:
        """Diagonal of the Jacobian-based covariance prediction."""
        return np.diag(self.predicted_covariance)


def run_monte_carlo(
    camera: Camera,
    object_points: np.ndarray,
    radius_m: float,
    viewing_angle_deg: float,
    noise_std_px: float,
    trials: int,
    seed: int,
    azimuth_deg: float = 0.0,
) -> MonteCarloSummary:
    """Estimate pose repeatedly after adding independent corner-pixel noise."""
    if trials < 2:
        raise ValueError("At least two trials are needed for an empirical variance")
    if noise_std_px < 0.0:
        raise ValueError("noise_std_px must be non-negative")

    rotation_gt, translation_gt = make_front_arc_pose(radius_m, viewing_angle_deg, azimuth_deg)
    image_points = _project_scene(camera, object_points, rotation_gt, translation_gt)
    diagnostics: JacobianDiagnostics = analyze_jacobian(
        camera, object_points, rotation_gt, translation_gt, noise_std_px
    )
    random = np.random.default_rng(seed)
    local_errors = np.empty((trials, 6), dtype=np.float64)
    for trial in range(trials):
        noisy_points = image_points + random.normal(0.0, noise_std_px, size=image_points.shape)
        rotation_estimate, translation_estimate, _, _ = ippe_square(
            camera, object_points, noisy_points
        )
        local_errors[trial, :3] = translation_estimate - translation_gt
        local_errors[trial, 3:] = so3_log(rotation_gt.T @ rotation_estimate)

    return MonteCarloSummary(
        viewing_angle_deg=viewing_angle_deg,
        trials=trials,
        noise_std_px=noise_std_px,
        empirical_covariance=np.cov(local_errors, rowvar=False, ddof=1),
        predicted_covariance=diagnostics.predicted_covariance,
        singular_values=diagnostics.singular_values,
        smallest_singular_value=diagnostics.smallest_singular_value,
        condition_number=diagnostics.condition_number,
    )


def run_monte_carlo_experiment(
    camera: Camera,
    object_points: np.ndarray,
    radius_m: float,
    viewing_angles_deg: tuple[float, ...],
    noise_std_px: float,
    trials: int,
    seed: int,
    azimuth_deg: float = 0.0,
) -> list[MonteCarloSummary]:
    """Run deterministic, independent Monte Carlo studies at several angles."""
    seed_sequence = np.random.SeedSequence(seed)
    child_sequences = seed_sequence.spawn(len(viewing_angles_deg))
    return [
        run_monte_carlo(
            camera,
            object_points,
            radius_m,
            angle,
            noise_std_px,
            trials,
            int(child_sequence.generate_state(1)[0]),
            azimuth_deg,
        )
        for angle, child_sequence in zip(viewing_angles_deg, child_sequences, strict=True)
    ]


def plot_monte_carlo_comparison(
    summaries: list[MonteCarloSummary], output_path: str | Path) -> Path:
    """Save empirical-versus-Jacobian local-pose variance bar charts."""
    if not summaries:
        raise ValueError("Cannot plot an empty Monte Carlo result")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        1, len(summaries), figsize=(6.0 * len(summaries), 4.5), constrained_layout=True
    )
    if len(summaries) == 1:
        axes = [axes]
    labels = [r"$t_x$", r"$t_y$", r"$t_z$", r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]
    positions = np.arange(6)
    for axis, summary in zip(axes, summaries, strict=True):
        empirical = np.maximum(summary.empirical_variance, np.finfo(float).tiny)
        predicted = np.maximum(summary.predicted_variance, np.finfo(float).tiny)
        axis.bar(positions - 0.2, empirical, width=0.4, label="empirical")
        axis.bar(positions + 0.2, predicted, width=0.4, label="Jacobian prediction")
        axis.set_yscale("log")
        axis.set_xticks(positions, labels)
        axis.set_ylabel("Variance (m² for t, rad² for ω)")
        axis.set_title(
            f"{summary.viewing_angle_deg:g}°: cond(J)={summary.condition_number:.2e}"
        )
        axis.grid(True, axis="y", alpha=0.3)
        axis.legend()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def write_monte_carlo_csv(summaries: list[MonteCarloSummary], output_path: str | Path) -> Path:
    """Write empirical and predicted variances plus conditioning diagnostics."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    names = ["tx", "ty", "tz", "omega_x", "omega_y", "omega_z"]
    with output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "viewing_angle_deg",
                "trials",
                "noise_std_px",
                "smallest_singular_value",
                "condition_number",
                *[f"empirical_var_{name}" for name in names],
                *[f"predicted_var_{name}" for name in names],
            ]
        )
        for summary in summaries:
            writer.writerow(
                [
                    summary.viewing_angle_deg,
                    summary.trials,
                    summary.noise_std_px,
                    summary.smallest_singular_value,
                    summary.condition_number,
                    *summary.empirical_variance,
                    *summary.predicted_variance,
                ]
            )
    return output
