"""Experiment 2: conditioning of the Jacobian, checked against Monte Carlo.

The 8x6 residual Jacobian at a viewpoint gives us singular values, a smallest
singular value and a condition number, which say how observable each direction
of the pose is. We then perturb the corners with Gaussian noise many times,
re-solve, and compare the covariance we actually measure with the first-order
prediction sigma_px^2 (J^T J)^-1.
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
from jacobian import JacobianDiagnostics, analyze_jacobian, so3_log
from viewpoint import make_front_arc_pose, project_scene


@dataclass(frozen=True)
class MonteCarloSummary:
    """Measured and predicted pose uncertainty at a single viewpoint."""

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
        """Just the diagonal of the measured covariance."""
        return np.diag(self.empirical_covariance)

    @property
    def predicted_variance(self) -> np.ndarray:
        """Just the diagonal of the predicted covariance."""
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
    """Re-solve the pose many times with fresh noise on the corners.

    Each trial perturbs the eight image coordinates independently, solves, and
    records the error as a local increment: translation straight up, rotation
    through the log map, so both match the Jacobian's parametrisation.
    """
    if trials < 2:
        raise ValueError("At least two trials are needed for an empirical variance")
    if noise_std_px < 0.0:
        raise ValueError("noise_std_px must be non-negative")

    rotation_gt, translation_gt = make_front_arc_pose(radius_m, viewing_angle_deg, azimuth_deg)
    image_points = project_scene(camera, object_points, rotation_gt, translation_gt)
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
    """One Monte Carlo per viewing angle, each with its own seed.

    Seeds are spawned from a single SeedSequence so the angles stay
    independent and the whole thing remains reproducible.
    """
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
    """Bar chart per angle: measured variance next to predicted variance."""
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
    """Same numbers as the plot, in CSV form."""
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


def main() -> None:
    """Run the noise study with the settings from config.py and save it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(config.OUTPUT_DIRECTORY),
        help=f"directory for generated plots and CSV files (default: {config.OUTPUT_DIRECTORY})",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
    plot_path = plot_monte_carlo_comparison(summaries, args.output_dir / "monte_carlo_variance.png")
    csv_path = write_monte_carlo_csv(summaries, args.output_dir / "monte_carlo_summary.csv")
    print("Experiment 2 --- Jacobian conditioning and Monte Carlo")
    print(f"  wrote {plot_path}")
    print(f"  wrote {csv_path}")
    for summary in summaries:
        print(
            f"  {summary.viewing_angle_deg:g}°: sigma_min={summary.smallest_singular_value:.6g}, "
            f"cond(J)={summary.condition_number:.6g}"
        )
        print("    empirical variance:", np.array2string(summary.empirical_variance, precision=4))
        print("    predicted variance:", np.array2string(summary.predicted_variance, precision=4))


if __name__ == "__main__":
    main()
