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
from jacobian import (
    JacobianDiagnostics,
    analyze_jacobian,
    jacobian_svd,
    reprojection_jacobian,
    so3_log,
)
from viewpoint import make_front_arc_pose, project_scene


POSE_PARAMETERS = ("tx", "ty", "tz", "omega_x", "omega_y", "omega_z")


@dataclass(frozen=True)
class ConditioningResult:
    """The Jacobian's SVD at one viewpoint of the arc."""

    viewing_angle_deg: float
    singular_values: np.ndarray
    right_singular_vectors: np.ndarray
    condition_number: float

    @property
    def weakest_direction(self) -> np.ndarray:
        """Least observable pose direction (right singular vector of sigma_6)."""
        return self.right_singular_vectors[-1]

    @property
    def second_weakest_direction(self) -> np.ndarray:
        """Right singular vector of sigma_5."""
        return self.right_singular_vectors[-2]


def run_conditioning_sweep(
    camera: Camera,
    object_points: np.ndarray,
    radius_m: float,
    min_angle_deg: float,
    max_angle_deg: float,
    samples: int,
    azimuth_deg: float = 0.0,
) -> list[ConditioningResult]:
    """Singular values, singular vectors and condition number at every angle of the arc.

    The Jacobian is evaluated at the true pose of each viewpoint, on exact data,
    which is where the first-order analysis applies. Translation is measured in
    units of the camera-marker distance, so all six parameters are dimensionless:
    singular values come out in pixels and the condition number is a pure number
    that depends on the geometry, not on the choice of units. Rotation directions
    are unaffected by this scaling.

    Near 0 degrees the two smallest singular values are nearly equal, so sigma_5
    and sigma_6 can swap their vectors from one angle to the next; the plane they
    span is stable.
    """
    if samples < 2:
        raise ValueError("samples must be at least two")
    results: list[ConditioningResult] = []
    for angle in np.linspace(min_angle_deg, max_angle_deg, samples):
        rotation, translation = make_front_arc_pose(radius_m, float(angle), azimuth_deg)
        distance = float(np.linalg.norm(translation))
        jacobian = reprojection_jacobian(camera, object_points, rotation, translation)
        scaled = jacobian @ np.diag([distance, distance, distance, 1.0, 1.0, 1.0])
        singular_values, right_singular_vectors, condition = jacobian_svd(scaled)
        results.append(
            ConditioningResult(
                viewing_angle_deg=float(angle),
                singular_values=singular_values,
                right_singular_vectors=right_singular_vectors,
                condition_number=condition,
            )
        )
    return results


def plot_conditioning_sweep(results: list[ConditioningResult], output_path: str | Path) -> Path:
    """Singular values, condition number, and what the weakest direction is made of."""
    if not results:
        raise ValueError("Cannot plot an empty conditioning sweep")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    angles = np.array([result.viewing_angle_deg for result in results])
    singular_values = np.array([result.singular_values for result in results])
    conditions = np.array([result.condition_number for result in results])
    weakest_share = np.array([result.weakest_direction**2 for result in results])

    figure, axes = plt.subplots(3, 1, figsize=(7.5, 11.5), constrained_layout=True)
    # Two pairs coincide (sigma_1 = sigma_2 within 0.02 %, sigma_3 = sigma_4 within
    # 3 %), so one line of each pair is shifted purely so that both can be seen;
    # the note on the plot says so. sigma_3 goes up rather than sigma_4 down, so
    # the drawn order never crosses sigma_5 and sigma_6.
    offsets = {1: 0.9, 2: 1.1}
    for k in range(6):
        factor = offsets.get(k, 1.0)
        label = rf"$\sigma_{k + 1}$" + (rf" (drawn $\times{factor}$)" if k in offsets else "")
        axes[0].plot(angles, singular_values[:, k] * factor, label=label)
    axes[0].text(
        0.02, 0.78,
        r"$\sigma_1 = \sigma_2$ and $\sigma_3 \approx \sigma_4$: $\sigma_2$ is drawn 10% lower and"
        r" $\sigma_3$ 10% higher, only for visibility."
        "\n"
        r"$\sigma_5 = \sigma_6$ at $0^\circ$ and separate as the view becomes oblique.",
        transform=axes[0].transAxes, fontsize="small", va="center",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$\sigma_k$")
    axes[0].set_title("Singular values")
    axes[0].legend(ncol=6, fontsize="small", loc="upper center", bbox_to_anchor=(0.5, -0.2))

    axes[1].plot(angles, conditions, color="black")
    axes[1].set_ylabel(r"$\kappa = \sigma_1 / \sigma_6$")
    axes[1].set_title("Condition number")

    labels = [r"$t_x$", r"$t_y$", r"$t_z$", r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]
    axes[2].stackplot(angles, weakest_share.T, labels=labels)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_ylabel("Share of each parameter")
    axes[2].set_title("Least observable direction (weakest singular vector)")
    axes[2].text(
        0.02, 0.05,
        r"$t_x$, $t_y$, $t_z$ and $\omega_y$ together stay below 0.01% at every angle,"
        "\n"
        r"so only $\omega_x$ and $\omega_z$ are visible.",
        transform=axes[2].transAxes, fontsize="small", va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    axes[2].legend(ncol=6, fontsize="small", loc="upper center", bbox_to_anchor=(0.5, -0.2))

    for axis in axes:
        axis.set_xlim(0.0, 90.0)
        axis.set_xticks(np.arange(0, 91, 10))
        axis.set_xlabel("Viewing angle from fronto-parallel (degrees)")
        axis.grid(True, alpha=0.3)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def write_conditioning_csv(results: list[ConditioningResult], output_path: str | Path) -> Path:
    """One row per angle: all six singular values, kappa, and the two weakest directions.

    Translation is in units of the camera-marker distance, so singular values
    are in pixels and the condition number is dimensionless.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "viewing_angle_deg",
                *[f"sigma_{k}" for k in range(1, 7)],
                "condition_number",
                *[f"v6_{name}" for name in POSE_PARAMETERS],
                *[f"v5_{name}" for name in POSE_PARAMETERS],
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.viewing_angle_deg,
                    *result.singular_values,
                    result.condition_number,
                    *result.weakest_direction,
                    *result.second_weakest_direction,
                ]
            )
    return output


def describe_direction(direction: np.ndarray, terms: int = 2) -> str:
    """Readable form of a pose direction, e.g. '+0.967 omega_x -0.256 omega_z'."""
    order = np.argsort(-np.abs(direction))[:terms]
    return " ".join(f"{direction[i]:+.3f} {POSE_PARAMETERS[i]}" for i in order)


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

    sweep = run_conditioning_sweep(
        make_camera(),
        make_square_marker(),
        config.SWEEP_RADIUS_M,
        config.SWEEP_MIN_ANGLE_DEG,
        config.SWEEP_MAX_ANGLE_DEG,
        config.SWEEP_SAMPLES,
        config.SWEEP_AZIMUTH_DEG,
    )
    sweep_plot = plot_conditioning_sweep(sweep, args.output_dir / "exp2_conditioning.png")
    sweep_csv = write_conditioning_csv(sweep, args.output_dir / "exp2_conditioning.csv")
    print("Experiment 2 --- Jacobian conditioning along the arc")
    print(f"  wrote {sweep_plot}")
    print(f"  wrote {sweep_csv}")
    print("  (translation in units of the camera-marker distance: kappa is dimensionless)")
    print("  angle   sigma_min    kappa    weakest direction")
    for result in sweep:
        if result.viewing_angle_deg in (0.0, 5.0, 15.0, 30.0, 45.0, 60.0, 75.0, 89.0):
            print(
                f"  {result.viewing_angle_deg:4.0f}°  {result.singular_values[-1]:9.3f}  "
                f"{result.condition_number:8.2f}   {describe_direction(result.weakest_direction)}"
            )
    print()

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
