"""Regression tests for the fully synthetic IPPE-square project."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from geometry.camera import make_camera
from geometry.marker import make_square_marker, transform_points
from experiment1 import plot_viewpoint_sweep, run_viewpoint_sweep
from experiment2 import run_conditioning_sweep, run_monte_carlo
from ippe_square import ippe_square
from jacobian import analyze_jacobian, perturb_pose, reprojection_jacobian, reprojection_residuals
from pipeline import generate_scene, make_ground_truth_pose, pose_error, validate_noiseless_scene


class SyntheticIPPEGeometryTests(unittest.TestCase):
    """Marker and projection conventions remain compatible with OpenCV IPPE."""

    def test_canonical_square_order_and_orientation(self) -> None:
        points = make_square_marker(0.10)
        expected = np.array(
            [
                [-0.05, 0.05, 0.0],
                [0.05, 0.05, 0.0],
                [0.05, -0.05, 0.0],
                [-0.05, -0.05, 0.0],
            ]
        )
        np.testing.assert_allclose(points, expected)
        signed_double_area = np.sum(
            points[:, 0] * np.roll(points[:, 1], -1)
            - points[:, 1] * np.roll(points[:, 0], -1)
        )
        self.assertLess(signed_double_area, 0.0, "OpenCV order is clockwise viewed from +Z")

    def test_projection_and_normalization_are_consistent(self) -> None:
        camera = make_camera()
        object_points = make_square_marker()
        rotation, translation = make_ground_truth_pose()
        camera_points = transform_points(object_points, rotation, translation)
        image_points = camera.project(camera_points)
        normalized_expected = camera_points[:, :2] / camera_points[:, 2:3]
        np.testing.assert_allclose(camera.normalize(image_points), normalized_expected, atol=1e-12)
        self.assertTrue(np.all(camera_points[:, 2] > 0.0))


class IPPEValidationTests(unittest.TestCase):
    """The custom solver agrees with OpenCV and keeps both hypotheses."""

    def test_custom_and_opencv_primary_candidates_match_noiseless_scene(self) -> None:
        scene = generate_scene()
        validate_noiseless_scene(scene)
        custom_solutions = scene["solutions_custom"]
        opencv_solutions = scene["solutions_cv"]
        self.assertEqual(len(custom_solutions), 2)
        self.assertEqual(len(opencv_solutions), 2)
        self.assertLessEqual(custom_solutions[0][2], custom_solutions[1][2])

        rotation_error, translation_error_mm = pose_error(
            scene["R_custom"], scene["t_custom"], scene["R_cv"], scene["t_cv"]
        )
        self.assertLess(rotation_error, 1e-5)
        self.assertLess(translation_error_mm, 0.05)
        self.assertLess(custom_solutions[0][2], 1e-8)
        self.assertAlmostEqual(custom_solutions[1][2], opencv_solutions[1][2], delta=2e-4)

    def test_noncanonical_square_is_rejected(self) -> None:
        scene = generate_scene()
        scrambled = scene["object_points"][[0, 3, 2, 1]]
        with self.assertRaises(ValueError):
            ippe_square(scene["camera"], scrambled, scene["image_points"])


class JacobianTests(unittest.TestCase):
    """Analytic local-pose derivatives agree with centred finite differences."""

    def test_jacobian_shape_svd_and_finite_difference(self) -> None:
        scene = generate_scene()
        camera = scene["camera"]
        object_points = scene["object_points"]
        rotation = scene["R_gt"]
        translation = scene["t_gt"]
        image_points = scene["image_points"]
        analytic = reprojection_jacobian(camera, object_points, rotation, translation)
        self.assertEqual(analytic.shape, (8, 6))

        numerical = np.empty_like(analytic)
        epsilon = 1e-7
        for column in range(6):
            delta = np.zeros(6)
            delta[column] = epsilon
            rotation_plus, translation_plus = perturb_pose(rotation, translation, delta)
            rotation_minus, translation_minus = perturb_pose(rotation, translation, -delta)
            numerical[:, column] = (
                reprojection_residuals(
                    camera, object_points, rotation_plus, translation_plus, image_points
                )
                - reprojection_residuals(
                    camera, object_points, rotation_minus, translation_minus, image_points
                )
            ) / (2.0 * epsilon)
        np.testing.assert_allclose(analytic, numerical, rtol=2e-6, atol=2e-6)

        diagnostics = analyze_jacobian(camera, object_points, rotation, translation, 0.5)
        self.assertEqual(diagnostics.jacobian.shape, (8, 6))
        self.assertEqual(diagnostics.singular_values.shape, (6,))
        self.assertGreater(diagnostics.smallest_singular_value, 0.0)
        self.assertGreater(diagnostics.condition_number, 1.0)


class ExperimentTests(unittest.TestCase):
    """Small deterministic experiment runs exercise the public interfaces."""

    def test_sweep_keeps_two_candidates_and_writes_plot(self) -> None:
        results = run_viewpoint_sweep(
            make_camera(),
            make_square_marker(),
            radius_m=0.5,
            min_angle_deg=5.0,
            max_angle_deg=60.0,
            samples=5,
        )
        self.assertEqual(len(results), 5)
        self.assertTrue(
            all(result.candidate_rmse_px[0] <= result.candidate_rmse_px[1] for result in results)
        )
        with tempfile.TemporaryDirectory() as directory:
            plot_path = plot_viewpoint_sweep(results, Path(directory) / "sweep.png")
            self.assertTrue(plot_path.is_file())
            self.assertGreater(plot_path.stat().st_size, 0)

    def test_conditioning_sweep_matches_fronto_parallel_closed_form(self) -> None:
        # Head-on, translation in units of Z: sigma_min = 2 f h^2 / Z^2 along an
        # out-of-plane tilt, and kappa = (Z / h)^2.
        f, half_side, distance = 800.0, 0.05, 0.5
        results = run_conditioning_sweep(
            make_camera(), make_square_marker(), distance, 0.0, 60.0, 7
        )
        head_on = results[0]
        self.assertAlmostEqual(
            head_on.singular_values[-1], 2 * f * half_side**2 / distance**2, delta=1e-2
        )
        self.assertAlmostEqual(head_on.condition_number, (distance / half_side) ** 2, delta=0.1)
        self.assertGreater(max(abs(head_on.weakest_direction[3:5])), 0.99)
        for result in results:
            self.assertTrue(np.all(np.diff(result.singular_values) <= 0.0))
            np.testing.assert_allclose(
                result.right_singular_vectors @ result.right_singular_vectors.T,
                np.eye(6),
                atol=1e-12,
            )
            self.assertAlmostEqual(
                result.condition_number,
                result.singular_values[0] / result.singular_values[-1],
            )
        smallest = [result.singular_values[-1] for result in results]
        self.assertTrue(np.all(np.diff(smallest) > 0.0), "obliquity should improve sigma_min")

    def test_monte_carlo_is_seeded_and_returns_covariances(self) -> None:
        kwargs = dict(
            camera=make_camera(),
            object_points=make_square_marker(),
            radius_m=0.5,
            viewing_angle_deg=60.0,
            noise_std_px=0.5,
            trials=60,
            seed=12345,
        )
        first = run_monte_carlo(**kwargs)
        second = run_monte_carlo(**kwargs)
        np.testing.assert_allclose(first.empirical_covariance, second.empirical_covariance)
        self.assertEqual(first.empirical_covariance.shape, (6, 6))
        self.assertEqual(first.predicted_covariance.shape, (6, 6))
        self.assertTrue(np.all(first.empirical_variance >= 0.0))
        self.assertTrue(np.all(first.predicted_variance >= 0.0))


if __name__ == "__main__":
    unittest.main()
