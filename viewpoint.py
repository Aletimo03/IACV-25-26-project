"""Where we put the camera: the (radius, viewing angle, azimuth) arc.

Both experiments move the camera the same way, so the parametrisation lives
here rather than in either of them.
"""

from __future__ import annotations

import numpy as np

from geometry.camera import Camera


def make_front_arc_pose(
    radius_m: float,
    viewing_angle_deg: float,
    azimuth_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pose of a marker seen by a camera sitting on an arc around it.

    ``viewing_angle_deg`` is measured between the marker normal and the line
    from marker to camera: 0 is fronto-parallel, larger values are more
    oblique. The camera stays on the front hemisphere (so it sees the face of
    the marker), always looks at the marker centre, and keeps image +Y down.
    90 degrees is excluded because there the camera lies in the marker plane
    and the square degenerates to a segment.
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
        # Only reachable if forward ever became parallel to marker_up.
        marker_up = np.array([1.0, 0.0, 0.0])
        right_marker = np.cross(forward_marker, marker_up)
    right_marker /= np.linalg.norm(right_marker)
    down_marker = np.cross(forward_marker, right_marker)

    # Columns are the camera axes written in marker coordinates, so the
    # transpose is what takes a marker point into the camera frame.
    camera_to_marker = np.column_stack([right_marker, down_marker, forward_marker])
    rotation_marker_to_camera = camera_to_marker.T
    translation_marker_to_camera = -rotation_marker_to_camera @ camera_center_marker
    return rotation_marker_to_camera, translation_marker_to_camera


def project_scene(
    camera: Camera,
    object_points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Project the corners, refusing poses that put the marker behind us."""
    points_camera = (rotation @ object_points.T).T + translation
    if np.any(points_camera[:, 2] <= 0.0):
        raise ValueError("The synthetic marker must remain in front of the camera")
    return camera.project(points_camera)
