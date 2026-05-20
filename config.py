"""
Global configuration for the synthetic pose-estimation pipeline (steps 1-5).

Edit the UPPERCASE constants below to change the synthetic scene used by
pipeline.py. No logic, no imports — just values. Grouped by pipeline step.

Out of scope for this file:
    - Pixel noise level (belongs to the Monte Carlo experiment, later)
    - Pose sweep grids (belong to the viewpoint-sweep experiment, later)
"""

# ──────────────────────────────────────────────────────────────────────────
# Step 2 — Marker geometry
# ──────────────────────────────────────────────────────────────────────────

MARKER_SIDE_M = 0.10              # square side length L, meters


# ──────────────────────────────────────────────────────────────────────────
# Step 3 — Camera intrinsics and image size
# ──────────────────────────────────────────────────────────────────────────

IMAGE_WIDTH  = 640                # image width,  pixels
IMAGE_HEIGHT = 480                # image height, pixels

CAMERA_FX = 800.0                 # focal length along x, pixels
CAMERA_FY = 800.0                 # focal length along y, pixels

# Principal point defaults to image center; changes if you edit the resolution.
CAMERA_CX = IMAGE_WIDTH  / 2.0    # principal point x, pixels
CAMERA_CY = IMAGE_HEIGHT / 2.0    # principal point y, pixels


# ──────────────────────────────────────────────────────────────────────────
# Step 4 — Ground-truth pose (marker → camera frame)
# ──────────────────────────────────────────────────────────────────────────

# Rotation around each axis, in degrees (intrinsic xyz Euler convention).
GT_ROTATION_AROUND_X_DEG = 15.0   # tilts the marker top toward/away from the camera
GT_ROTATION_AROUND_Y_DEG = 10.0   # swings the marker left/right (vertical axis)
GT_ROTATION_AROUND_Z_DEG =  5.0   # spins the marker flat, in its own plane

# Translation: position of the marker origin in the camera frame, meters.
GT_TRANSLATION_X_M = 0.05         # right of the optical axis
GT_TRANSLATION_Y_M = 0.02         # below the optical axis
GT_TRANSLATION_Z_M = 0.50         # distance in front of the camera
