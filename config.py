"""
Global configuration for the synthetic IPPE-square pose-estimation project.

Edit the UPPERCASE constants below to change the synthetic scene used by
pipeline.py. No logic, no imports — just values. Grouped by pipeline step.

This module intentionally contains values only.  Experiment logic belongs in
experiments.py and orchestration belongs in pipeline.py.
"""

# ──────────────────────────────────────────────────────────────────────────
# Marker geometry
# ──────────────────────────────────────────────────────────────────────────

MARKER_SIDE_M = 0.10              # square side length L, meters


# ──────────────────────────────────────────────────────────────────────────
# Camera intrinsics and image size
# ──────────────────────────────────────────────────────────────────────────

IMAGE_WIDTH  = 640                # image width,  pixels
IMAGE_HEIGHT = 480                # image height, pixels

CAMERA_FX = 800.0                 # focal length along x, pixels
CAMERA_FY = 800.0                 # focal length along y, pixels

# Principal point defaults to image center; changes if you edit the resolution.
CAMERA_CX = IMAGE_WIDTH  / 2.0    # principal point x, pixels
CAMERA_CY = IMAGE_HEIGHT / 2.0    # principal point y, pixels


# ──────────────────────────────────────────────────────────────────────────
# Ground-truth pose (marker → camera frame)
# ──────────────────────────────────────────────────────────────────────────

# Rotation around fixed X, Y, Z axes, in degrees.  SciPy's lower-case ``xyz``
# convention is extrinsic (fixed-axis), matching pipeline.py.
GT_ROTATION_AROUND_X_DEG = 15.0   # tilts the marker top toward/away from the camera
GT_ROTATION_AROUND_Y_DEG = 10.0   # swings the marker left/right (vertical axis)
GT_ROTATION_AROUND_Z_DEG =  5.0   # spins the marker flat, in its own plane

# Translation: position of the marker origin in the camera frame, meters.
GT_TRANSLATION_X_M = 0.05         # right of the optical axis
GT_TRANSLATION_Y_M = 0.02         # below the optical axis
GT_TRANSLATION_Z_M = 0.50         # distance in front of the camera


# ──────────────────────────────────────────────────────────────────────────
# Viewpoint-sweep experiment
# ──────────────────────────────────────────────────────────────────────────

SWEEP_RADIUS_M = 0.50
SWEEP_MIN_ANGLE_DEG = 0.0
SWEEP_MAX_ANGLE_DEG = 89.0
SWEEP_SAMPLES = 90                 # one view per degree: 0, 1, ..., 89
SWEEP_AZIMUTH_DEG = 0.0


# ──────────────────────────────────────────────────────────────────────────
# Jacobian and Monte Carlo experiment
# ──────────────────────────────────────────────────────────────────────────

CORNER_NOISE_STD_PX = 0.5
MONTE_CARLO_VIEWING_ANGLES_DEG = (5.0, 60.0)
MONTE_CARLO_TRIALS = 1000
MONTE_CARLO_SEED = 20260804


# ──────────────────────────────────────────────────────────────────────────
# Generated experiment artefacts
# ──────────────────────────────────────────────────────────────────────────

OUTPUT_DIRECTORY = "outputs"
