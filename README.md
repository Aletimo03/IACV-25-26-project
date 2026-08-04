# IACV 2025/26 — Synthetic IPPE-square pose estimation

This project reimplements the central geometric ideas behind OpenCV's
`SOLVEPNP_IPPE_SQUARE` in a fully synthetic setting. It uses only known
marker-corner 3D–2D correspondences: there is no image capture, marker
detection, distortion, or image-processing stage.

The custom solver estimates the two planar IPPE pose candidates. OpenCV is
used only as a reference for the noiseless validation experiment.

## Install

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```bash
# Noiseless custom-versus-OpenCV validation and Jacobian diagnostics
python pipeline.py

# 61-view front-facing camera sweep; writes PNG and CSV results
python pipeline.py --sweep

# 1,000 noisy trials at 5° and 60°; writes variance comparison PNG and CSV
python pipeline.py --monte-carlo

# Run all project experiments
python pipeline.py --all
```

Outputs are saved under `outputs/` by default (or set another location with
`--output-dir PATH`). The directory is intentionally ignored by Git.

Run the regression suite with the standard library test runner:

```bash
python -m unittest discover -s tests -v
```

## Geometry and conventions

- The camera frame is right-handed: +X right, +Y down, +Z forward.
- The pose maps marker coordinates to camera coordinates:
  `P_cam = R @ P_marker + t`.
- The centred square lies on marker `Z=0`; its required OpenCV order is
  `[-L/2,+L/2,0]`, `[+L/2,+L/2,0]`, `[+L/2,-L/2,0]`,
  `[-L/2,-L/2,0]`.
- The configured Euler sequence is SciPy lower-case `xyz`: extrinsic
  fixed-axis rotations. Estimation itself uses matrices and local rotation
  vectors, not Euler angles.
- The residual Jacobian has shape `(8, 6)` and parameter order
  `[t_x, t_y, t_z, omega_x, omega_y, omega_z]`. Its rotation increment is
  local and right-multiplied: `R_new = R @ exp([delta_omega]_x)`.
- Translation units are metres and rotation units are radians. Therefore the
  Jacobian singular values and condition number depend on this declared state
  scaling.

## Experiments

`--sweep` moves a camera on a fixed-radius, front-facing 0°–75° arc while
keeping it aimed at the marker centre. For every view, it keeps both custom
IPPE hypotheses and plots their OpenCV-compatible reprojection RMSE.

`--monte-carlo` adds independent zero-mean Gaussian noise to each image
corner coordinate, repeatedly solves the pose, and compares the empirical
local-pose covariance to the first-order prediction
`sigma_px^2 (J.T @ J)^-1` at near-frontal (5°) and oblique (60°) views.

All scene and experiment defaults live in `config.py`.

## Module layout

| File | Responsibility |
|---|---|
| `camera.py` | Calibrated pinhole projection and normalization |
| `marker.py` | Canonical square geometry and rigid transforms |
| `ippe_square.py` | From-scratch two-candidate IPPE-square solver |
| `jacobian.py` | Residual Jacobian, SVD diagnostics, covariance prediction |
| `experiments.py` | Viewpoint sweep, Monte Carlo study, plots, and CSV output |
| `pipeline.py` | Scene creation, OpenCV reference validation, and CLI |
| `tests/` | Geometry, solver, Jacobian, and experiment regressions |


