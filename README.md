# IACV 2025/26 — Synthetic Pose Estimation from a Square Marker

Synthetic pipeline for camera-marker pose estimation, mirroring the logic of
OpenCV's `SOLVEPNP_IPPE_SQUARE` but with no detection step — all 2D/3D
correspondences are generated mathematically.

## Status

**Steps 1–5 implemented** (forward simulation only):
1. Generate the marker plane (Z=0).
2. Place a square of side `L` on it.
3. Build a pinhole camera with known intrinsics `K`.
4. Choose a ground-truth pose `(R_gt, t_gt)`.
5. Project the four 3D corners into the image.

**Not yet implemented:** Step 6 (IPPE_SQUARE pose estimation), Step 7
(comparison vs. ground truth), Experiment 1 (two-pose ambiguity sweep),
Experiment 2 (Jacobian conditioning + Monte Carlo).

## Files

| File | Purpose |
|---|---|
| `config.py` | All scene parameters as UPPERCASE constants: marker side, image size, camera intrinsics, ground-truth rotation angles + translation. **Edit numbers here to change the scene.** No logic, no imports. |
| `camera.py` | Pinhole camera. `Camera` class (intrinsics `K`, `project()`, `normalize()`) + `make_camera()` factory that pulls defaults from `config.py`. |
| `marker.py` | Square marker geometry. `make_square_marker(L)` builds the 4 corner "object points" on the Z=0 plane (OpenCV ordering); `transform_points(pts, R, t)` applies a rigid transform. |
| `pipeline.py` | Orchestrator. `make_ground_truth_pose()` builds `(R_gt, t_gt)`; `generate_scene()` chains steps 1–5 via the factories and returns every intermediate quantity. Running it prints a verbose walkthrough with sanity checks. |
| `report.tex` | Overleaf-ready math write-up of steps 1–5: pinhole model, intrinsics & normalized coordinates, rigid transforms, `SO(3)` and Euler angles, the forward projection. |

## Architecture

Each object is built by a **config-aware factory**, so `generate_scene()` is a
thin orchestrator with no hardcoded parameters:

```
config.py  ──►  make_square_marker()     # marker corners
           ──►  make_camera()            # Camera object
           ──►  make_ground_truth_pose() # (R_gt, t_gt)
                        │
                        ▼
                 generate_scene()  ──►  scene dict
```

Dependency graph is acyclic: `config` ← `camera`, `marker` ← `pipeline`.

## Run

```bash
python pipeline.py
```

Edit `config.py` to change the synthetic scene. To vary parameters
programmatically (e.g. viewpoint sweeps later), call the factories directly
with overrides — they all accept arguments and fall back to `config` defaults.
