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
| `camera.py` | Pinhole camera model: intrinsic matrix `K`, `project()` (3D camera-frame → 2D pixels), `normalize()` (pixels → Z=1 ray directions via `K⁻¹`). |
| `marker.py` | The square marker geometry: `square_object_points(L)` returns the 4 corners on the Z=0 plane in the OpenCV ordering; `transform_points(R, t)` applies a rigid transform. |
| `config.py` | All scene parameters as UPPERCASE constants: marker side, image size, camera intrinsics, ground-truth Euler angles + translation. Edit numbers here to change the experiment. |
| `pipeline.py` | Orchestrator. `generate_scene()` chains steps 1–5 and returns every intermediate quantity. Running it prints a verbose walkthrough with sanity checks. |
| `pipeline_math.tex` | Full mathematical derivation of steps 1–7 (Overleaf-ready): pinhole model, rigid transforms, rotation representations (Euler / axis-angle / Rodrigues), DLT homography, IPPE_SQUARE pose extraction, error metrics. |

## Run

```bash
python pipeline.py
```

Edit `config.py` to change the synthetic scene.
