# IACV 2025/26 — Synthetic IPPE-square pose estimation

A from-scratch reimplementation of the ideas behind OpenCV's
`SOLVEPNP_IPPE_SQUARE`, in a fully synthetic setting: the 3D–2D corner
correspondences are generated mathematically, so there is no image capture, no
marker detection, no lens distortion and no image processing anywhere in the
project.

Our solver returns **both** planar IPPE pose candidates, unrefined. OpenCV is
used only once, as an external reference for the noiseless validation in
`pipeline.py`; nothing else in the project depends on it.

## Install

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

There are three independent entry points, one per deliverable.

```bash
python pipeline.py       # steps 1-7 on one scene, prints every intermediate value
python experiment1.py    # viewpoint sweep: the two-candidate ambiguity
python experiment2.py    # Jacobian conditioning and Monte Carlo (work in progress)
```

Plots and CSV files go to `outputs/`, which is not tracked by Git. Both
experiments accept `--output-dir PATH`.

Tests:

```bash
python -m unittest discover -s tests -v
```

## Module layout

| File | Responsibility |
|---|---|
| `config.py` | Every setting of the synthetic scene and of the experiments. Values only, no logic |
| `geometry/` | The scene objects; imported as `from geometry.camera import ...` (no `__init__.py`, it is a namespace package) |
| `geometry/camera.py` | Pinhole camera: `K`, projection, and normalization with `K_inv` |
| `geometry/marker.py` | The canonical square and the rigid transform `R @ P + t` |
| `ippe_square.py` | The solver: DLT homography, IPPE construction, linear translation |
| `viewpoint.py` | Camera placement on an arc, parametrised by radius, viewing angle and azimuth |
| `jacobian.py` | Reprojection Jacobian, SVD diagnostics, covariance prediction |
| `pipeline.py` | Steps 1-7: builds the scene, solves it, checks against OpenCV and ground truth |
| `experiment1.py` | Sweeps the viewpoint, keeps both candidates, writes plot and CSV |
| `experiment2.py` | Conditioning per viewpoint plus the Monte Carlo noise study |
| `tests/` | Regression tests for geometry, solver, Jacobian and experiments |
| `report.tex` | The report (LaTeX, compiles with pdfLaTeX; see `figures/`) |

Dependency direction: `config` → `geometry` → `ippe_square` / `jacobian` /
`viewpoint` → `pipeline` / `experiment1` / `experiment2`. Nothing imports an
entry point, so the three can run in any order.

## The pipeline (`pipeline.py`)

Running it walks through the seven steps on the scene described by `config.py`
and prints each intermediate quantity: the marker corners in metres, the
intrinsics, the ground-truth rotation matrix, the marker normal in the camera
frame, the viewing angle, the camera-frame coordinates and pixel of every
corner, both candidate poses from both solvers, and a per-corner residual table.

Notes on the implementation:

- **`make_ground_truth_pose` composes the Euler angles with a 180° flip about
  X.** Without it, small Euler angles would describe a marker whose normal
  points *away* from the camera, i.e. seen from behind. With the flip, the
  three angles in `config.py` mean what their comments say: a tilt away from
  the fronto-parallel view of a marker facing the camera, as a detected ArUco
  marker always is.
- **The solver validates its inputs.** `IPPE_SQUARE` is not a generic coplanar
  solver: it assumes the canonical centred square in a specific corner order.
  Anything else is rejected rather than silently turned into a meaningless
  pose.
- **Both candidates are returned, and neither is refined.** A free nonlinear
  refinement can pull both starting points to the same pose, which would hide
  the ambiguity that experiment 1 is about. The price is a difference from
  OpenCV in the fifth digit on the mirror candidate, since OpenCV does refine.
- **`generate_scene()` contains no Jacobian and no experiment code.** Steps 1-7
  and the two studies are kept apart on purpose.

On the default scene the primary candidate matches the ground truth to about
`1e-13` mm, with per-corner residuals near machine precision, and agrees with
OpenCV to the same order.

## Experiment 1 (`experiment1.py`)

The camera slides along an arc around the marker, always aimed at its centre,
one viewpoint per degree from 0° (fronto-parallel) to 89° (nearly edge-on). At
each viewpoint the solver runs and **both** candidates are kept, with their
reprojection error and their error against the ground truth.

- The arc has a single free parameter: azimuth and radius are fixed in
  `config.py` (0° and 0.5 m).
- No noise is added — the image points are the exact projections.
- Outputs: `outputs/viewpoint_sweep.png` and `outputs/viewpoint_sweep.csv`
  (90 rows: angle, both RMSEs, both pose errors, and two conditioning columns).
- `figures/viewpoint_sweep.png` is a tracked copy of the plot for the report.
  Refresh it when you rerun the sweep.

What comes out of it: the two candidates are identical head-on and separate as
the view becomes oblique, the wrong one being exactly twice the viewing angle
away in 3D while its image moves far more slowly. The curve flattens past 75°.
The report discusses this in section 3.

Two things to know about the code:

- `run_viewpoint_sweep` takes a `noise_std_px` argument but **does not add
  noise**; it only passes the value to `analyze_jacobian` to fill the two
  conditioning columns in the CSV.
- Candidates are sorted by reprojection error, so the plot labels them
  "candidate 1 (best)" and "candidate 2". Without noise the best one is always
  the true pose; with noise that would stop being true, and the labels would
  need to become "true" and "mirror".

## Experiment 2 (`experiment2.py`)

Work in progress. It currently computes the 8×6 Jacobian per viewpoint with its
singular values and condition number, then runs 1,000 noisy trials at 5° and
60° and compares the measured covariance with the first-order prediction.

Known gap: the prediction assumes a least-squares estimator, whereas the trials
re-solve with raw IPPE, which is algebraic and unrefined. The two agree well at
60° and not at 5°. Adding a Gauss-Newton refinement on top of IPPE is the next
step.

## Conventions

- Camera frame: +X right, +Y down, +Z forward. The pose maps marker to camera,
  `P_cam = R @ P_marker + t`.
- The square lies on marker `Z=0`, centred, in OpenCV's order:
  `[-L/2,+L/2,0]`, `[+L/2,+L/2,0]`, `[+L/2,-L/2,0]`, `[-L/2,-L/2,0]`
  (clockwise seen from +Z).
- Euler angles use SciPy's lower-case `xyz`, i.e. extrinsic fixed-axis
  rotations. They only *describe* the scene; estimation uses rotation matrices
  and local rotation vectors.
- **Viewing angle** always means the angle between the marker normal and the
  line from the marker to the camera: 0° is fronto-parallel, above 90° means
  the marker is seen from behind.
- Reprojection RMSE follows OpenCV's convention, averaging over the eight
  scalar residuals rather than the four corners.
- Lengths are metres and angles radians in the code, degrees only when printed.
  The Jacobian's condition number depends on that choice of units.
