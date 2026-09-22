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
| `docs/report.tex` | The report (LaTeX, compiles with pdfLaTeX) |
| `docs/ippe_derivation.tex` | Standalone IPPE derivation, every symbol defined, including the proof of what the ambiguity vector means geometrically. Not part of the report |

Dependency direction: `config` → `geometry` → `ippe_square` / `jacobian` /
`viewpoint` → `pipeline` / `experiment1` / `experiment2`. Nothing imports an
entry point, so the three can run in any order.

## The pipeline (`pipeline.py`)

Running it walks through the seven steps on the scene described by `config.py`
and prints each intermediate quantity: the marker corners in metres, the
intrinsics, the ground-truth rotation matrix, the marker normal in the camera
frame, the viewing angle, the camera-frame coordinates and pixel of every
corner, the IPPE intermediates (the homography, the image of the square centre,
and γ, whose inverse is the recovered depth), both candidate poses from both
solvers, and a per-corner residual table. Each intermediate is printed next to
the value the ground truth predicts for it.

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
  the ambiguity that experiment 1 is about.
- **Translation uses the same 8×3 system as OpenCV.** `solve_translation` keeps
  two of the three cross-product rows per corner. The third is redundant but
  reweights the least-squares fit, and keeping it made the mirror candidate
  differ from OpenCV in the fifth digit.
- **`generate_scene()` contains no Jacobian and no experiment code.** Steps 1-7
  and the two studies are kept apart on purpose.

On the default scene the primary candidate matches the ground truth to about
`1e-13` mm, with per-corner residuals near machine precision, and both
candidates agree with OpenCV.

**Known OpenCV limitation:** OpenCV's IPPE converts its rotation to a vector
without handling angles near 180°, so it returns wrong poses for a marker
facing the camera almost exactly. The baseline sits at 165.5° and is fine, but
setting all three Euler angles in `config.py` to zero gives exactly 180° and
makes the OpenCV check in `pipeline.py` fail while our solver stays exact. See
§2.7 of the report.

## Experiment 1 (`experiment1.py`)

The camera slides along an arc around the marker, always aimed at its centre,
one viewpoint per degree from 0° (fronto-parallel) to 89° (nearly edge-on). At
each viewpoint the solver runs and **both** candidates are kept, with their
reprojection error and their error against the ground truth.

- The arc has a single free parameter: azimuth and radius are fixed in
  `config.py` (0° and 0.5 m).
- No noise is added — the image points are the exact projections.
- Outputs: `outputs/exp1_sweep.png` and `outputs/exp1_sweep.csv`
  (90 rows: angle, both RMSEs, both pose errors).
- The report expects the plot as `figures/exp1_sweep.png` (ignored by Git,
  like `outputs/`). Upload it to Overleaf when you rerun the sweep.

What comes out of it:

- **Head-on, the two candidates are the same pose.** They separate as the view
  becomes oblique.
- **The wrong candidate is exactly twice the viewing angle away** in 3D, at
  every angle, while its image moves far more slowly: at 5° the two poses are
  10° apart and their images about 1 px apart.
- **The curve flattens past 75°**, where extra tilt no longer separates them.

**Why it happens.** The two IPPE candidates differ only by the sign of one
vector, and that vector turns out to be the line of sight projected onto the
marker plane. Its length is exactly the sine of the viewing angle, for any
pose. So the image tells the solver *how much* the marker is tilted but not
*towards which side*: two tilts, one on each side of the line of sight. The
proof is in `docs/ippe_derivation.tex`, the discussion in section 3 of the
report.

Two things to know about the code:

- The CSV has Windows-style (CRLF) line endings. That is Python's `csv` module
  default on every platform, not a bug; PyCharm warns about it if you try to
  commit the file.
- Candidates are sorted by reprojection error, so the plot labels them
  "candidate 1 (best)" and "candidate 2". Without noise the best one is always
  the true pose; with noise that would stop being true, and the labels would
  need to become "true" and "mirror".

## Experiment 2 (`experiment2.py`)

Work in progress. It has two parts.

**Conditioning along the arc (done).** At every viewing angle of the same
0°–89° arc as experiment 1, it evaluates the 8×6 Jacobian at the true pose and
reports all six singular values, the smallest one, the condition number, and
the singular vectors, which say *which* pose directions are poorly observed.

- Outputs: `outputs/exp2_conditioning.png` (singular values, condition number,
  and what the weakest direction is made of) and `outputs/exp2_conditioning.csv`
  (per angle: σ₁…σ₆, κ, and the two weakest directions).
- Translation is measured in units of the camera–marker distance, so all six
  pose parameters are dimensionless: singular values come out in pixels and the
  condition number is a pure number. In SI units (metres and radians) κ would
  change with the units chosen — at 30° it goes from 55 to 652 just by switching
  to millimetres — and two setups producing the same image (a 10 cm marker at
  0.5 m and a 20 cm one at 1 m) would get different values; with this scaling
  they get the same. Head-on, κ = (distance / half-side)² = 100.
- Singular vectors are sign-normalised (largest entry positive) so the same
  direction reads the same across angles.

**Monte Carlo (still to fix).** It runs 1,000 noisy trials at 5° and 60° and
compares the measured covariance with the first-order prediction.

Known gap: the prediction assumes a least-squares estimator, whereas the trials
re-solve with raw IPPE, which is algebraic and unrefined. The two agree well at
60° and not at 5°. Adding a Gauss-Newton refinement on top of IPPE is the next
step.

Link to experiment 1: near head-on, the two weakest directions of the Jacobian
are the two out-of-plane tilts, the same quantity the image cannot pin down in
experiment 1. The ambiguity and the poor conditioning are two views of the same
geometry.

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
