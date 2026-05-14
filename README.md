# Persistent Curve Memory: Structured Spline Representations for Sequential Scene Updates

Vineal Sunkara · Christina Zhang

This repository implements a structured scene representation in which 3D
geometry is encoded as a field of differentiable cubic B-splines, then
optimized sequentially under a moving camera with a persistent memory anchor.
The central question is whether *structured* primitives (curves) update more
stably across viewpoints than the unstructured alternatives (free point
clouds, 3D Gaussian splats) when each is given the same observations, the
same losses, and the same memory mechanism.

We evaluate on real hair geometry from the Cem Yuksel hair dataset, which
provides scenes whose thin-structure topology directly stresses memory
stability: even small per-point drift visibly destroys strand identity.

---

## Repository tour

```
gen-splines/
├── src/gensplines/        # Library code (installed editable as `gensplines`)
│   ├── spline.py          # Cubic B-spline evaluation + SplineField module
│   ├── memory.py          # EMA persistent memory (curve / point / Gaussian)
│   ├── losses.py          # Reprojection, tangent, anchor-proximity losses
│   ├── metrics.py         # CP drift, curvature deviation, reprojection error
│   ├── renderer.py        # Differentiable PyTorch3D point-cloud renderer
│   ├── render_utils.py    # Tube-mesh construction + OBJ/PLY export
│   ├── hair_loader.py     # Yuksel .hair parser + B-spline fitting
│   └── coordinates.py     # Yuksel → PyTorch3D axis convention
│
├── experiments/           # Runnable: each writes results to outputs/
│   ├── run_spline.py              # Main spline-memory pipeline
│   ├── run_pointcloud_baseline.py # Point-cloud baseline (matched protocol)
│   ├── run_gaussian_baseline.py   # Gaussian splat baseline
│   ├── run_world_model.py         # Explore → freeze → generate → revisit
│   ├── run_revisit_memory.py      # Multi-seed revisit-memory experiment
│   └── run_full_pipeline.py       # Orchestrates all three + evaluation
│
├── evaluation/            # Post-hoc analysis of saved results
│   ├── compare_baselines.py       # Drift / runtime comparison + plots
│   ├── evaluate_external.py       # Chamfer + held-out render MSE vs raw hair
│   └── export_gt.py               # Export GT control points to OBJ for viewer
│
├── scripts/               # SLURM submission + environment setup
│   ├── find_conda.sh              # Sourced by other scripts
│   ├── setup_env.sh               # One-shot environment setup
│   ├── slurm_amarel.sh            # Full-pipeline submission
│   └── slurm_amarel_revisit.sh    # Multi-seed revisit submission
│
├── viewer/                # Three.js viewer for OBJ/PLY artifacts
├── demos/                 # 2D pedagogical demos (build intuition for 3D)
├── data/                  # gitignored — Yuksel .hair files (placed here manually)
├── outputs/               # gitignored — experiment results
├── pyproject.toml         # Editable install: `pip install -e .`
└── requirements.txt       # Conda env spec (for reference)
```

The split is deliberate: `src/` is what *exists* as a library, `experiments/`
is what was *run* to produce results, `evaluation/` is how those results were
*measured*.

---

## Methodology

A scene is represented as N cubic B-splines, each parameterized by K control
points in ℝ³. Given a sequence of camera observations, control points are
optimized at each viewpoint against three losses:
(i) **rendered image loss** at the current view (PyTorch3D differentiable
point-cloud renderer),
(ii) **multi-view reprojection loss** against a buffer of recent views (the
geometric anchor — single-view rendering has depth ambiguity),
(iii) **anchor proximity loss** pulling control points toward an EMA of past
states (the persistent memory).
Two auxiliary regularizers — tangent consistency and curvature smoothness —
keep curves geometrically plausible during aggressive updates. Point-cloud
and Gaussian-splat baselines run the same protocol with the same losses
minus the curve-specific regularizers, so the only difference between
methods is the representation itself.

---

## Quick Start

### Prerequisites

- Linux with NVIDIA GPU (CUDA-capable). Experiments tested on A100 and L40s.
- conda (Miniconda or full Anaconda).
- Python 3.10.

### Hair data

The Yuksel hair `.hair` files are not redistributed with this repo. Download
them manually from
[www.cemyuksel.com/research/hairmodels/](http://www.cemyuksel.com/research/hairmodels/)
and place them under:

```
data/hairmodels/
├── wStraight.hair
├── wWavyThin.hair
├── wCurly.hair
└── ...
```

The loader reads from this directory directly — there is no automatic
download (the dataset host has scrape protection).

### Environment setup

```bash
git clone <repo-url> gen-splines
cd gen-splines

conda create -n spline_fields python=3.10 -y
conda activate spline_fields

# PyTorch. Pin 2.0.1 if you need glibc 2.17 compatibility (e.g. CentOS 7);
# otherwise the latest 2.x release works.
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 \
    -c pytorch -c nvidia -y

# PyTorch3D. Use the wheel matching your torch / CUDA combo.
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt201/download.html

# Remaining deps + the gensplines package itself
pip install matplotlib scipy imageio tqdm
pip install -e .
```

Verify:

```bash
python -c "import gensplines; print(gensplines.__version__)"   # 0.1.0
python -c "from gensplines import SplineField, evaluate_bspline; print('OK')"
```

### SLURM clusters

`scripts/setup_env.sh` automates the above for SLURM clusters that require
GPU-node allocation before package installation, and handles CentOS 7 /
glibc 2.17 constraints by pinning PyTorch 2.0.1.

We ran experiments on Rutgers' Amarel HPC; `scripts/slurm_amarel.sh` and
`scripts/slurm_amarel_revisit.sh` are written for that cluster but adapt
easily to any SLURM site — edit the `--partition`, `--gres`, and module-load
lines to match your environment.

Typical SLURM workflow:

```bash
# Request a GPU node interactively
srun --partition=<gpu_partition> --gres=gpu:1 --mem=32G \
     --time=02:00:00 --cpus-per-task=4 --pty bash

# One-time setup (idempotent — safe to re-run)
cd gen-splines
bash scripts/setup_env.sh

# Submit batch jobs
sbatch scripts/slurm_amarel.sh           # full pipeline, edit MODEL inside
sbatch scripts/slurm_amarel_revisit.sh   # multi-seed revisit
squeue -u $USER                          # check status
tail -f logs/gen-splines_<jobid>.out     # live output
```

---

## Reproducing the main results

All commands assume `conda activate spline_fields` from the repo root.

### 1. Main spline pipeline (single model)

```bash
python experiments/run_spline.py \
    --model-name wStraight \
    --num-curves 500 --K 12 \
    --num-views 72 --steps-per-view 80 \
    --output-dir outputs/spline_wStraight
```

Produces `opt_results.pt` (control-point trajectory + metrics),
`comparison_still.png`, and a 360° comparison video. ~10 min on one A100.

### 2. Full three-way comparison

```bash
python experiments/run_full_pipeline.py \
    --model-name wStraight \
    --output-root outputs/full_wStraight
```

Runs the spline, point-cloud, and Gaussian-splat methods in sequence, then
`compare_baselines.py` (drift/runtime) and `evaluate_external.py` (Chamfer +
held-out render MSE against raw hair strands). All artifacts land under
`outputs/full_wStraight/{spline,pointcloud,gaussian,compare,external_eval}/`.

For a fast smoke test:

```bash
python experiments/run_full_pipeline.py \
    --model-name wStraight \
    --output-root outputs/quick \
    --quick
```

### 3. World-model experiment (partial observability)

The camera observes only 0°–270°. Memory is then *frozen* and used to render
the full 360°, including the unobserved 270°–360° wedge. Measures how well
each representation extrapolates from the seen region.

```bash
python experiments/run_world_model.py \
    --model-name wStraight --explore-range 270 \
    --output-dir outputs/world_model_wStraight
```

Outputs include `exploration_timeline.png`, `generation_quality.png`,
`revisitation_consistency.png`, and a comparison video showing observed vs
unobserved regions side-by-side.

### 4. Multi-seed revisit-memory experiment

Camera follows trajectory A → B → C → D → A′ (returns to start). Measures
revisit consistency (does the representation render the same image at A′ as
it did at A?) and held-out generalization, aggregated across RNG seeds:

```bash
python experiments/run_revisit_memory.py \
    --model-name wWavyThin \
    --seeds 42,43,44,45,46 \
    --trajectory-azimuths 0,90,180,270,0 \
    --held-out-azimuths 45,135,225,315 \
    --output-dir outputs/revisit_wWavyThin
```

JSON with per-seed metrics and aggregate mean/std is written to
`revisit_results.json`, plus a comparison poster at `revisit_poster.png`.

---

## Key implementation notes

- **Coordinate convention.** Yuksel hair data is Y-up; PyTorch3D expects
  (x, z, -y). The axis flip happens *once* in `coordinates.orient_cp` /
  `orient_pts`, before any rendering. All snapshots saved by `run_spline.py`
  are already in the oriented frame — no double-flipping during analysis.
- **Rendering radius is the single most sensitive hyperparameter.** Too
  small → vanishing gradients (points cover too few pixels); too large →
  thin structures blur into blobs. `renderer.sweep_radius()` reports
  gradient norm and coverage across a sweep; the default 0.02 is the sweet
  spot at image size 256.
- **`bin_size=0` everywhere.** The coarse-bin rasterizer overflows on dense
  hair clouds at our scales. Naive rasterization is slower but reliable.
- **Memory anchor is EMA, not hard.** `memory.PersistentCurveMemory.update`
  blends new control points into the anchor at decay 0.8 by default. This
  is what `losses.anchor_proximity_loss` pulls toward — too aggressive and
  the representation freezes; too lax and it drifts.
- **Three baselines, one protocol.** Every representation runs the same view
  schedule, the same loss weights (where applicable), the same view buffer,
  and the same EMA. The only differences are the representation's degrees
  of freedom and which auxiliary regularizers are well-defined for it.
- **Held-out evaluation is intentionally fairer than chamfer-on-self.**
  `evaluate_external.py` compares both methods against *raw* Yuksel strand
  points, not the fitted GT used during optimization, and uses azimuths
  offset half-a-step from the training grid. This penalizes overfitting to
  the optimization signal.

---

## Demos

The `demos/` directory contains three standalone 2D scripts that build up
the intuition behind the 3D pipeline. They have no dependency on the
`gensplines` package and can be run with just NumPy + PyTorch + matplotlib:

```bash
python demos/01_basic_memory.py         # naive sequential update → drift
python demos/02_with_consistency.py     # consistency loss fixes drift
python demos/03_vs_point_baseline.py    # structured curve vs equal-DOF points
```

Each writes a `.png` to the cwd showing the curve evolution, drift
trajectory, and revisit error. Useful for explaining the project in slides
without needing a GPU.

---

## Viewer

`viewer/viewer.html` is a self-contained Three.js viewer for the OBJ/PLY
artifacts that `evaluation/export_gt.py` and `run_world_model.py` produce.
Open it in any modern browser, drop in a `.obj` (spline tubes), `.ply`
(point cloud), or GT mesh, and orbit. Useful for inspecting reconstruction
quality at close zoom levels that don't survive video compression.
