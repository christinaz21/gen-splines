# Generative Spline Fields with Persistent Curve Memory

**COS526/ECE576 — Neural Rendering Project**
Vineal Sunkara & Christina Zhang, Princeton University

---

## Quick Start (Copy-Paste Commands)

### 0. Upload to Amarel

```bash
# From your local machine:
scp -r spline_fields/ <netid>@amarel.rutgers.edu:~/spline_fields/

# SSH in:
ssh <netid>@amarel.rutgers.edu
```

### 1. Get a GPU node (interactive)

```bash
srun --partition=gpu --gres=gpu:1 --mem=32G --time=02:00:00 --cpus-per-task=4 --pty bash
```

### 2. Set up environment (ONCE)

```bash
cd ~/spline_fields
bash scripts/setup_env.sh
```

### 3. RUN THIS FIRST — Gradient Check

```bash
conda activate spline_fields
cd ~/spline_fields
mkdir -p logs outputs

python step0_gradient_check.py
```

**Read the output carefully.** It tells you:
- Whether gradients flow from pixels to control points
- The best rendering radius for your setup
- Whether optimization actually reduces loss

**If anything fails, STOP HERE and debug.** Common issues:
- PyTorch3D not built for your CUDA version → rebuild with matching CUDA
- Gradient norm is 0 → radius is too small, increase it
- Optimization doesn't reduce loss → try larger radius or higher learning rate

### 4. Generate synthetic dataset

```bash
python dataset.py --output-dir outputs/dataset
```

### 5. Run single-view optimization

```bash
python optimize.py \
    --radius 0.02 \
    --num-steps 500 \
    --lr 1e-3 \
    --output-dir outputs/single_view
```

### 6. Run sequential multi-view optimization (persistent memory)

```bash
python optimize_sequential.py \
    --radius 0.02 \
    --num-views 36 \
    --steps-per-view 50 \
    --output-dir outputs/sequential
```

### 7. (Alternative) Submit as batch jobs

```bash
mkdir -p logs
sbatch scripts/run_pipeline.slurm
# Check status: squeue -u $USER
# Check output: tail -f logs/optimize_*.out
```

---

## What Each Script Does

| Script | Purpose | Runtime |
|--------|---------|---------|
| `step0_gradient_check.py` | **RUN FIRST.** Verifies PyTorch3D gradients flow to control points. Sweeps radius. | ~30s |
| `dataset.py` | Generates synthetic helix/wave strand scenes with GT control points. | ~1min |
| `optimize.py` | Optimizes control points from a single viewpoint. Validates the render-update loop. | ~2min |
| `optimize_sequential.py` | **Core experiment.** Orbits 360° updating persistent control points sequentially. | ~10min |

## File Structure

```
spline_fields/
├── spline.py                  # Cubic B-spline math + SplineField module
├── renderer.py                # PyTorch3D point cloud rendering + radius sweep
├── dataset.py                 # Synthetic strand scene generation
├── metrics.py                 # CP drift, curvature deviation, reprojection error
├── step0_gradient_check.py    # PRE-FLIGHT: gradient verification
├── optimize.py                # Single-view optimization
├── optimize_sequential.py     # Multi-view persistent memory loop
├── scripts/
│   ├── setup_env.sh           # One-shot conda setup
│   ├── setup.slurm            # SLURM: environment test
│   └── run_pipeline.slurm     # SLURM: full pipeline
├── outputs/                   # All results go here
│   ├── dataset/               # GT control points + rendered images
│   ├── single_view/           # Single-view optimization results
│   └── sequential/            # Sequential optimization results
└── README.md                  # This file
```

## Understanding the Output

### `step0_gradient_check.py` output

```
  STEP 0c: Radius Sweep
    Radius    GradNorm   GradNZ%   AlphaPx    ImgMean    Verdict
    0.0020      0.0000      0.0%         0    0.00000   TOO SMALL
    0.0100      0.0234     12.3%      1847    0.02810       GOOD
    0.0200      0.0891     38.7%      7234    0.11040       GOOD    ← sweet spot
    0.0500      0.1204     67.2%     24891    0.38100    TOO BIG
```

**Pick the radius where GradNorm is high AND AlphaPx is reasonable.**
The recommended value will be printed. Use it in all subsequent runs.

### `optimize_sequential.py` output

The key plot is `outputs/sequential/sequential_optimization.png` showing:
1. **CP Drift vs Azimuth** — should decrease or stay flat as you orbit
2. **Loss per View** — should be low at each viewpoint
3. **Curvature Deviation** — structural shape consistency

**Good result:** Drift decreases monotonically, stays below initial value.
**Bad result:** Drift increases during second half of orbit (memory instability).

## Key Parameters to Tune

| Parameter | Flag | Default | Effect |
|-----------|------|---------|--------|
| **Radius** | `--radius` | 0.02 | Most critical. Controls gradient quality. |
| Steps/view | `--steps-per-view` | 50 | More = better convergence per view, slower |
| Learning rate | `--lr` | 5e-4 | Lower = more stable, slower convergence |
| Init noise | `--init-noise` | 0.15 | How far initial guess is from GT |
| Samples/curve | `--samples-per-curve` | 64 | Point density for rendering |
| Num views | `--num-views` | 36 | Angular resolution of orbit (36 = 10° steps) |

## Troubleshooting

**"Gradient norm is 0"**
→ Increase radius. Your points aren't covering enough pixels.

**"Loss decreases but CP drift doesn't"**
→ Multiple curve configurations can produce similar images (non-convex).
   Try: (a) add silhouette loss `--silhouette-weight 0.5`, (b) reduce init noise,
   (c) increase samples per curve.

**"CUDA out of memory"**
→ Reduce image size: `--image-size 128`, or reduce curves/samples.

**PyTorch3D build fails**
→ Check CUDA version alignment: `nvidia-smi` shows driver CUDA, `nvcc --version`
   shows toolkit CUDA. They must be compatible. On Amarel, try:
   `module load cuda/12.1` before building.

**Sequential drift increases**
→ Learning rate too high (overshooting). Try `--lr 1e-4`.
   Or increase `--steps-per-view` to converge more at each view.

## Next Steps (Weeks 3+)

After weeks 1-2, you should have:
- ✅ Verified gradient flow
- ✅ Found optimal radius
- ✅ Single-view optimization working
- ✅ Sequential 360° optimization showing stable/decreasing drift

**Week 3-4:** Train `SplineGenerator` (Gθ) to map latent codes to control points.
Use `spline.py::SplineGenerator` class — it's already implemented but unused.

**Week 5-6:** Implement frame-to-spline initialization (lifting 2D ridge
detections into 3D using monocular depth). Integrate into the render-update loop
with real video frames instead of synthetic GT.

**Week 7+:** Revisitation benchmark, comparison vs point-cloud baseline.
