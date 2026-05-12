## Quick answer on `hair_loader` vs `hair_loader_fast`

Yes, this is normal and explains why the imports work despite the apparent mismatch. Either:

1. **There's a `hair_loader.py` on disk that didn't make it into the listing you pasted.** This is most likely. The file exists, all the `from hair_loader import ...` lines resolve to it, and `hair_loader_fast.py` is a parallel/newer implementation that nothing actually imports. Run `ls *.py` on Amarel to confirm.

2. **Or `hair_loader_fast.py` is a near-clone with the same API** (same function names: `load_hair_file`, `subsample_strands`, `hair_to_spline_field`, `download_yuksel_hair`, `get_yuksel_hair_path`). Looking at it: yes, the public API matches what the other scripts import. So if there's a separate `hair_loader.py` and `hair_loader_fast.py` is the optimized version that was never wired up, removing `_fast` would also work — but only if you're sure the slow one isn't subtly different (different defaults, missing arguments).

**What this means for the cleanup:** there's a decision to make in Step 1, not just a rename. Either keep both files (if there really is a `hair_loader.py` and you want to retire `_fast`), or replace one with the other.

Here's the updated guide.

---

# Step-by-step cleanup guide (Amarel) — v2

## Before you start

```bash
ssh vss54@amarel.rutgers.edu
cd ~/spline_fields    # or wherever your local checkout lives

git status
git log --oneline -5

git checkout -b cleanup-reorg
git push -u origin cleanup-reorg
```

Get an interactive GPU node so import checks are real:

```bash
srun --partition=gpu --gres=gpu:1 --mem=16G --time=03:00:00 --cpus-per-task=4 --pty bash
source scripts/find_conda.sh
conda activate spline_fields

cd ~/spline_fields
python -c "import torch, pytorch3d; print(torch.__version__, pytorch3d.__version__)"
```

If that fails, fix the environment before touching anything else.

---

## Step 1 — Audit the hair_loader situation

Goal: figure out what's actually on disk vs what the code references, then resolve.

```bash
cd ~/spline_fields

# What files actually exist?
ls -la *.py | grep hair

# What do scripts import?
grep -rn "from hair_loader\|import hair_loader" --include="*.py" .
```

Three possible outcomes:

### Outcome A — both `hair_loader.py` and `hair_loader_fast.py` exist

The "slow" one is what's actually used. The "fast" one was developed but never wired in. Two options:

**Option A1 (recommended): keep only the fast one if its API matches.**

```bash
# Diff the public APIs to confirm compatibility
grep -E "^def " hair_loader.py | sort
grep -E "^def " hair_loader_fast.py | sort
```

If the same function names appear in both with the same signatures, the fast one is a drop-in replacement:

```bash
# Save a backup just in case
cp hair_loader.py /tmp/hair_loader_old.py

# Replace
git rm hair_loader.py
git mv hair_loader_fast.py hair_loader.py

# Make sure nothing references hair_loader_fast anymore
grep -rn "hair_loader_fast" --include="*.py" --include="*.sh" --include="*.md" .
# Should return zero matches.
```

**Option A2 (conservative): keep the slow one, delete the fast one.**

```bash
git rm hair_loader_fast.py
grep -rn "hair_loader_fast" --include="*.py" .
# Should already be zero matches since nothing imported it.
```

A1 is better if you trust that the fast version is correct (it has progress bars, vectorized FPS, no scipy dependency — a clear upgrade). A2 is safer if you've never run the fast version end-to-end. Your call.

### Outcome B — only `hair_loader_fast.py` exists, no `hair_loader.py`

Every `from hair_loader import` is currently broken. Rename:

```bash
git mv hair_loader_fast.py hair_loader.py
grep -rn "hair_loader_fast" --include="*.py" --include="*.sh" --include="*.md" .
# Should return zero matches.
```

### Outcome C — only `hair_loader.py` exists, no `hair_loader_fast.py`

The listing I was working from was incomplete. Nothing to do — the imports already work. Skip to verification.

### Verify (all outcomes)

```bash
python -c "from hair_loader import load_hair_file, hair_to_spline_field, download_yuksel_hair; print('ok')"

# Smoke-test the entry points (they'll fail later on other missing imports, that's expected)
python run_dense.py --help > /dev/null 2>&1 && echo "run_dense ok" || echo "run_dense FAIL (expected if other imports broken)"
```

### Commit

```bash
git add -A
git status   # confirm what you're committing
git commit -m "fix: resolve hair_loader/hair_loader_fast naming"
git push
```

If nothing changed (Outcome C), skip the commit and move on.

---

## Step 2 — Remove all grass scene functionality

The grass codepath was scoped out and never presented. Strip it.

```bash
cd ~/spline_fields

# Find every reference to grass
grep -rn "grass" --include="*.py" --include="*.sh" --include="*.md" .

# Files that will need editing:
#   world_model.py — has --scene-type grass option, conditional imports of grass_scene,
#                    grass-specific color/camera settings
#   Possibly: grass_scene.py itself if it exists on disk
ls *.py | grep -i grass
```

### 2a — Delete grass_scene.py if it exists

```bash
if [ -f grass_scene.py ]; then
    git rm grass_scene.py
fi
```

### 2b — Strip grass support from `world_model.py`

This is too surgical to script with `sed` safely. Open the file:

```bash
nano world_model.py
# or
vim world_model.py
```

Remove these sections:

1. The `grass_vis_colors` function (around line 60). Delete the entire function.

2. The `scene_colors` function should be simplified — since `_SCENE_TYPE` will always be `"hair"`, it just returns `hair_colors(n, seed)`. Or just delete `scene_colors` and `_SCENE_TYPE` entirely and change every call site (`scene_colors(...)` → `hair_colors(...)`).

3. The `BG_GRASS` constant — delete. Keep `BG_HAIR`. Either rename `BG_HAIR` to `BG` or leave both as-is.

4. The `--scene-type` and `--grass-type` argument definitions in `main()`'s argparse block. Delete both lines.

5. The grass branch in main():

```python
elif args.scene_type == "grass":
    from grass_scene import grass_to_spline_field
    log(f"\n  Generating grass field: ...")
    gt_cp = grass_to_spline_field(...)
    gt_cp = orient(gt_cp).to(args.device)
    args.vis_elev = 15.0
    args.vis_dist = 2.5
    args.opt_radius = 0.015
```

Delete the entire `elif` block. Change the surrounding logic so `args.scene_type == "hair"` is implicit (the `if` becomes unconditional code; remove the `if args.scene_type == "hair":` line and dedent).

6. Anywhere `_SCENE_TYPE` is set or referenced — remove. Same for the line `_SCENE_TYPE = args.scene_type` in main(), and the `BG = BG_GRASS if ... else BG_HAIR` line — replace with `BG = BG_HAIR` or just delete.

7. The metric `"scene_type": args.scene_type` and `"model": args.model_name if args.scene_type == "hair" else f"grass_{args.grass_type}"` — simplify to just `"model": args.model_name`.

8. The top-of-file docstring and the log line `log(f"\n  MINI WORLD MODEL — {args.scene_type.upper()} SCENE")` — change `{args.scene_type.upper()}` to `"HAIR"`.

After editing:

```bash
# Make sure no grass references remain
grep -n "grass\|scene_type\|_SCENE_TYPE\|BG_GRASS\|grass_vis_colors\|grass_to_spline_field" world_model.py
# Should return zero or only false-positives (e.g. inside a string like "background")
```

### 2c — README and any docs

```bash
grep -rn "grass\|scene-type\|world_model.*grass" README.md README2.md prog.md 2>/dev/null
# Edit each match. README2 and prog will be deleted later anyway.
```

### Verify

```bash
python world_model.py --help
# Should show options, no --scene-type or --grass-type, no errors.

# Make sure the file parses
python -c "import ast; ast.parse(open('world_model.py').read()); print('syntax ok')"
```

### Commit

```bash
git add -A
git commit -m "feat: remove grass scene support (dead feature, never presented)"
git push
```

---

## Step 3 — Promote the misnamed "old" files

`old_script/optimize_v2.py`, `old_script/renderer.py`, `old_script/metrics.py` are actively imported. They are not old.

```bash
cd ~/spline_fields

# Confirm they're actually used
grep -rn "from old_script\|import old_script" --include="*.py" .
grep -rn "from optimize_v2\|from renderer\|from metrics" --include="*.py" .

# Move them
git mv old_script/optimize_v2.py optimize_v2.py
git mv old_script/renderer.py renderer.py
git mv old_script/metrics.py metrics.py
```

Check the moved files' internal imports — they should already resolve flat:

```bash
grep -n "^import\|^from" optimize_v2.py renderer.py metrics.py
```

If any line says `from old_script.X`, fix:

```bash
sed -i 's/from old_script\.//g; s/import old_script\.//g' optimize_v2.py renderer.py metrics.py
```

Sweep the rest of the repo for stragglers:

```bash
grep -rln "from old_script\|import old_script" --include="*.py" .
# For each file that turns up, fix it. Most likely none remain after the move.

# Final check
grep -rn "old_script\." --include="*.py" .
# Should return zero.
```

### Verify

```bash
python -c "from optimize_v2 import PersistentCurveMemory, multi_view_reprojection_loss, tangent_consistency_loss, anchor_proximity_loss; print('optimize_v2 ok')"
python -c "from renderer import render_point_cloud, make_cameras; print('renderer ok')"
python -c "from metrics import control_point_drift, compute_all_metrics; print('metrics ok')"

for f in run_dense.py run_pointcloud_baseline.py run_gaussian_splat_baseline.py experiment_revisit_memory.py world_model.py run_full_baseline_pipeline.py compare_baselines.py evaluate_external_targets.py; do
    python $f --help > /dev/null 2>&1 && echo "$f ok" || echo "$f FAIL"
done
```

If any fail, fix before continuing. Don't move on.

### Commit

```bash
git add -A
git commit -m "refactor: promote optimize_v2/renderer/metrics out of old_script/"
git push
```

---

## Step 4 — Delete the genuinely dead `old_script/` files

```bash
cd ~/spline_fields

# Confirm nothing in the rest of the codebase imports from any remaining old_script file
for f in old_script/*.py; do
    base=$(basename "$f" .py)
    refs=$(grep -rn "from old_script\.$base\|import old_script\.$base\|from $base\b\|import $base\b" \
           --include="*.py" --exclude-dir="old_script" . | wc -l)
    echo "$f : $refs references"
done
```

Every line should say `: 0 references`. If anything is non-zero, investigate before deleting that file.

Delete:

```bash
git rm old_script/ablation.txt
git rm old_script/ablationSweep.sh
git rm old_script/dataset.py
git rm old_script/demo_and_video.py
git rm old_script/demo_end_to_end.py
git rm old_script/generate_training_data.py
git rm old_script/generator.py
git rm old_script/losses.py
git rm old_script/make_video.py
git rm old_script/optimize.py
git rm old_script/optimize_sequential.py
git rm old_script/step0_gradient_check.py
git rm old_script/train_generator.py

rmdir old_script/ 2>/dev/null || true
ls old_script/ 2>/dev/null
```

### Verify

```bash
for f in run_dense.py run_pointcloud_baseline.py run_gaussian_splat_baseline.py experiment_revisit_memory.py world_model.py run_full_baseline_pipeline.py compare_baselines.py evaluate_external_targets.py; do
    python $f --help > /dev/null 2>&1 && echo "$f ok" || echo "$f FAIL"
done
```

### Commit

```bash
git add -A
git commit -m "chore: remove dead old_script/ files (superseded by current pipeline)"
git push
```

---

## Step 5 — Delete duplicate README and stale progress report

```bash
cd ~/spline_fields

grep -rn "README2\|prog\.md\|PROGRESS_REPORT" --include="*.py" --include="*.sh" --include="*.md" .

git rm README2.md
git rm prog.md

git commit -m "chore: remove duplicate README and stale progress report"
git push
```

---

## Step 6 — Consolidate duplicated utilities

Three functions are copy-pasted across multiple files. Centralize them.

### 6a — Coordinate orientation

```bash
grep -rn "def orient_cp\|def orient_pts\|def orient(" --include="*.py" .
```

Create `coordinates.py`:

```bash
cat > coordinates.py << 'EOF'
"""
coordinates.py — Shared coordinate-system conversion.

The Yuksel hair dataset uses Y-up convention. PyTorch3D's camera convention
requires (x, z, -y), so curve and point data are reoriented before rendering.
"""

import torch


def orient_cp(cp: torch.Tensor) -> torch.Tensor:
    """Reorient (N, K, 3) control points from Yuksel to PyTorch3D convention."""
    out = cp.clone()
    new_y = out[..., 2].clone()
    new_z = -out[..., 1].clone()
    out[..., 1] = new_y
    out[..., 2] = new_z
    return out


def orient_pts(points: torch.Tensor) -> torch.Tensor:
    """Reorient (..., 3) points from Yuksel to PyTorch3D convention."""
    out = points.clone()
    new_y = out[..., 2].clone()
    new_z = -out[..., 1].clone()
    out[..., 1] = new_y
    out[..., 2] = new_z
    return out
EOF
```

Now remove the duplicates from each file and import. Do this **by hand** in `nano` or `vim` — `sed` is too fragile for multi-line removal. Open each file, locate the `def orient_cp` and `def orient_pts` blocks, delete them, add `from coordinates import orient_cp, orient_pts` near the other imports.

Files to edit:
- `run_dense.py` — has both `orient_cp` and `orient_pts`
- `run_pointcloud_baseline.py` — has both
- `run_gaussian_splat_baseline.py` — has `orient_cp` only
- `world_model.py` — has a single `orient()` function. Either rename it to `orient_pts` and import, or alias on import: `from coordinates import orient_pts as orient`

After each file:

```bash
python -c "import <module>; print('ok')"
python <file>.py --help > /dev/null && echo "ok"
```

`experiment_revisit_memory.py` imports `orient_cp` from `run_dense`:

```bash
grep -n "orient_cp" experiment_revisit_memory.py
```

If you see `from run_dense import ... orient_cp ...`, change it to `from coordinates import orient_cp`.

### Verify 6a

```bash
python -c "from coordinates import orient_cp, orient_pts; print('coords ok')"
for f in run_dense.py run_pointcloud_baseline.py run_gaussian_splat_baseline.py experiment_revisit_memory.py world_model.py; do
    python $f --help > /dev/null 2>&1 && echo "$f ok" || echo "$f FAIL"
done

# Make sure no duplicate definitions remain
grep -rn "^def orient_cp\|^def orient_pts\|^def orient(" --include="*.py" .
# Should show only coordinates.py
```

### 6b — Blonde colors

```bash
grep -rn "def blonde_colors" --include="*.py" .
```

Move one canonical copy into `render_utils.py` (which already has `make_tube_colors` and `make_point_colors`). Open `render_utils.py`, paste the `blonde_colors` definition somewhere sensible (near the other color helpers).

Then delete from `run_dense.py` and `run_pointcloud_baseline.py`, add `from render_utils import blonde_colors` to each.

### Verify 6b

```bash
python -c "from render_utils import blonde_colors; print('ok')"
grep -rn "^def blonde_colors" --include="*.py" .
# Should show only render_utils.py
for f in run_dense.py run_pointcloud_baseline.py; do
    python $f --help > /dev/null 2>&1 && echo "$f ok" || echo "$f FAIL"
done
```

### 6c — Persistent memory classes

```bash
grep -rn "class PersistentCurveMemory\|class PersistentPointMemory\|class PersistentGaussianMemory\|class PersistentMemory" --include="*.py" .
```

Note: `world_model.py` defines its own simpler `PersistentMemory` base class — that's the same pattern, slightly different name. Worth unifying.

Create `memory.py`:

```bash
cat > memory.py << 'EOF'
"""
memory.py — EMA-based persistent memory anchors.

All variants implement the same (update, get_anchor) interface but operate
on different primitive types: spline control points (N, K, 3), point cloud
positions (P, 3), or Gaussian splat means (P, 3). The behavior is identical;
the class hierarchy just makes call sites self-documenting.
"""

import torch


class _EMAMemory:
    def __init__(self, initial_params: torch.Tensor, ema_decay: float = 0.8):
        self.anchor = initial_params.clone().detach()
        self.ema_decay = ema_decay

    def update(self, new_params: torch.Tensor) -> None:
        self.anchor = (
            self.ema_decay * self.anchor
            + (1.0 - self.ema_decay) * new_params.detach()
        )

    def get_anchor(self) -> torch.Tensor:
        return self.anchor.clone()


class PersistentCurveMemory(_EMAMemory):
    """For spline control points (N, K, 3)."""


class PersistentPointMemory(_EMAMemory):
    """For point cloud positions (P, 3)."""


class PersistentGaussianMemory(_EMAMemory):
    """For Gaussian splat means (P, 3). Scale/opacity tracked separately."""
EOF
```

Edit each file to remove its local definition and import from `memory`:

- `optimize_v2.py` — defines `PersistentCurveMemory` inline. Delete the class, add `from memory import PersistentCurveMemory`.
- `run_pointcloud_baseline.py` — defines `PersistentPointMemory` inline. Same treatment.
- `run_gaussian_splat_baseline.py` — defines `PersistentGaussianMemory` inline. Same.
- `experiment_revisit_memory.py` — imports `PersistentCurveMemory` from `optimize_v2`. Change to `from memory import PersistentCurveMemory`. It also defines `PersistentPointMemory` inline — delete and import. Same for `PersistentGaussianMemory`.
- `world_model.py` — has a slightly different `PersistentMemory` class. Either replace usage with `PersistentCurveMemory`/`PersistentPointMemory` from `memory.py` (recommended for consistency), or leave it. If replacing, audit the call sites: in `world_model.py`, instances are created with `PersistentMemory(sp_pred_cp.data, args.ema_decay)` and `PersistentMemory(pc_pred.data, args.ema_decay)` and use `update_anchor()` not `update()`. To minimize churn, either:
 - Rename the method calls in `world_model.py` from `update_anchor` to `update`, then use `PersistentCurveMemory`/`PersistentPointMemory` from `memory.py`.
 - Or keep `world_model.py`'s local class (the duplication is small). Pragmatic choice.

I'd just import from `memory.py` and rename the method calls — 2 sites to fix.

### Verify 6c

```bash
python -c "from memory import PersistentCurveMemory, PersistentPointMemory, PersistentGaussianMemory; print('memory ok')"
grep -rn "^class Persistent" --include="*.py" .
# Should show only memory.py (and maybe world_model.py if you chose to keep its local class)

for f in optimize_v2.py run_dense.py run_pointcloud_baseline.py run_gaussian_splat_baseline.py experiment_revisit_memory.py world_model.py; do
    python -c "import ${f%.py}" 2>&1 | head -3 && echo "$f ok" || echo "$f FAIL"
done
```

### Run something real

A `--help` passing only proves imports resolve. Run the spline pipeline on a tiny problem:

```bash
mkdir -p outputs/cleanup_smoke_test

python run_dense.py \
    --model-name wStraight \
    --num-curves 50 \
    --K 8 \
    --num-views 12 \
    --steps-per-view 20 \
    --output-dir outputs/cleanup_smoke_test \
    --num-video-frames 6 \
    --fps 4

ls outputs/cleanup_smoke_test/
# Should contain comparison_still.png and opt_results.pt
```

If this completes end-to-end, the refactor is correct.

### Commit

```bash
git add -A
git commit -m "refactor: extract shared utilities (coordinates, blonde_colors, memory)"
git push
```

---

## Step 7 — Rename entry points

```bash
cd ~/spline_fields

git mv run_dense.py run_spline.py
git mv run_gaussian_splat_baseline.py run_gaussian_baseline.py
git mv run_full_baseline_pipeline.py run_full_pipeline.py
git mv world_model.py run_world_model.py
git mv viewershin.html viewer.html
```

Update every reference:

```bash
# Inside Python (run_full_pipeline.py shells out to these by filename)
grep -rn "run_dense\.py\|run_gaussian_splat_baseline\.py\|run_full_baseline_pipeline\.py\|world_model\.py" --include="*.py" .

sed -i 's/run_dense\.py/run_spline.py/g' run_full_pipeline.py
sed -i 's/run_gaussian_splat_baseline\.py/run_gaussian_baseline.py/g' run_full_pipeline.py
sed -i 's/world_model\.py/run_world_model.py/g' run_full_pipeline.py

# In shell scripts (SLURM jobs invoke these)
grep -rn "run_dense\.py\|run_gaussian_splat_baseline\.py\|run_full_baseline_pipeline\.py\|world_model\.py" scripts/

sed -i 's/run_dense\.py/run_spline.py/g' scripts/*.sh
sed -i 's/run_gaussian_splat_baseline\.py/run_gaussian_baseline.py/g' scripts/*.sh
sed -i 's/run_full_baseline_pipeline\.py/run_full_pipeline.py/g' scripts/*.sh
sed -i 's/world_model\.py/run_world_model.py/g' scripts/*.sh

# Inside experiment_revisit_memory.py — it imports from run_dense
grep -n "from run_dense\|import run_dense" experiment_revisit_memory.py
sed -i 's/from run_dense/from run_spline/g; s/import run_dense/import run_spline/g' experiment_revisit_memory.py

# Same for world_model.py if it imports anything from another renamed file
grep -n "from run_dense\|from world_model" run_world_model.py

# Viewer reference in README (will be updated later anyway)
grep -rn "viewershin" .
```

### Verify

```bash
grep -rn "run_dense\.py\|run_gaussian_splat_baseline\.py\|run_full_baseline_pipeline\.py\|world_model\.py\|viewershin" \
    --include="*.py" --include="*.sh" --include="*.md" --include="*.slurm" .
# Should return zero results.

for f in run_spline.py run_pointcloud_baseline.py run_gaussian_baseline.py run_world_model.py run_full_pipeline.py experiment_revisit_memory.py compare_baselines.py evaluate_external_targets.py; do
    python $f --help > /dev/null 2>&1 && echo "$f ok" || echo "$f FAIL"
done
```

### Commit

```bash
git add -A
git commit -m "refactor: rename entry points (run_spline, run_gaussian_baseline, run_world_model, viewer.html)"
git push
```

---

## Step 8 — Consolidate SLURM scripts

```bash
cd ~/spline_fields

# Delete the broken/outdated ones
git rm scripts/run_pipeline.slurm
git rm scripts/setup.slurm
git rm scripts/run_v2_experiments.slurm
git rm scripts/run_v2_experiments_della.slurm
git rm scripts/quick_run.sh
git rm scripts/quick_run_della.sh
git rm scripts/dense_della.sh
git rm scripts/revisit_della.sh
```

Create the two replacements:

```bash
cat > scripts/slurm_amarel.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=gen-splines
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

# Generative Spline Fields — full pipeline on Amarel.
# Submit: sbatch scripts/slurm_amarel.sh
# Edit MODEL below to switch hair model.

set -euo pipefail
mkdir -p logs

MODEL="wStraight"
OUTPUT_ROOT="outputs/full_pipeline_${MODEL}"

source ~/.bashrc
source scripts/find_conda.sh
conda activate spline_fields

echo "=========================================="
echo "  Job:    ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "  Node:   $(hostname)"
echo "  GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  Model:  ${MODEL}"
echo "  Output: ${OUTPUT_ROOT}"
echo "  Date:   $(date)"
echo "=========================================="

python run_full_pipeline.py \
    --model-name ${MODEL} \
    --device cuda \
    --output-root ${OUTPUT_ROOT} \
    --num-curves 500 \
    --K 12 \
    --num-views 72 \
    --steps-per-view 80 \
    --pc-points-per-curve 12 \
    --gs-points-per-curve 12

echo "DONE ($(date))"
echo "Results: ${OUTPUT_ROOT}/"
EOF

cat > scripts/slurm_amarel_revisit.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=revisit
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00

# Multi-seed revisit-memory experiment on Amarel.
# Submit: sbatch scripts/slurm_amarel_revisit.sh

set -euo pipefail
mkdir -p logs

MODEL="wWavyThin"
OUTPUT_DIR="outputs/revisit_${MODEL}"

source ~/.bashrc
source scripts/find_conda.sh
conda activate spline_fields

echo "Job ${SLURM_JOB_ID} on $(hostname): ${MODEL}"

python experiment_revisit_memory.py \
    --model-name ${MODEL} \
    --output-dir ${OUTPUT_DIR} \
    --device cuda \
    --seeds 42,43,44,45,46 \
    --held-out-azimuths 45,135,225,315 \
    --trajectory-azimuths 0,90,180,270,0 \
    --poster-seed 42

echo "Results: ${OUTPUT_DIR}/revisit_results.json"
EOF

chmod +x scripts/slurm_amarel.sh scripts/slurm_amarel_revisit.sh
```

### Verify

```bash
sbatch --test-only scripts/slurm_amarel.sh
sbatch --test-only scripts/slurm_amarel_revisit.sh
```

Both should report "Job validation succeeded" or similar.

### Commit

```bash
git add -A
git commit -m "refactor: consolidate SLURM scripts (slurm_amarel.sh, slurm_amarel_revisit.sh)"
git push
```

---

## Step 9 — Clean up `outputs/`

```bash
cd ~/spline_fields

git ls-files outputs/ | wc -l
git ls-files outputs/ | xargs du -ch 2>/dev/null | tail -1
```

Untrack heavy artifacts (without deleting from disk):

```bash
git rm --cached $(git ls-files outputs/ | grep -E '\.(pt|npz|obj)$') 2>/dev/null || true

find outputs -type d -name "video_frames" | while read d; do
    git rm -r --cached "$d" 2>/dev/null || true
done

# Adjust these to match what's actually in your outputs/
git rm -r --cached outputs/oldOutputs 2>/dev/null || true
git rm -r --cached outputs/ablation_anchor_* 2>/dev/null || true
git rm -r --cached outputs/quick_* 2>/dev/null || true
git rm -r --cached outputs/wm_10k_* 2>/dev/null || true
git rm -r --cached outputs/training_data 2>/dev/null || true
git rm -r --cached outputs/gen_conv 2>/dev/null || true
git rm -r --cached outputs/gen_percurve 2>/dev/null || true
git rm -r --cached outputs/dense_w* 2>/dev/null || true

git ls-files outputs/ | wc -l
git ls-files outputs/ | xargs du -ch 2>/dev/null | tail -1
```

Update `.gitignore`:

```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.pyc
*.pyo
*.pyd

# Conda/venv
.conda/
.venv/

# Logs
logs/

# Data (downloaded at runtime, not tracked)
data/hairmodels/

# Outputs: ignore by default, allowlist outputs/final/
outputs/*
!outputs/final/
!outputs/final/**

# Even in final/, don't track heavy binaries
outputs/final/**/*.pt
outputs/final/**/*.npz
outputs/final/**/*.obj
outputs/final/**/video_frames/

# OS junk
.DS_Store
Thumbs.db
EOF
```

Move curated outputs to `outputs/final/`:

```bash
mkdir -p outputs/final
# Example — move what you want to keep. Do this by hand based on what's actually
# in your outputs/ directory. Typically: one comparison_still.png per model,
# one .mp4 if short, the JSON metrics files.
#
# git mv outputs/full_pipeline_wWavy/spline/comparison_still.png outputs/final/wWavy_spline.png
# git mv outputs/full_pipeline_wWavy/compare/baseline_comparison.json outputs/final/wWavy_comparison.json
```

### Verify

```bash
git status
git ls-files outputs/
```

### Commit

```bash
git add -A
git commit -m "chore: untrack heavy outputs, allowlist outputs/final/"
git push
```

---

## Step 10 — Final audit

```bash
cd ~/spline_fields

# Anything still referencing the old layout?
grep -rn "old_script\|hair_loader_fast\|run_dense\.py\|run_gaussian_splat_baseline\|run_full_baseline_pipeline\|viewershin\|README2\|PROGRESS_REPORT\|grass\|_SCENE_TYPE" \
    --include="*.py" --include="*.sh" --include="*.md" --include="*.slurm" .
# Should return zero (or only innocuous false positives like "blade of grass" in a comment).

# Every entry point imports cleanly?
for f in run_spline.py run_pointcloud_baseline.py run_gaussian_baseline.py run_world_model.py run_full_pipeline.py experiment_revisit_memory.py compare_baselines.py evaluate_external_targets.py; do
    python $f --help > /dev/null 2>&1 && echo "$f ok" || echo "$f FAIL"
done

# Real end-to-end smoke test
rm -rf outputs/cleanup_smoke_test
python run_spline.py \
    --model-name wStraight \
    --num-curves 50 --K 8 \
    --num-views 12 --steps-per-view 20 \
    --output-dir outputs/cleanup_smoke_test \
    --num-video-frames 6 --fps 4

ls outputs/cleanup_smoke_test/
# Should contain comparison_still.png at minimum
```

### Replace the README

```bash
nano README.md
# Paste the README I gave you previously. Adjust invocations to flat layout:
# python run_spline.py ...        (NOT python -m experiments.run_spline)
# python run_world_model.py ...
# Remove any mention of --scene-type or grass.

git add README.md
git commit -m "docs: rewrite README for cleaned-up flat layout"
git push
```

### Merge to main

```bash
git checkout main
git merge cleanup-reorg
git push
```

Or open a PR on GitHub if Christina wants to review first.

---

## Recovery if something breaks

At any step, if verification fails:

```bash
git diff             # see what changed
git status

git reset --hard HEAD            # discard uncommitted changes
git revert HEAD                  # undo last commit (creates a new commit)
git reset --hard HEAD~1          # rewrite history (only if not pushed)
```

Every step ends in a commit, so the worst case is rolling back to the start of that step.

## Time estimate

- Steps 1–5 (the load-bearing ones): ~90 minutes
- Steps 6–10 (cosmetic but worth doing): ~2 hours
- Total: ~3.5 hours if everything is straightforward, ~5 hours if you hit a surprise

Steps 1–5 alone get you a working repo with the imports sorted and no dead code. If you run out of time, stop after Step 5 and the prof still gets something clean.
