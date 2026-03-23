#!/bin/bash
#SBATCH --job-name=spline_opt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/optimize_%j.out
#SBATCH --error=logs/optimize_%j.err

# ============================================================
# Full pipeline: dataset + single-view + sequential optimization
# Submit: sbatch scripts/run_pipeline.slurm
#
# IMPORTANT: Update --radius below with the value from your
#            step0_gradient_check.py radius sweep!
# ============================================================

set -euo pipefail

# ---- CONFIGURATION (edit these) ----
RADIUS=0.02    # <-- UPDATE with your best radius from step0
# ------------------------------------

echo "=========================================="
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Date: $(date)"
echo "  Radius: ${RADIUS}"
echo "=========================================="

cd "$(dirname "$0")/.."
source scripts/find_conda.sh
conda activate spline_fields

echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo "=========================================="

# ----------------------------------------------------------
# Step 1: Generate synthetic dataset
# ----------------------------------------------------------
echo ""
echo ">>> Step 1: Generate dataset"
python dataset.py \
    --num-helix 25 \
    --num-wave 15 \
    --K 8 \
    --num-views 36 \
    --samples-per-curve 128 \
    --image-size 256 \
    --radius "${RADIUS}" \
    --output-dir outputs/dataset

# ----------------------------------------------------------
# Step 2: Single-view optimization
# ----------------------------------------------------------
echo ""
echo ">>> Step 2: Single-view optimization"
python optimize.py \
    --num-helix 20 \
    --num-wave 10 \
    --K 8 \
    --radius "${RADIUS}" \
    --num-steps 500 \
    --lr 1e-3 \
    --init-noise 0.15 \
    --output-dir outputs/single_view

# ----------------------------------------------------------
# Step 3: Sequential multi-view optimization
# ----------------------------------------------------------
echo ""
echo ">>> Step 3: Sequential optimization (persistent memory)"
python optimize_sequential.py \
    --num-helix 20 \
    --num-wave 10 \
    --K 8 \
    --radius "${RADIUS}" \
    --num-views 36 \
    --steps-per-view 50 \
    --lr 5e-4 \
    --init-noise 0.15 \
    --output-dir outputs/sequential

echo ""
echo ">>> Pipeline complete. Results in outputs/"
echo "=========================================="
