#!/bin/bash
#SBATCH --job-name=spline_setup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err

# ============================================================
# Gradient check + radius sweep
# Submit: sbatch scripts/setup.slurm
# ============================================================

set -euo pipefail

echo "=========================================="
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Date: $(date)"
echo "=========================================="

cd "$(dirname "$0")/.."
source scripts/find_conda.sh
conda activate spline_fields

echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo "=========================================="

python step0_gradient_check.py

echo "Setup check complete."
