#!/bin/bash
#SBATCH --job-name=spline_v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/spline_v2_%j.out
#SBATCH --error=logs/spline_v2_%j.err

# ============================================================
#  Spline Fields v2 — Real Data Experiments
#  Run all Yuksel models + ablation study
# ============================================================

mkdir -p logs

module load anaconda3
conda activate spline_fields

cd /path/to/spline_fields  # <-- UPDATE THIS PATH

echo "============================================================"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  Date: $(date)"
echo "  Python: $(python --version)"
echo "============================================================"

# Step 1: Download Yuksel hair models (only needs network once)
echo ""
echo ">>> Step 1: Downloading hair models..."
python -c "
from hair_loader import download_yuksel_hair
for m in ['straight', 'wCurly', 'wWavy', 'wStraight']:
    try:
        download_yuksel_hair(m, save_dir='data')
    except Exception as e:
        print(f'  Failed to download {m}: {e}')
"

# Step 2: Quick sanity check — load one model and verify
echo ""
echo ">>> Step 2: Verifying hair loader..."
python -c "
from hair_loader import load_hair_file, hair_to_spline_field
import torch
strands = load_hair_file('data/straight.hair')
cp = hair_to_spline_field(strands, num_curves=50, K=8)
print(f'  Sanity check passed: {cp.shape}')
assert cp.shape == (50, 8, 3)
"

# Step 3: Run ablation on 'straight' model (all configs)
echo ""
echo ">>> Step 3: Running ablation study on 'straight'..."
python run_experiments.py --ablation-only --output-base outputs/experiments

# Step 4: Run v2_full on all Yuksel models
echo ""
echo ">>> Step 4: Running v2_full on all Yuksel models..."
python run_experiments.py --yuksel-only --output-base outputs/experiments

# Step 5: Collect results
echo ""
echo ">>> Step 5: Final results..."
python run_experiments.py --collect --output-base outputs/experiments

echo ""
echo "============================================================"
echo "  DONE. Results in outputs/experiments/"
echo "============================================================"
