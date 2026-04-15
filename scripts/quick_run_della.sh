#!/bin/bash
#SBATCH --job-name=spline       
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=59:00
#SBATCH --cpus-per-task=4

# quick_run.sh — Interactive quick-start for testing on a GPU node
#
# Usage:
#   srun --partition=gpu --gres=gpu:1 --mem=32G --time=02:00:00 --pty bash
#   conda activate spline_fields
#   bash quick_run.sh [model_name]
#
# Examples:
#   bash quick_run.sh straight     # fast, smallest model
#   bash quick_run.sh wCurly       # challenging curly hair
#   bash quick_run.sh wWavy        # wavy hair


# load modules or conda environments here
source ~/.bashrc
eval "$(conda shell.bash hook)"  # this is needed to load python packages correctly

conda activate /scratch/gpfs/CHIJ/christina/envs/spline_fields
module load cudatoolkit/11.8
MODEL="wStraight"
# MODEL=${1:-straight}
echo "============================================================"
echo "  Quick Run: ${MODEL}"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "============================================================"

# Download if needed
# python -c "from hair_loader import download_yuksel_hair; download_yuksel_hair('${MODEL}')"

# Run v2_full
python optimize_v2.py \
    --data-source yuksel \
    --model-name ${MODEL} \
    --num-curves 50 \
    --K 8 \
    --num-views 36 \
    --steps-per-view 80 \
    --reproj-weight 2.0 \
    --curv-weight 0.1 \
    --tangent-weight 0.05 \
    --anchor-weight 0.01 \
    --view-buffer 3 \
    --ema-decay 0.7 \
    --output-dir outputs/quick_${MODEL}

echo ""
echo "  Done! Results in outputs/quick_${MODEL}/"
echo "  Key file: outputs/quick_${MODEL}/results.png"
