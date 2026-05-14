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
