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
