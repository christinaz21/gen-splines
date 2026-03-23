#!/bin/bash
# ============================================================
# find_conda.sh — Shared conda activation logic.
# Source this from any SLURM script or interactive session:
#   source scripts/find_conda.sh
#   conda activate spline_fields
# ============================================================

ENV_NAME="spline_fields"
MINICONDA_DIR="$HOME/NeuralRenderingECE576/Assignment1/nrad_assignment/miniconda3"

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
elif [ -d "${MINICONDA_DIR}" ]; then
    source "${MINICONDA_DIR}/etc/profile.d/conda.sh"
else
    echo "ERROR: No conda found. Expected at: ${MINICONDA_DIR}"
    exit 1
fi
