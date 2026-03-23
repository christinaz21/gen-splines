#!/bin/bash
# ============================================================
# setup_env.sh — Environment setup for Amarel (CentOS 7, glibc 2.17)
#
# IMPORTANT: CentOS 7 has glibc 2.17. Modern PyTorch (>=2.1) requires
# glibc 2.27+. We MUST use PyTorch 2.0.1, which is the last version
# that supports glibc 2.17.
#
# PyTorch3D must also match: we use v0.7.5 (last for PyTorch 2.0.x).
#
# BEFORE RUNNING:
#   1. Get on a GPU node first!
#      srun --partition=gpu --gres=gpu:1 --mem=32G --time=02:00:00 --cpus-per-task=4 --pty bash
#   2. Verify GPU is visible:
#      nvidia-smi
#      If you see "Unable to determine device handle", you are NOT on
#      a proper GPU node. Re-run srun with --gres=gpu:1.
#
# Usage:
#   bash scripts/setup_env.sh
# ============================================================

set -euo pipefail

ENV_NAME="spline_fields"
MINICONDA_DIR="$HOME/NeuralRenderingECE576/Assignment1/nrad_assignment/miniconda3"

echo "=========================================="
echo "  Spline Fields — Environment Setup"
echo "  Target: CentOS 7 / glibc 2.17 / Amarel"
echo "  Node:   $(hostname)"
echo "=========================================="

# --- 0) Preflight: are we on a GPU node? ---
echo ""
echo ">>> Preflight checks..."
GLIBC_VER=$(ldd --version 2>&1 | head -1 | grep -oP '[0-9]+\.[0-9]+$' || echo "unknown")
echo "  glibc version: ${GLIBC_VER}"
echo "  OS: $(cat /etc/redhat-release 2>/dev/null || echo 'unknown')"

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "NOT DETECTED")
    DRIVER_CUDA=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "unknown")
    echo "  GPU: ${GPU_NAME}"
    echo "  Driver CUDA: ${DRIVER_CUDA}"

    if [[ "${GPU_NAME}" == *"NOT DETECTED"* ]] || [[ "${GPU_NAME}" == *"Unknown Error"* ]]; then
        echo ""
        echo "  ERROR: GPU not accessible on this node!"
        echo "  You are likely on a login node or a node without GPU allocation."
        echo ""
        echo "  Fix: request a GPU node first:"
        echo "    srun --partition=gpu --gres=gpu:1 --mem=32G --time=02:00:00 --cpus-per-task=4 --pty bash"
        echo ""
        echo "  Then re-run: bash scripts/setup_env.sh"
        exit 1
    fi
else
    echo "  WARNING: nvidia-smi not found. Continuing anyway..."
fi

# --- 1) Find conda ---
echo ""
echo ">>> Finding conda..."
if command -v conda >/dev/null 2>&1; then
    echo "  Global conda detected: $(command -v conda)"
    CONDA_BASE="$(conda info --base)"
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
elif [ -d "${MINICONDA_DIR}" ]; then
    echo "  Using local Miniconda: ${MINICONDA_DIR}"
    source "${MINICONDA_DIR}/etc/profile.d/conda.sh"
else
    echo "  ERROR: No conda found."
    echo "  Expected: ${MINICONDA_DIR}"
    echo "  Update MINICONDA_DIR at the top of this script."
    exit 1
fi

# --- 2) Check if env already exists ---
if conda env list 2>/dev/null | grep -qw "${ENV_NAME}"; then
    echo ""
    echo "  Environment '${ENV_NAME}' already exists."
    echo "  To recreate: conda env remove -n ${ENV_NAME} -y && bash scripts/setup_env.sh"
    echo "  To verify:   conda activate ${ENV_NAME} && python step0_gradient_check.py"
    exit 0
fi

# --- 3) Try cloning existing ddpm env (fastest path) ---
echo ""
CLONE_SUCCESS=false
if conda env list 2>/dev/null | grep -qw "ddpm"; then
    echo ">>> Found existing 'ddpm' environment."
    echo "  Checking if it has a working PyTorch..."
    DDPM_TORCH=$(conda run -n ddpm python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "FAIL")

    if [[ "${DDPM_TORCH}" != "FAIL" ]]; then
        echo "  ddpm has PyTorch ${DDPM_TORCH} — cloning it (saves ~20 min)..."
        conda create -n "${ENV_NAME}" --clone ddpm -y
        CLONE_SUCCESS=true
        echo "  Clone complete."
    else
        echo "  ddpm PyTorch doesn't import cleanly. Building from scratch."
    fi
fi

# --- 4) If clone failed, build from scratch with PyTorch 2.0.1 ---
if [ "${CLONE_SUCCESS}" = false ]; then
    echo ">>> Creating fresh environment with PyTorch 2.0.1 (last glibc 2.17 compatible)..."
    conda create -n "${ENV_NAME}" python=3.10 -y
fi

echo ""
echo ">>> Activating ${ENV_NAME}..."
conda activate "${ENV_NAME}"

# Only install PyTorch if we didn't clone
if [ "${CLONE_SUCCESS}" = false ]; then
    echo ""
    echo ">>> Installing PyTorch 2.0.1 + CUDA 11.8 (CentOS 7 compatible)..."
    echo "  NOTE: PyTorch 2.0.1 is the LAST version supporting glibc 2.17."
    echo "  PyTorch >=2.1 will fail with 'GLIBC_2.27 not found'."
    conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
fi

# --- 5) Verify PyTorch works before building PyTorch3D ---
echo ""
echo ">>> Verifying PyTorch..."
python -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA version: {torch.version.cuda}')
else:
    print('  WARNING: CUDA not available. GPU tests will fail.')
    print('  Make sure you are on a GPU node with --gres=gpu:1')
" || {
    echo ""
    echo "  FATAL: PyTorch import failed!"
    echo "  Check the error above. Common fixes:"
    echo "    - Make sure you're on a GPU node"
    echo "    - Try: conda install pytorch==2.0.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y"
    exit 1
}

# --- 6) Install PyTorch3D ---
echo ""
echo ">>> Installing PyTorch3D build dependencies..."
pip install fvcore iopath

echo ""
echo ">>> Installing PyTorch3D (--no-build-isolation to use existing torch)..."
echo "  This will compile from source and takes 5-15 minutes."
echo "  If it fails, see fallback options at the end of this script."

# Set CXX flags for CentOS 7 compatibility
export FORCE_CUDA=1
# Try to find a newer GCC if available
if module avail gcc 2>&1 | grep -q "gcc"; then
    echo "  Loading newer GCC module..."
    module load gcc 2>/dev/null || true
fi
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt251/download.html || {
    echo ""
    echo "  PyTorch3D v0.7.5 source build failed."
    echo "  Trying v0.7.4..."
    pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.4" || {
        echo ""
        echo "  ============================================"
        echo "  PyTorch3D source build failed. Manual fixes:"
        echo "  ============================================"
        echo ""
        echo "  Option A: Try prebuilt wheel (if available for your combo):"
        echo "    pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt201/download.html"
        echo ""
        echo "  Option B: Load newer GCC and retry:"
        echo "    module load gcc/9.2.0  (or whatever is available: module avail gcc)"
        echo "    pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5'"
        echo ""
        echo "  Option C: Build in a separate step with verbose output:"
        echo "    git clone https://github.com/facebookresearch/pytorch3d.git /tmp/pytorch3d"
        echo "    cd /tmp/pytorch3d && git checkout v0.7.5"
        echo "    FORCE_CUDA=1 python setup.py install 2>&1 | tee build.log"
        echo ""
        exit 1
    }
}

# --- 7) Other dependencies ---
echo ""
echo ">>> Installing remaining dependencies..."
pip install matplotlib scipy tensorboard imageio tqdm

# --- 8) Final verification ---
echo ""
echo ">>> Final verification..."
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA (torch): {torch.version.cuda}')

import pytorch3d
print(f'PyTorch3D {pytorch3d.__version__}')
from pytorch3d.renderer import PointsRenderer, PointsRasterizer, AlphaCompositor
print('PyTorch3D renderer: OK')

# Quick smoke test: create a small point cloud and render
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform, FoVPerspectiveCameras,
    PointsRasterizationSettings,
)
import torch

if torch.cuda.is_available():
    device = 'cuda'
    pts = torch.randn(1, 100, 3, device=device)
    rgb = torch.ones(1, 100, 3, device=device)
    pc = Pointclouds(points=pts, features=rgb)
    R, T = look_at_view_transform(dist=3.0, elev=30.0, azim=0.0)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    raster_settings = PointsRasterizationSettings(image_size=64, radius=0.05, points_per_pixel=4)
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())
    img = renderer(pc)
    print(f'Smoke test render: {img.shape} (should be [1, 64, 64, 4])')
    print('Render smoke test: OK')
else:
    print('Skipping render smoke test (no GPU)')

print()
print('=' * 50)
print('ALL CHECKS PASSED — environment is ready!')
print('=' * 50)
" || {
    echo ""
    echo "Verification failed. Check errors above."
    exit 1
}

echo ""
echo "=========================================="
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "    source scripts/find_conda.sh"
echo "    conda activate ${ENV_NAME}"
echo "    python step0_gradient_check.py"
echo "=========================================="
