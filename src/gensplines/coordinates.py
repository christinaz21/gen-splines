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
