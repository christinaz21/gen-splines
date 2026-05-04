"""
losses.py — Differentiable loss functions for spline optimization.

THE CORE PROBLEM:
  PyTorch3D's point renderer flattens all curve points into a single unstructured
  point cloud, destroying the curve topology. At small radius, gradients are sparse.
  At large radius, everything blurs. Either way, many 3D configs produce similar
  images (depth ambiguity), so the rendered loss alone cannot recover geometry.

THE FIX:
  Use multi-view REPROJECTION loss instead of (or alongside) rendered image loss.
  
  Reprojection loss:
    1. Project GT curves to 2D screen coords via camera.transform_points_screen()
    2. Project predicted curves to 2D screen coords from the same camera
    3. Loss = L2 between corresponding 2D points, summed over multiple views
  
  This works because:
    - Curves have FIXED CORRESPONDENCE (curve i maps to curve i, point j to point j)
    - Multiple views break depth ambiguity (a point can't slide along a ray if
      it must also match from a perpendicular view)
    - Gradients flow directly from 2D error to 3D control points through the
      camera projection — no rasterization bottleneck
    - This IS what the paper proposes: backpropagate reconstruction error to
      update control points P^(t+1) = P^(t) - η∇_P L

  The rendered loss can be used as a secondary signal for appearance matching.
"""

import torch
import torch.nn.functional as F
from renderer import make_cameras


def reprojection_loss(gt_points_3d: torch.Tensor,
                      pred_points_3d: torch.Tensor,
                      azimuths: list,
                      elevation: float = 30.0,
                      dist: float = 4.0,
                      image_size: int = 256,
                      device: str = "cuda") -> torch.Tensor:
    """
    Multi-view reprojection loss: project both GT and predicted curve points
    to 2D screen space from multiple viewpoints and measure L2 distance.

    This is the key loss function that makes the optimization work.

    Args:
        gt_points_3d: (N, M, 3) GT sampled curve points (N curves, M samples each)
        pred_points_3d: (N, M, 3) predicted sampled curve points
        azimuths: list of azimuth angles (degrees) to project from
        elevation: camera elevation
        dist: camera distance
        image_size: screen size for projection normalization
        device: compute device

    Returns:
        Scalar loss (mean L2 reprojection error across all views)
    """
    N, M, D = gt_points_3d.shape
    total_loss = 0.0

    # Flatten to (N*M, 3) for projection
    gt_flat = gt_points_3d.reshape(-1, 3)
    pred_flat = pred_points_3d.reshape(-1, 3)

    for az in azimuths:
        cameras = make_cameras(azimuth=az, elevation=elevation,
                                dist=dist, device=device)

        # Project to screen space: (1, P, 3) -> (1, P, 3) where [:,:,:2] are x,y
        gt_screen = cameras.transform_points_screen(
            gt_flat.unsqueeze(0), image_size=(image_size, image_size)
        )[0, :, :2]  # (N*M, 2)

        pred_screen = cameras.transform_points_screen(
            pred_flat.unsqueeze(0), image_size=(image_size, image_size)
        )[0, :, :2]  # (N*M, 2)

        # L2 distance in screen space, normalized by image size
        view_loss = ((gt_screen - pred_screen) / image_size).pow(2).sum(dim=-1).mean()
        total_loss = total_loss + view_loss

    return total_loss / len(azimuths)


def smoothness_loss(control_points: torch.Tensor) -> torch.Tensor:
    """
    Penalize non-smooth curves (large second derivatives of control points).
    Preserves the inherent smoothness of the B-spline representation.

    Args:
        control_points: (N, K, 3)

    Returns:
        Scalar smoothness penalty
    """
    d1 = control_points[:, 1:] - control_points[:, :-1]  # first diff
    d2 = d1[:, 1:] - d1[:, :-1]                          # second diff
    return (d2 ** 2).mean()


def anchor_loss(control_points: torch.Tensor,
                anchor: torch.Tensor) -> torch.Tensor:
    """
    Persistent memory regularization: penalize control points for drifting
    from their anchor state.

    This implements the "persistent geometric states correct geometry instead
    of replacing it" principle from the paper.

    Args:
        control_points: (N, K, 3) current control points
        anchor: (N, K, 3) anchor (initial or memory) state

    Returns:
        Scalar anchor penalty
    """
    return F.mse_loss(control_points, anchor)


def curve_length_regularization(control_points: torch.Tensor) -> torch.Tensor:
    """
    Penalize curves for changing length too much (stretching/compressing).
    Helps maintain topology.

    Args:
        control_points: (N, K, 3)

    Returns:
        Scalar variance of segment lengths
    """
    segments = control_points[:, 1:] - control_points[:, :-1]  # (N, K-1, 3)
    lengths = segments.norm(dim=-1)  # (N, K-1)
    # Penalize variance in segment lengths (uniform spacing is ideal for B-splines)
    return lengths.var(dim=-1).mean()
