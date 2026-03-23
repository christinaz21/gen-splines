"""
metrics.py — Geometry-Aware Memory Metrics.

These metrics measure structural consistency in parameter space,
not pixel space. This is the key differentiator from SSIM/LPIPS.

Metrics:
  1. Control-Point Drift: L2 distance between GT and predicted control points
  2. Curvature Deviation: change in curvature profile over time
  3. Reprojection Error: 2D pixel error when projecting curves to a viewpoint
"""

import torch
import torch.nn.functional as F
from spline import SplineField, evaluate_bspline


def control_point_drift(gt_cp: torch.Tensor, pred_cp: torch.Tensor,
                         per_curve: bool = False) -> torch.Tensor:
    """
    Compute L2 distance between ground-truth and predicted control points.

    Args:
        gt_cp: (N, K, 3) ground-truth control points
        pred_cp: (N, K, 3) predicted control points
        per_curve: if True, return (N,) per-curve drift; else scalar mean

    Returns:
        Scalar or (N,) tensor of drift values
    """
    assert gt_cp.shape == pred_cp.shape, f"Shape mismatch: {gt_cp.shape} vs {pred_cp.shape}"
    drift = (gt_cp - pred_cp).norm(dim=-1)  # (N, K)
    if per_curve:
        return drift.mean(dim=-1)  # (N,)
    return drift.mean()


def curvature_deviation(gt_cp: torch.Tensor, pred_cp: torch.Tensor,
                         num_samples: int = 64) -> torch.Tensor:
    """
    Compare curvature profiles of ground-truth and predicted curves.

    Args:
        gt_cp: (N, K, 3)
        pred_cp: (N, K, 3)
        num_samples: samples along each curve for curvature computation

    Returns:
        Scalar mean absolute curvature deviation
    """
    gt_field = SplineField(gt_cp.shape[0], gt_cp.shape[1])
    gt_field.control_points.data = gt_cp
    pred_field = SplineField(pred_cp.shape[0], pred_cp.shape[1])
    pred_field.control_points.data = pred_cp

    gt_curv = gt_field.compute_curvature(num_samples)
    pred_curv = pred_field.compute_curvature(num_samples)

    return (gt_curv - pred_curv).abs().mean()


def reprojection_error(gt_cp: torch.Tensor, pred_cp: torch.Tensor,
                        cameras, num_samples: int = 64) -> torch.Tensor:
    """
    Project both GT and predicted curves to 2D and measure pixel distance.

    Args:
        gt_cp: (N, K, 3)
        pred_cp: (N, K, 3)
        cameras: PyTorch3D cameras object
        num_samples: samples per curve

    Returns:
        Mean 2D reprojection error in pixels
    """
    gt_points = evaluate_bspline(gt_cp, num_samples).reshape(-1, 3)
    pred_points = evaluate_bspline(pred_cp, num_samples).reshape(-1, 3)

    # Project to screen space
    gt_screen = cameras.transform_points_screen(gt_points.unsqueeze(0))[0, :, :2]
    pred_screen = cameras.transform_points_screen(pred_points.unsqueeze(0))[0, :, :2]

    return (gt_screen - pred_screen).norm(dim=-1).mean()


def compute_all_metrics(gt_cp: torch.Tensor, pred_cp: torch.Tensor,
                         num_samples: int = 64) -> dict:
    """
    Compute all geometry-aware metrics.

    Returns:
        dict with keys: cp_drift, curvature_dev, cp_drift_per_curve
    """
    device = gt_cp.device

    cp_drift = control_point_drift(gt_cp, pred_cp)
    cp_drift_per = control_point_drift(gt_cp, pred_cp, per_curve=True)

    # Curvature on CPU to avoid memory issues
    curv_dev = curvature_deviation(gt_cp.cpu(), pred_cp.cpu(), num_samples)

    return {
        "cp_drift": cp_drift.item(),
        "curvature_deviation": curv_dev.item(),
        "cp_drift_per_curve": cp_drift_per.cpu().numpy(),
        "cp_drift_max": cp_drift_per.max().item(),
        "cp_drift_min": cp_drift_per.min().item(),
    }


def track_drift_over_time(gt_cp: torch.Tensor, cp_history: list) -> dict:
    """
    Given a history of control point snapshots, compute drift trajectory.

    Args:
        gt_cp: (N, K, 3) ground-truth
        cp_history: list of (N, K, 3) tensors at different optimization steps

    Returns:
        dict with drift_trajectory, curvature_trajectory
    """
    drift_traj = []
    curv_traj = []

    for cp in cp_history:
        drift_traj.append(control_point_drift(gt_cp, cp).item())
        curv_traj.append(curvature_deviation(gt_cp.cpu(), cp.cpu()).item())

    return {
        "drift_trajectory": drift_traj,
        "curvature_trajectory": curv_traj,
    }
