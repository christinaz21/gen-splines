"""
spline.py — Cubic B-Spline parameterization for Generative Spline Fields.

Implements:
  - SplineField: a collection of N cubic B-spline curves, each with K control points.
  - Fully differentiable: gradients flow from sampled 3D points back to control points.
  - Curvature computation for drift metrics.
"""

import torch
import torch.nn as nn


# Cubic B-spline basis matrix (uniform knots)
# Converts [u^3, u^2, u, 1] into basis function weights for 4 consecutive control points
BSPLINE_MATRIX = (1.0 / 6.0) * torch.tensor([
    [-1.0,  3.0, -3.0,  1.0],
    [ 3.0, -6.0,  3.0,  0.0],
    [-3.0,  0.0,  3.0,  0.0],
    [ 1.0,  4.0,  1.0,  0.0],
], dtype=torch.float32)


def evaluate_bspline(control_points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Evaluate cubic B-spline curves.

    Args:
        control_points: (N, K, 3) — N curves, each with K control points in R^3
        num_samples: number of points to sample along each curve

    Returns:
        (N, num_samples, 3) — sampled curve points
    """
    N, K, D = control_points.shape
    n_seg = K - 3  # number of curve segments for cubic B-spline
    assert n_seg >= 1, f"Need at least 4 control points, got {K}"

    device = control_points.device
    M = BSPLINE_MATRIX.to(device)

    # Parameter values along the curve
    t = torch.linspace(0.0, 1.0, num_samples, device=device)
    t_scaled = t * n_seg
    # Clamp to avoid out-of-bounds at t=1.0
    seg_idx = t_scaled.long().clamp(0, n_seg - 1)
    u = t_scaled - seg_idx.float()

    # Basis: [u^3, u^2, u, 1] @ M -> (num_samples, 4)
    u_powers = torch.stack([u ** 3, u ** 2, u, torch.ones_like(u)], dim=-1)
    basis = u_powers @ M  # (num_samples, 4)

    # Gather 4 consecutive control points for each segment
    # idx: (num_samples, 4) — indices into the K control points
    offsets = torch.arange(4, device=device).unsqueeze(0)  # (1, 4)
    idx = seg_idx.unsqueeze(-1) + offsets  # (num_samples, 4)

    # Gather for all N curves: control_points[:, idx, :] -> (N, num_samples, 4, 3)
    P = control_points[:, idx, :]  # advanced indexing broadcasts over N

    # Weighted sum: basis (num_samples, 4) x P (N, num_samples, 4, 3) -> (N, num_samples, 3)
    points = torch.einsum('si, nsi d -> ns d', basis, P).contiguous()
    # Fix einsum spacing — use proper notation
    points = torch.einsum('si,nsid->nsd', basis, P)

    return points


class SplineField(nn.Module):
    """
    A collection of N cubic B-spline curves representing a 3D scene.

    Each curve has K control points in R^3. The control points are the
    learnable parameters — gradients from rendering flow back to them.

    Args:
        num_curves: number of curves in the scene
        control_points_per_curve: K, number of control points per curve (>= 4)
        dim: spatial dimension (default 3)
    """

    def __init__(self, num_curves: int = 50, control_points_per_curve: int = 8, dim: int = 3):
        super().__init__()
        self.num_curves = num_curves
        self.K = control_points_per_curve
        self.dim = dim

        # Initialize control points — small random values centered at origin
        self.control_points = nn.Parameter(
            torch.randn(num_curves, control_points_per_curve, dim) * 0.3
        )

    def forward(self, num_samples_per_curve: int = 64) -> torch.Tensor:
        """
        Sample all curves and return a flat point cloud.

        Args:
            num_samples_per_curve: M, points per curve

        Returns:
            (N * M, 3) flattened point cloud
        """
        points = evaluate_bspline(self.control_points, num_samples_per_curve)
        return points.reshape(-1, self.dim)

    def forward_per_curve(self, num_samples_per_curve: int = 64) -> torch.Tensor:
        """
        Sample all curves, keeping curve identity.

        Returns:
            (N, M, 3) — points organized by curve
        """
        return evaluate_bspline(self.control_points, num_samples_per_curve)

    def get_curve_colors(self) -> torch.Tensor:
        """
        Generate a unique, deterministic RGB color for each curve.
        Uses evenly-spaced hues so colors are maximally distinguishable.

        Returns:
            (N, 3) tensor of RGB colors in [0, 1]
        """
        N = self.num_curves
        device = self.control_points.device
        # Evenly spaced hues, full saturation and value
        hues = torch.linspace(0, 1, N + 1, device=device)[:N]
        colors = torch.zeros(N, 3, device=device)
        for i in range(N):
            h = hues[i].item()
            # HSV to RGB (simplified)
            c = 1.0
            x = 1.0 - abs((h * 6) % 2 - 1)
            if h < 1/6:
                r, g, b = c, x, 0
            elif h < 2/6:
                r, g, b = x, c, 0
            elif h < 3/6:
                r, g, b = 0, c, x
            elif h < 4/6:
                r, g, b = 0, x, c
            elif h < 5/6:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            colors[i] = torch.tensor([r, g, b], device=device)
        return colors

    def forward_with_colors(self, num_samples_per_curve: int = 64):
        """
        Sample all curves and return points + per-point colors.

        Each curve gets a unique RGB color. All points on the same curve
        share that color. This lets the renderer encode curve identity,
        which is CRITICAL for gradients to route to the correct control points.

        Returns:
            points: (N * M, 3) flattened point cloud
            colors: (N * M, 3) per-point RGB colors
        """
        per_curve = evaluate_bspline(self.control_points, num_samples_per_curve)  # (N, M, 3)
        N, M, D = per_curve.shape

        curve_colors = self.get_curve_colors()  # (N, 3)
        # Expand: each point on curve i gets color i
        point_colors = curve_colors.unsqueeze(1).expand(N, M, 3).reshape(-1, 3)

        return per_curve.reshape(-1, D), point_colors

    def compute_curvature(self, num_samples: int = 64) -> torch.Tensor:
        """
        Compute discrete curvature along each curve.
        Useful for the geometry-aware memory metric.

        Returns:
            (N, M-2) curvature values along each curve
        """
        points = self.forward_per_curve(num_samples)  # (N, M, 3)
        dt = 1.0 / (num_samples - 1)

        # First derivative (finite differences)
        d1 = (points[:, 1:] - points[:, :-1]) / dt  # (N, M-1, 3)

        # Second derivative
        d2 = (d1[:, 1:] - d1[:, :-1]) / dt  # (N, M-2, 3)

        # Curvature = |d1 x d2| / |d1|^3
        cross = torch.cross(d1[:, :-1], d2, dim=-1)  # (N, M-2, 3)
        curvature = cross.norm(dim=-1) / (d1[:, :-1].norm(dim=-1) ** 3 + 1e-8)

        return curvature

    def total_arc_length(self, num_samples: int = 64) -> torch.Tensor:
        """Compute arc length of each curve. Returns (N,)."""
        points = self.forward_per_curve(num_samples)
        diffs = points[:, 1:] - points[:, :-1]
        return diffs.norm(dim=-1).sum(dim=-1)


class SplineGenerator(nn.Module):
    """
    Generative model: latent vector z -> control points.
    This is Gθ(z, c) from the paper.

    For weeks 1-2 this is not needed — we optimize control points directly.
    This will be used in weeks 3-4 for training the generative prior.
    """

    def __init__(self, latent_dim: int = 64, num_curves: int = 50,
                 control_points_per_curve: int = 8, hidden_dim: int = 256):
        super().__init__()
        self.num_curves = num_curves
        self.K = control_points_per_curve
        out_dim = num_curves * control_points_per_curve * 3

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim) latent codes

        Returns:
            (B, num_curves, K, 3) control points
        """
        out = self.net(z)
        return out.view(-1, self.num_curves, self.K, 3)
