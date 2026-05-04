"""
optimize_v2.py — Enhanced sequential optimization with techniques to push
drift reduction well beyond 53% (reprojection) / 66% (total).

New techniques beyond v1:
  1. Multi-view reprojection loss with OVERLAPPING view pairs (not just current view)
  2. Curvature regularization — penalize curvature changes from anchor state
  3. Adaptive learning rate per-curve based on gradient confidence
  4. Momentum-damped anchor updates (EMA blending instead of hard replacement)

Usage:
    python optimize_v2.py --data-source yuksel --model-name straight --num-curves 50
    python optimize_v2.py --data-source synthetic --num-curves 30
"""

import os
import sys
import time
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# These imports assume existing codebase files are in the same directory
from spline import SplineField, evaluate_bspline


# =========================================================================
# Loss functions — the core improvements
# =========================================================================

def multi_view_reprojection_loss(pred_cp, gt_projections, cameras_list,
                                  num_samples=64, image_size=256):
    """
    Reprojection loss against MULTIPLE views simultaneously.

    Instead of optimizing against only the current view, we maintain a buffer
    of recent views and compute reprojection error across all of them.
    This provides much stronger geometric constraints — a single view has
    depth ambiguity, but 2-3 views triangulate geometry.

    Args:
        pred_cp: (N, K, 3) predicted control points
        gt_projections: list of (P, 2) ground-truth 2D projections per view
        cameras_list: list of PyTorch3D cameras
        num_samples: samples per curve for projection
        image_size: int, image dimension for screen projection

    Returns:
        scalar loss
    """
    pred_points = evaluate_bspline(pred_cp, num_samples)  # (N, M, 3)
    pred_flat = pred_points.reshape(-1, 3)  # (N*M, 3)

    total_loss = 0.0
    for gt_2d, cam in zip(gt_projections, cameras_list):
        pred_2d = cam.transform_points_screen(
            pred_flat.unsqueeze(0),
            image_size=((image_size, image_size),)
        )[0, :, :2]
        total_loss = total_loss + F.mse_loss(pred_2d, gt_2d)

    return total_loss / len(cameras_list)


def curvature_regularization(pred_cp, anchor_cp, num_samples=64, weight=0.1):
    """
    Penalize curvature deviation from the anchor (persistent memory) state.

    This prevents optimization from producing geometrically valid but
    topologically different curves. A curve can match an image perfectly
    while having completely wrong curvature — this loss prevents that.

    Args:
        pred_cp: (N, K, 3) current control points
        anchor_cp: (N, K, 3) anchor control points (from memory)
        num_samples: samples for curvature computation

    Returns:
        scalar curvature deviation loss
    """
    def _compute_curvature(cp):
        points = evaluate_bspline(cp, num_samples)  # (N, M, 3)
        dt = 1.0 / (num_samples - 1)
        d1 = (points[:, 1:] - points[:, :-1]) / dt
        d2 = (d1[:, 1:] - d1[:, :-1]) / dt
        cross = torch.cross(d1[:, :-1], d2, dim=-1)
        curvature = cross.norm(dim=-1) / (d1[:, :-1].norm(dim=-1) ** 3 + 1e-8)
        return curvature

    pred_curv = _compute_curvature(pred_cp)
    anchor_curv = _compute_curvature(anchor_cp.detach())

    return weight * F.mse_loss(pred_curv, anchor_curv)


def tangent_consistency_loss(pred_cp, num_samples=64, weight=0.05):
    """
    Encourage smooth tangent fields along each curve.

    Sudden tangent direction changes indicate optimization has
    pushed control points into geometrically implausible configurations.

    Returns:
        scalar tangent smoothness loss
    """
    points = evaluate_bspline(pred_cp, num_samples)  # (N, M, 3)
    tangents = points[:, 1:] - points[:, :-1]  # (N, M-1, 3)
    tangents = F.normalize(tangents, dim=-1)

    # Cosine similarity between consecutive tangent vectors
    cos_sim = (tangents[:, 1:] * tangents[:, :-1]).sum(dim=-1)  # (N, M-2)
    # Penalize sharp turns (cos_sim close to -1)
    return weight * (1.0 - cos_sim).mean()


def anchor_proximity_loss(pred_cp, anchor_cp, weight=0.01):
    """
    Soft elastic constraint: control points should stay near their
    anchor positions. Prevents catastrophic drift during aggressive optimization.

    This is DIFFERENT from just reducing the learning rate — it creates
    a potential well around the anchor that gets stronger as you drift further.
    """
    return weight * (pred_cp - anchor_cp.detach()).norm(dim=-1).mean()


# =========================================================================
# Enhanced persistent memory with EMA anchors
# =========================================================================

class PersistentCurveMemory:
    """
    Manages the persistent curve memory with EMA (exponential moving average)
    anchor updates instead of hard replacement.

    v1 problem: After optimizing at view t, we set anchor = optimized_cp.
    This means early noisy estimates propagate forward.

    v2 solution: anchor = alpha * optimized_cp + (1-alpha) * old_anchor
    This smooths the memory updates and prevents single-view noise from
    corrupting the anchor.
    """

    def __init__(self, initial_cp: torch.Tensor, ema_decay: float = 0.7):
        """
        Args:
            initial_cp: (N, K, 3) initial control points
            ema_decay: weight for old anchor (0.7 = 70% old, 30% new)
        """
        self.anchor = initial_cp.clone().detach()
        self.ema_decay = ema_decay
        self.update_count = 0
        self.history = [initial_cp.clone().cpu()]

    def update(self, new_cp: torch.Tensor):
        """Update anchor with EMA blending."""
        self.anchor = (
            self.ema_decay * self.anchor +
            (1 - self.ema_decay) * new_cp.detach()
        )
        self.update_count += 1
        self.history.append(new_cp.clone().cpu())

    def get_anchor(self) -> torch.Tensor:
        return self.anchor.clone()


# =========================================================================
# Per-curve adaptive learning rate
# =========================================================================

class PerCurveAdaptiveLR:
    """
    Curves that receive strong, consistent gradients get higher LR.
    Curves in occluded or ambiguous regions get lower LR to avoid drift.

    This prevents the well-observed problem where some curves get pushed
    aggressively by sparse pixel gradients while others barely move.
    """

    def __init__(self, num_curves: int, K: int, base_lr: float = 5e-4,
                 momentum: float = 0.9, min_scale: float = 0.1, max_scale: float = 3.0):
        self.base_lr = base_lr
        self.momentum = momentum
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.grad_ema = torch.zeros(num_curves)  # running avg of grad norms per curve

    def compute_scales(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grad: (N, K, 3) gradient tensor

        Returns:
            (N, 1, 1) per-curve LR scale factors
        """
        grad_norms = grad.reshape(grad.shape[0], -1).norm(dim=-1).cpu()  # (N,)

        self.grad_ema = self.momentum * self.grad_ema + (1 - self.momentum) * grad_norms

        # Normalize: mean gradient gets scale=1
        mean_grad = self.grad_ema.mean() + 1e-8
        scales = self.grad_ema / mean_grad
        scales = scales.clamp(self.min_scale, self.max_scale)

        return scales.to(grad.device).reshape(-1, 1, 1)

    def step(self, param: nn.Parameter, grad: torch.Tensor):
        """Apply per-curve scaled gradient update."""
        scales = self.compute_scales(grad)
        param.data -= self.base_lr * scales * grad


# =========================================================================
# Main optimization loop
# =========================================================================

def run_enhanced_sequential(args):
    device = args.device

    print(f"\n{'='*70}")
    print(f"  Enhanced Sequential Optimization v2")
    print(f"  Data source: {args.data_source} | Model: {args.model_name}")
    print(f"{'='*70}")

    # ------------------------------------------------------------------
    # 1. Load data — real or synthetic
    # ------------------------------------------------------------------
    if args.data_source == "yuksel":
        from hair_loader import download_yuksel_hair, load_hair_file, hair_to_spline_field
        hair_path = download_yuksel_hair(args.model_name, save_dir=args.data_dir)
        strands = load_hair_file(hair_path)
        gt_cp = hair_to_spline_field(
            strands, num_curves=args.num_curves, K=args.K,
            seed=args.seed, strategy=args.strand_strategy
        ).to(device)
    elif args.data_source == "synthetic":
        from dataset import create_combined_scene
        gt_cp = create_combined_scene(
            num_helix=args.num_curves // 2,
            num_wave=args.num_curves - args.num_curves // 2,
            K=args.K, seed=args.seed
        ).to(device)
    elif args.data_source == "file":
        gt_cp = torch.load(args.cp_file).to(device)
    else:
        raise ValueError(f"Unknown data source: {args.data_source}")

    N, K, _ = gt_cp.shape
    print(f"  Scene: {N} curves × {K} control points")

    # ------------------------------------------------------------------
    # 2. Build GT SplineField and render 360° views
    # ------------------------------------------------------------------
    from renderer import render_point_cloud
    from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras

    gt_field = SplineField(N, K).to(device)
    gt_field.control_points.data = gt_cp.clone()

    config = {"radius": args.radius, "image_size": args.image_size}
    azimuths = torch.linspace(0, 360 - 360 / args.num_views, args.num_views)

    print(f"  Rendering {args.num_views} GT views ...")
    gt_images = []
    gt_cameras = []
    gt_projections = []

    num_samples = args.samples_per_curve
    gt_points_3d = gt_field.forward_per_curve(num_samples)  # (N, M, 3)
    gt_flat = gt_points_3d.reshape(-1, 3)

    for az in azimuths:
        # Render image
        with torch.no_grad():
            pts = gt_field(num_samples)
            img = render_point_cloud(pts, azimuth=az.item(), config=config, device=device)
            gt_images.append(img[..., :3] if img.shape[-1] >= 3 else img)

        # Camera for reprojection
        R, T = look_at_view_transform(dist=4.0, elev=30.0, azim=az.item())
        cam = FoVPerspectiveCameras(
            device=device, R=R, T=T,
            fov=60.0, aspect_ratio=1.0,
            znear=0.1, zfar=100.0
        )
        gt_cameras.append(cam)

        # GT 2D projections for reprojection loss
        with torch.no_grad():
            proj_2d = cam.transform_points_screen(
                gt_flat.unsqueeze(0),
                image_size=((args.image_size, args.image_size),)
            )[0, :, :2]
            gt_projections.append(proj_2d)

    gt_images = [img.detach() for img in gt_images]
    print(f"  Done. Image shape: {gt_images[0].shape}")

    # ------------------------------------------------------------------
    # 3. Initialize prediction (noisy) + memory
    # ------------------------------------------------------------------
    pred_field = SplineField(N, K).to(device)
    pred_field.control_points.data = gt_cp.clone() + args.init_noise * torch.randn_like(gt_cp)

    from metrics import control_point_drift, compute_all_metrics
    initial_drift = control_point_drift(gt_cp, pred_field.control_points.data).item()
    print(f"  Initial CP drift: {initial_drift:.4f}")

    # Initialize enhanced components
    memory = PersistentCurveMemory(pred_field.control_points.data, ema_decay=args.ema_decay)
    adaptive_lr = PerCurveAdaptiveLR(N, K, base_lr=args.lr)

    # ------------------------------------------------------------------
    # 4. Sequential optimization loop
    # ------------------------------------------------------------------
    view_drifts = []
    view_losses = []
    curvature_devs = []
    cp_snapshots = []

    total_steps = 0
    view_buffer_size = args.view_buffer  # how many recent views to use for reprojection

    print(f"\n  Starting sequential optimization:")
    print(f"    Views: {args.num_views}, Steps/view: {args.steps_per_view}")
    print(f"    View buffer: {view_buffer_size}, EMA decay: {args.ema_decay}")
    print(f"    Curvature reg: {args.curv_weight}, Tangent reg: {args.tangent_weight}")
    print(f"    Anchor proximity: {args.anchor_weight}")

    for v_idx in range(args.num_views):
        az = azimuths[v_idx].item()

        # Determine which views to use for reprojection (current + recent)
        buffer_start = max(0, v_idx - view_buffer_size + 1)
        buffer_indices = list(range(buffer_start, v_idx + 1))

        # Reset optimizer each view (important: fresh momentum)
        optimizer = torch.optim.Adam([pred_field.control_points], lr=args.lr)

        for step in range(args.steps_per_view):
            optimizer.zero_grad()

            pred_points = pred_field(num_samples)

            # --- Loss 1: Rendered image loss (current view) ---
            pred_image = render_point_cloud(
                pred_points, azimuth=az, config=config, device=device
            )
            pred_rgb = pred_image[..., :3] if pred_image.shape[-1] >= 3 else pred_image
            loss_render = F.mse_loss(pred_rgb, gt_images[v_idx])

            # --- Loss 2: Multi-view reprojection (buffered views) ---
            loss_reproj = multi_view_reprojection_loss(
                pred_field.control_points,
                [gt_projections[i] for i in buffer_indices],
                [gt_cameras[i] for i in buffer_indices],
                num_samples=num_samples,
                image_size=args.image_size
            )

            # --- Loss 3: Curvature regularization ---
            loss_curv = curvature_regularization(
                pred_field.control_points, memory.get_anchor(),
                num_samples=num_samples, weight=args.curv_weight
            )

            # --- Loss 4: Tangent consistency ---
            loss_tangent = tangent_consistency_loss(
                pred_field.control_points, num_samples=num_samples,
                weight=args.tangent_weight
            )

            # --- Loss 5: Anchor proximity ---
            loss_anchor = anchor_proximity_loss(
                pred_field.control_points, memory.get_anchor(),
                weight=args.anchor_weight
            )

            # Total loss
            loss = (args.render_weight * loss_render +
                    args.reproj_weight * loss_reproj +
                    loss_curv + loss_tangent + loss_anchor)

            loss.backward()

            # Apply per-curve adaptive LR (optional)
            if args.use_adaptive_lr:
                grad = pred_field.control_points.grad.data.clone()
                optimizer.zero_grad()
                adaptive_lr.step(pred_field.control_points, grad)
            else:
                optimizer.step()

            total_steps += 1

        # Update persistent memory (EMA)
        memory.update(pred_field.control_points.data)

        # Track metrics
        with torch.no_grad():
            drift = control_point_drift(gt_cp, pred_field.control_points.data).item()
            view_drifts.append(drift)
            view_losses.append(loss.item())

            from spline import SplineField as SF_temp
            gt_f = SF_temp(N, K); gt_f.control_points.data = gt_cp.cpu()
            pr_f = SF_temp(N, K); pr_f.control_points.data = pred_field.control_points.data.cpu()
            c_dev = (gt_f.compute_curvature(64) - pr_f.compute_curvature(64)).abs().mean().item()
            curvature_devs.append(c_dev)

            cp_snapshots.append(pred_field.control_points.data.clone().cpu())

        if v_idx % max(1, args.num_views // 10) == 0 or v_idx == args.num_views - 1:
            print(f"    View {v_idx:3d}/{args.num_views} | az={az:6.1f}° | "
                  f"drift={drift:.4f} | loss={loss.item():.6f} | "
                  f"buf={len(buffer_indices)} views")

    # ------------------------------------------------------------------
    # 5. Revisitation test
    # ------------------------------------------------------------------
    print(f"\n  --- Revisitation Test ---")
    with torch.no_grad():
        revisit_pts = pred_field(num_samples)
        revisit_img = render_point_cloud(
            revisit_pts, azimuth=azimuths[0].item(), config=config, device=device
        )
        revisit_rgb = revisit_img[..., :3] if revisit_img.shape[-1] >= 3 else revisit_img
        revisit_loss = F.mse_loss(revisit_rgb, gt_images[0]).item()
    print(f"  Revisitation loss: {revisit_loss:.6f}")

    # ------------------------------------------------------------------
    # 6. Final results
    # ------------------------------------------------------------------
    final_metrics = compute_all_metrics(
        gt_cp.cpu(), pred_field.control_points.data.cpu(), num_samples=64
    )

    drift_reduction_pct = (1 - final_metrics['cp_drift'] / initial_drift) * 100

    print(f"\n  {'='*60}")
    print(f"  RESULTS — Enhanced Sequential v2")
    print(f"  {'='*60}")
    print(f"  Data: {args.data_source} / {args.model_name}")
    print(f"  Initial CP drift:       {initial_drift:.4f}")
    print(f"  Final CP drift:         {final_metrics['cp_drift']:.4f}")
    print(f"  Drift reduction:        {drift_reduction_pct:.1f}%")
    print(f"  Curvature deviation:    {final_metrics['curvature_deviation']:.6f}")
    print(f"  Revisitation loss:      {revisit_loss:.6f}")
    print(f"  Total steps:            {total_steps}")
    print(f"  {'='*60}")

    # ------------------------------------------------------------------
    # 7. Save everything
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    results = {
        "data_source": args.data_source,
        "model_name": args.model_name,
        "num_curves": N,
        "K": K,
        "initial_drift": initial_drift,
        "final_drift": final_metrics["cp_drift"],
        "drift_reduction_pct": drift_reduction_pct,
        "curvature_deviation": final_metrics["curvature_deviation"],
        "revisitation_loss": revisit_loss,
        "view_drifts": view_drifts,
        "view_losses": view_losses,
        "curvature_devs": curvature_devs,
        "azimuths": azimuths.tolist(),
        "args": vars(args),
    }

    torch.save(results, os.path.join(args.output_dir, "results.pt"))
    torch.save(cp_snapshots, os.path.join(args.output_dir, "cp_snapshots.pt"))
    torch.save(gt_cp.cpu(), os.path.join(args.output_dir, "gt_cp.pt"))
    torch.save(pred_field.control_points.data.cpu(),
               os.path.join(args.output_dir, "final_cp.pt"))

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({k: v for k, v in results.items()
                    if not isinstance(v, (list, np.ndarray))}, f, indent=2)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 4, figsize=(20, 4))

        axes[0].plot(azimuths.numpy(), view_drifts, "b-o", markersize=2)
        axes[0].axhline(y=initial_drift, color="r", linestyle="--",
                         label=f"Initial ({initial_drift:.3f})", alpha=0.7)
        axes[0].set_xlabel("Azimuth (°)")
        axes[0].set_ylabel("CP Drift")
        axes[0].set_title("Control-Point Drift")
        axes[0].legend()

        axes[1].plot(azimuths.numpy(), view_losses, "g-o", markersize=2)
        axes[1].set_xlabel("Azimuth (°)")
        axes[1].set_ylabel("Total Loss")
        axes[1].set_title("Loss per View")

        axes[2].plot(azimuths.numpy(), curvature_devs, "m-o", markersize=2)
        axes[2].set_xlabel("Azimuth (°)")
        axes[2].set_ylabel("Curvature Dev")
        axes[2].set_title("Curvature Deviation")

        # Drift reduction over time
        reductions = [(1 - d / initial_drift) * 100 for d in view_drifts]
        axes[3].plot(azimuths.numpy(), reductions, "r-o", markersize=2)
        axes[3].axhline(y=53, color="gray", linestyle=":", label="v1 reproj (53%)", alpha=0.5)
        axes[3].axhline(y=66, color="gray", linestyle="--", label="v1 total (66%)", alpha=0.5)
        axes[3].set_xlabel("Azimuth (°)")
        axes[3].set_ylabel("Drift Reduction (%)")
        axes[3].set_title("Improvement vs v1")
        axes[3].legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "results.png"), dpi=150)
        plt.close()
        print(f"  Plot saved to {args.output_dir}/results.png")
    except Exception as e:
        print(f"  Plot failed: {e}")

    return results


# =========================================================================
# CLI
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced sequential optimization v2")

    # Data
    parser.add_argument("--data-source", type=str, default="yuksel",
                        choices=["yuksel", "synthetic", "file"])
    parser.add_argument("--model-name", type=str, default="straight",
                        help="Yuksel model: straight, wCurly, wWavy, wStraight, etc.")
    parser.add_argument("--cp-file", type=str, default=None,
                        help="Path to pre-computed control points .pt file")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--strand-strategy", type=str, default="diverse",
                        choices=["random", "diverse", "longest"])
    parser.add_argument("--num-curves", type=int, default=50)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    # Rendering
    parser.add_argument("--radius", type=float, default=0.02)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--samples-per-curve", type=int, default=64)

    # Optimization
    parser.add_argument("--num-views", type=int, default=36)
    parser.add_argument("--steps-per-view", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--init-noise", type=float, default=0.15)

    # Loss weights
    parser.add_argument("--render-weight", type=float, default=1.0)
    parser.add_argument("--reproj-weight", type=float, default=2.0,
                        help="Reprojection loss weight (higher = more geometric)")
    parser.add_argument("--curv-weight", type=float, default=0.1,
                        help="Curvature regularization weight")
    parser.add_argument("--tangent-weight", type=float, default=0.05,
                        help="Tangent consistency weight")
    parser.add_argument("--anchor-weight", type=float, default=0.01,
                        help="Anchor proximity weight")

    # Enhanced features
    parser.add_argument("--view-buffer", type=int, default=3,
                        help="Number of recent views for multi-view reprojection")
    parser.add_argument("--ema-decay", type=float, default=0.7,
                        help="EMA decay for anchor updates (0.7 = smooth)")
    parser.add_argument("--use-adaptive-lr", action="store_true",
                        help="Enable per-curve adaptive learning rate")

    # Output
    parser.add_argument("--output-dir", type=str, default="outputs/v2_enhanced")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    run_enhanced_sequential(args)
