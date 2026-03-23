"""
optimize.py — Control point optimization via multi-view reprojection.

Implements the paper's update rule: P^(t+1) = P^(t) - η∇_P L

The critical change: instead of comparing rendered IMAGES (which destroys
curve structure and has massive depth ambiguity), we compare projected
CURVE POINTS in 2D screen space across multiple views.

Why this works:
  - Fixed curve correspondence (curve i ↔ curve i, point j ↔ point j)
  - Multiple views break depth ambiguity
  - Gradients flow directly: 2D error → 3D control points via camera projection
  - No rasterization bottleneck

Usage:
    # Standard run:
    python optimize.py --num-views 8 --num-steps 1000

    # Easy test first:
    python optimize.py --init-noise 0.05 --num-views 8 --num-steps 500

    # Compare with/without anchor:
    python optimize.py --anchor-weight 0.0 --output-dir outputs/no_anchor
    python optimize.py --anchor-weight 0.05 --output-dir outputs/with_anchor
"""

import os
import time
import argparse
import torch

from spline import SplineField, evaluate_bspline
from renderer import render_point_cloud
from dataset import create_combined_scene
from metrics import control_point_drift, compute_all_metrics
from losses import reprojection_loss, smoothness_loss, anchor_loss


def optimize(args):
    device = args.device
    print(f"\n{'='*65}")
    print(f"  Spline Optimization via Multi-View Reprojection")
    print(f"  {args.num_views} views  |  anchor={args.anchor_weight}  |  "
          f"smooth={args.smooth_weight}")
    print(f"{'='*65}")

    # ------------------------------------------------------------------
    # 1. Create ground-truth scene
    # ------------------------------------------------------------------
    gt_cp = create_combined_scene(
        num_helix=args.num_helix, num_wave=args.num_wave,
        K=args.K, seed=args.seed
    ).to(device)
    N = gt_cp.shape[0]
    print(f"  GT scene: {N} curves, {args.K} control points each")

    # Sample GT curves into 3D points (keeping curve structure)
    gt_curve_points = evaluate_bspline(gt_cp, args.samples_per_curve)  # (N, M, 3)
    print(f"  GT curve points: {gt_curve_points.shape}")

    # Camera viewpoints spread around the object
    azimuths = torch.linspace(0, 360, args.num_views + 1)[:-1].tolist()
    print(f"  Viewpoints: {len(azimuths)} azimuths")

    # ------------------------------------------------------------------
    # 2. Initialize predicted field (perturbed from GT)
    # ------------------------------------------------------------------
    pred_field = SplineField(N, args.K).to(device)
    pred_field.control_points.data = gt_cp.clone() + args.init_noise * torch.randn_like(gt_cp)
    initial_drift = control_point_drift(gt_cp, pred_field.control_points.data).item()
    print(f"  Initial noise: {args.init_noise}, initial CP drift: {initial_drift:.4f}")

    # Anchor for persistent memory
    anchor_cp = pred_field.control_points.data.clone().detach()

    # ------------------------------------------------------------------
    # 3. Optimization: P^(t+1) = P^(t) - η∇_P L
    # ------------------------------------------------------------------
    optimizer = torch.optim.Adam(pred_field.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)

    os.makedirs(args.output_dir, exist_ok=True)
    losses_log = []
    drifts_log = []
    component_log = {"reproj": [], "smooth": [], "anchor": []}

    t_start = time.time()

    for step in range(args.num_steps):
        optimizer.zero_grad()

        # Sample predicted curves (preserving curve structure)
        pred_curve_points = evaluate_bspline(
            pred_field.control_points, args.samples_per_curve
        )  # (N, M, 3)

        # --- Reprojection loss (primary) ---
        loss_reproj = reprojection_loss(
            gt_curve_points.detach(), pred_curve_points,
            azimuths=azimuths, device=device
        )

        # --- Smoothness loss ---
        loss_smooth = smoothness_loss(pred_field.control_points)

        # --- Anchor loss (persistent memory) ---
        loss_anch = anchor_loss(pred_field.control_points, anchor_cp)

        # --- Total: the paper's L that we differentiate ---
        loss = loss_reproj + args.smooth_weight * loss_smooth + args.anchor_weight * loss_anch

        loss.backward()
        optimizer.step()
        scheduler.step()

        # Track
        with torch.no_grad():
            drift = control_point_drift(gt_cp, pred_field.control_points.data).item()
            losses_log.append(loss.item())
            drifts_log.append(drift)
            component_log["reproj"].append(loss_reproj.item())
            component_log["smooth"].append(loss_smooth.item())
            component_log["anchor"].append(loss_anch.item())

        if step % args.log_every == 0:
            elapsed = time.time() - t_start
            print(f"  Step {step:>5d}/{args.num_steps}  "
                  f"loss={loss.item():.6f} (reproj={loss_reproj.item():.6f})  "
                  f"CP_drift={drift:.4f}  [{elapsed:.1f}s]")

    # ------------------------------------------------------------------
    # 4. Results
    # ------------------------------------------------------------------
    final_metrics = compute_all_metrics(
        gt_cp.cpu(), pred_field.control_points.data.cpu(), num_samples=64
    )

    print(f"\n  {'='*60}")
    print(f"  RESULTS ({args.num_views} views, reprojection loss)")
    print(f"  {'='*60}")
    print(f"  CP drift:  {initial_drift:.4f} -> {final_metrics['cp_drift']:.4f}")
    drift_pct = (1 - final_metrics['cp_drift'] / initial_drift) * 100
    print(f"  Drift change:        {drift_pct:+.1f}%  "
          f"({'IMPROVED' if drift_pct > 0 else 'WORSE'})")
    print(f"  Curvature deviation: {final_metrics['curvature_deviation']:.6f}")
    print(f"  CP drift max/min:    {final_metrics['cp_drift_max']:.4f} / "
          f"{final_metrics['cp_drift_min']:.4f}")

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    results = {
        "losses": losses_log,
        "drifts": drifts_log,
        "components": component_log,
        "final_metrics": final_metrics,
        "initial_drift": initial_drift,
        "args": vars(args),
    }
    torch.save(results, os.path.join(args.output_dir, "results.pt"))
    torch.save(pred_field.control_points.data.cpu(),
               os.path.join(args.output_dir, "pred_control_points.pt"))
    torch.save(gt_cp.cpu(), os.path.join(args.output_dir, "gt_control_points.pt"))

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(component_log["reproj"], label="Reprojection", alpha=0.8)
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Reprojection Loss")
        axes[0].set_yscale("log")

        axes[1].plot(drifts_log, "b-")
        axes[1].axhline(y=initial_drift, color="r", linestyle="--",
                         label=f"Initial ({initial_drift:.3f})", alpha=0.7)
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("CP Drift (L2)")
        axes[1].set_title("Control-Point Drift")
        axes[1].legend()

        axes[2].plot(component_log["reproj"], label="Reproj", alpha=0.7)
        axes[2].plot(component_log["smooth"], label="Smooth", alpha=0.7)
        axes[2].plot(component_log["anchor"], label="Anchor", alpha=0.7)
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("Loss")
        axes[2].set_title("Loss Components")
        axes[2].set_yscale("log")
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "optimization.png"), dpi=150)
        plt.close()
        print(f"  Saved plot to {args.output_dir}/optimization.png")
    except Exception as e:
        print(f"  Could not save plot: {e}")

    # Render comparison images
    try:
        import imageio
        import numpy as np
        config = {"radius": 0.005, "image_size": args.image_size}
        with torch.no_grad():
            gt_pts = evaluate_bspline(gt_cp, args.samples_per_curve).reshape(-1, 3)
            pred_pts = pred_field(args.samples_per_curve)
            for az in [0.0, 90.0, 180.0, 270.0]:
                gt_img = render_point_cloud(gt_pts, azimuth=az, config=config,
                                            device=device)[..., :3]
                pred_img = render_point_cloud(pred_pts, azimuth=az, config=config,
                                              device=device)[..., :3]
                # Side by side
                combined = torch.cat([gt_img, pred_img], dim=1)
                combined_np = (combined.cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
                imageio.imwrite(
                    os.path.join(args.output_dir, f"compare_az{az:.0f}.png"), combined_np)
        print(f"  Saved GT|Pred comparison images from 4 viewpoints")
    except Exception as e:
        print(f"  Could not save images: {e}")

    print(f"\n  All results saved to {args.output_dir}/")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spline optimization via reprojection")
    # Scene
    parser.add_argument("--num-helix", type=int, default=20)
    parser.add_argument("--num-wave", type=int, default=10)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    # Rendering / projection
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--samples-per-curve", type=int, default=128)
    parser.add_argument("--num-views", type=int, default=8,
                        help="Number of viewpoints for reprojection (more = less ambiguity)")
    # Optimization
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--init-noise", type=float, default=0.15)
    # Regularization
    parser.add_argument("--anchor-weight", type=float, default=0.0,
                        help="Persistent memory strength (try 0, 0.01, 0.05)")
    parser.add_argument("--smooth-weight", type=float, default=0.001,
                        help="Smoothness regularization")
    # Logging
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="outputs/reproj")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    optimize(args)
