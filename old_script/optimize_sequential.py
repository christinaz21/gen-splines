"""
optimize_sequential.py — Sequential persistent curve memory via reprojection.

Simulates the real use case from the paper:
  1. Camera orbits a scene sequentially
  2. At each new viewpoint, optimize control points to match the observation
  3. Persistent memory (anchor regularization) prevents forgetting previous views
  4. Fixed topology: only control point POSITIONS change, not the curve structure

The key comparison for the paper:
  - Run WITHOUT anchor (anchor_weight=0): memory-free baseline
  - Run WITH anchor: persistent curve memory
  - The drift plot should show anchor version is more stable

Usage:
    # Without persistent memory (baseline):
    python optimize_sequential.py --anchor-weight 0.0 --output-dir outputs/seq_no_memory

    # With persistent memory:
    python optimize_sequential.py --anchor-weight 0.05 --output-dir outputs/seq_memory

    # Comparison with different anchor strengths:
    python optimize_sequential.py --anchor-weight 0.01 --output-dir outputs/seq_weak
    python optimize_sequential.py --anchor-weight 0.1 --output-dir outputs/seq_strong
"""

import os
import time
import argparse
import torch

from spline import SplineField, evaluate_bspline
from renderer import render_point_cloud
from dataset import create_combined_scene
from metrics import control_point_drift, curvature_deviation, compute_all_metrics
from losses import reprojection_loss, smoothness_loss, anchor_loss


def sequential_optimization(args):
    device = args.device
    print(f"\n{'='*65}")
    print(f"  Sequential Persistent Curve Memory (Reprojection)")
    print(f"  Views: {args.num_views}  |  Window: {args.view_window}  |  "
          f"Anchor: {args.anchor_weight}")
    print(f"{'='*65}")

    # ------------------------------------------------------------------
    # 1. Ground truth
    # ------------------------------------------------------------------
    gt_cp = create_combined_scene(
        num_helix=args.num_helix, num_wave=args.num_wave,
        K=args.K, seed=args.seed
    ).to(device)
    N = gt_cp.shape[0]
    print(f"  GT scene: {N} curves, {args.K} control points each")

    # Pre-compute GT curve points for all views
    gt_curve_points = evaluate_bspline(gt_cp, args.samples_per_curve)  # (N, M, 3)
    azimuths = torch.linspace(0, 360, args.num_views + 1)[:-1]
    print(f"  {args.num_views} viewpoints, {args.steps_per_view} steps each")

    # ------------------------------------------------------------------
    # 2. Initialize predicted field
    # ------------------------------------------------------------------
    pred_field = SplineField(N, args.K).to(device)
    pred_field.control_points.data = gt_cp.clone() + args.init_noise * torch.randn_like(gt_cp)
    initial_drift = control_point_drift(gt_cp, pred_field.control_points.data).item()
    print(f"  Initial CP drift: {initial_drift:.4f}")

    # Persistent memory anchor
    anchor_cp = pred_field.control_points.data.clone().detach()

    # ------------------------------------------------------------------
    # 3. Sequential loop
    # ------------------------------------------------------------------
    optimizer = torch.optim.Adam(pred_field.parameters(), lr=args.lr)
    os.makedirs(args.output_dir, exist_ok=True)

    view_drifts = []
    view_losses = []
    curvature_devs = []
    cp_snapshots = []

    t_start = time.time()
    total_steps = 0

    for view_idx in range(args.num_views):
        az = azimuths[view_idx].item()

        # Build view window: current + recent previous views
        window_azimuths = []
        for w in range(args.view_window):
            idx = view_idx - w
            if idx >= 0:
                window_azimuths.append(azimuths[idx].item())

        for step in range(args.steps_per_view):
            optimizer.zero_grad()

            pred_curve_points = evaluate_bspline(
                pred_field.control_points, args.samples_per_curve
            )

            # Reprojection loss over view window
            loss_reproj = reprojection_loss(
                gt_curve_points.detach(), pred_curve_points,
                azimuths=window_azimuths, device=device
            )

            # Smoothness
            loss_smooth = smoothness_loss(pred_field.control_points)

            # Anchor (persistent memory)
            loss_anch = anchor_loss(pred_field.control_points, anchor_cp)

            loss = loss_reproj + args.smooth_weight * loss_smooth + \
                   args.anchor_weight * loss_anch

            loss.backward()
            optimizer.step()
            total_steps += 1

        # Update anchor after each view (memory consolidation)
        with torch.no_grad():
            anchor_cp = (1 - args.anchor_momentum) * anchor_cp + \
                         args.anchor_momentum * pred_field.control_points.data.clone()

        # Record metrics
        with torch.no_grad():
            drift = control_point_drift(gt_cp, pred_field.control_points.data).item()
            curv_dev = curvature_deviation(
                gt_cp.cpu(), pred_field.control_points.data.cpu()
            ).item()

        view_drifts.append(drift)
        view_losses.append(loss_reproj.item())
        curvature_devs.append(curv_dev)
        cp_snapshots.append(pred_field.control_points.data.clone().cpu())

        elapsed = time.time() - t_start
        print(f"  View {view_idx+1:>3d}/{args.num_views}  "
              f"az={az:>6.1f}°  reproj={loss_reproj.item():.6f}  "
              f"CP_drift={drift:.4f}  [win={len(window_azimuths)}]  [{elapsed:.1f}s]")

    # ------------------------------------------------------------------
    # 4. Revisitation test
    # ------------------------------------------------------------------
    print(f"\n  --- Revisitation Test ---")
    revisit_drifts = {}
    with torch.no_grad():
        pred_curve_pts = evaluate_bspline(
            pred_field.control_points, args.samples_per_curve
        )
        for test_az in [0.0, 90.0, 180.0, 270.0]:
            rl = reprojection_loss(
                gt_curve_points, pred_curve_pts,
                azimuths=[test_az], device=device
            ).item()
            revisit_drifts[test_az] = rl
            print(f"  Azimuth {test_az:>5.0f}°: reproj loss = {rl:.6f}")

    avg_revisit = sum(revisit_drifts.values()) / len(revisit_drifts)

    # ------------------------------------------------------------------
    # 5. Final metrics
    # ------------------------------------------------------------------
    final_metrics = compute_all_metrics(
        gt_cp.cpu(), pred_field.control_points.data.cpu(), num_samples=64
    )

    drift_pct = (1 - final_metrics['cp_drift'] / initial_drift) * 100
    print(f"\n  {'='*60}")
    print(f"  RESULTS (sequential, {args.num_views} views)")
    print(f"  anchor_w={args.anchor_weight}, steps/view={args.steps_per_view}")
    print(f"  {'='*60}")
    print(f"  CP drift:  {initial_drift:.4f} -> {final_metrics['cp_drift']:.4f}  "
          f"({drift_pct:+.1f}%)")
    print(f"  Curvature deviation:    {final_metrics['curvature_deviation']:.6f}")
    print(f"  Avg revisit reproj:     {avg_revisit:.6f}")
    print(f"  Total steps:            {total_steps}")

    # Stability check
    half = len(view_drifts) // 2
    avg_first = sum(view_drifts[:half]) / half
    avg_second = sum(view_drifts[half:]) / (len(view_drifts) - half)
    if avg_second <= avg_first * 1.05:
        print(f"  Memory: STABLE (2nd half {avg_second:.4f} <= 1st half {avg_first:.4f})")
    else:
        print(f"  Memory: DRIFTING (2nd half {avg_second:.4f} > 1st half {avg_first:.4f})")

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    results = {
        "view_drifts": view_drifts,
        "view_losses": view_losses,
        "curvature_devs": curvature_devs,
        "revisit_losses": revisit_drifts,
        "initial_drift": initial_drift,
        "final_metrics": final_metrics,
        "azimuths": azimuths.tolist(),
        "args": vars(args),
    }
    torch.save(results, os.path.join(args.output_dir, "sequential_results.pt"))
    torch.save(cp_snapshots, os.path.join(args.output_dir, "cp_snapshots.pt"))
    torch.save(gt_cp.cpu(), os.path.join(args.output_dir, "gt_control_points.pt"))

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(azimuths.numpy(), view_drifts, "b-o", markersize=3)
        axes[0].axhline(y=initial_drift, color="r", linestyle="--",
                         label=f"Initial ({initial_drift:.3f})", alpha=0.7)
        axes[0].set_xlabel("Azimuth (degrees)")
        axes[0].set_ylabel("Control-Point Drift")
        axes[0].set_title("CP Drift During 360° Orbit")
        axes[0].legend()

        axes[1].plot(azimuths.numpy(), view_losses, "g-o", markersize=3)
        axes[1].set_xlabel("Azimuth (degrees)")
        axes[1].set_ylabel("Reprojection Loss")
        axes[1].set_title("Reprojection Loss per View")

        axes[2].plot(azimuths.numpy(), curvature_devs, "m-o", markersize=3)
        axes[2].set_xlabel("Azimuth (degrees)")
        axes[2].set_ylabel("Curvature Deviation")
        axes[2].set_title("Curvature Deviation During Orbit")

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "sequential_optimization.png"), dpi=150)
        plt.close()
        print(f"  Saved plot to {args.output_dir}/sequential_optimization.png")
    except Exception as e:
        print(f"  Could not save plot: {e}")

    print(f"\n  All results saved to {args.output_dir}/")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential spline optimization")
    # Scene
    parser.add_argument("--num-helix", type=int, default=20)
    parser.add_argument("--num-wave", type=int, default=10)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    # Projection
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--samples-per-curve", type=int, default=128)
    # Sequential
    parser.add_argument("--num-views", type=int, default=36)
    parser.add_argument("--steps-per-view", type=int, default=100)
    parser.add_argument("--view-window", type=int, default=5,
                        help="Views per optimization step (current + previous)")
    # Optimization
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--init-noise", type=float, default=0.15)
    # Regularization
    parser.add_argument("--anchor-weight", type=float, default=0.05,
                        help="0 = no memory (baseline), >0 = persistent memory")
    parser.add_argument("--smooth-weight", type=float, default=0.001)
    parser.add_argument("--anchor-momentum", type=float, default=0.3,
                        help="How fast anchor tracks (0=fixed, 1=instant)")
    # Output
    parser.add_argument("--output-dir", type=str, default="outputs/sequential")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    sequential_optimization(args)
