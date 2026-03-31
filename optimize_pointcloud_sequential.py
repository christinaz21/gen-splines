"""
optimize_pointcloud_sequential.py — Point-cloud baseline for sequential memory.

Baseline objective:
  - Use the same multi-view reprojection setup as spline optimization.
  - Remove spline structure entirely: optimize an unconstrained point cloud.
  - Compare memory stability and revisitation behavior against spline control points.
"""

import os
import time
import argparse
import torch
import torch.nn as nn

from spline import evaluate_bspline
from renderer import render_point_cloud
from dataset import create_combined_scene
from losses import reprojection_loss, anchor_loss


def chamfer_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Symmetric Chamfer distance between two point clouds."""
    dist = torch.cdist(x, y)
    return dist.min(dim=1).values.mean() + dist.min(dim=0).values.mean()


def local_shape_loss(points: torch.Tensor, nbr_idx: torch.Tensor, init_rel: torch.Tensor) -> torch.Tensor:
    """
    Preserve local neighborhood structure (a weak smoothness prior).
    This gives a fairer baseline than fully unconstrained points.
    """
    nbr_pts = points[nbr_idx]  # (P, k, 3)
    cur_rel = nbr_pts - points.unsqueeze(1)  # (P, k, 3)
    return (cur_rel - init_rel).pow(2).mean()


class PointCloudField(nn.Module):
    def __init__(self, init_points: torch.Tensor):
        super().__init__()
        self.points = nn.Parameter(init_points.clone())

    def forward(self) -> torch.Tensor:
        return self.points


def sequential_pointcloud_baseline(args):
    device = args.device
    print(f"\n{'='*68}")
    print("  Point-Cloud Baseline (Sequential Multi-View Reprojection)")
    print(f"  Views: {args.num_views}  |  Window: {args.view_window}  |  Anchor: {args.anchor_weight}")
    print(f"{'='*68}")

    # ------------------------------------------------------------------
    # 1) Build GT point cloud from spline scene (same source as spline method)
    # ------------------------------------------------------------------
    gt_cp = create_combined_scene(
        num_helix=args.num_helix, num_wave=args.num_wave, K=args.K, seed=args.seed
    ).to(device)
    gt_points = evaluate_bspline(gt_cp, args.samples_per_curve).reshape(-1, 3).detach()
    P = gt_points.shape[0]
    print(f"  GT scene: {gt_cp.shape[0]} curves -> {P} sampled points")

    azimuths = torch.linspace(0, 360, args.num_views + 1, device=device)[:-1]

    # ------------------------------------------------------------------
    # 2) Baseline init: noisy unconstrained points
    # ------------------------------------------------------------------
    init_points = gt_points + args.init_noise * torch.randn_like(gt_points)
    field = PointCloudField(init_points).to(device)
    anchor_pts = field.points.detach().clone()
    initial_point_drift = (field.points.detach() - gt_points).norm(dim=-1).mean().item()
    print(f"  Initial point drift: {initial_point_drift:.4f}")

    # Neighborhood prior from the initial state
    k = min(args.knn_k, P - 1)
    with torch.no_grad():
        dmat = torch.cdist(init_points, init_points)
        nn_idx = dmat.argsort(dim=1)[:, 1:k + 1]  # skip self
        init_rel = init_points[nn_idx] - init_points.unsqueeze(1)

    # ------------------------------------------------------------------
    # 3) Sequential optimization loop
    # ------------------------------------------------------------------
    optimizer = torch.optim.Adam(field.parameters(), lr=args.lr)
    os.makedirs(args.output_dir, exist_ok=True)

    view_reproj_losses = []
    view_chamfer = []
    view_point_drift = []
    view_local = []

    t_start = time.time()
    total_steps = 0

    for view_idx in range(args.num_views):
        az = azimuths[view_idx].item()

        window_azimuths = []
        for w in range(args.view_window):
            idx = view_idx - w
            if idx >= 0:
                window_azimuths.append(azimuths[idx].item())

        for _ in range(args.steps_per_view):
            optimizer.zero_grad()

            pred_points = field()
            gt_struct = gt_points.unsqueeze(0)    # (1, P, 3)
            pred_struct = pred_points.unsqueeze(0)

            loss_reproj = reprojection_loss(
                gt_struct, pred_struct, azimuths=window_azimuths, device=device
            )
            loss_anchor = anchor_loss(pred_points, anchor_pts)
            loss_local = local_shape_loss(pred_points, nn_idx, init_rel)

            loss = (
                loss_reproj
                + args.anchor_weight * loss_anchor
                + args.local_weight * loss_local
            )
            loss.backward()
            optimizer.step()
            total_steps += 1

        with torch.no_grad():
            anchor_pts = (
                (1.0 - args.anchor_momentum) * anchor_pts
                + args.anchor_momentum * field.points.detach().clone()
            )
            pred_now = field.points.detach()
            p_drift = (pred_now - gt_points).norm(dim=-1).mean().item()
            chamf = chamfer_distance(pred_now, gt_points).item()
            loc = local_shape_loss(pred_now, nn_idx, init_rel).item()

        view_point_drift.append(p_drift)
        view_chamfer.append(chamf)
        view_reproj_losses.append(loss_reproj.item())
        view_local.append(loc)

        elapsed = time.time() - t_start
        print(
            f"  View {view_idx+1:>3d}/{args.num_views} az={az:>6.1f}°  "
            f"reproj={loss_reproj.item():.6f}  "
            f"p_drift={p_drift:.4f}  chamfer={chamf:.4f}  [{elapsed:.1f}s]"
        )

    # ------------------------------------------------------------------
    # 4) Revisitation test
    # ------------------------------------------------------------------
    print("\n  --- Revisitation Test ---")
    revisit_losses = {}
    with torch.no_grad():
        pred_struct = field.points.detach().unsqueeze(0)
        gt_struct = gt_points.unsqueeze(0)
        for test_az in [0.0, 90.0, 180.0, 270.0]:
            rl = reprojection_loss(
                gt_struct, pred_struct, azimuths=[test_az], device=device
            ).item()
            revisit_losses[test_az] = rl
            print(f"  Azimuth {test_az:>5.0f}°: reproj loss = {rl:.6f}")
    avg_revisit = sum(revisit_losses.values()) / len(revisit_losses)

    # ------------------------------------------------------------------
    # 5) Final report + save
    # ------------------------------------------------------------------
    final_point_drift = view_point_drift[-1]
    drift_pct = (1.0 - final_point_drift / initial_point_drift) * 100.0
    print(f"\n  {'='*60}")
    print(f"  BASELINE RESULTS ({args.num_views} views)")
    print(f"  {'='*60}")
    print(f"  Point drift: {initial_point_drift:.4f} -> {final_point_drift:.4f} ({drift_pct:+.1f}%)")
    print(f"  Final Chamfer:        {view_chamfer[-1]:.6f}")
    print(f"  Avg revisit reproj:   {avg_revisit:.6f}")
    print(f"  Total steps:          {total_steps}")

    results = {
        "view_reproj_losses": view_reproj_losses,
        "view_point_drift": view_point_drift,
        "view_chamfer": view_chamfer,
        "view_local_shape": view_local,
        "revisit_losses": revisit_losses,
        "avg_revisit": avg_revisit,
        "initial_point_drift": initial_point_drift,
        "final_point_drift": final_point_drift,
        "final_chamfer": view_chamfer[-1],
        "args": vars(args),
    }
    torch.save(results, os.path.join(args.output_dir, "pointcloud_baseline_results.pt"))
    torch.save(field.points.detach().cpu(), os.path.join(args.output_dir, "pred_points.pt"))
    torch.save(gt_points.detach().cpu(), os.path.join(args.output_dir, "gt_points.pt"))

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        az_np = azimuths.detach().cpu().numpy()
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(az_np, view_point_drift, "b-o", markersize=3, label="Point drift")
        axes[0].axhline(y=initial_point_drift, color="r", linestyle="--", alpha=0.7, label="Initial")
        axes[0].set_xlabel("Azimuth (degrees)")
        axes[0].set_ylabel("Mean Point Drift")
        axes[0].set_title("Point Drift During Orbit")
        axes[0].legend()

        axes[1].plot(az_np, view_reproj_losses, "g-o", markersize=3)
        axes[1].set_xlabel("Azimuth (degrees)")
        axes[1].set_ylabel("Reprojection Loss")
        axes[1].set_title("Reprojection Loss per View")

        axes[2].plot(az_np, view_chamfer, "m-o", markersize=3, label="Chamfer")
        axes[2].plot(az_np, view_local, "c-", alpha=0.8, label="Local-shape loss")
        axes[2].set_xlabel("Azimuth (degrees)")
        axes[2].set_ylabel("Loss")
        axes[2].set_title("3D Geometry Metrics")
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "pointcloud_baseline.png"), dpi=150)
        plt.close()
        print(f"  Saved plot to {args.output_dir}/pointcloud_baseline.png")
    except Exception as e:
        print(f"  Could not save plot: {e}")

    # Render comparison images
    try:
        import imageio
        import numpy as np

        config = {"radius": args.render_radius, "image_size": args.image_size}
        with torch.no_grad():
            pred_pts = field.points.detach()
            for az in [0.0, 90.0, 180.0, 270.0]:
                gt_img = render_point_cloud(gt_points, azimuth=az, config=config, device=device)[..., :3]
                pred_img = render_point_cloud(pred_pts, azimuth=az, config=config, device=device)[..., :3]
                combined = torch.cat([gt_img, pred_img], dim=1)
                combined_np = (combined.cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
                imageio.imwrite(
                    os.path.join(args.output_dir, f"compare_az{az:.0f}.png"),
                    combined_np,
                )
        print("  Saved GT|Pred comparison images from 4 viewpoints")
    except Exception as e:
        print(f"  Could not save images: {e}")

    print(f"\n  All results saved to {args.output_dir}/")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Point-cloud sequential memory baseline")
    # Scene
    parser.add_argument("--num-helix", type=int, default=20)
    parser.add_argument("--num-wave", type=int, default=10)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    # Data / projection
    parser.add_argument("--samples-per-curve", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-views", type=int, default=36)
    parser.add_argument("--steps-per-view", type=int, default=100)
    parser.add_argument("--view-window", type=int, default=5)
    # Optimization
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--init-noise", type=float, default=0.15)
    # Regularization
    parser.add_argument("--anchor-weight", type=float, default=0.05)
    parser.add_argument("--anchor-momentum", type=float, default=0.3)
    parser.add_argument("--local-weight", type=float, default=0.02)
    parser.add_argument("--knn-k", type=int, default=8)
    # Output
    parser.add_argument("--render-radius", type=float, default=0.005)
    parser.add_argument("--output-dir", type=str, default="outputs/pointcloud_baseline")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    sequential_pointcloud_baseline(args)
