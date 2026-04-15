"""
run_pointcloud_baseline.py — Point-cloud baseline for dense hair reconstruction.

Purpose:
  Compare a direct point-cloud parameterization against spline control-point
  optimization under a similar multi-view optimization setup.

This script:
  1) Loads a Yuksel hair model and fits spline control points (same upstream data prep
     as run_dense.py so comparisons are apples-to-apples).
  2) Converts GT splines to a GT point cloud.
  3) Initializes a noisy predicted point cloud and optimizes it across views using:
       - Render loss (current view)
       - Multi-view reprojection loss (view buffer)
       - EMA anchor proximity (persistent memory analogue)
  4) Saves metrics + plots + a comparison still.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hair_loader import download_yuksel_hair, load_hair_file, hair_to_spline_field
from renderer import render_point_cloud
from spline import evaluate_bspline


def log(msg):
    print(msg, flush=True)


def orient_cp(cp):
    """Yuksel Y-up -> PyTorch3D convention (x, z, -y)."""
    out = cp.clone()
    new_y = out[..., 2].clone()
    new_z = -out[..., 1].clone()
    out[..., 1] = new_y
    out[..., 2] = new_z
    return out


def orient_pts(points):
    out = points.clone()
    new_y = out[..., 2].clone()
    new_z = -out[..., 1].clone()
    out[..., 1] = new_y
    out[..., 2] = new_z
    return out


BLONDE = (0.82, 0.72, 0.42)


def blonde_colors(num_points_per_curve_list, base=BLONDE):
    """Generate blonde per-point colors with subtle strand variation."""
    if isinstance(num_points_per_curve_list, torch.Tensor):
        n, m, _ = num_points_per_curve_list.shape
        counts = [m] * n
    else:
        counts = num_points_per_curve_list
    rng = np.random.RandomState(42)
    cols = []
    for m in counts:
        jitter = rng.uniform(-0.06, 0.06, size=3)
        strand_color = np.clip(np.array(base) + jitter, 0, 1)
        t = np.linspace(0, 1, m)
        darken = 1.0 - 0.15 * t
        cols.append(np.outer(darken, strand_color))
    return torch.tensor(np.concatenate(cols), dtype=torch.float32)


def render_pts(points, colors, az, image_size, radius, device, elev=25.0, dist=3.5):
    from pytorch3d.renderer import (
        AlphaCompositor,
        FoVPerspectiveCameras,
        PointsRasterizationSettings,
        PointsRasterizer,
        PointsRenderer,
        look_at_view_transform,
    )
    from pytorch3d.structures import Pointclouds

    r, t = look_at_view_transform(dist=dist, elev=elev, azim=az)
    cam = FoVPerspectiveCameras(device=device, R=r, T=t)
    rs = PointsRasterizationSettings(
        image_size=image_size,
        radius=radius,
        points_per_pixel=10,
        bin_size=0,
    )
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cam, raster_settings=rs),
        compositor=AlphaCompositor(background_color=(0.03, 0.03, 0.05)),
    )
    pc = Pointclouds(points=[points.to(device)], features=[colors.to(device)])
    return renderer(pc)[0, ..., :3].cpu().numpy()


def render_blonde_points(points_flat, points_per_curve, az, args):
    colors = blonde_colors([points_per_curve] * (points_flat.shape[0] // points_per_curve))
    return render_pts(
        points_flat,
        colors,
        az=az,
        image_size=args.vis_image_size,
        radius=args.vis_radius,
        device=args.device,
        elev=args.elevation,
        dist=args.dist,
    )


def render_dense_gt_blonde(strands, az, args):
    from hair_loader import subsample_strands

    n_dense = min(args.dense_gt_strands, len(strands))
    sel = subsample_strands(strands, n_dense, strategy="random", seed=99, min_length=5)
    pts_list, pt_counts = [], []
    for s in sel:
        m = len(s)
        p = s[np.linspace(0, m - 1, min(args.dense_gt_points_per_strand, m), dtype=int)] if m > args.dense_gt_points_per_strand else s
        pts_list.append(p)
        pt_counts.append(len(p))

    points = torch.tensor(np.concatenate(pts_list), dtype=torch.float32)
    colors = blonde_colors(pt_counts)

    c = points.mean(0)
    points -= c
    md = points.norm(dim=-1).max()
    if md > 1e-6:
        points /= md
    points = orient_pts(points)

    return render_pts(
        points,
        colors,
        az=az,
        image_size=args.vis_image_size,
        radius=args.vis_radius * 0.5,
        device=args.device,
        elev=args.elevation,
        dist=args.dist,
    )


def make_render_config(args):
    return {
        "image_size": args.image_size,
        "radius": args.radius,
        "points_per_pixel": args.points_per_pixel,
        "dist": args.cam_dist,
        "fov": args.fov,
        "compositor": "alpha",
    }


def make_camera(azimuth, device, dist, elev, fov):
    from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras

    r, t = look_at_view_transform(dist=dist, elev=elev, azim=azimuth)
    return FoVPerspectiveCameras(device=device, R=r, T=t, fov=fov, aspect_ratio=1.0)


def multi_view_reprojection_loss_points(pred_points, gt_projections, cameras, image_size):
    total = 0.0
    for gt_2d, cam in zip(gt_projections, cameras):
        pred_2d = cam.transform_points_screen(
            pred_points.unsqueeze(0), image_size=((image_size, image_size),)
        )[0, :, :2]
        total = total + F.mse_loss(pred_2d, gt_2d)
    return total / len(cameras)


def anchor_proximity_loss_points(pred_points, anchor_points, weight):
    return weight * (pred_points - anchor_points.detach()).norm(dim=-1).mean()


class PersistentPointMemory:
    def __init__(self, initial_points, ema_decay):
        self.anchor = initial_points.clone().detach()
        self.ema_decay = ema_decay

    def update(self, new_points):
        self.anchor = self.ema_decay * self.anchor + (1.0 - self.ema_decay) * new_points.detach()

    def get_anchor(self):
        return self.anchor.clone()


def paired_point_drift(gt_points, pred_points):
    return (gt_points - pred_points).norm(dim=-1).mean()


def chamfer_distance_symmetric(a, b):
    """
    Symmetric Chamfer distance between two point sets.
    O(P^2) cdist implementation; intended for moderate point counts.
    """
    dists = torch.cdist(a.unsqueeze(0), b.unsqueeze(0), p=2)[0]
    return dists.min(dim=1).values.mean() + dists.min(dim=0).values.mean()


def build_gt_assets(gt_points, azimuths, args, device):
    cfg = make_render_config(args)
    gt_images, gt_cameras, gt_projections = [], [], []

    log(f"  Rendering {len(azimuths)} GT views...")
    for az in azimuths:
        az_val = az.item()
        cam = make_camera(az_val, device, args.cam_dist, args.cam_elev, args.fov)
        gt_cameras.append(cam)

        with torch.no_grad():
            img = render_point_cloud(gt_points, azimuth=az_val, elevation=args.cam_elev, config=cfg, device=device)
            gt_images.append(img[..., :3].detach())

            proj = cam.transform_points_screen(
                gt_points.unsqueeze(0), image_size=((args.image_size, args.image_size),)
            )[0, :, :2]
            gt_projections.append(proj.detach())

    return gt_images, gt_cameras, gt_projections


def optimize_points(gt_points, args, device):
    azimuths = torch.linspace(0, 360 - 360 / args.num_views, args.num_views)
    gt_images, gt_cameras, gt_projections = build_gt_assets(gt_points, azimuths, args, device)

    pred_points = nn.Parameter(gt_points.clone() + args.init_noise * torch.randn_like(gt_points))
    memory = PersistentPointMemory(pred_points.data, args.ema_decay)

    init_drift = paired_point_drift(gt_points, pred_points.data).item()
    init_chamfer = chamfer_distance_symmetric(gt_points, pred_points.data).item()
    noisy_points = pred_points.data.clone().cpu()

    log(f"  Initial paired drift: {init_drift:.4f}")
    log(f"  Initial chamfer:      {init_chamfer:.4f}")

    view_drifts = []
    view_chamfers = []
    view_losses = []
    snapshots = []
    t0 = time.time()

    log(f"\n  {'View':>5} {'Az':>7} {'Drift':>8} {'Chamfer':>10} {'Loss':>12} {'Time':>7}")
    log(f"  {'─'*5} {'─'*7} {'─'*8} {'─'*10} {'─'*12} {'─'*7}")

    for vi in range(args.num_views):
        az = azimuths[vi].item()
        buffer_start = max(0, vi - args.view_buffer + 1)
        buffer_idx = list(range(buffer_start, vi + 1))

        optimizer = torch.optim.Adam([pred_points], lr=args.lr)

        for _ in range(args.steps_per_view):
            optimizer.zero_grad()
            cfg = make_render_config(args)
            pred_img = render_point_cloud(
                pred_points, azimuth=az, elevation=args.cam_elev, config=cfg, device=device
            )[..., :3]
            loss_render = F.mse_loss(pred_img, gt_images[vi])
            loss_reproj = multi_view_reprojection_loss_points(
                pred_points,
                [gt_projections[i] for i in buffer_idx],
                [gt_cameras[i] for i in buffer_idx],
                args.image_size,
            )
            loss_anchor = anchor_proximity_loss_points(
                pred_points, memory.get_anchor(), weight=args.anchor_weight
            )
            loss = args.render_weight * loss_render + args.reproj_weight * loss_reproj + loss_anchor
            loss.backward()
            optimizer.step()

        memory.update(pred_points.data)

        with torch.no_grad():
            drift = paired_point_drift(gt_points, pred_points.data).item()
            chamfer = chamfer_distance_symmetric(gt_points, pred_points.data).item()
            view_drifts.append(drift)
            view_chamfers.append(chamfer)
            view_losses.append(loss.item())
            snapshots.append(pred_points.data.clone().cpu())

        elapsed = time.time() - t0
        if vi % max(1, args.num_views // 12) == 0 or vi == args.num_views - 1:
            log(
                f"  {vi:5d} {az:6.1f}° {drift:8.4f} {chamfer:10.4f} "
                f"{loss.item():12.4f} {elapsed:6.0f}s"
            )

    final_drift = view_drifts[-1]
    final_chamfer = view_chamfers[-1]
    drift_reduction = (1.0 - final_drift / init_drift) * 100.0

    log(f"\n  {'═'*56}")
    log(
        f"  DONE: {gt_points.shape[0]} pts | drift {init_drift:.4f}->{final_drift:.4f} "
        f"({drift_reduction:.1f}%)"
    )
    log(f"  chamfer {init_chamfer:.4f}->{final_chamfer:.4f}")
    log(f"  {'═'*56}")

    return {
        "gt_points": gt_points.cpu(),
        "noisy_points": noisy_points,
        "final_points": pred_points.data.cpu(),
        "point_history": snapshots,
        "azimuths": azimuths.tolist(),
        "view_drifts": view_drifts,
        "view_chamfers": view_chamfers,
        "view_losses": view_losses,
        "initial_drift": init_drift,
        "final_drift": final_drift,
        "drift_reduction_pct": drift_reduction,
        "initial_chamfer": init_chamfer,
        "final_chamfer": final_chamfer,
        "time_seconds": time.time() - t0,
    }


def save_outputs(results, strands, args):
    os.makedirs(args.output_dir, exist_ok=True)

    torch.save(results, os.path.join(args.output_dir, "point_baseline_results.pt"))
    with open(os.path.join(args.output_dir, "point_baseline_results.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "num_curves": args.num_curves,
                "K": args.K,
                "pc_points_per_curve": args.pc_points_per_curve,
                "num_points": int(results["gt_points"].shape[0]),
                "num_views": args.num_views,
                "steps_per_view": args.steps_per_view,
                "initial_drift": results["initial_drift"],
                "final_drift": results["final_drift"],
                "drift_reduction_pct": results["drift_reduction_pct"],
                "initial_chamfer": results["initial_chamfer"],
                "final_chamfer": results["final_chamfer"],
                "time_seconds": results["time_seconds"],
            },
            f,
            indent=2,
        )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        az = np.array(results["azimuths"])
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        axes[0].plot(az, results["view_drifts"], "b-o", markersize=2)
        axes[0].set_title("Paired Point Drift")
        axes[0].set_xlabel("Azimuth (deg)")
        axes[0].set_ylabel("L2")
        axes[1].plot(az, results["view_chamfers"], "m-o", markersize=2)
        axes[1].set_title("Chamfer Distance")
        axes[1].set_xlabel("Azimuth (deg)")
        axes[1].set_ylabel("Symmetric")
        axes[2].plot(az, results["view_losses"], "g-o", markersize=2)
        axes[2].set_title("Optimization Loss")
        axes[2].set_xlabel("Azimuth (deg)")
        axes[2].set_ylabel("Loss")
        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, "point_baseline_curves.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        log(f"  Plot saved to {plot_path}")
    except Exception as exc:
        log(f"  Plot failed: {exc}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        points_per_curve = args.pc_points_per_curve
        gt = render_dense_gt_blonde(strands, az=30.0, args=args)
        noisy = render_blonde_points(results["noisy_points"], points_per_curve, az=30.0, args=args)
        final = render_blonde_points(results["final_points"], points_per_curve, az=30.0, args=args)

        fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor="#080810")
        fig.subplots_adjust(wspace=0.02, left=0.01, right=0.99, top=0.82, bottom=0.08)
        for ax, (im, title) in zip(
            axes,
            [
                (gt, f"Ground Truth ({min(args.dense_gt_strands, len(strands))} strands)"),
                (noisy, f"Initial — Noisy ({args.num_curves} point-curves, σ={args.init_noise})"),
                (final, f"After Persistent Point Memory ({args.num_curves} point-curves)"),
            ],
        ):
            ax.imshow(np.clip(im, 0, 1))
            ax.axis("off")
            ax.set_title(title, color="#e0d0a0", fontsize=12, fontweight="bold", pad=10)

        fig.suptitle(
            f"Generative Spline Fields — Point-Cloud Baseline | {args.model_name} | "
            f"{results['drift_reduction_pct']:.1f}% Drift Reduction",
            color="white", fontsize=14,
            fontweight="bold",
            y=0.94,
        )
        fig.text(
            0.5,
            0.03,
            f"Drift: {results['initial_drift']:.4f} → {results['final_drift']:.4f} | "
            f"{args.num_curves} curves × {args.pc_points_per_curve} pts | {args.num_views} views",
            color="#666",
            fontsize=9,
            ha="center",
        )

        still_path = os.path.join(args.output_dir, "point_baseline_still.png")
        plt.savefig(still_path, dpi=150, facecolor="#080810", edgecolor="none")
        plt.close()
        log(f"  Still saved to {still_path}")

        # Keep a run_dense-compatible still filename.
        compat_still = os.path.join(args.output_dir, "comparison_still.png")
        if compat_still != still_path:
            import shutil

            shutil.copyfile(still_path, compat_still)
            log(f"  Also saved: {compat_still}")
    except Exception as exc:
        log(f"  Still render failed: {exc}")

    if args.skip_video:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        frames_dir = os.path.join(args.output_dir, "video_frames")
        os.makedirs(frames_dir, exist_ok=True)
        azimuths = np.linspace(0, 360, args.num_video_frames, endpoint=False)
        points_per_curve = args.pc_points_per_curve
        n_dense = min(args.dense_gt_strands, len(strands))

        log(f"  Rendering {args.num_video_frames} video frames...")
        t0 = time.time()
        for i, az in enumerate(azimuths):
            gt = render_dense_gt_blonde(strands, az=float(az), args=args)
            noisy = render_blonde_points(results["noisy_points"], points_per_curve, az=float(az), args=args)
            final = render_blonde_points(results["final_points"], points_per_curve, az=float(az), args=args)

            fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor="#080810")
            fig.subplots_adjust(wspace=0.02, left=0.01, right=0.99, top=0.82, bottom=0.08)
            for ax, (im, title) in zip(
                axes,
                [
                    (gt, f"Ground Truth ({n_dense} strands)"),
                    (noisy, f"Initial — Noisy ({args.num_curves} point-curves, σ={args.init_noise})"),
                    (final, f"After Persistent Point Memory ({args.num_curves} point-curves)"),
                ],
            ):
                ax.imshow(np.clip(im, 0, 1))
                ax.axis("off")
                ax.set_title(title, color="#e0d0a0", fontsize=12, fontweight="bold", pad=10)

            fig.suptitle(
                "Generative Spline Fields with Persistent Point Memory",
                color="white",
                fontsize=15,
                fontweight="bold",
                y=0.94,
            )
            fig.text(
                0.5,
                0.87,
                f"{args.model_name} | Azimuth: {az:.0f}° | "
                f"Drift Reduction: {results['drift_reduction_pct']:.1f}%",
                color="#aaa",
                fontsize=11,
                ha="center",
            )
            fig.text(
                0.5,
                0.03,
                f"Drift: {results['initial_drift']:.4f} → {results['final_drift']:.4f} | "
                f"{args.num_curves} curves × {args.pc_points_per_curve} pts | {args.num_views} views",
                color="#666",
                fontsize=9,
                ha="center",
            )

            plt.savefig(
                os.path.join(frames_dir, f"frame_{i:04d}.png"),
                dpi=args.dpi,
                facecolor="#080810",
                edgecolor="none",
            )
            plt.close()
            if i % max(1, args.num_video_frames // 8) == 0:
                log(f"    Frame {i:4d}/{args.num_video_frames} | {time.time() - t0:.0f}s")

        video_path = os.path.join(args.output_dir, "point_baseline_video.mp4")
        cmd = (
            f"ffmpeg -y -framerate {args.fps} -i {frames_dir}/frame_%04d.png "
            f"-c:v libopenh264 -b:v 6M -pix_fmt yuv420p -crf 18 "
            f"-vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' {video_path}"
        )
        ret = os.system(cmd)
        if ret == 0:
            log(f"  Video saved: {video_path}")
            compat_video = os.path.join(args.output_dir, "comparison_video.mp4")
            if compat_video != video_path:
                import shutil

                shutil.copyfile(video_path, compat_video)
                log(f"  Also saved: {compat_video}")
            if not args.keep_frames:
                import shutil

                shutil.rmtree(frames_dir)
        else:
            log(f"  ffmpeg failed. Frames kept in {frames_dir}/")
    except Exception as exc:
        log(f"  Video render failed: {exc}")


def main():
    p = argparse.ArgumentParser(description="Point-cloud baseline for dense hair task")
    p.add_argument("--model-name", default="wStraight")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--num-curves", type=int, default=500)
    p.add_argument("--K", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pc-points-per-curve", type=int, default=12)
    p.add_argument("--num-views", type=int, default=72)
    p.add_argument("--steps-per-view", type=int, default=80)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--init-noise", type=float, default=0.35)
    p.add_argument("--render-weight", type=float, default=0.5)
    p.add_argument("--reproj-weight", type=float, default=1.5)
    p.add_argument("--anchor-weight", type=float, default=0.02)
    p.add_argument("--view-buffer", type=int, default=5)
    p.add_argument("--ema-decay", type=float, default=0.8)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--radius", type=float, default=0.02)
    p.add_argument("--points-per-pixel", type=int, default=8)
    p.add_argument("--cam-elev", type=float, default=30.0)
    p.add_argument("--cam-dist", type=float, default=4.0)
    p.add_argument("--fov", type=float, default=60.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--vis-image-size", type=int, default=512)
    p.add_argument("--vis-radius", type=float, default=0.006)
    p.add_argument("--elevation", type=float, default=25.0)
    p.add_argument("--dist", type=float, default=3.5)
    p.add_argument("--dense-gt-strands", type=int, default=3000)
    p.add_argument("--dense-gt-points-per-strand", type=int, default=48)
    p.add_argument("--num-video-frames", type=int, default=72)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--keep-frames", action="store_true")
    p.add_argument("--skip-video", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    if args.output_dir is None:
        args.output_dir = f"outputs/point_baseline_{args.model_name}_{args.num_curves}"
    if args.quick:
        args.num_views = 36
        args.steps_per_view = 40
        args.num_video_frames = 36

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log(f"\n{'═'*68}")
    log("  POINT-CLOUD BASELINE")
    log(
        f"  Model: {args.model_name} | Curves: {args.num_curves} | K: {args.K} | "
        f"Pts/curve: {args.pc_points_per_curve}"
    )
    log(
        f"  Views: {args.num_views} | Steps/view: {args.steps_per_view} | "
        f"Noise sigma={args.init_noise}"
    )
    log(f"{'═'*68}")

    hair_path = download_yuksel_hair(args.model_name, save_dir=args.data_dir)
    strands = load_hair_file(hair_path)

    log(f"\nFitting spline GT ({args.num_curves} curves, K={args.K})...")
    gt_cp = hair_to_spline_field(
        strands, num_curves=args.num_curves, K=args.K, seed=args.seed, strategy="diverse"
    )
    gt_cp = orient_cp(gt_cp).to(args.device)

    with torch.no_grad():
        gt_points = evaluate_bspline(gt_cp, args.pc_points_per_curve).reshape(-1, 3)
    log(f"  GT point cloud size: {gt_points.shape[0]}")

    log("\nOptimizing point cloud baseline...")
    results = optimize_points(gt_points, args, args.device)
    save_outputs(results, strands, args)

    log(f"\n{'═'*68}")
    log(
        f"  DONE | drift reduction {results['drift_reduction_pct']:.1f}% | "
        f"chamfer {results['initial_chamfer']:.4f}->{results['final_chamfer']:.4f}"
    )
    log(f"  Output: {args.output_dir}/")
    log(f"{'═'*68}")


if __name__ == "__main__":
    main()
