"""
run_gaussian_splat_baseline.py — Isotropic Gaussian splat baseline.

This baseline follows the same optimization protocol as run_pointcloud_baseline.py,
but parameterizes each primitive as a 3D Gaussian splat with learnable:
  - mean (xyz)
  - scale (isotropic screen-space radius proxy)
  - opacity
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
from spline import evaluate_bspline


def log(msg):
    print(msg, flush=True)


def orient_cp(cp):
    out = cp.clone()
    new_y = out[..., 2].clone()
    new_z = -out[..., 1].clone()
    out[..., 1] = new_y
    out[..., 2] = new_z
    return out


def gaussian_splat_render(
    means,
    log_scales,
    opacities,
    camera,
    image_size,
    points_per_pixel,
    device,
    bg=(0.03, 0.03, 0.05),
):
    from pytorch3d.renderer import PointsRasterizationSettings, PointsRasterizer
    from pytorch3d.structures import Pointclouds

    n_points = means.shape[0]
    feats = torch.ones((n_points, 3), device=device, dtype=means.dtype)
    cloud = Pointclouds(points=[means], features=[feats])

    # Use a reasonably large raster radius and learn effective gaussian width per point.
    rs = PointsRasterizationSettings(
        image_size=image_size,
        radius=0.04,
        points_per_pixel=points_per_pixel,
        bin_size=0,
    )
    rasterizer = PointsRasterizer(cameras=camera, raster_settings=rs)
    fragments = rasterizer(cloud)
    idx = fragments.idx[0]       # (H, W, K), -1 for empty
    dists = fragments.dists[0]   # (H, W, K), normalized squared distance

    h, w, k = idx.shape
    valid = idx >= 0
    safe_idx = idx.clamp(min=0)

    scales = F.softplus(log_scales).clamp(min=1e-3, max=0.25)
    alpha = torch.sigmoid(opacities)

    gathered_scales = scales[safe_idx]      # (H,W,K)
    gathered_alpha = alpha[safe_idx]        # (H,W,K)
    gathered_feats = feats[safe_idx]        # (H,W,K,3)

    # Guard invalid fragments to avoid inf*0 => NaN in weighted sums.
    safe_dists = torch.where(valid, dists.clamp(min=0.0, max=10.0), torch.zeros_like(dists))
    exponent = -safe_dists / (2.0 * gathered_scales ** 2 + 1e-8)
    w_gauss = torch.exp(exponent)
    w_eff = torch.where(valid, w_gauss * gathered_alpha, torch.zeros_like(w_gauss))

    denom = w_eff.sum(dim=-1, keepdim=True) + 1e-8
    rgb = (w_eff.unsqueeze(-1) * gathered_feats).sum(dim=-2) / denom

    acc = w_eff.sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
    bg_tensor = torch.tensor(bg, device=device, dtype=means.dtype).view(1, 1, 3)
    rgb = rgb * acc + bg_tensor * (1.0 - acc)

    return rgb


def render_gaussian_view(means, log_scales, opacities, azimuth, args, image_size, points_per_pixel):
    cam = make_camera(azimuth, args.device, args.cam_dist, args.cam_elev, args.fov)
    with torch.no_grad():
        im = gaussian_splat_render(
            means,
            log_scales,
            opacities,
            camera=cam,
            image_size=image_size,
            points_per_pixel=points_per_pixel,
            device=args.device,
        )
    return im.detach().cpu().numpy()


def make_camera(azimuth, device, dist, elev, fov):
    from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform

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


def chamfer_distance_symmetric(a, b):
    dists = torch.cdist(a.unsqueeze(0), b.unsqueeze(0), p=2)[0]
    return dists.min(dim=1).values.mean() + dists.min(dim=0).values.mean()


class PersistentGaussianMemory:
    def __init__(self, initial_points, ema_decay):
        self.anchor = initial_points.clone().detach()
        self.ema_decay = ema_decay

    def update(self, new_points):
        self.anchor = self.ema_decay * self.anchor + (1.0 - self.ema_decay) * new_points.detach()

    def get_anchor(self):
        return self.anchor.clone()


def build_gt_assets(gt_points, azimuths, args, device):
    gt_images, gt_cameras, gt_projections = [], [], []
    with torch.no_grad():
        gt_log_scales = torch.full((gt_points.shape[0],), np.log(args.gs_init_scale), device=device)
        opacity_logit = float(torch.logit(torch.tensor(args.gs_init_opacity)).item())
        gt_opacities = torch.full((gt_points.shape[0],), opacity_logit, device=device)
        for az in azimuths:
            az_val = az.item()
            cam = make_camera(az_val, device, args.cam_dist, args.cam_elev, args.fov)
            gt_cameras.append(cam)
            img = gaussian_splat_render(
                gt_points,
                gt_log_scales,
                gt_opacities,
                camera=cam,
                image_size=args.image_size,
                points_per_pixel=args.points_per_pixel,
                device=device,
            )
            gt_images.append(img.detach())
            proj = cam.transform_points_screen(
                gt_points.unsqueeze(0), image_size=((args.image_size, args.image_size),)
            )[0, :, :2]
            gt_projections.append(proj.detach())
    return gt_images, gt_cameras, gt_projections


def optimize_gaussians(gt_points, args, device):
    azimuths = torch.linspace(0, 360 - 360 / args.num_views, args.num_views)
    gt_images, gt_cameras, gt_projections = build_gt_assets(gt_points, azimuths, args, device)

    pred_means = nn.Parameter(gt_points.clone() + args.init_noise * torch.randn_like(gt_points))
    pred_log_scales = nn.Parameter(torch.full((gt_points.shape[0],), np.log(args.gs_init_scale), device=device))
    opacity_logit = float(torch.logit(torch.tensor(args.gs_init_opacity)).item())
    pred_logits_opacity = nn.Parameter(torch.full((gt_points.shape[0],), opacity_logit, device=device))
    memory = PersistentGaussianMemory(pred_means.data, args.ema_decay)

    init_drift = (gt_points - pred_means.data).norm(dim=-1).mean().item()
    init_chamfer = chamfer_distance_symmetric(gt_points, pred_means.data).item()
    view_drifts, view_chamfers, view_losses = [], [], []
    snapshots = []
    t0 = time.time()

    for vi in range(args.num_views):
        az = azimuths[vi].item()
        buffer_start = max(0, vi - args.view_buffer + 1)
        buffer_idx = list(range(buffer_start, vi + 1))
        cam = gt_cameras[vi]

        optimizer = torch.optim.Adam([pred_means, pred_log_scales, pred_logits_opacity], lr=args.lr)
        for _ in range(args.steps_per_view):
            optimizer.zero_grad()
            pred_img = gaussian_splat_render(
                pred_means,
                pred_log_scales,
                pred_logits_opacity,
                camera=cam,
                image_size=args.image_size,
                points_per_pixel=args.points_per_pixel,
                device=device,
            )
            loss_render = F.mse_loss(pred_img, gt_images[vi])
            loss_reproj = multi_view_reprojection_loss_points(
                pred_means,
                [gt_projections[i] for i in buffer_idx],
                [gt_cameras[i] for i in buffer_idx],
                args.image_size,
            )
            loss_anchor = anchor_proximity_loss_points(pred_means, memory.get_anchor(), weight=args.anchor_weight)

            scale_reg = args.gs_scale_reg * F.softplus(pred_log_scales).mean()
            opacity_reg = args.gs_opacity_reg * torch.sigmoid(pred_logits_opacity).mean()
            loss = (
                args.render_weight * loss_render
                + args.reproj_weight * loss_reproj
                + loss_anchor
                + scale_reg
                + opacity_reg
            )
            if not torch.isfinite(loss):
                log("[GS] non-finite loss detected; applying safety reset.")
                with torch.no_grad():
                    pred_means.data = torch.nan_to_num(pred_means.data, nan=0.0, posinf=1.0, neginf=-1.0)
                    pred_log_scales.data = torch.nan_to_num(pred_log_scales.data, nan=np.log(args.gs_init_scale))
                    pred_logits_opacity.data = torch.nan_to_num(
                        pred_logits_opacity.data, nan=float(torch.logit(torch.tensor(args.gs_init_opacity)).item())
                    )
                    pred_log_scales.data.clamp_(min=np.log(1e-3), max=np.log(0.25))
                    pred_logits_opacity.data.clamp_(min=-8.0, max=8.0)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_([pred_means, pred_log_scales, pred_logits_opacity], max_norm=5.0)
            optimizer.step()
            with torch.no_grad():
                pred_means.data = torch.nan_to_num(pred_means.data, nan=0.0, posinf=1.0, neginf=-1.0)
                pred_log_scales.data = torch.nan_to_num(pred_log_scales.data, nan=np.log(args.gs_init_scale))
                pred_logits_opacity.data = torch.nan_to_num(
                    pred_logits_opacity.data, nan=float(torch.logit(torch.tensor(args.gs_init_opacity)).item())
                )
                pred_log_scales.data.clamp_(min=np.log(1e-3), max=np.log(0.25))
                pred_logits_opacity.data.clamp_(min=-8.0, max=8.0)

        memory.update(pred_means.data)
        with torch.no_grad():
            drift = (gt_points - pred_means.data).norm(dim=-1).mean().item()
            chamfer = chamfer_distance_symmetric(gt_points, pred_means.data).item()
            view_drifts.append(drift)
            view_chamfers.append(chamfer)
            view_losses.append(loss.item())
            snapshots.append(pred_means.data.clone().cpu())

        if vi % max(1, args.num_views // 12) == 0 or vi == args.num_views - 1:
            elapsed = time.time() - t0
            log(
                f"[GS] view {vi:3d}/{args.num_views-1:3d} az={az:6.1f} "
                f"drift={drift:.4f} chamfer={chamfer:.4f} loss={loss.item():.4f} t={elapsed:.0f}s"
            )

    final_drift = view_drifts[-1]
    drift_reduction = (1.0 - final_drift / init_drift) * 100.0
    noisy_scales = torch.full_like(pred_log_scales.detach().cpu(), args.gs_init_scale)
    noisy_opacity = torch.full_like(pred_logits_opacity.detach().cpu(), args.gs_init_opacity)
    return {
        "gt_points": gt_points.cpu(),
        "noisy_points": (gt_points + args.init_noise * torch.randn_like(gt_points)).detach().cpu(),
        "final_points": pred_means.data.cpu(),
        "point_history": snapshots,
        "noisy_scales": noisy_scales,
        "noisy_opacity": noisy_opacity,
        "final_scales": F.softplus(pred_log_scales).detach().cpu(),
        "final_opacity": torch.sigmoid(pred_logits_opacity).detach().cpu(),
        "azimuths": azimuths.tolist(),
        "view_drifts": view_drifts,
        "view_chamfers": view_chamfers,
        "view_losses": view_losses,
        "initial_drift": init_drift,
        "final_drift": final_drift,
        "drift_reduction_pct": drift_reduction,
        "initial_chamfer": init_chamfer,
        "final_chamfer": view_chamfers[-1],
        "time_seconds": time.time() - t0,
    }


def save_outputs(results, args):
    os.makedirs(args.output_dir, exist_ok=True)
    pt_path = os.path.join(args.output_dir, "gaussian_baseline_results.pt")
    js_path = os.path.join(args.output_dir, "gaussian_baseline_results.json")
    torch.save(results, pt_path)
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "num_curves": args.num_curves,
                "K": args.K,
                "gs_points_per_curve": args.gs_points_per_curve,
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
    log(f"  Saved: {pt_path}")
    log(f"  Saved: {js_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        az = np.array(results["azimuths"])
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        axes[0].plot(az, results["view_drifts"], "g-o", markersize=2)
        axes[0].set_title("Gaussian Drift")
        axes[0].set_xlabel("Azimuth (deg)")
        axes[0].set_ylabel("L2")
        axes[1].plot(az, results["view_chamfers"], "m-o", markersize=2)
        axes[1].set_title("Gaussian Chamfer")
        axes[1].set_xlabel("Azimuth (deg)")
        axes[1].set_ylabel("Symmetric")
        axes[2].plot(az, results["view_losses"], "c-o", markersize=2)
        axes[2].set_title("Optimization Loss")
        axes[2].set_xlabel("Azimuth (deg)")
        axes[2].set_ylabel("Loss")
        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, "gaussian_baseline_curves.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        log(f"  Plot saved to {plot_path}")
    except Exception as exc:
        log(f"  Plot failed: {exc}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        gt_im = render_gaussian_view(
            results["gt_points"].to(args.device),
            torch.full((results["gt_points"].shape[0],), np.log(args.gs_init_scale), device=args.device),
            torch.full(
                (results["gt_points"].shape[0],),
                float(torch.logit(torch.tensor(args.gs_init_opacity)).item()),
                device=args.device,
            ),
            azimuth=30.0,
            args=args,
            image_size=args.vis_image_size,
            points_per_pixel=args.vis_points_per_pixel,
        )
        noisy_im = render_gaussian_view(
            results["noisy_points"].to(args.device),
            torch.log(results["noisy_scales"].to(args.device).clamp(min=1e-3)),
            torch.logit(results["noisy_opacity"].to(args.device).clamp(1e-4, 1.0 - 1e-4)),
            azimuth=30.0,
            args=args,
            image_size=args.vis_image_size,
            points_per_pixel=args.vis_points_per_pixel,
        )
        final_im = render_gaussian_view(
            results["final_points"].to(args.device),
            torch.log(results["final_scales"].to(args.device).clamp(min=1e-3)),
            torch.logit(results["final_opacity"].to(args.device).clamp(1e-4, 1.0 - 1e-4)),
            azimuth=30.0,
            args=args,
            image_size=args.vis_image_size,
            points_per_pixel=args.vis_points_per_pixel,
        )

        fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor="#080810")
        fig.subplots_adjust(wspace=0.02, left=0.01, right=0.99, top=0.82, bottom=0.08)
        for ax, (im, title) in zip(
            axes,
            [
                (gt_im, "Gaussian GT Proxy"),
                (noisy_im, f"Initial Noisy Gaussians (sigma={args.init_noise})"),
                (final_im, "Final Gaussian Splats"),
            ],
        ):
            ax.imshow(np.clip(im, 0, 1))
            ax.axis("off")
            ax.set_title(title, color="#e0d0a0", fontsize=12, fontweight="bold", pad=10)

        fig.suptitle(
            f"Gaussian Splat Baseline | {args.model_name} | {results['drift_reduction_pct']:.1f}% Drift Reduction",
            color="white",
            fontsize=14,
            fontweight="bold",
            y=0.94,
        )
        still_path = os.path.join(args.output_dir, "gaussian_baseline_still.png")
        plt.savefig(still_path, dpi=150, facecolor="#080810", edgecolor="none")
        plt.close()
        log(f"  Still saved to {still_path}")
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
        gt_log_scales = torch.full((results["gt_points"].shape[0],), np.log(args.gs_init_scale), device=args.device)
        gt_opacity_logits = torch.full(
            (results["gt_points"].shape[0],),
            float(torch.logit(torch.tensor(args.gs_init_opacity)).item()),
            device=args.device,
        )
        noisy_log_scales = torch.log(results["noisy_scales"].to(args.device).clamp(min=1e-3))
        noisy_opacity_logits = torch.logit(results["noisy_opacity"].to(args.device).clamp(1e-4, 1.0 - 1e-4))
        final_log_scales = torch.log(results["final_scales"].to(args.device).clamp(min=1e-3))
        final_opacity_logits = torch.logit(results["final_opacity"].to(args.device).clamp(1e-4, 1.0 - 1e-4))

        log(f"  Rendering {args.num_video_frames} gaussian video frames...")
        for i, az in enumerate(azimuths):
            gt_im = render_gaussian_view(
                results["gt_points"].to(args.device), gt_log_scales, gt_opacity_logits, float(az),
                args, args.vis_image_size, args.vis_points_per_pixel
            )
            noisy_im = render_gaussian_view(
                results["noisy_points"].to(args.device), noisy_log_scales, noisy_opacity_logits, float(az),
                args, args.vis_image_size, args.vis_points_per_pixel
            )
            final_im = render_gaussian_view(
                results["final_points"].to(args.device), final_log_scales, final_opacity_logits, float(az),
                args, args.vis_image_size, args.vis_points_per_pixel
            )

            fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor="#080810")
            fig.subplots_adjust(wspace=0.02, left=0.01, right=0.99, top=0.82, bottom=0.08)
            for ax, (im, title) in zip(
                axes,
                [
                    (gt_im, "Gaussian GT Proxy"),
                    (noisy_im, f"Initial Noisy Gaussians (sigma={args.init_noise})"),
                    (final_im, "Final Gaussian Splats"),
                ],
            ):
                ax.imshow(np.clip(im, 0, 1))
                ax.axis("off")
                ax.set_title(title, color="#e0d0a0", fontsize=12, fontweight="bold", pad=10)
            fig.suptitle(
                f"Gaussian Splat Baseline — {args.model_name} | Azimuth {az:.0f}°",
                color="white",
                fontsize=14,
                fontweight="bold",
                y=0.94,
            )
            plt.savefig(
                os.path.join(frames_dir, f"frame_{i:04d}.png"),
                dpi=args.dpi,
                facecolor="#080810",
                edgecolor="none",
            )
            plt.close()

        video_path = os.path.join(args.output_dir, "gaussian_baseline_video.mp4")
        cmd = (
            f"ffmpeg -y -framerate {args.fps} -i {frames_dir}/frame_%04d.png "
            f"-c:v libopenh264 -b:v 6M -pix_fmt yuv420p -crf 18 "
            f"-vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' {video_path}"
        )
        ret = os.system(cmd)
        if ret == 0:
            log(f"  Video saved: {video_path}")
            if not args.keep_frames:
                import shutil

                shutil.rmtree(frames_dir)
        else:
            log(f"  ffmpeg failed. Frames kept in {frames_dir}/")
    except Exception as exc:
        log(f"  Video render failed: {exc}")


def main():
    p = argparse.ArgumentParser(description="Gaussian splat baseline for dense hair task")
    p.add_argument("--model-name", default="wStraight")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--num-curves", type=int, default=500)
    p.add_argument("--K", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gs-points-per-curve", type=int, default=12)
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
    p.add_argument("--points-per-pixel", type=int, default=10)
    p.add_argument("--cam-elev", type=float, default=30.0)
    p.add_argument("--cam-dist", type=float, default=4.0)
    p.add_argument("--fov", type=float, default=60.0)
    p.add_argument("--gs-init-scale", type=float, default=0.015)
    p.add_argument("--gs-init-opacity", type=float, default=0.85)
    p.add_argument("--gs-scale-reg", type=float, default=1e-3)
    p.add_argument("--gs-opacity-reg", type=float, default=2e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--vis-image-size", type=int, default=512)
    p.add_argument("--vis-points-per-pixel", type=int, default=10)
    p.add_argument("--num-video-frames", type=int, default=72)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--keep-frames", action="store_true")
    p.add_argument("--skip-video", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    if args.output_dir is None:
        args.output_dir = f"outputs/gaussian_baseline_{args.model_name}_{args.num_curves}"
    if args.quick:
        args.num_views = 36
        args.steps_per_view = 40

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    hair_path = download_yuksel_hair(args.model_name, save_dir=args.data_dir)
    strands = load_hair_file(hair_path)
    log(f"Fitting spline GT ({args.num_curves} curves, K={args.K})...")
    gt_cp = hair_to_spline_field(
        strands, num_curves=args.num_curves, K=args.K, seed=args.seed, strategy="diverse"
    )
    gt_cp = orient_cp(gt_cp).to(args.device)
    with torch.no_grad():
        gt_points = evaluate_bspline(gt_cp, args.gs_points_per_curve).reshape(-1, 3)

    log("Optimizing gaussian splat baseline...")
    results = optimize_gaussians(gt_points, args, args.device)
    save_outputs(results, args)
    log(
        f"DONE | drift reduction {results['drift_reduction_pct']:.1f}% "
        f"| chamfer {results['initial_chamfer']:.4f}->{results['final_chamfer']:.4f}"
    )


if __name__ == "__main__":
    main()
