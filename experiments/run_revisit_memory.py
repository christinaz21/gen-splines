"""
experiment_revisit_memory.py — Mini world-model experiment: persistent memory under camera motion.

Protocol (Yuksel hair, single scene):
  Camera path A → B → C → D → A′ (revisit). Views are optimized sequentially; memory persists.

Compares:
  • Spline memory (control points + PersistentCurveMemory / EMA anchor)
  • Point-cloud memory (same schedule; PersistentPointMemory)
  • Gaussian splat memory (means + scale + opacity; same schedule)

Metrics:
  1. Held-out view error at azimuth(s) H (never in trajectory): MSE, PSNR (+ SSIM if skimage available)
  2. Revisit consistency: MSE between render(M₁, A) and render(Mₜ, A) — lower ⇒ less drift at same pose
  3. Memory size (parameter count & bytes)

Robustness:
  • --seeds: repeat full experiment with different RNG seeds; report mean/std in aggregate.

Outputs:
  revisit_results.json, revisit_poster.png, optional per-method .pt snapshots
"""

from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gensplines.hair_loader import download_yuksel_hair, load_hair_file, hair_to_spline_field
from gensplines.metrics import control_point_drift
from gensplines.losses import (
    multi_view_reprojection_loss,
    tangent_consistency_loss,
    anchor_proximity_loss,
)
from gensplines.renderer import render_point_cloud
from gensplines.spline import SplineField, evaluate_bspline

from gensplines.coordinates import orient_cp
from gensplines.memory import PersistentCurveMemory, PersistentPointMemory, PersistentGaussianMemory

# Reuse rendering helpers (safe import; does not run run_spline main)
from run_spline import (
    blonde_colors,
    render_pts,
    render_cp_blonde,
    render_dense_gt_blonde,
)
from run_gaussian_baseline import gaussian_splat_render


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_trajectory(s: str) -> list[float]:
    parts = [float(x.strip()) for x in s.split(",") if x.strip()]
    if len(parts) < 2:
        raise ValueError("Trajectory must have at least 2 azimuths (include a revisit at the end).")
    return parts


def make_camera(azimuth: float, device: str, dist: float, elev: float, fov: float):
    from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform

    r, t = look_at_view_transform(dist=dist, elev=elev, azim=azimuth)
    return FoVPerspectiveCameras(device=device, R=r, T=t, fov=fov, aspect_ratio=1.0)


def render_pc_blonde(points_flat: torch.Tensor, points_per_curve: int, az: float, args, device: str):
    """Blonde visualization for point cloud (matches point baseline style)."""
    n = points_flat.shape[0] // points_per_curve
    colors = blonde_colors([points_per_curve] * n)
    return render_pts(
        points_flat,
        colors,
        az,
        args.vis_image_size,
        args.vis_radius,
        device,
        elev=args.elevation,
        dist=args.dist,
    )


def mse_psnr_ssim(a: np.ndarray, b: np.ndarray):
    a = np.clip(a.astype(np.float64), 0.0, 1.0)
    b = np.clip(b.astype(np.float64), 0.0, 1.0)
    mse = float(np.mean((a - b) ** 2))
    psnr = float(10.0 * np.log10(1.0 / (mse + 1e-12)))
    ssim = None
    try:
        from skimage.metrics import structural_similarity as ssim_fn

        ssim = float(ssim_fn(a, b, channel_axis=-1, data_range=1.0))
    except Exception:
        pass
    return mse, psnr, ssim


def make_render_cfg(image_size: int, radius: float, ppp: int):
    return {
        "image_size": image_size,
        "radius": radius,
        "points_per_pixel": ppp,
        "dist": 4.0,
        "fov": 60.0,
        "compositor": "alpha",
        "bin_size": 0,
    }


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


def optimize_spline_trajectory(gt_cp: torch.Tensor, traj_azs: list[float], args, device: str):
    from pytorch3d.renderer import (
        look_at_view_transform,
        FoVPerspectiveCameras,
        PointsRasterizationSettings,
        PointsRenderer,
        PointsRasterizer,
        AlphaCompositor,
    )
    from pytorch3d.structures import Pointclouds

    N, K, _ = gt_cp.shape
    gt_o = orient_cp(gt_cp).to(device)
    gt_field = SplineField(N, K).to(device)
    gt_field.control_points.data = gt_o.clone()
    ns = args.samples_per_curve
    rs = PointsRasterizationSettings(
        image_size=args.opt_image_size,
        radius=args.opt_radius,
        points_per_pixel=8,
        bin_size=0,
    )
    azs = torch.tensor(traj_azs, dtype=torch.float32, device=device)

    gt_imgs, gt_cams, gt_projs = [], [], []
    gt_flat = gt_field.forward_per_curve(ns).reshape(-1, 3)

    log(f"  [Spline] Caching {len(traj_azs)} trajectory GT views...")
    for az in azs:
        az_val = float(az.item())
        R, T = look_at_view_transform(dist=4.0, elev=30.0, azim=az_val)
        cam = FoVPerspectiveCameras(
            device=device, R=R, T=T, fov=60.0, aspect_ratio=1.0, znear=0.1, zfar=100.0
        )
        gt_cams.append(cam)
        with torch.no_grad():
            pts = gt_field(ns)
            rgb = torch.ones_like(pts)
            rend = PointsRenderer(
                rasterizer=PointsRasterizer(cameras=cam, raster_settings=rs),
                compositor=AlphaCompositor(),
            )
            img = rend(Pointclouds(points=[pts], features=[rgb]))[0, ..., :3]
            gt_imgs.append(img.detach())
            proj = cam.transform_points_screen(
                gt_flat.unsqueeze(0), image_size=((args.opt_image_size, args.opt_image_size),)
            )[0, :, :2]
            gt_projs.append(proj.detach())

    pred = SplineField(N, K).to(device)
    pred.control_points.data = gt_o.clone() + args.init_noise * torch.randn_like(gt_o)
    init_d = control_point_drift(gt_o, pred.control_points.data).item()
    mem = PersistentCurveMemory(pred.control_points.data, ema_decay=args.ema_decay)

    snapshots = []
    drifts = []
    t0 = time.time()

    for vi in range(len(traj_azs)):
        az = float(azs[vi].item())
        buffer_start = max(0, vi - args.view_buffer + 1)
        bi = list(range(buffer_start, vi + 1))
        opt = torch.optim.Adam([pred.control_points], lr=args.lr)

        for _ in range(args.steps_per_view):
            opt.zero_grad()
            pts = pred(ns)
            rgb = torch.ones_like(pts)
            R, T = look_at_view_transform(dist=4.0, elev=30.0, azim=az)
            cam = FoVPerspectiveCameras(
                device=device, R=R, T=T, fov=60.0, aspect_ratio=1.0, znear=0.1, zfar=100.0
            )
            rend = PointsRenderer(
                rasterizer=PointsRasterizer(cameras=cam, raster_settings=rs),
                compositor=AlphaCompositor(),
            )
            pimg = rend(Pointclouds(points=[pts], features=[rgb]))[0, ..., :3]
            lr_ = F.mse_loss(pimg, gt_imgs[vi])
            lp = multi_view_reprojection_loss(
                pred.control_points,
                [gt_projs[i] for i in bi],
                [gt_cams[i] for i in bi],
                num_samples=ns,
                image_size=args.opt_image_size,
            )
            lt = tangent_consistency_loss(pred.control_points, num_samples=ns, weight=args.tangent_weight)
            la = anchor_proximity_loss(pred.control_points, mem.get_anchor(), weight=args.anchor_weight)
            loss = args.render_weight * lr_ + args.reproj_weight * lp + lt + la
            loss.backward()
            opt.step()

        mem.update(pred.control_points.data)
        with torch.no_grad():
            drifts.append(control_point_drift(gt_o, pred.control_points.data).item())
        snapshots.append(pred.control_points.data.detach().cpu().clone())

        log(f"  [Spline] step {vi + 1}/{len(traj_azs)} az={az:6.1f}° drift={drifts[-1]:.4f}")

    elapsed = time.time() - t0
    fd = drifts[-1]
    red = (1.0 - fd / max(init_d, 1e-12)) * 100.0

    return {
        "method": "spline",
        "gt_cp": gt_o.cpu(),
        "snapshots_cp": snapshots,
        "trajectory_azimuths": traj_azs,
        "initial_drift": init_d,
        "final_drift": fd,
        "drift_reduction_pct": red,
        "view_drifts": drifts,
        "time_seconds": elapsed,
        "num_params": int(N * K * 3),
        "bytes_estimate": int(N * K * 3 * 4),
    }


def optimize_point_trajectory(gt_points: torch.Tensor, traj_azs: list[float], args, device: str):
    cfg = make_render_cfg(args.opt_image_size, args.radius, args.points_per_pixel)
    azs_tensor = torch.tensor(traj_azs, dtype=torch.float32)

    gt_images, gt_cameras, gt_projections = [], [], []
    log(f"  [Point] Caching {len(traj_azs)} trajectory GT views...")
    for az in azs_tensor:
        az_val = float(az.item())
        cam = make_camera(az_val, device, args.cam_dist, args.cam_elev, args.fov)
        gt_cameras.append(cam)
        with torch.no_grad():
            img = render_point_cloud(
                gt_points, azimuth=az_val, elevation=args.cam_elev, config=cfg, device=device
            )
            gt_images.append(img[..., :3].detach())
            proj = cam.transform_points_screen(
                gt_points.unsqueeze(0), image_size=((args.opt_image_size, args.opt_image_size),)
            )[0, :, :2]
            gt_projections.append(proj.detach())

    pred_points = nn.Parameter(gt_points.clone() + args.init_noise * torch.randn_like(gt_points))
    memory = PersistentPointMemory(pred_points.data, args.ema_decay)
    init_d = (gt_points - pred_points.data).norm(dim=-1).mean().item()

    snapshots = []
    drifts = []
    t0 = time.time()

    for vi in range(len(traj_azs)):
        az = float(azs_tensor[vi].item())
        buffer_start = max(0, vi - args.view_buffer + 1)
        buffer_idx = list(range(buffer_start, vi + 1))
        optimizer = torch.optim.Adam([pred_points], lr=args.lr)

        for _ in range(args.steps_per_view):
            optimizer.zero_grad()
            pred_img = render_point_cloud(
                pred_points, azimuth=az, elevation=args.cam_elev, config=cfg, device=device
            )[..., :3]
            loss_render = F.mse_loss(pred_img, gt_images[vi])
            loss_reproj = multi_view_reprojection_loss_points(
                pred_points,
                [gt_projections[i] for i in buffer_idx],
                [gt_cameras[i] for i in buffer_idx],
                args.opt_image_size,
            )
            loss_anchor = anchor_proximity_loss_points(
                pred_points, memory.get_anchor(), weight=args.anchor_weight
            )
            loss = args.render_weight * loss_render + args.reproj_weight * loss_reproj + loss_anchor
            loss.backward()
            optimizer.step()

        memory.update(pred_points.data)
        with torch.no_grad():
            drifts.append((gt_points - pred_points.data).norm(dim=-1).mean().item())
        snapshots.append(pred_points.data.detach().cpu().clone())

        log(f"  [Point] step {vi + 1}/{len(traj_azs)} az={az:6.1f}° drift={drifts[-1]:.4f}")

    elapsed = time.time() - t0
    fd = drifts[-1]
    red = (1.0 - fd / max(init_d, 1e-12)) * 100.0
    n_pts = gt_points.shape[0]

    return {
        "method": "pointcloud",
        "gt_points": gt_points.cpu(),
        "snapshots_pts": snapshots,
        "trajectory_azimuths": traj_azs,
        "points_per_curve": args.pc_points_per_curve,
        "initial_drift": init_d,
        "final_drift": fd,
        "drift_reduction_pct": red,
        "view_drifts": drifts,
        "time_seconds": elapsed,
        "num_params": int(n_pts * 3),
        "bytes_estimate": int(n_pts * 3 * 4),
    }


def optimize_gaussian_trajectory(gt_points: torch.Tensor, traj_azs: list[float], args, device: str):
    """Sequential Gaussian splat optimization on a fixed azimuth trajectory (same protocol as full GS baseline)."""
    azs_tensor = torch.tensor(traj_azs, dtype=torch.float32, device=device)
    gt_images, gt_cameras, gt_projections = [], [], []
    opacity_logit = float(torch.logit(torch.tensor(args.gs_init_opacity)).item())
    with torch.no_grad():
        gt_log_scales = torch.full((gt_points.shape[0],), np.log(args.gs_init_scale), device=device)
        gt_opacities = torch.full((gt_points.shape[0],), opacity_logit, device=device)

    log(f"  [Gaussian] Caching {len(traj_azs)} trajectory GT views...")
    for i in range(len(traj_azs)):
        az_val = float(azs_tensor[i].item())
        cam = make_camera(az_val, device, args.cam_dist, args.cam_elev, args.fov)
        gt_cameras.append(cam)
        with torch.no_grad():
            img = gaussian_splat_render(
                gt_points,
                gt_log_scales,
                gt_opacities,
                camera=cam,
                image_size=args.opt_image_size,
                points_per_pixel=args.gs_points_per_pixel,
                device=device,
            )
            gt_images.append(img.detach())
            proj = cam.transform_points_screen(
                gt_points.unsqueeze(0), image_size=((args.opt_image_size, args.opt_image_size),)
            )[0, :, :2]
            gt_projections.append(proj.detach())

    pred_means = nn.Parameter(gt_points.clone() + args.init_noise * torch.randn_like(gt_points))
    pred_log_scales = nn.Parameter(
        torch.full((gt_points.shape[0],), np.log(args.gs_init_scale), device=device)
    )
    pred_logits_opacity = nn.Parameter(torch.full((gt_points.shape[0],), opacity_logit, device=device))
    memory = PersistentGaussianMemory(pred_means.data, args.ema_decay)

    init_d = (gt_points - pred_means.data).norm(dim=-1).mean().item()
    snapshots = []
    drifts = []
    t0 = time.time()
    n_pts = gt_points.shape[0]

    for vi in range(len(traj_azs)):
        az = float(azs_tensor[vi].item())
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
                image_size=args.opt_image_size,
                points_per_pixel=args.gs_points_per_pixel,
                device=device,
            )
            loss_render = F.mse_loss(pred_img, gt_images[vi])
            loss_reproj = multi_view_reprojection_loss_points(
                pred_means,
                [gt_projections[i] for i in buffer_idx],
                [gt_cameras[i] for i in buffer_idx],
                args.opt_image_size,
            )
            loss_anchor = anchor_proximity_loss_points(
                pred_means, memory.get_anchor(), weight=args.anchor_weight
            )
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
                with torch.no_grad():
                    pred_means.data = torch.nan_to_num(pred_means.data, nan=0.0, posinf=1.0, neginf=-1.0)
                    pred_log_scales.data = torch.nan_to_num(
                        pred_log_scales.data, nan=np.log(args.gs_init_scale)
                    )
                    pred_logits_opacity.data = torch.nan_to_num(
                        pred_logits_opacity.data, nan=opacity_logit
                    )
                    pred_log_scales.data.clamp_(min=np.log(1e-3), max=np.log(0.25))
                    pred_logits_opacity.data.clamp_(min=-8.0, max=8.0)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [pred_means, pred_log_scales, pred_logits_opacity], max_norm=5.0
            )
            optimizer.step()
            with torch.no_grad():
                pred_means.data = torch.nan_to_num(pred_means.data, nan=0.0, posinf=1.0, neginf=-1.0)
                pred_log_scales.data = torch.nan_to_num(
                    pred_log_scales.data, nan=np.log(args.gs_init_scale)
                )
                pred_logits_opacity.data = torch.nan_to_num(
                    pred_logits_opacity.data, nan=opacity_logit
                )
                pred_log_scales.data.clamp_(min=np.log(1e-3), max=np.log(0.25))
                pred_logits_opacity.data.clamp_(min=-8.0, max=8.0)

        memory.update(pred_means.data)
        with torch.no_grad():
            drifts.append((gt_points - pred_means.data).norm(dim=-1).mean().item())
        snapshots.append(
            {
                "means": pred_means.data.clone().cpu(),
                "log_scales": pred_log_scales.data.clone().cpu(),
                "logits_opacity": pred_logits_opacity.data.clone().cpu(),
            }
        )
        log(f"  [Gaussian] step {vi + 1}/{len(traj_azs)} az={az:6.1f}° drift={drifts[-1]:.4f}")

    elapsed = time.time() - t0
    fd = drifts[-1]
    red = (1.0 - fd / max(init_d, 1e-12)) * 100.0
    # Learnable DOF: xyz + log_scale + logit_opacity per primitive (report for fairness).
    num_gs_params = int(n_pts * 5)

    return {
        "method": "gaussian",
        "gt_points": gt_points.cpu(),
        "snapshots_gs": snapshots,
        "trajectory_azimuths": traj_azs,
        "points_per_curve": args.pc_points_per_curve,
        "initial_drift": init_d,
        "final_drift": fd,
        "drift_reduction_pct": red,
        "view_drifts": drifts,
        "time_seconds": elapsed,
        "num_params": num_gs_params,
        "bytes_estimate": int(num_gs_params * 4),
        "num_points_primitives": int(n_pts),
    }


def render_gs_numpy(snap: dict, az: float, vis, args, device: str) -> np.ndarray:
    means = snap["means"].to(device)
    log_sc = snap["log_scales"].to(device)
    logit = snap["logits_opacity"].to(device)
    n_pts = means.shape[0]
    ppc = int(args.pc_points_per_curve)
    n_curves = n_pts // ppc
    if n_curves * ppc != n_pts:
        raise ValueError(f"Gaussian means length {n_pts} not divisible by pc_points_per_curve={ppc}")
    feats = blonde_colors([ppc] * n_curves).to(device=device, dtype=means.dtype)
    cam = make_camera(az, device, args.cam_dist, args.cam_elev, args.fov)
    with torch.no_grad():
        im = gaussian_splat_render(
            means,
            log_sc,
            logit,
            camera=cam,
            image_size=vis.vis_image_size,
            points_per_pixel=args.gs_points_per_pixel,
            device=device,
            features=feats,
        )
    return np.clip(im.detach().cpu().numpy(), 0.0, 1.0)


def parse_held_out_azimuths(s: str) -> list[float]:
    parts = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not parts:
        raise ValueError("Need at least one held-out azimuth.")
    return parts


def parse_seeds(s: str) -> list[int]:
    parts = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not parts:
        raise ValueError("Need at least one seed (e.g. --seeds 42 or --seeds 42,43,44).")
    return parts


AGG_KEYS = (
    "revisit_consistency_mse_M1_vs_MT_at_A",
    "held_out_mse_vs_gt_mean_over_H",
    "held_out_psnr_vs_gt_mean_over_H",
    "held_out_ssim_vs_gt_mean_over_H",
    "render_at_A_after_trajectory_mse_vs_gt",
    "render_at_A_after_trajectory_psnr_vs_gt",
    "time_seconds",
)


def aggregate_across_seeds(per_seed: list[dict]) -> dict:
    """Mean / std across RNG seeds for scalar method metrics."""
    methods = ("spline", "pointcloud", "gaussian")
    agg: dict = {}
    for m in methods:
        blocks = [p[m] for p in per_seed if p.get(m) is not None]
        if not blocks:
            continue
        agg[m] = {}
        for k in AGG_KEYS:
            vals = []
            for b in blocks:
                if k not in b or b[k] is None:
                    continue
                try:
                    vals.append(float(b[k]))
                except (TypeError, ValueError):
                    continue
            if vals:
                agg[m][k] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n": len(vals),
                }
    ratios = ("memory_compression_point_over_spline_bytes", "memory_compression_gaussian_over_spline_bytes")
    for rk in ratios:
        vals = []
        for p in per_seed:
            if rk in p and p[rk] is not None:
                try:
                    vals.append(float(p[rk]))
                except (TypeError, ValueError):
                    pass
        if vals:
            agg[rk] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
    return agg


def _agg_mean(aggregate: dict, method: str, metric_key: str):
    block = aggregate.get(method)
    if not isinstance(block, dict) or metric_key not in block:
        return None
    v = block[metric_key].get("mean")
    if v is None or not np.isfinite(v):
        return None
    return float(v)


def _pick_best(means: dict[str, float | None], lower_is_better: bool, eps: float):
    valid = {k: float(v) for k, v in means.items() if v is not None and np.isfinite(v)}
    if not valid:
        return {"winner": None, "tie": True, "tied_methods": [], "means": means}
    if lower_is_better:
        best_val = min(valid.values())
    else:
        best_val = max(valid.values())
    winners = [k for k, v in valid.items() if abs(v - best_val) <= eps]
    tie = len(winners) > 1
    return {
        "winner": "tie" if tie else winners[0],
        "tie": tie,
        "tied_methods": winners if tie else [],
        "best_value": float(best_val),
        "means": valid,
    }


def build_winner_summary(aggregate: dict, eps: float = 1e-10) -> dict:
    """
    Pick winners from aggregate_over_seeds means (lower MSE / time, higher PSNR/SSIM).
    Only methods present under aggregate (spline / pointcloud / gaussian) are compared.
    """
    methods = [m for m in ("spline", "pointcloud", "gaussian") if isinstance(aggregate.get(m), dict)]
    metric_specs = [
        ("revisit_consistency_mse_M1_vs_MT_at_A", True, "Revisit: MSE(render@A step1, render@A final); lower = more stable at same pose."),
        ("held_out_mse_vs_gt_mean_over_H", True, "Held-out views: mean MSE vs dense GT; lower = better generalization."),
        ("held_out_psnr_vs_gt_mean_over_H", False, "Held-out views: mean PSNR vs dense GT; higher = better."),
        ("held_out_ssim_vs_gt_mean_over_H", False, "Held-out views: mean SSIM vs dense GT; higher = better (requires scikit-image)."),
        ("render_at_A_after_trajectory_mse_vs_gt", True, "Final render at A vs GT: MSE; lower = better fit at revisit pose."),
        ("render_at_A_after_trajectory_psnr_vs_gt", False, "Final render at A vs GT: PSNR; higher = better."),
        ("time_seconds", True, "Wall-clock trajectory optimization; lower = faster."),
    ]

    by_metric: dict = {}
    for key, lower, desc in metric_specs:
        means = {m: _agg_mean(aggregate, m, key) for m in methods}
        if all(v is None for v in means.values()):
            continue
        row = _pick_best(means, lower_is_better=lower, eps=eps)
        row["lower_is_better"] = lower
        row["description"] = desc
        by_metric[key] = row

    table_rows = []
    for key, lower, desc in metric_specs:
        if key not in by_metric:
            continue
        row = by_metric[key]
        table_rows.append(
            {
                "metric": key,
                "lower_is_better": lower,
                "description": desc,
                "winner": row["winner"],
                "tie": row["tie"],
                "tied_methods": row["tied_methods"],
                "means": row["means"],
            }
        )

    return {
        "tie_epsilon": eps,
        "methods_compared": methods,
        "by_metric": by_metric,
        "table_rows_for_paper": table_rows,
    }


def render_gt_at_az(strands, az: float, vis):
    return render_dense_gt_blonde(
        strands,
        az,
        vis.vis_image_size,
        vis.vis_radius,
        vis.device,
        vis.dense_gt_strands,
        elev=vis.elevation,
        dist=vis.dist,
    )


def build_metrics_and_poster(
    strands,
    spline_out: dict | None,
    point_out: dict | None,
    gaussian_out: dict | None,
    traj_azs: list[float],
    held_out_azimuths: list[float],
    vis,
    args,
):
    A_az = float(traj_azs[0])
    primary_H = float(held_out_azimuths[0])

    gt_A = render_gt_at_az(strands, A_az, vis)
    gt_H_primary = render_gt_at_az(strands, primary_H, vis)

    out: dict = {
        "trajectory_azimuths": traj_azs,
        "held_out_azimuths": held_out_azimuths,
        "held_out_primary_for_poster": primary_H,
        "viewpoint_A_azimuth": A_az,
        "gt": {},
        "spline": None,
        "pointcloud": None,
        "gaussian": None,
    }

    def pack_method(key: str, method_dict: dict, mode: str):
        if mode == "spline":
            M1 = method_dict["snapshots_cp"][0]
            MT = method_dict["snapshots_cp"][-1]
            r1_A = render_cp_blonde(
                M1,
                A_az,
                vis.vis_image_size,
                vis.vis_radius,
                vis.device,
                vis.samples_per_curve,
                vis.elevation,
                vis.dist,
            )
            rt_A = render_cp_blonde(
                MT,
                A_az,
                vis.vis_image_size,
                vis.vis_radius,
                vis.device,
                vis.samples_per_curve,
                vis.elevation,
                vis.dist,
            )
        elif mode == "pointcloud":
            M1 = method_dict["snapshots_pts"][0]
            MT = method_dict["snapshots_pts"][-1]
            ppc = int(method_dict["points_per_curve"])
            r1_A = render_pc_blonde(M1.to(vis.device), ppc, A_az, vis, vis.device)
            rt_A = render_pc_blonde(MT.to(vis.device), ppc, A_az, vis, vis.device)
        else:
            M1 = method_dict["snapshots_gs"][0]
            MT = method_dict["snapshots_gs"][-1]
            r1_A = render_gs_numpy(M1, A_az, vis, args, vis.device)
            rt_A = render_gs_numpy(MT, A_az, vis, args, vis.device)

        rev_cons_mse, _, _ = mse_psnr_ssim(r1_A, rt_A)
        mse_A_final_gt, psnr_A_final, _ = mse_psnr_ssim(rt_A, gt_A)

        held_mses, held_psnrs, held_ssims = [], [], []
        per_h: dict = {}
        for h_az in held_out_azimuths:
            gt_h = render_gt_at_az(strands, h_az, vis)
            if mode == "spline":
                rt_h = render_cp_blonde(
                    MT,
                    h_az,
                    vis.vis_image_size,
                    vis.vis_radius,
                    vis.device,
                    vis.samples_per_curve,
                    vis.elevation,
                    vis.dist,
                )
            elif mode == "pointcloud":
                rt_h = render_pc_blonde(
                    MT.to(vis.device), int(method_dict["points_per_curve"]), h_az, vis, vis.device
                )
            else:
                rt_h = render_gs_numpy(MT, h_az, vis, args, vis.device)
            hm, hp, hs = mse_psnr_ssim(rt_h, gt_h)
            held_mses.append(hm)
            held_psnrs.append(hp)
            if hs is not None:
                held_ssims.append(hs)
            per_h[f"{h_az:.2f}"] = {"mse": hm, "psnr": hp, "ssim": hs}

        block = {
            "revisit_consistency_mse_M1_vs_MT_at_A": rev_cons_mse,
            "held_out_mse_vs_gt_mean_over_H": float(np.mean(held_mses)),
            "held_out_psnr_vs_gt_mean_over_H": float(np.mean(held_psnrs)),
            "held_out_ssim_vs_gt_mean_over_H": float(np.mean(held_ssims)) if held_ssims else None,
            "held_out_per_azimuth": per_h,
            "render_at_A_after_trajectory_mse_vs_gt": mse_A_final_gt,
            "render_at_A_after_trajectory_psnr_vs_gt": psnr_A_final,
            "num_params": method_dict["num_params"],
            "bytes_estimate": method_dict["bytes_estimate"],
            "time_seconds": method_dict["time_seconds"],
        }
        out[key] = block

        rt_H_vis = None
        if mode == "spline":
            rt_H_vis = render_cp_blonde(
                MT,
                primary_H,
                vis.vis_image_size,
                vis.vis_radius,
                vis.device,
                vis.samples_per_curve,
                vis.elevation,
                vis.dist,
            )
        elif mode == "pointcloud":
            rt_H_vis = render_pc_blonde(
                MT.to(vis.device), int(method_dict["points_per_curve"]), primary_H, vis, vis.device
            )
        else:
            rt_H_vis = render_gs_numpy(MT, primary_H, vis, args, vis.device)

        return {
            "r1_A": np.clip(r1_A, 0, 1),
            "rt_A": np.clip(rt_A, 0, 1),
            "rt_H": np.clip(rt_H_vis, 0, 1),
        }

    panels = {"gt_A": np.clip(gt_A, 0, 1), "gt_H": np.clip(gt_H_primary, 0, 1)}

    if spline_out is not None:
        panels["spline"] = pack_method("spline", spline_out, "spline")
    if point_out is not None:
        panels["point"] = pack_method("pointcloud", point_out, "pointcloud")
    if gaussian_out is not None:
        panels["gaussian"] = pack_method("gaussian", gaussian_out, "gaussian")

    if spline_out is not None and point_out is not None:
        out["memory_compression_point_over_spline_bytes"] = float(
            point_out["bytes_estimate"] / max(spline_out["bytes_estimate"], 1)
        )
    if gaussian_out is not None and spline_out is not None:
        out["memory_compression_gaussian_over_spline_bytes"] = float(
            gaussian_out["bytes_estimate"] / max(spline_out["bytes_estimate"], 1)
        )

    return out, panels


def save_poster(
    panels,
    traj_azs,
    held_out_primary: float,
    held_out_all: list[float],
    output_path: str,
    model_name: str,
):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        log(f"  Poster skipped: {e}")
        return

    gt_A = panels["gt_A"]
    gt_H = panels["gt_H"]
    h_label = f"{held_out_primary:.0f}°"
    if len(held_out_all) > 1:
        h_label = f"{held_out_primary:.0f}° (1/{len(held_out_all)} held-out evals)"

    rows = [
        ("Ground truth", [gt_A, gt_H, gt_A]),
    ]
    row_order = [
        ("Spline memory", "spline"),
        ("Point-cloud memory", "point"),
        ("Gaussian splat memory", "gaussian"),
    ]
    for label, key in row_order:
        if key in panels:
            p = panels[key]
            rows.append((label, [p["r1_A"], p["rt_H"], p["rt_A"]]))

    ncol = 3
    nrow = len(rows)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow), facecolor="#080810")
    if nrow == 1:
        axes = np.array([axes])

    col_titles = [
        f"View A ({traj_azs[0]:.0f}°)\nmemory after 1st timestep",
        f"Held-out H ({h_label})\nmemory after full trajectory",
        f"Revisit A ({traj_azs[0]:.0f}°)\nmemory after full trajectory",
    ]

    for r, (row_title, imgs) in enumerate(rows):
        for c in range(ncol):
            ax = axes[r, c]
            ax.imshow(np.clip(imgs[c], 0, 1))
            ax.axis("off")
            if r == 0:
                ax.set_title(col_titles[c], color="#cccccc", fontsize=10, pad=8)
        axes[r, 0].text(
            -0.08,
            0.5,
            row_title,
            transform=axes[r, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            color="#e0d0a0",
            fontsize=11,
            fontweight="bold",
        )

    fig.suptitle(
        f"Persistent memory under camera motion — {model_name}\n"
        f"Path: {' → '.join(f'{a:.0f}°' for a in traj_azs)}",
        color="white",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0.03, 0.02, 0.98, 0.92])
    plt.savefig(output_path, dpi=160, facecolor="#080810", edgecolor="none")
    plt.close()
    log(f"  Poster saved: {output_path}")


def main():
    p = argparse.ArgumentParser(
        description="Revisit-memory experiment: spline vs point vs Gaussian (multi-seed robustness)."
    )
    p.add_argument("--model-name", default="wWavyThin")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", default="outputs/revisit_memory")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--seeds",
        default="42",
        help="Comma-separated RNG seeds (full experiment repeated per seed; aggregate mean/std).",
    )
    p.add_argument(
        "--poster-seed",
        type=int,
        default=None,
        help="Which seed's tensors to use for the poster (default: first seed in --seeds).",
    )
    p.add_argument("--num-curves", type=int, default=500)
    p.add_argument("--K", type=int, default=12)
    p.add_argument("--pc-points-per-curve", type=int, default=12)
    p.add_argument(
        "--trajectory-azimuths",
        default="0,90,180,270,0",
        help="Comma-separated azimuth sequence (include revisit of first view at end for standard protocol).",
    )
    p.add_argument(
        "--held-out-azimuths",
        default="45",
        help="Comma-separated held-out azimuths (must not appear in trajectory); metrics averaged over all.",
    )
    p.add_argument("--steps-per-view", type=int, default=80)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--init-noise", type=float, default=0.35)
    p.add_argument("--render-weight", type=float, default=0.5)
    p.add_argument("--reproj-weight", type=float, default=1.5)
    p.add_argument("--tangent-weight", type=float, default=0.1)
    p.add_argument("--anchor-weight", type=float, default=0.02)
    p.add_argument("--view-buffer", type=int, default=5)
    p.add_argument("--ema-decay", type=float, default=0.8)
    p.add_argument("--opt-image-size", type=int, default=256)
    p.add_argument("--opt-radius", type=float, default=0.02)
    p.add_argument("--samples-per-curve", type=int, default=96)
    p.add_argument("--radius", type=float, default=0.02)
    p.add_argument("--points-per-pixel", type=int, default=8)
    p.add_argument("--cam-elev", type=float, default=30.0)
    p.add_argument("--cam-dist", type=float, default=4.0)
    p.add_argument("--fov", type=float, default=60.0)
    p.add_argument("--vis-image-size", type=int, default=512)
    p.add_argument("--vis-radius", type=float, default=0.006)
    p.add_argument("--eval-image-size", type=int, default=256)
    p.add_argument("--eval-radius", type=float, default=0.02)
    p.add_argument("--elevation", type=float, default=25.0)
    p.add_argument("--dist", type=float, default=3.5)
    p.add_argument("--dense-gt-strands", type=int, default=3000)
    p.add_argument("--gs-init-scale", type=float, default=0.015)
    p.add_argument("--gs-init-opacity", type=float, default=0.85)
    p.add_argument("--gs-scale-reg", type=float, default=1e-3)
    p.add_argument("--gs-opacity-reg", type=float, default=2e-4)
    p.add_argument("--gs-points-per-pixel", type=int, default=10)
    p.add_argument("--no-spline", action="store_true", help="Skip spline memory run.")
    p.add_argument("--no-point", action="store_true", help="Skip point-cloud memory run.")
    p.add_argument("--no-gaussian", action="store_true", help="Skip Gaussian splat memory run.")
    args = p.parse_args()

    run_spline = not args.no_spline
    run_point = not args.no_point
    run_gaussian = not args.no_gaussian
    if not run_spline and not run_point and not run_gaussian:
        raise SystemExit("Enable at least one method (do not pass --no-spline --no-point --no-gaussian).")

    seeds = parse_seeds(args.seeds)
    poster_seed = args.poster_seed if args.poster_seed is not None else seeds[0]
    if poster_seed not in seeds:
        raise SystemExit(f"--poster-seed {poster_seed} must appear in --seeds {seeds}.")

    traj = parse_trajectory(args.trajectory_azimuths)
    held_outs = parse_held_out_azimuths(args.held_out_azimuths)
    traj_set = set(round(float(x), 6) for x in traj)
    for ho in held_outs:
        if round(ho, 6) in traj_set:
            raise ValueError(
                f"Held-out azimuth {ho} collides with trajectory. Change --held-out-azimuths or trajectory."
            )

    os.makedirs(args.output_dir, exist_ok=True)

    hair_path = download_yuksel_hair(args.model_name, save_dir=args.data_dir)
    strands = load_hair_file(hair_path)

    vis = SimpleNamespace(
        vis_image_size=args.vis_image_size,
        vis_radius=args.vis_radius,
        eval_image_size=args.eval_image_size,
        eval_radius=args.eval_radius,
        samples_per_curve=args.samples_per_curve,
        elevation=args.elevation,
        dist=args.dist,
        dense_gt_strands=args.dense_gt_strands,
        device=args.device,
    )

    per_seed: list[dict] = []
    panels_poster = None

    for seed in seeds:
        log(
            f"\n{'═' * 60}\n  Seed {seed} | {args.model_name}\n"
            f"  Trajectory (deg): {traj}\n  Held-out H (deg): {held_outs}\n{'═' * 60}"
        )
        torch.manual_seed(seed)
        np.random.seed(seed)

        gt_cp = hair_to_spline_field(
            strands, num_curves=args.num_curves, K=args.K, seed=seed, strategy="diverse"
        )
        gt_cp_o = orient_cp(gt_cp).to(args.device)
        with torch.no_grad():
            gt_points = evaluate_bspline(gt_cp_o, args.pc_points_per_curve).reshape(-1, 3)

        spline_out = None
        point_out = None
        gaussian_out = None

        if run_spline:
            spline_out = optimize_spline_trajectory(gt_cp, traj, args, args.device)
            torch.save(spline_out, os.path.join(args.output_dir, f"revisit_spline_seed{seed}.pt"))

        if run_point:
            point_out = optimize_point_trajectory(gt_points, traj, args, args.device)
            torch.save(point_out, os.path.join(args.output_dir, f"revisit_point_seed{seed}.pt"))

        if run_gaussian:
            gaussian_out = optimize_gaussian_trajectory(gt_points, traj, args, args.device)
            torch.save(gaussian_out, os.path.join(args.output_dir, f"revisit_gaussian_seed{seed}.pt"))

        metrics, panels = build_metrics_and_poster(
            strands,
            spline_out,
            point_out,
            gaussian_out,
            traj,
            held_outs,
            vis,
            args,
        )
        metrics["seed"] = seed
        metrics["model_name"] = args.model_name
        per_seed.append(metrics)
        if seed == poster_seed:
            panels_poster = panels

    if panels_poster is None:
        raise RuntimeError("poster_seed did not match any run.")

    aggregate = aggregate_across_seeds(per_seed)
    winner_summary = build_winner_summary(aggregate)
    final_out = {
        "seeds": seeds,
        "poster_seed": poster_seed,
        "aggregate_over_seeds": aggregate,
        "winner_summary": winner_summary,
        "per_seed": per_seed,
        "notes": {
            "revisit_consistency": (
                "MSE between render(memory_after_step1, A) and render(memory_after_full_trajectory, A); "
                "lower ⇒ less drift at the same pose."
            ),
            "held_out": (
                "Mean over held-out azimuths: MSE/PSNR/SSIM of final memory vs dense GT (install scikit-image for SSIM)."
            ),
            "memory": (
                "Spline: N×K×3 scalars; point: N×ppc×3; Gaussian: N×(3+1+1) (mean + log_scale + logit_opacity)."
            ),
            "fairness": (
                "Gaussian has more DOF per primitive than point/spline position-only counts; interpret jointly with "
                "revisit + held-out metrics."
            ),
            "winner_summary": (
                "winner_summary uses aggregate_over_seeds means; best metric wins; "
                "tie if values are within winner_summary.tie_epsilon."
            ),
        },
    }

    json_path = os.path.join(args.output_dir, "revisit_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_out, f, indent=2)

    poster_path = os.path.join(args.output_dir, "revisit_poster.png")
    save_poster(
        panels_poster,
        traj,
        held_outs[0],
        held_outs,
        poster_path,
        args.model_name,
    )

    log(f"\n  Results: {json_path}")
    log(f"  Poster:  {poster_path} (seed {poster_seed})")
    log(f"  Seeds:   {seeds} | aggregate keys in JSON under aggregate_over_seeds")
    log("  Winner summary (aggregate means):")
    for row in winner_summary.get("table_rows_for_paper", []):
        log(f"    {row['metric']}: winner={row['winner']}  means={row['means']}")
    log("  Done.\n")


if __name__ == "__main__":
    main()
