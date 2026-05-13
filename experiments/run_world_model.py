"""
run_world_model.py — Mini World Model with Persistent Curve Memory

Demonstrates splines as spatial memory for a world model using
Cem Yuksel hair data (via hair_loader.py).

Produces (N, K, 3) control points → evaluate_bspline → renderer → losses
→ sequential optimization loop.

Three phases:

  Phase 1 — EXPLORATION: Camera observes 0° to explore_range° (partial).
    Predict-observe-update loop. Memory builds incrementally.

  Phase 2 — VIDEO GENERATION: Memory FROZEN. Camera does full 360°.
    Renders include UNOBSERVED angles → proves spatial memory generalization.

  Phase 3 — REVISITATION: Return to explored viewpoints.
    Measure temporal consistency.

Run on Amarel:
    python run_world_model.py --model-name wStraight \
        --num-curves 500 --output-dir outputs/world_model_hair

    # Quick test:
    python run_world_model.py --model-name wStraight --quick \
        --output-dir outputs/wm_quick
"""

import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gensplines.coordinates import orient_pts as orient


def log(msg):
    print(msg, flush=True)


def hair_colors(n, seed=42):
    rng = np.random.RandomState(seed)
    base = np.array([0.78, 0.62, 0.35])
    j = rng.uniform(-0.04, 0.04, size=(n, 3))
    return torch.tensor(np.clip(base + j, 0.05, 1), dtype=torch.float32)


BG = (0.12, 0.12, 0.15)


class CachedTubeRenderer:
    """
    Builds tube mesh ONCE, renders from multiple angles WITHOUT rebuilding.

    Visual improvements:
    - Curve interpolation: densifies N trained curves to densify_factor*N for rendering
    - Thinner tubes (0.0015 default vs 0.003 before)
    - Better lighting: low ambient, high diffuse, grazing angle for strand highlights
    - More arc-length samples (vis_samples=128 default)
    """
    def __init__(self, cp, vis_samples, device, tube_radius=0.0015, n_sides=4,
                 seed=42, densify_factor=1):
        from gensplines.spline import evaluate_bspline
        from gensplines.render_utils import build_tube_mesh, make_tube_colors

        N, K, D = cp.shape

        # Densify: interpolate between adjacent curves for denser visualization
        if densify_factor > 1:
            dense_cps = [cp]
            for f in range(1, densify_factor):
                alpha = f / densify_factor
                # Interpolate each curve with its neighbor (circular wrap)
                shifted = torch.roll(cp, -1, dims=0)
                interp = (1 - alpha) * cp + alpha * shifted
                dense_cps.append(interp)
            cp_vis = torch.cat(dense_cps, dim=0)
            N_vis = cp_vis.shape[0]
        else:
            cp_vis = cp
            N_vis = N

        tr = tube_radius * (500 / N_vis) ** 0.18

        with torch.no_grad():
            curve_pts = evaluate_bspline(cp_vis, vis_samples)

        self.verts, self.faces = build_tube_mesh(curve_pts, radius=tr, n_sides=n_sides)
        self.vcols = make_tube_colors(N_vis, vis_samples, n_sides, seed).to(device)
        self.verts = self.verts.to(device)
        self.faces = self.faces.to(device)
        self.device = device

        # Pre-build mesh and texture (reused every render)
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import TexturesVertex
        tex = TexturesVertex(verts_features=[self.vcols])
        self._mesh = Meshes(verts=[self.verts], faces=[self.faces], textures=tex)

        # Pre-build raster settings
        from pytorch3d.renderer import RasterizationSettings
        self._rset = RasterizationSettings(image_size=384, blur_radius=0.0,
                                           faces_per_pixel=2, bin_size=0)

    def render(self, az, image_size, elev=25.0, dist=3.0):
        from pytorch3d.renderer import (
            look_at_view_transform, FoVPerspectiveCameras,
            MeshRenderer, MeshRasterizer,
            SoftPhongShader, PointLights)

        # Update raster settings if image size changed
        if self._rset.image_size != image_size:
            from pytorch3d.renderer import RasterizationSettings
            self._rset = RasterizationSettings(image_size=image_size, blur_radius=0.0,
                                               faces_per_pixel=2, bin_size=0)

        R, T = look_at_view_transform(dist=dist, elev=elev, azim=az)
        cam = FoVPerspectiveCameras(device=self.device, R=R, T=T)

        # Improved lighting: lower ambient, higher diffuse, grazing angle
        lights = PointLights(
            device=self.device, location=[[2., 4., 4.]],
            ambient_color=[[0.32, 0.28, 0.24]],
            diffuse_color=[[0.72, 0.65, 0.52]],
            specular_color=[[0.30, 0.25, 0.18]])

        rend = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cam, raster_settings=self._rset),
            shader=SoftPhongShader(device=self.device, cameras=cam, lights=lights))

        res = rend(self._mesh)
        img = res[0, ..., :3].cpu().numpy()
        alpha = res[0, ..., 3:4].cpu().numpy()
        return np.clip(img * alpha + np.array(BG).reshape(1, 1, 3) * (1 - alpha), 0, 1)


def render_vis_tubes(cp, az, args, seed=42):
    """Render spline CPs as tube meshes (for spline memory + GT visualization)."""
    try:
        from gensplines.render_utils import render_spline_tubes
        N = cp.shape[0]
        tube_r = 0.003 * (500 / N) ** 0.2
        return render_spline_tubes(cp, args.vis_samples, float(az),
                                   args.vis_size, args.device,
                                   tube_radius=tube_r, n_sides=4,
                                   elev=args.vis_elev, dist=args.vis_dist, seed=seed)
    except Exception:
        # Fallback to point rendering if render_utils not available
        from gensplines.spline import evaluate_bspline
        with torch.no_grad():
            pts = evaluate_bspline(cp, args.vis_samples).reshape(-1, 3)
        return render_vis_dots(pts, az, args, seed)


def render_vis_dots(points, az, args, seed=99):
    """Render points as dots (for point cloud memory visualization)."""
    from pytorch3d.renderer import (
        AlphaCompositor, FoVPerspectiveCameras,
        PointsRasterizationSettings, PointsRasterizer,
        PointsRenderer, look_at_view_transform)
    from pytorch3d.structures import Pointclouds

    colors = hair_colors(points.shape[0], seed)
    R, T = look_at_view_transform(dist=args.vis_dist, elev=args.vis_elev, azim=az)
    cam = FoVPerspectiveCameras(device=args.device, R=R, T=T)
    rset = PointsRasterizationSettings(
        image_size=args.vis_size, radius=args.vis_radius,
        points_per_pixel=10, bin_size=0)
    rend = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cam, raster_settings=rset),
        compositor=AlphaCompositor(background_color=BG))
    pc = Pointclouds(points=[points.to(args.device)], features=[colors.to(args.device)])
    return np.clip(rend(pc)[0, ..., :3].cpu().numpy(), 0, 1)


def make_camera(az, device, dist=4.0, elev=30.0, fov=60.0):
    from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=az)
    return FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov)


class CachedRenderer:
    """
    Cached differentiable point renderer.
    Caches raster settings, compositor, AND the per-view camera/rasterizer/renderer.
    Within a view's optimization steps, azimuth is constant — no rebuild needed.
    """
    def __init__(self, config, device):
        from pytorch3d.renderer import (
            PointsRasterizationSettings, AlphaCompositor)

        self.device = device
        self.config = config
        self.raster_settings = PointsRasterizationSettings(
            image_size=config["image_size"],
            radius=config["radius"],
            points_per_pixel=config.get("points_per_pixel", 4),
            bin_size=0,
        )
        self.compositor = AlphaCompositor()
        self._white_cache = {}
        self._last_az = None
        self._renderer = None

    def render(self, points, azimuth, features=None):
        from pytorch3d.renderer import (
            look_at_view_transform, FoVPerspectiveCameras,
            PointsRasterizer, PointsRenderer)
        from pytorch3d.structures import Pointclouds

        # Only rebuild renderer when azimuth changes
        if self._last_az != azimuth:
            cfg = self.config
            R, T = look_at_view_transform(
                dist=cfg.get("dist", 4.0), elev=cfg.get("elev", 30.0), azim=azimuth)
            cam = FoVPerspectiveCameras(device=self.device, R=R, T=T)
            rasterizer = PointsRasterizer(cameras=cam, raster_settings=self.raster_settings)
            self._renderer = PointsRenderer(rasterizer=rasterizer, compositor=self.compositor)
            self._last_az = azimuth

        if features is None:
            n = points.shape[0]
            if n not in self._white_cache:
                self._white_cache[n] = torch.ones(n, 3, device=self.device)
            features = self._white_cache[n]

        pc = Pointclouds(points=[points], features=[features])
        return self._renderer(pc)[0]


_cached_renderer = None

def render_for_loss(points, az, config, device):
    """Fast render using cached settings. Only camera changes per call."""
    global _cached_renderer
    if _cached_renderer is None or _cached_renderer.config != config:
        _cached_renderer = CachedRenderer(config, device)
    return _cached_renderer.render(points, az)


# ══════════════════════════════════════════════════════════════
# Core: Persistent Memory with Predict-Observe-Update loop
# ══════════════════════════════════════════════════════════════

from gensplines.memory import PersistentCurveMemory, PersistentPointMemory


def explore_phase(gt_cp, gt_points, args):
    """
    Phase 1: Sequential exploration with predict-observe-update loop.

    At each step:
      1. PREDICT: render current memory state from the upcoming viewpoint
      2. OBSERVE: receive GT view
      3. UPDATE: optimize memory to match observation
      4. STABILIZE: EMA anchor update

    Returns history of predictions, memory states, and drift.
    """
    from gensplines.spline import evaluate_bspline

    device = args.device
    N, K = gt_cp.shape[0], gt_cp.shape[1]
    M_opt = args.opt_samples    # fewer samples = faster optimization
    M_vis = args.vis_samples    # more samples = prettier visualization

    # Exploration trajectory: partial orbit (0° to explore_range°)
    explore_azs = torch.linspace(0, args.explore_range,
                                 args.explore_views + 1)[:-1]

    # Render config for optimization
    opt_config = {
        "image_size": args.opt_size, "radius": args.opt_radius,
        "points_per_pixel": args.opt_ppp, "dist": 4.0, "fov": 60.0,
        "compositor": "alpha", "elev": 30.0,
    }

    # Pre-render all GT images and projections
    log("  Pre-rendering GT exploration views ...")
    gt_images, gt_cameras, gt_projections = [], [], []
    for az in explore_azs:
        az_val = az.item()
        cam = make_camera(az_val, device)
        gt_cameras.append(cam)
        with torch.no_grad():
            img = render_for_loss(gt_points, az_val, opt_config, device)
            gt_images.append(img[..., :3].detach())
            proj = cam.transform_points_screen(
                gt_points.unsqueeze(0),
                image_size=((args.opt_size, args.opt_size),)
            )[0, :, :2]
            gt_projections.append(proj.detach())

    # ── SPLINE MEMORY ──
    log("\n  === SPLINE MEMORY (exploration) ===")
    sp_pred_cp = nn.Parameter(gt_cp.clone() + args.init_noise * torch.randn_like(gt_cp))
    sp_memory = PersistentCurveMemory(sp_pred_cp.data, args.ema_decay)

    sp_history = {
        "predictions": [],   # rendered BEFORE update (world model output)
        "post_update": [],   # rendered AFTER update
        "drifts": [],
        "prediction_errors": [],  # MSE between prediction and GT
    }

    for vi in range(args.explore_views):
        az = explore_azs[vi].item()
        buffer_start = max(0, vi - args.view_buffer + 1)
        buffer_idx = list(range(buffer_start, vi + 1))

        # STEP 1: PREDICT — render current memory BEFORE seeing GT
        with torch.no_grad():
            pred_pts = evaluate_bspline(sp_pred_cp.data, M_opt).reshape(-1, 3)
            pred_img = render_for_loss(pred_pts, az, opt_config, device)[..., :3]
            pred_error = F.mse_loss(pred_img, gt_images[vi]).item()
            sp_history["prediction_errors"].append(pred_error)

            # Visual render for timeline
            if vi % max(1, args.explore_views // 8) == 0 or vi == 0:
                vis_img = render_vis_dots(evaluate_bspline(sp_pred_cp.data, args.vis_samples).reshape(-1, 3), az, args, seed=42)
                sp_history["predictions"].append((vi, az, vis_img))

        # STEP 2: OBSERVE + UPDATE
        optimizer = torch.optim.Adam([sp_pred_cp], lr=args.lr)
        for _ in range(args.steps_per_view):
            optimizer.zero_grad()

            # Single B-spline evaluation — reused for ALL losses
            pts_curve = evaluate_bspline(sp_pred_cp, M_opt)  # (N, M, 3)
            pts = pts_curve.reshape(-1, 3)

            # Render loss
            rendered = render_for_loss(pts, az, opt_config, device)[..., :3]
            loss_render = F.mse_loss(rendered, gt_images[vi])

            # Reprojection loss — HUBER instead of MSE (Fix #3)
            # Linear penalty for large errors prevents outlier curves from
            # dominating gradient updates
            loss_reproj = 0
            for bi in buffer_idx:
                pred_2d = gt_cameras[bi].transform_points_screen(
                    pts.unsqueeze(0),
                    image_size=((args.opt_size, args.opt_size),)
                )[0, :, :2]
                loss_reproj = loss_reproj + F.huber_loss(
                    pred_2d, gt_projections[bi], delta=args.huber_delta)
            loss_reproj = loss_reproj / len(buffer_idx)

            # Anchor memory loss
            loss_anchor = args.anchor_weight * F.mse_loss(
                sp_pred_cp, sp_memory.get_anchor())

            # Tangent consistency — penalizes sharp kinks (reuses pts_curve)
            tangents = pts_curve[:, 1:] - pts_curve[:, :-1]
            tangents = F.normalize(tangents, dim=-1)
            cos_sim = (tangents[:, 1:] * tangents[:, :-1]).sum(dim=-1)
            loss_tangent = args.tangent_weight * (1.0 - cos_sim).mean()

            # Curvature regularization — second-difference of CPs
            second_diff = sp_pred_cp[:, 2:] - 2*sp_pred_cp[:, 1:-1] + sp_pred_cp[:, :-2]
            loss_curv = args.curv_weight * second_diff.norm(dim=-1).mean()

            loss = (args.render_weight * loss_render +
                    args.reproj_weight * loss_reproj +
                    loss_anchor + loss_tangent + loss_curv)
            loss.backward()

            # Per-curve gradient norm clipping (Fix #2)
            # Prevents outlier curves from dominating the gradient
            if sp_pred_cp.grad is not None:
                with torch.no_grad():
                    grad = sp_pred_cp.grad
                    per_curve_norm = grad.reshape(N, -1).norm(dim=-1)
                    threshold = per_curve_norm.median() * args.grad_clip_factor
                    scale = (threshold / per_curve_norm.clamp(min=threshold))
                    sp_pred_cp.grad *= scale.reshape(-1, 1, 1)

            optimizer.step()

            # Hard radius clamp on anchor distance (Fix #4)
            # Safety net: no curve can drift more than max_drift from anchor
            with torch.no_grad():
                anchor = sp_memory.get_anchor()
                delta = sp_pred_cp.data - anchor
                delta_norm = delta.reshape(N, -1).norm(dim=-1, keepdim=True)  # (N, 1)
                max_d = args.max_drift
                # Only clamp curves that exceeded max_drift
                exceeded = delta_norm > max_d
                if exceeded.any():
                    clamped_scale = (max_d / delta_norm.clamp(min=1e-8))
                    # Reshape for broadcasting: (N, 1) -> (N, 1, 1) for (N, K, 3)
                    clamped_scale = clamped_scale.unsqueeze(-1)
                    sp_pred_cp.data = torch.where(
                        exceeded.unsqueeze(-1).expand_as(sp_pred_cp),
                        anchor + delta * clamped_scale.expand_as(delta),
                        sp_pred_cp.data
                    )

        # STEP 3: STABILIZE
        sp_memory.update(sp_pred_cp.data)

        with torch.no_grad():
            drift = (gt_cp - sp_pred_cp.data).norm(dim=-1).mean().item()
            sp_history["drifts"].append(drift)

            if vi % max(1, args.explore_views // 8) == 0:
                vis_img = render_vis_dots(evaluate_bspline(sp_pred_cp.data, args.vis_samples).reshape(-1, 3), az, args, seed=42)
                sp_history["post_update"].append((vi, az, vis_img))

        if vi % max(1, args.explore_views // 6) == 0:
            log(f"    View {vi:3d}/{args.explore_views} | az={az:5.1f}° | "
                f"drift={drift:.4f} | pred_err={pred_error:.5f}")

    sp_final_cp = sp_pred_cp.data.clone()

    # ── POINT CLOUD MEMORY ──
    log("\n  === POINT CLOUD MEMORY (exploration) ===")
    pc_pred = nn.Parameter(gt_points.clone() + args.init_noise * torch.randn_like(gt_points))
    pc_memory = PersistentPointMemory(pc_pred.data, args.ema_decay)

    pc_history = {
        "predictions": [],
        "post_update": [],
        "drifts": [],
        "prediction_errors": [],
    }

    for vi in range(args.explore_views):
        az = explore_azs[vi].item()
        buffer_start = max(0, vi - args.view_buffer + 1)
        buffer_idx = list(range(buffer_start, vi + 1))

        # PREDICT
        with torch.no_grad():
            pred_img = render_for_loss(pc_pred.data, az, opt_config, device)[..., :3]
            pred_error = F.mse_loss(pred_img, gt_images[vi]).item()
            pc_history["prediction_errors"].append(pred_error)

            if vi % max(1, args.explore_views // 8) == 0 or vi == 0:
                vis_img = render_vis_dots(pc_pred.data, az, args, seed=99)
                pc_history["predictions"].append((vi, az, vis_img))

        # OBSERVE + UPDATE (same losses as spline, minus tangent/curvature which are spline-specific)
        optimizer = torch.optim.Adam([pc_pred], lr=args.lr)
        for _ in range(args.steps_per_view):
            optimizer.zero_grad()
            pred_img = render_for_loss(pc_pred, az, opt_config, device)[..., :3]
            loss_render = F.mse_loss(pred_img, gt_images[vi])

            # Huber reprojection (same as spline for fair comparison)
            loss_reproj = 0
            for bi in buffer_idx:
                pred_2d = gt_cameras[bi].transform_points_screen(
                    pc_pred.unsqueeze(0),
                    image_size=((args.opt_size, args.opt_size),)
                )[0, :, :2]
                loss_reproj = loss_reproj + F.huber_loss(
                    pred_2d, gt_projections[bi], delta=args.huber_delta)
            loss_reproj = loss_reproj / len(buffer_idx)

            loss_anchor = args.anchor_weight * F.mse_loss(
                pc_pred, pc_memory.get_anchor())

            loss = (args.render_weight * loss_render +
                    args.reproj_weight * loss_reproj + loss_anchor)
            loss.backward()
            optimizer.step()

        pc_memory.update(pc_pred.data)

        with torch.no_grad():
            drift = (gt_points - pc_pred.data).norm(dim=-1).mean().item()
            pc_history["drifts"].append(drift)

            if vi % max(1, args.explore_views // 8) == 0:
                vis_img = render_vis_dots(pc_pred.data, az, args, seed=99)
                pc_history["post_update"].append((vi, az, vis_img))

        if vi % max(1, args.explore_views // 6) == 0:
            log(f"    View {vi:3d}/{args.explore_views} | az={az:5.1f}° | "
                f"drift={drift:.4f} | pred_err={pred_error:.5f}")

    pc_final = pc_pred.data.clone()

    return (sp_final_cp, sp_history, pc_final, pc_history,
            explore_azs, gt_images, gt_cameras, gt_projections)


def generate_video_phase(sp_cp, pc_points, gt_cp, gt_points, args):
    """
    Phase 2: Video generation from FROZEN memory.

    Memory is not updated. Camera follows a full 360° trajectory.
    At each angle, render from the frozen memory state.
    This IS video generation supported by spatial memory.
    """
    from gensplines.spline import evaluate_bspline

    device = args.device
    M_vis = args.vis_samples

    gen_azs = torch.linspace(0, 360, args.gen_views + 1)[:-1]

    log("\n  === VIDEO GENERATION FROM FROZEN MEMORY ===")
    log(f"  Generating {args.gen_views} frames over full 360°")
    log(f"  Memory was trained on 0°-{args.explore_range}° only")

    sp_frames, pc_frames, gt_frames = [], [], []
    sp_mse, pc_mse = [], []

    with torch.no_grad():
        sp_pts = evaluate_bspline(sp_cp, M_vis).reshape(-1, 3)
        gt_pts_full = evaluate_bspline(gt_cp, M_vis).reshape(-1, 3)

    for i, az in enumerate(gen_azs):
        az_val = az.item()

        # Use DOT renders for speed — MSE is the same, generation is ~100x faster
        sp_img = render_vis_dots(sp_pts, az_val, args, seed=42)
        pc_img = render_vis_dots(pc_points, az_val, args, seed=99)
        gt_img = render_vis_dots(gt_pts_full, az_val, args, seed=42)

        sp_frames.append(sp_img)
        pc_frames.append(pc_img)
        gt_frames.append(gt_img)

        # Quantitative: MSE of rendered images
        sp_mse.append(((sp_img - gt_img) ** 2).mean())
        pc_mse.append(((pc_img - gt_img) ** 2).mean())

        if i % max(1, args.gen_views // 6) == 0:
            in_range = "OBSERVED" if az_val <= args.explore_range else "UNOBSERVED"
            log(f"    Frame {i:3d} | az={az_val:5.1f}° [{in_range}] | "
                f"sp_mse={sp_mse[-1]:.5f} | pc_mse={pc_mse[-1]:.5f}")

    return {
        "azimuths": gen_azs.tolist(),
        "sp_frames": sp_frames,
        "pc_frames": pc_frames,
        "gt_frames": gt_frames,
        "sp_mse": sp_mse,
        "pc_mse": pc_mse,
        "explore_range": args.explore_range,
    }


def revisitation_phase(sp_cp, pc_points, gt_cp, gt_points,
                       explore_azs, args):
    """
    Phase 3: Revisitation — return to explored viewpoints.
    Compare current memory rendering vs GT at previously seen angles.
    """
    from gensplines.spline import evaluate_bspline

    M_vis = args.vis_samples
    log("\n  === REVISITATION CONSISTENCY ===")

    revisit_azs = explore_azs[::max(1, len(explore_azs) // 8)]

    sp_revisit, pc_revisit, gt_revisit = [], [], []
    sp_errors, pc_errors = [], []

    # Reuse cached tube renderers (NO densification for speed/stability)
    log("  Building tube meshes for revisitation...")
    try:
        sp_tube = CachedTubeRenderer(sp_cp, min(M_vis, 64), args.device, seed=42,
                                     tube_radius=args.tube_radius, n_sides=args.tube_sides,
                                     densify_factor=1)
        gt_tube = CachedTubeRenderer(gt_cp, min(M_vis, 64), args.device, seed=42,
                                     tube_radius=args.tube_radius, n_sides=args.tube_sides,
                                     densify_factor=1)
        use_tubes = True
        log(f"    Mesh: {sp_tube.verts.shape[0]:,} verts, {sp_tube.faces.shape[0]:,} faces")
    except Exception as e:
        log(f"    Tube build failed ({e}), using dots")
        use_tubes = False

    with torch.no_grad():
        if not use_tubes:
            sp_pts = evaluate_bspline(sp_cp, M_vis).reshape(-1, 3)
            gt_pts = evaluate_bspline(gt_cp, M_vis).reshape(-1, 3)

    for az in revisit_azs:
        az_val = az.item()
        if use_tubes:
            sp_img = sp_tube.render(az_val, args.vis_size, args.vis_elev, args.vis_dist)
            gt_img = gt_tube.render(az_val, args.vis_size, args.vis_elev, args.vis_dist)
        else:
            sp_img = render_vis_dots(sp_pts, az_val, args, seed=42)
            gt_img = render_vis_dots(gt_pts, az_val, args, seed=42)
        pc_img = render_vis_dots(pc_points, az_val, args, seed=99)

        sp_revisit.append((az_val, sp_img))
        pc_revisit.append((az_val, pc_img))
        gt_revisit.append((az_val, gt_img))

        sp_err = ((sp_img - gt_img) ** 2).mean()
        pc_err = ((pc_img - gt_img) ** 2).mean()
        sp_errors.append(sp_err)
        pc_errors.append(pc_err)

        log(f"    Revisit az={az_val:5.1f}° | sp_err={sp_err:.5f} | pc_err={pc_err:.5f}")

    return {
        "sp_revisit": sp_revisit,
        "pc_revisit": pc_revisit,
        "gt_revisit": gt_revisit,
        "sp_errors": sp_errors,
        "pc_errors": pc_errors,
        "revisit_azs": [a.item() for a in revisit_azs],
    }


# ══════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════

def plot_exploration(sp_hist, pc_hist, explore_azs, args):
    """Timeline of memory building during exploration."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=BG)
    fig.subplots_adjust(hspace=0.25, wspace=0.3)

    azs = [a.item() for a in explore_azs]

    # Prediction error (THE correct world model metric)
    ax = axes[0, 0]
    ax.plot(azs, sp_hist["prediction_errors"], "b-", lw=2.5, label="Spline memory")
    ax.plot(azs, pc_hist["prediction_errors"], "r-", lw=2.5, label="Point cloud")
    ax.set_xlabel("Observation azimuth (°)", color="white")
    ax.set_ylabel("Prediction error (MSE)", color="white")
    ax.set_title("World Model Prediction Quality\n(lower = better prediction BEFORE observing)",
                 color="white", fontsize=12, fontweight="bold")
    ax.legend(facecolor="#222", edgecolor="white", labelcolor="white")
    ax.tick_params(colors="white")
    ax.set_facecolor("#1a1a1a")
    ax.grid(alpha=0.2, color="white")

    # Prediction improvement rate
    ax = axes[0, 1]
    sp_errs = np.array(sp_hist["prediction_errors"])
    pc_errs = np.array(pc_hist["prediction_errors"])
    # Compute spline advantage percentage at each step
    advantage = (pc_errs - sp_errs) / (pc_errs + 1e-10) * 100
    ax.plot(azs, advantage, "g-", lw=2, label="Spline advantage (%)")
    ax.axhline(y=0, color="white", ls="--", alpha=0.3)
    ax.fill_between(azs, 0, advantage, where=advantage > 0, alpha=0.2, color="green")
    ax.fill_between(azs, 0, advantage, where=advantage < 0, alpha=0.2, color="red")
    ax.set_xlabel("Observation azimuth (°)", color="white")
    ax.set_ylabel("Spline better ← 0 → Points better", color="white")
    ax.set_title("Spline Advantage Over Time\n(% lower prediction error than points)",
                 color="white", fontsize=11, fontweight="bold")
    ax.tick_params(colors="white")
    ax.set_facecolor("#1a1a1a")
    ax.grid(alpha=0.2, color="white")

    # Prediction snapshots — spline
    ax = axes[1, 0]
    if sp_hist["predictions"]:
        n = min(4, len(sp_hist["predictions"]))
        strip = np.concatenate([sp_hist["predictions"][i][2]
                                for i in range(n)], axis=1)
        ax.imshow(strip)
        titles = " | ".join([f"t={sp_hist['predictions'][i][0]}, "
                             f"az={sp_hist['predictions'][i][1]:.0f}°"
                             for i in range(n)])
        ax.set_title(f"Spline Predictions: {titles}",
                     color="#88ccff", fontsize=9, fontweight="bold")
    ax.axis("off")

    # Prediction snapshots — point cloud
    ax = axes[1, 1]
    if pc_hist["predictions"]:
        n = min(4, len(pc_hist["predictions"]))
        strip = np.concatenate([pc_hist["predictions"][i][2]
                                for i in range(n)], axis=1)
        ax.imshow(strip)
        titles = " | ".join([f"t={pc_hist['predictions'][i][0]}, "
                             f"az={pc_hist['predictions'][i][1]:.0f}°"
                             for i in range(n)])
        ax.set_title(f"Point Cloud Predictions: {titles}",
                     color="#ff9999", fontsize=9, fontweight="bold")
    ax.axis("off")

    fig.suptitle("Phase 1: Exploration — Sequential Memory Building\n"
                 "Camera observes one view at a time, memory predicts BEFORE observing",
                 color="white", fontsize=13, fontweight="bold", y=0.98)

    p = os.path.join(args.output_dir, "exploration_timeline.png")
    plt.savefig(p, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    log(f"  {p}")


def plot_generation(gen_data, args):
    """Show video generation quality — observed vs unobserved regions."""
    azs = gen_data["azimuths"]
    sp_mse = gen_data["sp_mse"]
    pc_mse = gen_data["pc_mse"]
    boundary = gen_data["explore_range"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor=BG)
    fig.subplots_adjust(wspace=0.3)

    # MSE over full 360°
    ax = axes[0]
    ax.plot(azs, sp_mse, "b-", lw=2, label="Spline memory")
    ax.plot(azs, pc_mse, "r-", lw=2, label="Point cloud")
    ax.axvspan(boundary, 360, alpha=0.15, color="yellow",
               label=f"Unobserved region ({boundary}°-360°)")
    ax.axvspan(0, boundary, alpha=0.08, color="green", label=f"Observed (0°-{boundary}°)")
    ax.set_xlabel("Generation azimuth (°)", color="white")
    ax.set_ylabel("Render MSE vs GT", color="white")
    ax.set_title("Video Generation Quality: Observed vs Unobserved Regions",
                 color="white", fontsize=12, fontweight="bold")
    ax.legend(facecolor="#222", edgecolor="white", labelcolor="white", fontsize=9)
    ax.tick_params(colors="white")
    ax.set_facecolor("#1a1a1a")
    ax.grid(alpha=0.2, color="white")

    # Average MSE in observed vs unobserved
    ax = axes[1]
    obs_sp = np.mean([m for a, m in zip(azs, sp_mse) if a <= boundary])
    obs_pc = np.mean([m for a, m in zip(azs, pc_mse) if a <= boundary])
    unobs_sp = np.mean([m for a, m in zip(azs, sp_mse) if a > boundary])
    unobs_pc = np.mean([m for a, m in zip(azs, pc_mse) if a > boundary])

    x = np.arange(2)
    w = 0.35
    ax.bar(x - w/2, [obs_sp, unobs_sp], w, label="Spline", color="#4472C4")
    ax.bar(x + w/2, [obs_pc, unobs_pc], w, label="Point Cloud", color="#C0504D")
    ax.set_xticks(x)
    ax.set_xticklabels(["Observed\nRegion", "Unobserved\nRegion"], color="white")
    ax.set_ylabel("Mean Render MSE", color="white")
    ax.set_title("Generation Quality by Region", color="white",
                 fontsize=12, fontweight="bold")
    ax.legend(facecolor="#222", edgecolor="white", labelcolor="white")
    ax.tick_params(colors="white")
    ax.set_facecolor("#1a1a1a")

    # Annotate differences
    for i, (s, p) in enumerate([(obs_sp, obs_pc), (unobs_sp, unobs_pc)]):
        winner = "Spline" if s < p else "Point"
        pct = abs(s - p) / max(s, p) * 100
        ax.text(i, max(s, p) * 1.05, f"{winner} +{pct:.1f}%",
                ha="center", color="#66ff66" if winner == "Spline" else "#ff6666",
                fontsize=10, fontweight="bold")

    fig.suptitle("Phase 2: Video Generation from Frozen Memory",
                 color="white", fontsize=14, fontweight="bold", y=1.02)

    p = os.path.join(args.output_dir, "generation_quality.png")
    plt.savefig(p, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    log(f"  {p}")


def plot_revisitation(revisit_data, args):
    """Show revisitation consistency."""
    fig, axes = plt.subplots(3, len(revisit_data["revisit_azs"]),
                             figsize=(4 * len(revisit_data["revisit_azs"]), 12),
                             facecolor=BG)
    fig.subplots_adjust(wspace=0.03, hspace=0.1, left=0.04, right=0.99,
                        top=0.88, bottom=0.04)

    for col, (az, gt, sp, pc) in enumerate(zip(
        revisit_data["revisit_azs"],
        revisit_data["gt_revisit"],
        revisit_data["sp_revisit"],
        revisit_data["pc_revisit"],
    )):
        axes[0, col].imshow(gt[1]); axes[0, col].axis("off")
        axes[0, col].set_title(f"{az:.0f}°", color="white", fontsize=10)

        axes[1, col].imshow(sp[1]); axes[1, col].axis("off")
        sp_err = revisit_data["sp_errors"][col]
        axes[1, col].set_title(f"err={sp_err:.5f}", color="#88ccff", fontsize=9)

        axes[2, col].imshow(pc[1]); axes[2, col].axis("off")
        pc_err = revisit_data["pc_errors"][col]
        axes[2, col].set_title(f"err={pc_err:.5f}", color="#ff9999", fontsize=9)

    if len(revisit_data["revisit_azs"]) > 0:
        axes[0, 0].set_ylabel("Ground\nTruth", color="white", fontsize=11,
                              fontweight="bold", rotation=0, labelpad=40)
        axes[1, 0].set_ylabel("Spline\nMemory", color="#88ccff", fontsize=11,
                              fontweight="bold", rotation=0, labelpad=40)
        axes[2, 0].set_ylabel("Point\nCloud", color="#ff9999", fontsize=11,
                              fontweight="bold", rotation=0, labelpad=40)

    avg_sp = np.mean(revisit_data["sp_errors"])
    avg_pc = np.mean(revisit_data["pc_errors"])
    fig.suptitle(
        f"Phase 3: Revisitation Consistency\n"
        f"Spline avg error: {avg_sp:.5f} | Point cloud avg error: {avg_pc:.5f}",
        color="white", fontsize=13, fontweight="bold", y=0.95)

    p = os.path.join(args.output_dir, "revisitation_consistency.png")
    plt.savefig(p, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    log(f"  {p}")


def make_comparison_video(gen_data, args):
    """Side-by-side video: GT | Spline Memory | Point Cloud Memory."""
    log(f"\n  Generating comparison video ({len(gen_data['sp_frames'])} frames) ...")
    fd = os.path.join(args.output_dir, "video_frames")
    os.makedirs(fd, exist_ok=True)

    boundary = gen_data["explore_range"]
    t0 = time.time()

    for i, (az, sp, pc, gt) in enumerate(zip(
        gen_data["azimuths"], gen_data["sp_frames"],
        gen_data["pc_frames"], gen_data["gt_frames"]
    )):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), facecolor=BG)
        fig.subplots_adjust(wspace=0.03, left=0.01, right=0.99, top=0.82, bottom=0.06)

        axes[0].imshow(gt); axes[0].axis("off")
        axes[0].set_title("Ground Truth", color="#66ff66", fontsize=14, fontweight="bold")

        axes[1].imshow(sp); axes[1].axis("off")
        axes[1].set_title("Spline Memory\n(generated from frozen memory)",
                          color="#88ccff", fontsize=13, fontweight="bold")

        axes[2].imshow(pc); axes[2].axis("off")
        axes[2].set_title("Point Cloud Memory\n(generated from frozen memory)",
                          color="#ff9999", fontsize=13, fontweight="bold")

        region = "OBSERVED" if az <= boundary else "UNOBSERVED"
        region_color = "#66ff66" if az <= boundary else "#ffcc00"
        fig.suptitle(
            f"Video Generation from Spatial Memory | az={az:.0f}° [{region}]\n"
            f"Memory trained on 0°-{boundary:.0f}° only",
            color="white", fontsize=13, fontweight="bold", y=0.95)
        fig.text(0.5, 0.02,
                 f"Region: {region} | Spline MSE: {gen_data['sp_mse'][i]:.5f} | "
                 f"Point MSE: {gen_data['pc_mse'][i]:.5f}",
                 color=region_color, fontsize=10, ha="center", fontfamily="monospace")

        plt.savefig(os.path.join(fd, f"f_{i:04d}.png"), dpi=100,
                    facecolor=BG, bbox_inches="tight", pad_inches=0.05)
        plt.close()

        if i % max(1, len(gen_data["sp_frames"]) // 6) == 0:
            log(f"    Frame {i}/{len(gen_data['sp_frames'])} | {time.time()-t0:.0f}s")

    vp = os.path.join(args.output_dir, "world_model_video.mp4")
    for codec in ["libx264", "libopenh264"]:
        cmd = (f"ffmpeg -y -framerate {args.fps} -i {fd}/f_%04d.png "
               f"-c:v {codec} -b:v 6M -pix_fmt yuv420p -crf 18 "
               f"-vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' {vp} 2>/dev/null")
        if os.system(cmd) == 0:
            log(f"  Video: {vp}")
            if not args.keep_frames:
                import shutil; shutil.rmtree(fd)
            return
    log(f"  ffmpeg failed — frames in {fd}/")


def plot_summary(sp_hist, pc_hist, gen_data, revisit_data, args):
    """Single summary figure for the presentation."""
    fig = plt.figure(figsize=(20, 12), facecolor=BG)

    boundary = gen_data["explore_range"]

    # Top left: prediction quality (the correct world model metric)
    ax1 = fig.add_axes([0.05, 0.55, 0.4, 0.35])
    azs_exp = np.linspace(0, boundary, len(sp_hist["prediction_errors"]))
    ax1.plot(azs_exp, sp_hist["prediction_errors"], "b-", lw=2, label="Spline")
    ax1.plot(azs_exp, pc_hist["prediction_errors"], "r-", lw=2, label="Points")
    ax1.set_title("Phase 1: Prediction Quality (lower = better)", color="white",
                  fontsize=11, fontweight="bold")
    ax1.legend(facecolor="#222", edgecolor="white", labelcolor="white")
    ax1.tick_params(colors="white"); ax1.set_facecolor("#1a1a1a")
    ax1.grid(alpha=0.2, color="white")
    ax1.set_xlabel("Azimuth (°)", color="white")
    ax1.set_ylabel("Prediction MSE", color="white")

    # Top right: generation quality
    ax2 = fig.add_axes([0.55, 0.55, 0.4, 0.35])
    ax2.plot(gen_data["azimuths"], gen_data["sp_mse"], "b-", lw=2, label="Spline")
    ax2.plot(gen_data["azimuths"], gen_data["pc_mse"], "r-", lw=2, label="Points")
    ax2.axvspan(boundary, 360, alpha=0.15, color="yellow", label="Unobserved")
    ax2.set_title("Phase 2: Video Generation Quality", color="white",
                  fontsize=11, fontweight="bold")
    ax2.legend(facecolor="#222", edgecolor="white", labelcolor="white", fontsize=8)
    ax2.tick_params(colors="white"); ax2.set_facecolor("#1a1a1a")
    ax2.grid(alpha=0.2, color="white")
    ax2.set_xlabel("Generation azimuth (°)", color="white")

    # Bottom: sample frames from generated video
    n_frames = min(6, len(gen_data["sp_frames"]))
    frame_indices = np.linspace(0, len(gen_data["sp_frames"]) - 1, n_frames, dtype=int)

    for col, fi in enumerate(frame_indices):
        ax_gt = fig.add_axes([0.02 + col * 0.16, 0.27, 0.15, 0.2])
        ax_gt.imshow(gen_data["gt_frames"][fi]); ax_gt.axis("off")
        az = gen_data["azimuths"][fi]
        region = "obs" if az <= boundary else "unobs"
        ax_gt.set_title(f"{az:.0f}° [{region}]", color="white", fontsize=8)

        ax_sp = fig.add_axes([0.02 + col * 0.16, 0.05, 0.15, 0.2])
        ax_sp.imshow(gen_data["sp_frames"][fi]); ax_sp.axis("off")

    fig.text(0.01, 0.37, "GT", color="#66ff66", fontsize=10, fontweight="bold", rotation=90, va="center")
    fig.text(0.01, 0.15, "Spline\nMemory", color="#88ccff", fontsize=9, fontweight="bold", rotation=90, va="center")

    fig.suptitle(
        "Mini World Model: Persistent Curve Memory for Consistent Video Generation\n"
        f"Explored 0°-{boundary:.0f}° → Generated full 360° from frozen memory",
        color="white", fontsize=14, fontweight="bold", y=0.98)

    p = os.path.join(args.output_dir, "world_model_summary.png")
    plt.savefig(p, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    log(f"  {p}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="wStraight",
                   help="Cem Yuksel hair model name (e.g. wStraight, wWavy, wCurly)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--num-curves", type=int, default=500)
    p.add_argument("--K", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)

    # Phase 1: Exploration
    p.add_argument("--explore-views", type=int, default=36,
                   help="Number of views during exploration")
    p.add_argument("--explore-range", type=float, default=270.0,
                   help="Angular range of exploration (degrees). <360 means partial observability")
    p.add_argument("--steps-per-view", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--init-noise", type=float, default=0.08,
                   help="Init noise (0.08 for clean visuals, 0.15 for robustness test)")
    p.add_argument("--render-weight", type=float, default=0.5)
    p.add_argument("--reproj-weight", type=float, default=1.5)
    p.add_argument("--anchor-weight", type=float, default=0.05,
                   help="Anchor pull strength (0.05-0.1 for real hair)")
    p.add_argument("--tangent-weight", type=float, default=0.05)
    p.add_argument("--curv-weight", type=float, default=0.03)
    p.add_argument("--huber-delta", type=float, default=0.05,
                   help="Huber loss delta for reprojection (linear penalty beyond this)")
    p.add_argument("--grad-clip-factor", type=float, default=5.0,
                   help="Per-curve gradient clip at median * this factor")
    p.add_argument("--max-drift", type=float, default=0.3,
                   help="Hard ceiling on how far any curve can drift from anchor")
    p.add_argument("--ema-decay", type=float, default=0.8)
    p.add_argument("--view-buffer", type=int, default=5)

    # Phase 2: Generation
    p.add_argument("--gen-views", type=int, default=72,
                   help="Number of views for video generation (full 360)")

    # Rendering — SEPARATE settings for optimization vs visualization
    p.add_argument("--opt-size", type=int, default=128,
                   help="Image size for optimization loss (smaller = faster, 128 is fine)")
    p.add_argument("--opt-radius", type=float, default=0.02)
    p.add_argument("--opt-samples", type=int, default=16,
                   help="Spline samples per curve during optimization (fewer = faster)")
    p.add_argument("--opt-ppp", type=int, default=4,
                   help="Points per pixel during optimization (fewer = faster)")
    p.add_argument("--vis-size", type=int, default=512,
                   help="Image size for visualization renders")
    p.add_argument("--vis-radius", type=float, default=0.006,
                   help="Point cloud dot radius for visualization")
    p.add_argument("--vis-samples", type=int, default=128,
                   help="Spline samples per curve for visualization (more = smoother)")
    p.add_argument("--tube-radius", type=float, default=0.0015,
                   help="Tube radius for spline rendering (thinner = more natural)")
    p.add_argument("--tube-sides", type=int, default=4)
    p.add_argument("--densify-factor", type=int, default=2,
                   help="Multiply curve count for visualization (interpolates between curves)")
    p.add_argument("--vis-elev", type=float, default=25.0)
    p.add_argument("--vis-dist", type=float, default=3.0)

    p.add_argument("--device", default="cuda")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--keep-frames", action="store_true")
    p.add_argument("--skip-video", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--output-dir", default="outputs/world_model")
    args = p.parse_args()

    if args.quick:
        args.explore_views = 18
        args.gen_views = 36
        args.steps_per_view = 30
        args.vis_size = 384
        args.opt_samples = 12
        args.vis_samples = 64
        args.densify_factor = 1  # skip densification in quick mode

    os.makedirs(args.output_dir, exist_ok=True)

    log(f"\n{'='*60}")
    log(f"  MINI WORLD MODEL — HAIR SCENE")
    log(f"  Phase 1: Explore 0°-{args.explore_range}° ({args.explore_views} views)")
    log(f"  Phase 2: Generate full 360° ({args.gen_views} views)")
    log(f"  Phase 3: Revisit explored viewpoints")
    log(f"  {args.num_curves} curves | K={args.K}")
    log(f"{'='*60}")

    from gensplines.spline import evaluate_bspline
    from gensplines.hair_loader import download_yuksel_hair, load_hair_file, hair_to_spline_field

    log(f"\n  Loading hair data: {args.model_name} ...")
    hp = download_yuksel_hair(args.model_name, save_dir=args.data_dir)
    strands = load_hair_file(hp)
    gt_cp = hair_to_spline_field(strands, num_curves=args.num_curves, K=args.K,
                                 seed=args.seed, strategy="diverse")
    gt_cp = orient(gt_cp).to(args.device)

    with torch.no_grad():
        gt_points = evaluate_bspline(gt_cp, args.opt_samples).reshape(-1, 3)

    t_total = time.time()

    # Phase 1
    log("\n" + "="*60)
    log("  PHASE 1: EXPLORATION")
    log("="*60)
    (sp_cp, sp_hist, pc_pts, pc_hist,
     explore_azs, gt_imgs, gt_cams, gt_projs) = explore_phase(gt_cp, gt_points, args)
    plot_exploration(sp_hist, pc_hist, explore_azs, args)

    # Phase 2
    log("\n" + "="*60)
    log("  PHASE 2: VIDEO GENERATION")
    log("="*60)
    gen_data = generate_video_phase(sp_cp, pc_pts, gt_cp, gt_points, args)
    plot_generation(gen_data, args)

    # Phase 3
    log("\n" + "="*60)
    log("  PHASE 3: REVISITATION")
    log("="*60)
    revisit_data = revisitation_phase(sp_cp, pc_pts, gt_cp, gt_points,
                                      explore_azs, args)
    plot_revisitation(revisit_data, args)

    # Summary
    plot_summary(sp_hist, pc_hist, gen_data, revisit_data, args)

    # Video
    if not args.skip_video:
        make_comparison_video(gen_data, args)

    # Save raw data for re-rendering and viewer export
    torch.save(sp_cp.cpu(), os.path.join(args.output_dir, "spline_final_cp.pt"))
    torch.save(pc_pts.cpu(), os.path.join(args.output_dir, "pointcloud_final_pts.pt"))
    torch.save(gt_cp.cpu(), os.path.join(args.output_dir, "gt_cp.pt"))
    log(f"  Saved CPs and points for re-rendering/export")

    # Export for live viewer and archival
    try:
        from gensplines.render_utils import (export_tubes_obj, export_points_ply,
                                  export_control_points_json, export_control_points_bin)

        # Compact OBJ for quick loading (32 samples, 3 sides → ~15MB)
        export_tubes_obj(sp_cp, args.vis_samples,
                         os.path.join(args.output_dir, "spline_tubes_compact.obj"),
                         n_sides=4, tube_radius=args.tube_radius, compact=True)

        # Demo-quality OBJ for live zooming (64 samples, 4 sides → ~30MB)
        export_tubes_obj(sp_cp, 64,
                         os.path.join(args.output_dir, "spline_tubes_demo.obj"),
                         n_sides=4, tube_radius=args.tube_radius, compact=False)

        # Full-quality OBJ (128 samples, 4 sides → ~95MB, for extreme close-ups)
        export_tubes_obj(sp_cp, 128,
                         os.path.join(args.output_dir, "spline_tubes_full.obj"),
                         n_sides=4, tube_radius=args.tube_radius, compact=False)

        # Point cloud
        export_points_ply(pc_pts,
                          os.path.join(args.output_dir, "pointcloud.ply"))

        # Control points — the REAL storage format (288KB for 2K curves)
        export_control_points_json(sp_cp,
                                   os.path.join(args.output_dir, "control_points.json"))
        export_control_points_bin(sp_cp,
                                  os.path.join(args.output_dir, "control_points.npz"))

        log(f"\n  Export summary:")
        log(f"    control_points.npz     — what a real system stores")
        log(f"    spline_tubes_compact.obj — quick viewer loading")
        log(f"    spline_tubes_demo.obj  — smooth at close zoom")
        log(f"    spline_tubes_full.obj  — maximum quality")
        log(f"    pointcloud.ply         — point cloud baseline")
    except ImportError:
        log("  render_utils.py not found — skipping export")

    # Save metrics
    metrics = {
        "model": args.model_name,
        "num_curves": args.num_curves,
        "explore_range": args.explore_range,
        "explore_views": args.explore_views,
        "gen_views": args.gen_views,
        "sp_final_drift": sp_hist["drifts"][-1] if sp_hist["drifts"] else 0,
        "pc_final_drift": pc_hist["drifts"][-1] if pc_hist["drifts"] else 0,
        "sp_final_pred_error": sp_hist["prediction_errors"][-1] if sp_hist["prediction_errors"] else 0,
        "pc_final_pred_error": pc_hist["prediction_errors"][-1] if pc_hist["prediction_errors"] else 0,
        "sp_avg_pred_error": float(np.mean(sp_hist["prediction_errors"])) if sp_hist["prediction_errors"] else 0,
        "pc_avg_pred_error": float(np.mean(pc_hist["prediction_errors"])) if pc_hist["prediction_errors"] else 0,
        "sp_avg_gen_mse_observed": float(np.mean([m for a, m in zip(gen_data["azimuths"], gen_data["sp_mse"]) if a <= args.explore_range])),
        "pc_avg_gen_mse_observed": float(np.mean([m for a, m in zip(gen_data["azimuths"], gen_data["pc_mse"]) if a <= args.explore_range])),
        "sp_avg_gen_mse_unobserved": float(np.mean([m for a, m in zip(gen_data["azimuths"], gen_data["sp_mse"]) if a > args.explore_range])),
        "pc_avg_gen_mse_unobserved": float(np.mean([m for a, m in zip(gen_data["azimuths"], gen_data["pc_mse"]) if a > args.explore_range])),
        "sp_avg_revisit_error": float(np.mean(revisit_data["sp_errors"])),
        "pc_avg_revisit_error": float(np.mean(revisit_data["pc_errors"])),
        "total_time_seconds": time.time() - t_total,
    }

    mp = os.path.join(args.output_dir, "world_model_metrics.json")
    with open(mp, "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"\n  Metrics: {mp}")

    log(f"\n{'='*60}")
    log(f"  COMPLETE — {time.time()-t_total:.0f}s total")
    log(f"  exploration_timeline.png")
    log(f"  generation_quality.png")
    log(f"  revisitation_consistency.png")
    log(f"  world_model_summary.png")
    if not args.skip_video:
        log(f"  world_model_video.mp4")
    log(f"  world_model_metrics.json")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
