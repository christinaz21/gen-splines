"""
run_dense.py — Dense pipeline: fit, optimize, render presentation video.
All panels render as blonde hair for visual clarity.
"""

import os, sys, argparse, time, torch, numpy as np
import torch.nn.functional as F
from spline import SplineField, evaluate_bspline
from hair_loader import download_yuksel_hair, load_hair_file, hair_to_spline_field
from metrics import control_point_drift
from optimize_v2 import (multi_view_reprojection_loss, tangent_consistency_loss,
                          anchor_proximity_loss, PersistentCurveMemory)

def log(msg): print(msg, flush=True)

BLONDE = (0.82, 0.72, 0.42)

# === Orientation: Yuksel Y-up -> PyTorch3D (x, z, -y) ===
def orient_cp(cp):
    f=cp.clone(); ny=f[...,2].clone(); nz=-f[...,1].clone(); f[...,1]=ny; f[...,2]=nz; return f
def orient_pts(p):
    f=p.clone(); ny=f[:,2].clone(); nz=-f[:,1].clone(); f[:,1]=ny; f[:,2]=nz; return f

# === Coloring (blonde with per-strand variation + root-to-tip gradient) ===
def blonde_colors(num_points_per_curve_list, base=BLONDE):
    """Generate blonde colors. Accepts list of point counts or (N,M,3) tensor."""
    if isinstance(num_points_per_curve_list, torch.Tensor):
        N,M,_ = num_points_per_curve_list.shape
        counts = [M]*N
    else:
        counts = num_points_per_curve_list
    rng=np.random.RandomState(42); cs=[]
    for M in counts:
        v=rng.uniform(-0.06,0.06,size=3)
        sc=np.clip(np.array(base)+v, 0, 1)
        t=np.linspace(0,1,M); darken=1.0-0.15*t
        cs.append(np.outer(darken, sc))
    return torch.tensor(np.concatenate(cs), dtype=torch.float32)

# === Rendering (all bin_size=0) ===
def render_pts(points, colors, az, image_size, radius, device, elev=25.0, dist=3.5):
    from pytorch3d.renderer import (look_at_view_transform, FoVPerspectiveCameras,
        PointsRasterizationSettings, PointsRenderer, PointsRasterizer, AlphaCompositor)
    from pytorch3d.structures import Pointclouds
    R,T = look_at_view_transform(dist=dist, elev=elev, azim=az)
    cam = FoVPerspectiveCameras(device=device, R=R, T=T)
    rs = PointsRasterizationSettings(image_size=image_size, radius=radius,
                                      points_per_pixel=10, bin_size=0)
    rend = PointsRenderer(rasterizer=PointsRasterizer(cameras=cam, raster_settings=rs),
                           compositor=AlphaCompositor(background_color=(0.03,0.03,0.05)))
    pc = Pointclouds(points=[points.to(device)], features=[colors.to(device)])
    return rend(pc)[0,...,:3].cpu().numpy()

def render_cp_blonde(cp, az, image_size, radius, device, num_samples=96,
                      elev=25.0, dist=3.5):
    """Render already-oriented control points as blonde hair. NO orientation applied."""
    N,K,_ = cp.shape
    field = SplineField(N,K); field.control_points.data = cp  # already oriented
    with torch.no_grad():
        ppc = field.forward_per_curve(num_samples)  # (N, M, 3)
    colors = blonde_colors(ppc)
    flat = ppc.reshape(-1, 3)
    return render_pts(flat, colors, az, image_size, radius, device, elev, dist)

def render_dense_gt_blonde(strands, az, image_size, radius, device, num_strands=3000,
                            num_pts=48, elev=25.0, dist=3.5):
    """Render raw strands as dense blonde hair with orientation fix."""
    from hair_loader import subsample_strands
    sel = subsample_strands(strands, num_strands, strategy="random", seed=99, min_length=5)
    pts_list, pt_counts = [], []
    for s in sel:
        M = len(s)
        p = s[np.linspace(0, M-1, min(num_pts, M), dtype=int)] if M > num_pts else s
        pts_list.append(p)
        pt_counts.append(len(p))

    points = torch.tensor(np.concatenate(pts_list), dtype=torch.float32)
    colors = blonde_colors(pt_counts)

    # Normalize
    c = points.mean(0); points -= c
    md = points.norm(dim=-1).max()
    if md > 1e-6: points /= md

    # Apply orientation (raw strands need it)
    points = orient_pts(points)

    return render_pts(points, colors, az, image_size, radius*0.5, device, elev, dist)

# === Optimization ===
def run_opt(gt_cp, args, device="cuda"):
    from pytorch3d.renderer import (look_at_view_transform, FoVPerspectiveCameras,
        PointsRasterizationSettings, PointsRenderer, PointsRasterizer, AlphaCompositor)
    from pytorch3d.structures import Pointclouds

    N,K,_ = gt_cp.shape
    # Orient once — all subsequent data is in this coordinate system
    gt_o = orient_cp(gt_cp).to(device)
    gt_field = SplineField(N,K).to(device); gt_field.control_points.data = gt_o.clone()
    ns = args.samples_per_curve
    rs = PointsRasterizationSettings(image_size=args.opt_image_size, radius=args.opt_radius,
                                      points_per_pixel=8, bin_size=0)
    azs = torch.linspace(0, 360-360/args.num_views, args.num_views)

    log(f"  Rendering {args.num_views} GT views...")
    gt_imgs, gt_cams, gt_projs = [], [], []
    gt_flat = gt_field.forward_per_curve(ns).reshape(-1,3)

    for az in azs:
        R,T = look_at_view_transform(dist=4.0, elev=30.0, azim=az.item())
        cam = FoVPerspectiveCameras(device=device, R=R, T=T, fov=60.0, aspect_ratio=1.0,
                                     znear=0.1, zfar=100.0)
        gt_cams.append(cam)
        with torch.no_grad():
            pts=gt_field(ns); rgb=torch.ones_like(pts)
            rend = PointsRenderer(rasterizer=PointsRasterizer(cameras=cam, raster_settings=rs),
                                   compositor=AlphaCompositor())
            img = rend(Pointclouds(points=[pts], features=[rgb]))[0,...,:3]
            gt_imgs.append(img.detach())
            proj = cam.transform_points_screen(gt_flat.unsqueeze(0),
                image_size=((args.opt_image_size, args.opt_image_size),))[0,:,:2]
            gt_projs.append(proj)

    # Noisy init (already in oriented space)
    pred = SplineField(N,K).to(device)
    pred.control_points.data = gt_o.clone() + args.init_noise*torch.randn_like(gt_o)
    init_d = control_point_drift(gt_o, pred.control_points.data).item()
    noisy_cp = pred.control_points.data.clone().cpu()  # already oriented
    log(f"  Initial drift: {init_d:.4f} | Points/frame: {N*ns}")

    mem = PersistentCurveMemory(pred.control_points.data, ema_decay=args.ema_decay)
    drifts = []; t0 = time.time()

    log(f"\n  {'View':>5} {'Az':>7} {'Drift':>8} {'Red%':>7} {'Loss':>12} {'Time':>7}")
    log(f"  {'─'*5} {'─'*7} {'─'*8} {'─'*7} {'─'*12} {'─'*7}")

    for vi in range(args.num_views):
        az = azs[vi].item()
        bi = list(range(max(0, vi-args.view_buffer+1), vi+1))
        opt = torch.optim.Adam([pred.control_points], lr=args.lr)

        for _ in range(args.steps_per_view):
            opt.zero_grad()
            pts=pred(ns); rgb=torch.ones_like(pts)
            R,T = look_at_view_transform(dist=4.0, elev=30.0, azim=az)
            cam = FoVPerspectiveCameras(device=device, R=R, T=T, fov=60.0,
                                         aspect_ratio=1.0, znear=0.1, zfar=100.0)
            rend = PointsRenderer(rasterizer=PointsRasterizer(cameras=cam, raster_settings=rs),
                                   compositor=AlphaCompositor())
            pimg = rend(Pointclouds(points=[pts], features=[rgb]))[0,...,:3]
            lr_ = F.mse_loss(pimg, gt_imgs[vi])
            lp = multi_view_reprojection_loss(pred.control_points,
                [gt_projs[i] for i in bi], [gt_cams[i] for i in bi],
                num_samples=ns, image_size=args.opt_image_size)
            lt = tangent_consistency_loss(pred.control_points, num_samples=ns,
                                          weight=args.tangent_weight)
            la = anchor_proximity_loss(pred.control_points, mem.get_anchor(),
                                        weight=args.anchor_weight)
            loss = args.render_weight*lr_ + args.reproj_weight*lp + lt + la
            loss.backward(); opt.step()

        mem.update(pred.control_points.data)
        with torch.no_grad():
            d = control_point_drift(gt_o, pred.control_points.data).item()
            drifts.append(d)
        red = (1-d/init_d)*100; el = time.time()-t0
        if vi % max(1, args.num_views//12)==0 or vi==args.num_views-1:
            log(f"  {vi:5d} {az:6.1f}° {d:8.4f} {red:6.1f}% {loss.item():12.4f} {el:6.0f}s")

    fd=drifts[-1]; fr=(1-fd/init_d)*100; tt=time.time()-t0
    log(f"\n  {'═'*50}")
    log(f"  DONE: {N} curves | {fr:.1f}% reduction | {tt:.0f}s")
    log(f"  {'═'*50}")

    # All outputs are ALREADY in oriented coordinate system
    return {"gt_cp":gt_o.cpu(), "noisy_cp":noisy_cp, "final_cp":pred.control_points.data.cpu(),
            "initial_drift":init_d, "final_drift":fd, "drift_reduction":fr,
            "view_drifts":drifts, "azimuths":azs.tolist(), "time_seconds":tt}

# === Video ===
def make_video(res, strands, args, device="cuda"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    # These are ALREADY oriented from run_opt
    gt_cp = res["gt_cp"]
    noisy_cp = res["noisy_cp"]
    final_cp = res["final_cp"]
    red = res["drift_reduction"]
    N = gt_cp.shape[0]
    dn = min(args.dense_gt_strands, len(strands))

    fd_dir = os.path.join(args.output_dir, "video_frames")
    os.makedirs(fd_dir, exist_ok=True)
    azs = np.linspace(0, 360, args.num_video_frames, endpoint=False)

    log(f"  Rendering {args.num_video_frames} frames (GT:{dn} strands, model:{N} curves)...")
    t0 = time.time()

    for i, az in enumerate(azs):
        # GT: dense raw strands (orient_pts applied inside render_dense_gt_blonde)
        g = render_dense_gt_blonde(strands, float(az), args.vis_image_size, args.vis_radius,
                                    device, dn, elev=args.elevation, dist=args.dist)

        # Noisy: already oriented, render directly as blonde
        n = render_cp_blonde(noisy_cp, float(az), args.vis_image_size, args.vis_radius,
                              device, args.samples_per_curve, args.elevation, args.dist)

        # Final: already oriented, render directly as blonde
        f = render_cp_blonde(final_cp, float(az), args.vis_image_size, args.vis_radius,
                              device, args.samples_per_curve, args.elevation, args.dist)

        fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor="#080810")
        fig.subplots_adjust(wspace=0.02, left=0.01, right=0.99, top=0.82, bottom=0.08)

        for ax, (im, title) in zip(axes, [
            (g, f"Ground Truth ({dn} strands)"),
            (n, f"Initial — Noisy ({N} curves, σ={args.init_noise})"),
            (f, f"After Persistent Curve Memory ({N} curves)"),
        ]):
            ax.imshow(np.clip(im, 0, 1))
            ax.axis("off")
            ax.set_title(title, color="#e0d0a0", fontsize=12, fontweight="bold", pad=10)

        fig.suptitle("Generative Spline Fields with Persistent Curve Memory",
                     color="white", fontsize=15, fontweight="bold", y=0.94)
        fig.text(0.5, 0.87,
                 f"{args.model_name} | Azimuth: {az:.0f}° | "
                 f"Drift Reduction: {red:.1f}%",
                 color="#aaa", fontsize=11, ha="center")
        fig.text(0.5, 0.03,
                 f"Drift: {res['initial_drift']:.4f} → {res['final_drift']:.4f} | "
                 f"{N} curves × {gt_cp.shape[1]} CP | {args.num_views} views",
                 color="#666", fontsize=9, ha="center")

        plt.savefig(os.path.join(fd_dir, f"frame_{i:04d}.png"),
                    dpi=args.dpi, facecolor="#080810", edgecolor="none")
        plt.close()

        if i % max(1, args.num_video_frames//8) == 0:
            log(f"    Frame {i:4d}/{args.num_video_frames} | {time.time()-t0:.0f}s")

    # Stitch
    vp = os.path.join(args.output_dir, "comparison_video.mp4")
    cmd = (f"ffmpeg -y -framerate {args.fps} -i {fd_dir}/frame_%04d.png "
           f"-c:v libx264 -pix_fmt yuv420p -crf 18 "
           f"-vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' {vp}")
    log("  Stitching video...")
    ret = os.system(cmd)
    if ret == 0:
        log(f"  Video: {vp} ({os.path.getsize(vp)/1024/1024:.1f} MB)")
        if not args.keep_frames:
            import shutil; shutil.rmtree(fd_dir)
    else:
        log(f"  ffmpeg failed. Frames in {fd_dir}/")

    # Still at az=30
    g = render_dense_gt_blonde(strands, 30.0, args.vis_image_size, args.vis_radius,
                                device, dn, elev=args.elevation, dist=args.dist)
    n = render_cp_blonde(noisy_cp, 30.0, args.vis_image_size, args.vis_radius,
                          device, args.samples_per_curve, args.elevation, args.dist)
    f2 = render_cp_blonde(final_cp, 30.0, args.vis_image_size, args.vis_radius,
                           device, args.samples_per_curve, args.elevation, args.dist)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor="#080810")
    fig.subplots_adjust(wspace=0.02, left=0.01, right=0.99, top=0.82, bottom=0.08)
    for ax, (im, title) in zip(axes, [
        (g, f"Ground Truth ({dn} strands)"),
        (n, f"Initial — Noisy ({N} curves)"),
        (f2, f"After Persistent Memory ({N} curves)"),
    ]):
        ax.imshow(np.clip(im, 0, 1))
        ax.axis("off")
        ax.set_title(title, color="#e0d0a0", fontsize=12, fontweight="bold", pad=10)

    fig.suptitle(f"Generative Spline Fields — {args.model_name} | {red:.1f}% Drift Reduction",
                 color="white", fontsize=14, fontweight="bold", y=0.94)
    sp = os.path.join(args.output_dir, "comparison_still.png")
    plt.savefig(sp, dpi=150, facecolor="#080810", edgecolor="none")
    plt.close()
    log(f"  Still: {sp}")

# === Main ===
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="wCurly")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--num-curves", type=int, default=500)
    p.add_argument("--K", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-views", type=int, default=72)
    p.add_argument("--steps-per-view", type=int, default=80)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--init-noise", type=float, default=0.35)
    p.add_argument("--opt-radius", type=float, default=0.02)
    p.add_argument("--opt-image-size", type=int, default=256)
    p.add_argument("--samples-per-curve", type=int, default=96)
    p.add_argument("--render-weight", type=float, default=0.5)
    p.add_argument("--reproj-weight", type=float, default=1.5)
    p.add_argument("--tangent-weight", type=float, default=0.1)
    p.add_argument("--anchor-weight", type=float, default=0.02)
    p.add_argument("--view-buffer", type=int, default=5)
    p.add_argument("--ema-decay", type=float, default=0.8)
    p.add_argument("--vis-image-size", type=int, default=512)
    p.add_argument("--vis-radius", type=float, default=0.006)
    p.add_argument("--elevation", type=float, default=25.0)
    p.add_argument("--dist", type=float, default=3.5)
    p.add_argument("--dense-gt-strands", type=int, default=3000)
    p.add_argument("--num-video-frames", type=int, default=72)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--keep-frames", action="store_true")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    if a.output_dir is None: a.output_dir = f"outputs/dense_{a.model_name}_{a.num_curves}"
    if a.quick: a.steps_per_view=40; a.num_views=36; a.num_video_frames=36
    os.makedirs(a.output_dir, exist_ok=True)

    log(f"\n{'═'*65}")
    log(f"  DENSE HAIR PIPELINE")
    log(f"  Model: {a.model_name} | Curves: {a.num_curves} | K: {a.K}")
    log(f"  Noise: σ={a.init_noise} | Views: {a.num_views} | Steps: {a.steps_per_view}")
    log(f"{'═'*65}")

    hair_path = download_yuksel_hair(a.model_name, save_dir=a.data_dir)
    strands = load_hair_file(hair_path)

    log(f"\nFitting {a.num_curves} B-splines...")
    t0 = time.time()
    gt_cp = hair_to_spline_field(strands, num_curves=a.num_curves, K=a.K,
                                  seed=a.seed, strategy="diverse")
    log(f"  Done in {time.time()-t0:.1f}s")

    log(f"\nOptimizing...")
    res = run_opt(gt_cp, a, device=a.device)
    torch.save(res, os.path.join(a.output_dir, "opt_results.pt"))

    log(f"\nRendering video...")
    make_video(res, strands, a, device=a.device)

    log(f"\n{'═'*65}")
    log(f"  ALL DONE | {a.model_name} | {a.num_curves} curves | {res['drift_reduction']:.1f}%")
    log(f"  Output: {a.output_dir}/")
    log(f"{'═'*65}")

if __name__ == "__main__": main()
