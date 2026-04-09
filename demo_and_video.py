"""
demo_and_video.py — Full Pipeline Demo + 360° Triptych Video

All-in-one: runs the end-to-end pipeline, saves all results,
renders a 360° rotating triptych video with colored curves.

Usage:
    python demo_and_video.py --generator-path outputs/gen_percurve/best_generator.pt

    # Skip video if you just want the numbers:
    python demo_and_video.py --generator-path outputs/gen_percurve/best_generator.pt --skip-video
"""

import os
import time
import argparse
import colorsys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from generator import load_generator
from spline import SplineField, evaluate_bspline
from dataset import create_combined_scene
from metrics import control_point_drift, curvature_deviation, compute_all_metrics
from losses import reprojection_loss, smoothness_loss, anchor_loss


# =========================================================================
# Rendering helpers
# =========================================================================

def hsv_to_rgb_tuple(h, s, v):
    return colorsys.hsv_to_rgb(h, s, v)


def get_curve_colors(N, device="cuda"):
    """Generate N distinct colors using golden-angle hue spacing."""
    colors = []
    for i in range(N):
        hue = (i * 137.508) % 360
        sat = 0.7 + 0.3 * ((i * 7) % 10) / 10
        val = 0.75 + 0.25 * ((i * 3) % 10) / 10
        r, g, b = hsv_to_rgb_tuple(hue / 360, sat, val)
        colors.append([r, g, b])
    return torch.tensor(colors, device=device)


def render_colored_tubes(control_points, azimuth=0.0, elevation=30.0,
                          image_size=512, tube_radius=0.012, samples=128,
                          bg_color=(0.08, 0.08, 0.12), device="cuda"):
    """Render curves as colored tube meshes on dark background."""
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        look_at_view_transform, FoVPerspectiveCameras,
        RasterizationSettings, MeshRasterizer, MeshRenderer,
        SoftPhongShader, PointLights, TexturesVertex,
    )

    curve_points = evaluate_bspline(control_points, samples)
    N, M, _ = curve_points.shape
    curve_colors = get_curve_colors(N, device)
    n_sides = 6

    all_verts = []
    all_faces = []
    all_colors = []
    vert_offset = 0

    for i in range(N):
        pts = curve_points[i]
        r, g, b = curve_colors[i].tolist()

        for j in range(M - 1):
            d = pts[j + 1] - pts[j]
            d_len = d.norm() + 1e-8
            d_norm = d / d_len

            up = torch.tensor([0.0, 1.0, 0.0], device=device)
            if abs(torch.dot(d_norm, up)) > 0.99:
                up = torch.tensor([1.0, 0.0, 0.0], device=device)

            right = torch.cross(d_norm, up)
            right = right / (right.norm() + 1e-8)
            up2 = torch.cross(right, d_norm)
            up2 = up2 / (up2.norm() + 1e-8)

            for center in [pts[j], pts[j + 1]]:
                for s in range(n_sides):
                    angle = 2 * np.pi * s / n_sides
                    offset = tube_radius * (np.cos(angle) * right + np.sin(angle) * up2)
                    all_verts.append(center + offset)
                    all_colors.append([r, g, b])

            base = vert_offset
            for s in range(n_sides):
                s_next = (s + 1) % n_sides
                all_faces.append([base + s, base + n_sides + s, base + n_sides + s_next])
                all_faces.append([base + s, base + n_sides + s_next, base + s_next])
            vert_offset += 2 * n_sides

    verts = torch.stack(all_verts).to(device)
    faces = torch.tensor(all_faces, dtype=torch.long, device=device)
    vert_colors = torch.tensor(all_colors, dtype=torch.float32, device=device).unsqueeze(0)

    R, T = look_at_view_transform(dist=4.0, elev=elevation, azim=azimuth)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1, bin_size=0,
    )
    lights = PointLights(
        device=device, location=[[3.0, 3.0, 3.0]],
        ambient_color=[[0.5, 0.5, 0.5]], diffuse_color=[[0.5, 0.5, 0.5]],
        specular_color=[[0.15, 0.15, 0.15]],
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )
    textures = TexturesVertex(verts_features=vert_colors)
    mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
    images = renderer(mesh)

    img = images[0, ..., :3].cpu().numpy()
    alpha = images[0, ..., 3:4].cpu().numpy()
    bg = np.array(bg_color).reshape(1, 1, 3)
    composited = img * alpha + bg * (1 - alpha)
    return (np.clip(composited, 0, 1) * 255).astype(np.uint8)


def render_colored_points(control_points, azimuth=0.0, elevation=30.0,
                            image_size=512, samples=128,
                            bg_color=(0.08, 0.08, 0.12), device="cuda"):
    """Fallback: colored points on dark background."""
    from renderer import render_point_cloud

    curve_points = evaluate_bspline(control_points, samples)
    N, M, _ = curve_points.shape
    curve_colors = get_curve_colors(N, device)
    point_colors = curve_colors.unsqueeze(1).expand(N, M, 3).reshape(-1, 3)
    pts = curve_points.reshape(-1, 3)

    config = {"radius": 0.008, "image_size": image_size}
    img = render_point_cloud(pts, features=point_colors, azimuth=azimuth,
                              elevation=elevation, config=config, device=device)
    img_np = img[..., :3].cpu().numpy()
    mask = (img_np.sum(axis=-1, keepdims=True) < 0.05).astype(np.float32)
    bg = np.array(bg_color).reshape(1, 1, 3)
    composited = img_np * (1 - mask) + bg * mask
    return (np.clip(composited, 0, 1) * 255).astype(np.uint8)


def make_render_fn(use_tubes, device, image_size, tube_radius, samples):
    """Create a render function with error fallback."""
    def render_fn(cp, az):
        if use_tubes:
            try:
                return render_colored_tubes(cp, azimuth=az, image_size=image_size,
                                             tube_radius=tube_radius, samples=samples,
                                             device=device)
            except Exception:
                pass
        return render_colored_points(cp, azimuth=az, image_size=image_size,
                                      samples=samples, device=device)
    return render_fn


def make_triptych_frame(gt_img, gen_img, final_img, azimuth,
                          drift_initial, drift_final,
                          frame_width=1536, frame_height=580):
    """Combine three renders into a labeled triptych frame."""
    panel_w = frame_width // 3
    panel_h = frame_height - 68

    gt_pil = Image.fromarray(gt_img).resize((panel_w, panel_h), Image.LANCZOS)
    gen_pil = Image.fromarray(gen_img).resize((panel_w, panel_h), Image.LANCZOS)
    final_pil = Image.fromarray(final_img).resize((panel_w, panel_h), Image.LANCZOS)

    frame = Image.new("RGB", (frame_width, frame_height), (20, 20, 30))
    frame.paste(gt_pil, (0, 0))
    frame.paste(gen_pil, (panel_w, 0))
    frame.paste(final_pil, (panel_w * 2, 0))

    draw = ImageDraw.Draw(frame)
    try:
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (OSError, IOError):
        font_label = ImageFont.load_default()
        font_small = font_label

    label_y = panel_h + 8
    meter_y = panel_h + 32

    draw.text((panel_w // 2 - 60, label_y), "Ground Truth",
              fill=(180, 220, 255), font=font_label)
    draw.text((panel_w + panel_w // 2 - 80, label_y), "Generated Gθ(z)",
              fill=(255, 200, 100), font=font_label)
    draw.text((panel_w * 2 + panel_w // 2 - 90, label_y), "After Memory",
              fill=(100, 255, 150), font=font_label)

    drift_pct = (1 - drift_final / drift_initial) * 100 if drift_initial > 0 else 0
    info = (f"Azimuth: {azimuth:.0f}\u00b0    |    "
            f"Drift: {drift_initial:.3f} \u2192 {drift_final:.3f}  ({drift_pct:+.1f}%)")
    draw.text((frame_width // 2 - 220, meter_y), info,
              fill=(200, 200, 200), font=font_small)

    for x in [panel_w, panel_w * 2]:
        draw.line([(x, 0), (x, panel_h)], fill=(60, 60, 80), width=2)

    return np.array(frame)


# =========================================================================
# Main: Demo + Video combined
# =========================================================================

def run(args):
    device = args.device
    print(f"\n{'='*65}")
    print(f"  END-TO-END DEMO + VIDEO")
    print(f"  Generative Spline Fields with Persistent Curve Memory")
    print(f"{'='*65}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ==================================================================
    # STAGE 1: z ~ N(0,I) → Gθ(z) → initial spline scene
    # ==================================================================
    print(f"\n  --- Stage 1: Generate Scene via Gθ(z) ---")

    generator = load_generator(
        args.generator_path, device=device,
        latent_dim=args.latent_dim, num_curves=args.num_curves,
        K=args.K, hidden_dim=args.hidden_dim
    )

    norm_path = os.path.join(os.path.dirname(args.generator_path), "normalization.pt")
    if os.path.exists(norm_path):
        norm = torch.load(norm_path, weights_only=True)
        data_mean = norm["mean"].to(device)
        data_std = norm["std"].to(device)
    else:
        data_mean = torch.zeros(3, device=device)
        data_std = torch.ones(3, device=device)

    torch.manual_seed(args.seed)
    z = torch.randn(1, args.latent_dim, device=device)
    with torch.no_grad():
        gen_cp_norm = generator(z)
        generated_cp = (gen_cp_norm * data_std + data_mean).squeeze(0)

    N, K, D = generated_cp.shape
    print(f"  Generated: {N} curves, {K} CPs each")

    # ==================================================================
    # STAGE 2: Ground truth
    # ==================================================================
    print(f"\n  --- Stage 2: Ground Truth ---")
    gt_cp = create_combined_scene(num_helix=20, num_wave=20, K=K, seed=args.gt_seed).to(device)
    gt_cp = gt_cp[:N]
    print(f"  GT: {gt_cp.shape}")

    # ==================================================================
    # STAGE 3: Initialize persistent curve memory
    # ==================================================================
    print(f"\n  --- Stage 3: Initialize Persistent Memory ---")
    memory_field = SplineField(N, K).to(device)
    memory_field.control_points.data = generated_cp.clone()

    initial_drift = control_point_drift(gt_cp, memory_field.control_points.data).item()
    initial_curv = curvature_deviation(gt_cp.cpu(), memory_field.control_points.data.cpu()).item()
    print(f"  Initial drift: {initial_drift:.4f}")

    anchor_cp = memory_field.control_points.data.clone().detach()

    # ==================================================================
    # STAGE 4: Sequential 360° optimization
    # ==================================================================
    print(f"\n  --- Stage 4: Sequential Optimization ({args.num_views} views) ---")

    azimuths = torch.linspace(0, 360, args.num_views + 1)[:-1]
    gt_curve_points = evaluate_bspline(gt_cp, args.samples_per_curve)
    optimizer = torch.optim.Adam(memory_field.parameters(), lr=args.lr)

    view_drifts = []
    view_losses = []
    curvature_devs = []
    t_start = time.time()

    for view_idx in range(args.num_views):
        az = azimuths[view_idx].item()
        window_az = [azimuths[max(0, view_idx - w)].item() for w in range(args.view_window)]

        for step in range(args.steps_per_view):
            optimizer.zero_grad()
            pred_curve_points = evaluate_bspline(
                memory_field.control_points, args.samples_per_curve
            )
            loss_reproj = reprojection_loss(
                gt_curve_points.detach(), pred_curve_points,
                azimuths=window_az, device=device
            )
            loss_anch = anchor_loss(memory_field.control_points, anchor_cp)
            loss_smooth = smoothness_loss(memory_field.control_points)
            loss = loss_reproj + args.anchor_weight * loss_anch + args.smooth_weight * loss_smooth
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            anchor_cp = (1 - args.anchor_momentum) * anchor_cp + \
                         args.anchor_momentum * memory_field.control_points.data.clone()
            drift = control_point_drift(gt_cp, memory_field.control_points.data).item()
            curv = curvature_deviation(gt_cp.cpu(), memory_field.control_points.data.cpu()).item()

        view_drifts.append(drift)
        view_losses.append(loss_reproj.item())
        curvature_devs.append(curv)

        if view_idx % 6 == 0 or view_idx == args.num_views - 1:
            print(f"  View {view_idx+1:>3d}/{args.num_views}  "
                  f"az={az:>6.1f}°  drift={drift:.4f}  [{time.time()-t_start:.1f}s]")

    # ==================================================================
    # STAGE 5: Revisitation
    # ==================================================================
    print(f"\n  --- Stage 5: Revisitation ---")
    revisit = {}
    with torch.no_grad():
        pred_pts = evaluate_bspline(memory_field.control_points, args.samples_per_curve)
        for test_az in [0.0, 90.0, 180.0, 270.0]:
            rl = reprojection_loss(gt_curve_points, pred_pts,
                                    azimuths=[test_az], device=device).item()
            revisit[test_az] = rl
            print(f"  {test_az:>5.0f}°: {rl:.6f}")

    # ==================================================================
    # STAGE 6: Results
    # ==================================================================
    fm = compute_all_metrics(gt_cp.cpu(), memory_field.control_points.data.cpu())
    drift_pct = (1 - fm['cp_drift'] / initial_drift) * 100
    avg_revisit = sum(revisit.values()) / len(revisit)
    half = len(view_drifts) // 2
    avg1 = sum(view_drifts[:half]) / half
    avg2 = sum(view_drifts[half:]) / (len(view_drifts) - half)
    stable = "STABLE" if avg2 <= avg1 * 1.05 else "DRIFTING"

    print(f"\n  {'='*60}")
    print(f"  END-TO-END RESULTS")
    print(f"  {'='*60}")
    print(f"  Generated (Gθ):   drift={initial_drift:.4f}")
    print(f"  After memory:     drift={fm['cp_drift']:.4f}")
    print(f"  Improvement:      {drift_pct:+.1f}%")
    print(f"  Revisitation:     {avg_revisit:.6f}")
    print(f"  Memory:           {stable} (1st={avg1:.4f}, 2nd={avg2:.4f})")
    print(f"  {'='*60}")

    # ==================================================================
    # SAVE ALL .pt FILES
    # ==================================================================
    print(f"\n  --- Saving .pt files ---")

    torch.save(gt_cp.cpu(), os.path.join(args.output_dir, "gt_cp.pt"))
    print(f"  Saved gt_cp.pt")

    torch.save(generated_cp.cpu(), os.path.join(args.output_dir, "generated_cp.pt"))
    print(f"  Saved generated_cp.pt")

    torch.save(memory_field.control_points.data.cpu(),
               os.path.join(args.output_dir, "final_cp.pt"))
    print(f"  Saved final_cp.pt")

    results = {
        "initial_drift": initial_drift, "initial_curv": initial_curv,
        "final_metrics": fm, "view_drifts": view_drifts,
        "view_losses": view_losses, "curvature_devs": curvature_devs,
        "revisit_losses": revisit, "drift_pct": drift_pct,
        "azimuths": azimuths.tolist(), "args": vars(args),
    }
    torch.save(results, os.path.join(args.output_dir, "demo_results.pt"))
    print(f"  Saved demo_results.pt")

    # Verify saves
    for fname in ["gt_cp.pt", "generated_cp.pt", "final_cp.pt", "demo_results.pt"]:
        fpath = os.path.join(args.output_dir, fname)
        assert os.path.exists(fpath), f"SAVE FAILED: {fpath}"
    print(f"  All .pt files verified!")

    # ==================================================================
    # PLOT
    # ==================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes[0].plot(azimuths.numpy(), view_drifts, "b-o", markersize=3)
        axes[0].axhline(y=initial_drift, color="r", linestyle="--",
                         label=f"Gθ: {initial_drift:.3f}", alpha=0.7)
        axes[0].set_xlabel("Azimuth (°)"); axes[0].set_ylabel("CP Drift")
        axes[0].set_title("Drift During Orbit"); axes[0].legend()

        axes[1].plot(azimuths.numpy(), view_losses, "g-o", markersize=3)
        axes[1].set_xlabel("Azimuth (°)"); axes[1].set_ylabel("Reproj Loss")
        axes[1].set_title("Reprojection Loss")

        axes[2].plot(azimuths.numpy(), curvature_devs, "m-o", markersize=3)
        axes[2].set_xlabel("Azimuth (°)"); axes[2].set_ylabel("Curvature Dev")
        axes[2].set_title("Curvature Deviation"); axes[2].legend()

        plt.suptitle(f"End-to-End: Drift {initial_drift:.3f} → {fm['cp_drift']:.3f} ({drift_pct:+.1f}%)",
                      fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "demo_results.png"), dpi=150)
        plt.close()
        print(f"  Saved demo_results.png")
    except Exception as e:
        print(f"  Plot error: {e}")

    # ==================================================================
    # VIDEO (unless --skip-video)
    # ==================================================================
    if args.skip_video:
        print(f"\n  Video skipped (--skip-video)")
        return

    print(f"\n  --- Generating 360° Triptych Video ---")

    # Detect renderer
    use_tubes = not args.no_tubes
    if use_tubes:
        try:
            from pytorch3d.renderer import MeshRenderer
            _ = render_colored_tubes(gt_cp[:3], azimuth=0, image_size=128,
                                      device=device)
            print(f"  Renderer: colored tube meshes")
        except Exception as e:
            print(f"  Tubes unavailable ({e}), using colored points")
            use_tubes = False
    else:
        print(f"  Renderer: colored points (--no-tubes)")

    render_fn = make_render_fn(use_tubes, device, args.video_size,
                                args.tube_radius, args.samples_per_curve)

    gt_cp_d = gt_cp
    gen_cp_d = generated_cp
    final_cp_d = memory_field.control_points.data

    num_frames = args.num_frames
    video_azimuths = np.linspace(0, 360, num_frames, endpoint=False)

    frames_dir = os.path.join(args.output_dir, "video_frames")
    os.makedirs(frames_dir, exist_ok=True)

    print(f"  Rendering {num_frames} frames...")
    frames = []
    for i, az in enumerate(video_azimuths):
        if i % 15 == 0:
            print(f"    Frame {i+1}/{num_frames} (az={az:.1f}°)")

        with torch.no_grad():
            gt_img = render_fn(gt_cp_d, az)
            gen_img = render_fn(gen_cp_d, az)
            final_img = render_fn(final_cp_d, az)

        frame = make_triptych_frame(gt_img, gen_img, final_img,
                                      azimuth=az,
                                      drift_initial=initial_drift,
                                      drift_final=fm['cp_drift'])
        frames.append(frame)
        Image.fromarray(frame).save(os.path.join(frames_dir, f"frame_{i:04d}.png"))

    # Assemble MP4
    video_path = os.path.join(args.output_dir, "triptych_360.mp4")
    print(f"\n  Assembling video ({args.fps} fps)...")

    try:
        import imageio.v2 as imageio
        writer = imageio.get_writer(video_path, fps=args.fps, codec='libx264',
                                     quality=8, pixelformat='yuv420p')
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"  Saved: {video_path} ({num_frames / args.fps:.1f}s)")
    except Exception as e:
        print(f"  MP4 failed ({e}), saving GIF...")
        gif_path = video_path.replace(".mp4", ".gif")
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                            duration=int(1000 / args.fps), loop=0)
        print(f"  Saved: {gif_path}")

    print(f"\n  All outputs in {args.output_dir}/")
    print(f"  Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo + Video")
    # Generator
    parser.add_argument("--generator-path", type=str, required=True)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-curves", type=int, default=40)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gt-seed", type=int, default=42)
    # Optimization
    parser.add_argument("--num-views", type=int, default=36)
    parser.add_argument("--steps-per-view", type=int, default=100)
    parser.add_argument("--view-window", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--anchor-weight", type=float, default=0.05)
    parser.add_argument("--smooth-weight", type=float, default=0.001)
    parser.add_argument("--anchor-momentum", type=float, default=0.3)
    parser.add_argument("--samples-per-curve", type=int, default=128)
    # Video
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--no-tubes", action="store_true",
                        help="Use point rendering instead of tube meshes")
    parser.add_argument("--video-size", type=int, default=512)
    parser.add_argument("--tube-radius", type=float, default=0.012)
    parser.add_argument("--num-frames", type=int, default=90)
    parser.add_argument("--fps", type=int, default=15)
    # Output
    parser.add_argument("--output-dir", type=str, default="outputs/demo")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run(args)
