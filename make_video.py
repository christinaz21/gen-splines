"""
make_video.py — Generate a 360° rotating triptych video.

Renders GT | Generated | After Memory side-by-side with:
  - Dark background for contrast
  - Per-curve colors (each strand a different hue)
  - Larger tube radius for visibility
  - Labels and a drift meter
  - Smooth 360° rotation assembled into MP4

Usage:
    python make_video.py --demo-dir outputs/demo --output outputs/demo/triptych_360.mp4
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from spline import evaluate_bspline


def render_colored_curves(control_points, azimuth=0.0, elevation=30.0,
                           image_size=512, tube_radius=0.012, samples=128,
                           bg_color=(0.08, 0.08, 0.12), device="cuda"):
    """
    Render spline curves as colored tube meshes on a dark background.

    Each curve gets a unique color from a perceptually uniform palette.
    """
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        look_at_view_transform, FoVPerspectiveCameras,
        RasterizationSettings, MeshRasterizer, MeshRenderer,
        SoftPhongShader, PointLights, TexturesVertex,
    )

    curve_points = evaluate_bspline(control_points, samples)  # (N, M, 3)
    N, M, D = curve_points.shape

    # Generate per-curve colors (golden angle hue spacing for max distinction)
    colors_per_curve = []
    for i in range(N):
        hue = (i * 137.508) % 360  # golden angle
        sat = 0.7 + 0.3 * ((i * 7) % 10) / 10  # vary saturation
        val = 0.75 + 0.25 * ((i * 3) % 10) / 10  # vary brightness
        r, g, b = hsv_to_rgb(hue / 360, sat, val)
        colors_per_curve.append((r, g, b))

    # Build tube meshes with per-curve colors
    all_verts = []
    all_faces = []
    all_colors = []
    n_sides = 6
    vert_offset = 0

    for i in range(N):
        pts = curve_points[i]  # (M, 3)
        r, g, b = colors_per_curve[i]

        for j in range(M - 1):
            # Direction along curve
            d = pts[j + 1] - pts[j]
            d_len = d.norm() + 1e-8
            d_norm = d / d_len

            # Perpendicular vectors
            up = torch.tensor([0.0, 1.0, 0.0], device=device)
            if abs(torch.dot(d_norm, up)) > 0.99:
                up = torch.tensor([1.0, 0.0, 0.0], device=device)

            right = torch.cross(d_norm, up)
            right = right / (right.norm() + 1e-8)
            up2 = torch.cross(right, d_norm)
            up2 = up2 / (up2.norm() + 1e-8)

            # Circle vertices at start and end of segment
            for k, center in enumerate([pts[j], pts[j + 1]]):
                for s in range(n_sides):
                    angle = 2 * np.pi * s / n_sides
                    offset = tube_radius * (np.cos(angle) * right + np.sin(angle) * up2)
                    all_verts.append(center + offset)
                    all_colors.append([r, g, b])

            # Triangle faces connecting the two rings
            base = vert_offset
            for s in range(n_sides):
                s_next = (s + 1) % n_sides
                # Two triangles per quad
                all_faces.append([base + s, base + n_sides + s, base + n_sides + s_next])
                all_faces.append([base + s, base + n_sides + s_next, base + s_next])

            vert_offset += 2 * n_sides

    verts = torch.stack(all_verts).to(device)
    faces = torch.tensor(all_faces, dtype=torch.long, device=device)
    vert_colors = torch.tensor(all_colors, dtype=torch.float32, device=device).unsqueeze(0)

    # Render
    R, T = look_at_view_transform(dist=4.0, elev=elevation, azim=azimuth)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,  # naive rasterization — handles large meshes
    )

    lights = PointLights(
        device=device,
        location=[[3.0, 3.0, 3.0]],
        ambient_color=[[0.5, 0.5, 0.5]],
        diffuse_color=[[0.5, 0.5, 0.5]],
        specular_color=[[0.15, 0.15, 0.15]],
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )

    textures = TexturesVertex(verts_features=vert_colors)
    mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
    images = renderer(mesh)  # (1, H, W, 4)

    img = images[0, ..., :3].cpu().numpy()

    # Composite onto dark background
    alpha = images[0, ..., 3:4].cpu().numpy()
    bg = np.array(bg_color).reshape(1, 1, 3)
    composited = img * alpha + bg * (1 - alpha)

    return (np.clip(composited, 0, 1) * 255).astype(np.uint8)


def render_colored_points_fallback(control_points, azimuth=0.0, elevation=30.0,
                                     image_size=512, samples=128,
                                     bg_color=(0.08, 0.08, 0.12), device="cuda"):
    """Fallback: colored point rendering if mesh rendering fails."""
    from renderer import render_point_cloud

    curve_points = evaluate_bspline(control_points, samples)
    N, M, D = curve_points.shape

    # Per-curve colors
    colors = torch.zeros(N, M, 3, device=device)
    for i in range(N):
        hue = (i * 137.508) % 360
        r, g, b = hsv_to_rgb(hue / 360, 0.8, 0.9)
        colors[i, :] = torch.tensor([r, g, b], device=device)

    pts = curve_points.reshape(-1, 3)
    clrs = colors.reshape(-1, 3)

    config = {"radius": 0.008, "image_size": image_size}
    img = render_point_cloud(pts, features=clrs, azimuth=azimuth, elevation=elevation,
                              config=config, device=device)
    img_np = img[..., :3].cpu().numpy()

    # Dark background composite (approximate via replacing near-black)
    mask = (img_np.sum(axis=-1, keepdims=True) < 0.05).astype(np.float32)
    bg = np.array(bg_color).reshape(1, 1, 3)
    composited = img_np * (1 - mask) + bg * mask

    return (np.clip(composited, 0, 1) * 255).astype(np.uint8)


def hsv_to_rgb(h, s, v):
    """Convert HSV to RGB (all 0-1 range)."""
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return r, g, b


def make_triptych_frame(gt_img, gen_img, final_img, azimuth, drift_initial,
                          drift_final, frame_width=1536, frame_height=580):
    """
    Combine three renders into a labeled triptych frame.
    """
    panel_w = frame_width // 3
    panel_h = frame_height - 68  # leave room for labels

    # Resize panels
    gt_pil = Image.fromarray(gt_img).resize((panel_w, panel_h), Image.LANCZOS)
    gen_pil = Image.fromarray(gen_img).resize((panel_w, panel_h), Image.LANCZOS)
    final_pil = Image.fromarray(final_img).resize((panel_w, panel_h), Image.LANCZOS)

    # Create frame with dark background
    frame = Image.new("RGB", (frame_width, frame_height), (20, 20, 30))
    frame.paste(gt_pil, (0, 0))
    frame.paste(gen_pil, (panel_w, 0))
    frame.paste(final_pil, (panel_w * 2, 0))

    # Draw labels
    draw = ImageDraw.Draw(frame)
    try:
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (OSError, IOError):
        font_label = ImageFont.load_default()
        font_small = font_label

    label_y = panel_h + 8
    meter_y = panel_h + 32

    # Panel labels
    draw.text((panel_w // 2 - 60, label_y), "Ground Truth", fill=(180, 220, 255), font=font_label)
    draw.text((panel_w + panel_w // 2 - 80, label_y), "Gθ(z) Generated", fill=(255, 200, 100), font=font_label)
    draw.text((panel_w * 2 + panel_w // 2 - 90, label_y), "After Memory", fill=(100, 255, 150), font=font_label)

    # Drift info
    drift_pct = (1 - drift_final / drift_initial) * 100 if drift_initial > 0 else 0
    info = f"Azimuth: {azimuth:.0f}°    |    Drift: {drift_initial:.3f} → {drift_final:.3f}  ({drift_pct:+.1f}%)"
    draw.text((frame_width // 2 - 220, meter_y), info, fill=(200, 200, 200), font=font_small)

    # Separator lines between panels
    for x in [panel_w, panel_w * 2]:
        draw.line([(x, 0), (x, panel_h)], fill=(60, 60, 80), width=2)

    return np.array(frame)


def main(args):
    device = args.device
    print(f"\n{'='*60}")
    print(f"  Generating 360° Triptych Video")
    print(f"{'='*60}")

    # Load saved control points from demo
    gt_cp = torch.load(os.path.join(args.demo_dir, "gt_cp.pt"), weights_only=True).to(device)
    gen_cp = torch.load(os.path.join(args.demo_dir, "generated_cp.pt"), weights_only=True).to(device)
    final_cp = torch.load(os.path.join(args.demo_dir, "final_cp.pt"), weights_only=True).to(device)

    # Load drift values
    results = torch.load(os.path.join(args.demo_dir, "demo_results.pt"), weights_only=True)
    drift_initial = results["initial_drift"]
    drift_final = results["final_metrics"]["cp_drift"]

    print(f"  GT:    {gt_cp.shape}")
    print(f"  Gen:   {gen_cp.shape}")
    print(f"  Final: {final_cp.shape}")
    print(f"  Drift: {drift_initial:.4f} → {drift_final:.4f}")

    # Choose renderer
    use_tubes = True
    try:
        from pytorch3d.renderer import MeshRenderer
        test_img = render_colored_curves(gt_cp[:5], azimuth=0, image_size=128,
                                          device=device)
        print(f"  Renderer: tube meshes (colored)")
    except Exception as e:
        print(f"  Tube rendering unavailable ({e}), using colored points")
        use_tubes = False

    def render_fn(cp, az):
        if use_tubes:
            try:
                return render_colored_curves(cp, azimuth=az, image_size=args.image_size,
                                              tube_radius=args.tube_radius,
                                              samples=args.samples, device=device)
            except Exception:
                return render_colored_points_fallback(cp, azimuth=az,
                                                       image_size=args.image_size,
                                                       samples=args.samples, device=device)
        else:
            return render_colored_points_fallback(cp, azimuth=az,
                                                   image_size=args.image_size,
                                                   samples=args.samples, device=device)

    # Generate frames
    num_frames = args.num_frames
    azimuths = np.linspace(0, 360, num_frames, endpoint=False)

    frames_dir = os.path.join(args.demo_dir, "video_frames")
    os.makedirs(frames_dir, exist_ok=True)

    print(f"  Rendering {num_frames} frames...")
    frames = []
    for i, az in enumerate(azimuths):
        if i % 10 == 0:
            print(f"    Frame {i+1}/{num_frames} (az={az:.1f}°)")

        with torch.no_grad():
            gt_img = render_fn(gt_cp, az)
            gen_img = render_fn(gen_cp, az)
            final_img = render_fn(final_cp, az)

        frame = make_triptych_frame(gt_img, gen_img, final_img,
                                      azimuth=az,
                                      drift_initial=drift_initial,
                                      drift_final=drift_final)
        frames.append(frame)

        # Save individual frame
        Image.fromarray(frame).save(os.path.join(frames_dir, f"frame_{i:04d}.png"))

    # Assemble MP4
    print(f"\n  Assembling MP4 ({args.fps} fps)...")
    output_path = args.output

    try:
        import imageio
        writer = imageio.get_writer(output_path, fps=args.fps, codec='libx264',
                                     quality=8, pixelformat='yuv420p')
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"  Saved video: {output_path}")
        print(f"  Duration: {num_frames / args.fps:.1f}s")
    except Exception as e:
        print(f"  imageio MP4 failed ({e}), trying with PIL GIF...")
        gif_path = output_path.replace(".mp4", ".gif")
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                            duration=int(1000 / args.fps), loop=0)
        print(f"  Saved GIF: {gif_path}")

    print(f"  Frames saved to: {frames_dir}/")
    print(f"\n  Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 360° triptych video")
    parser.add_argument("--demo-dir", type=str, default="outputs/demo",
                        help="Directory with demo results (gt_cp.pt, generated_cp.pt, final_cp.pt)")
    parser.add_argument("--output", type=str, default="outputs/demo/triptych_360.mp4")
    # Rendering
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--tube-radius", type=float, default=0.012,
                        help="Thicker = more visible strands")
    parser.add_argument("--samples", type=int, default=128)
    # Video
    parser.add_argument("--num-frames", type=int, default=90,
                        help="Total frames (90 at 15fps = 6 second loop)")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    main(args)
