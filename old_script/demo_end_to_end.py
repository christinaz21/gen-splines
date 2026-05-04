"""
demo_end_to_end.py — Full Pipeline Demo

Complete pipeline from slides:
  1. z ~ N(0, I) → Gθ(z) → initial spline scene
  2. Render via PyTorch3D (tube meshes for smooth curves)
  3. Initialize persistent curve memory
  4. Sequential 360° orbit with memory updates
  5. Evaluate drift, curvature, revisitation
  6. Render before/after comparison

Rendering upgrade: tube meshes instead of point splatting.
Each B-spline curve → cylindrical mesh → PyTorch3D mesh renderer.
This produces smooth, continuous strand renders matching the proposal's
vision of "Smooth, Defined, and Controllable Curves."

Usage:
    python demo_end_to_end.py \
        --generator-path outputs/gen_percurve/best_generator.pt \
        --output-dir outputs/demo
"""

import os
import time
import argparse
import torch
import torch.nn.functional as F
import numpy as np

from generator import load_generator
from spline import SplineField, evaluate_bspline
from dataset import create_combined_scene
from metrics import control_point_drift, curvature_deviation, compute_all_metrics
from losses import reprojection_loss, smoothness_loss, anchor_loss


# =========================================================================
# Tube Mesh Rendering — makes curves look like actual strands
# =========================================================================

def curves_to_tube_mesh(curve_points, tube_radius=0.008, n_sides=6):
    """
    Convert sampled curve points into tube meshes for smooth rendering.

    Each curve → cylindrical mesh with n_sides cross-section.
    This produces smooth, continuous strand renders instead of scattered dots.

    Args:
        curve_points: (N, M, 3) — N curves, M samples each
        tube_radius: radius of each tube
        n_sides: polygon sides for cross-section (6 = hexagonal)

    Returns:
        verts: (V, 3) all mesh vertices
        faces: (F, 3) triangle faces
    """
    N, M, D = curve_points.shape
    device = curve_points.device

    all_verts = []
    all_faces = []
    vert_offset = 0

    # Circle template for cross-section
    angles = torch.linspace(0, 2 * np.pi, n_sides + 1, device=device)[:-1]
    circle_x = torch.cos(angles)  # (n_sides,)
    circle_y = torch.sin(angles)

    for i in range(N):
        pts = curve_points[i]  # (M, 3)

        # Compute tangent vectors along the curve
        tangents = pts[1:] - pts[:-1]  # (M-1, 3)
        tangents = F.normalize(tangents, dim=-1)
        # Pad last tangent
        tangents = torch.cat([tangents, tangents[-1:]], dim=0)  # (M, 3)

        # Compute normal and binormal via arbitrary vector
        arbitrary = torch.tensor([0.0, 0.0, 1.0], device=device)
        # If tangent is nearly parallel to arbitrary, use different one
        normals = torch.cross(tangents, arbitrary.expand_as(tangents), dim=-1)
        small_norm = normals.norm(dim=-1) < 1e-4
        if small_norm.any():
            alt = torch.tensor([0.0, 1.0, 0.0], device=device)
            normals[small_norm] = torch.cross(
                tangents[small_norm], alt.expand(small_norm.sum(), -1), dim=-1
            )
        normals = F.normalize(normals, dim=-1)
        binormals = torch.cross(tangents, normals, dim=-1)
        binormals = F.normalize(binormals, dim=-1)

        # Generate tube vertices: for each point along curve, place a circle
        # verts shape: (M, n_sides, 3)
        tube_verts = (
            pts.unsqueeze(1)  # (M, 1, 3)
            + tube_radius * circle_x[None, :, None] * normals.unsqueeze(1)
            + tube_radius * circle_y[None, :, None] * binormals.unsqueeze(1)
        )  # (M, n_sides, 3)

        verts_flat = tube_verts.reshape(-1, 3)  # (M * n_sides, 3)

        # Generate faces: connect adjacent circles
        faces = []
        for j in range(M - 1):
            for k in range(n_sides):
                k_next = (k + 1) % n_sides
                v0 = j * n_sides + k + vert_offset
                v1 = j * n_sides + k_next + vert_offset
                v2 = (j + 1) * n_sides + k + vert_offset
                v3 = (j + 1) * n_sides + k_next + vert_offset
                faces.append([v0, v2, v1])
                faces.append([v1, v2, v3])

        all_verts.append(verts_flat)
        all_faces.extend(faces)
        vert_offset += len(verts_flat)

    verts = torch.cat(all_verts, dim=0)
    faces = torch.tensor(all_faces, dtype=torch.long, device=device)
    return verts, faces


def render_tube_mesh(verts, faces, azimuth=0.0, elevation=30.0,
                      image_size=512, device="cuda"):
    """
    Render tube meshes via PyTorch3D's mesh renderer.

    Returns: (H, W, 3) RGB image
    """
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        look_at_view_transform, FoVPerspectiveCameras,
        RasterizationSettings, MeshRenderer, MeshRasterizer,
        SoftPhongShader, PointLights, TexturesVertex,
    )

    R, T = look_at_view_transform(dist=4.0, elev=elevation, azim=azimuth)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
	bin_size=0,
    )

    lights = PointLights(
        device=device,
        location=[[2.0, 2.0, 2.0]],
        ambient_color=[[0.4, 0.4, 0.4]],
        diffuse_color=[[0.6, 0.6, 0.6]],
        specular_color=[[0.2, 0.2, 0.2]],
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )

    # White vertex colors
    vert_colors = torch.ones_like(verts).unsqueeze(0)
    textures = TexturesVertex(verts_features=vert_colors)

    mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
    images = renderer(mesh)

    return images[0, ..., :3]  # (H, W, 3)


def render_scene(control_points, samples_per_curve=128, tube_radius=0.008,
                  azimuth=0.0, image_size=512, device="cuda", use_tubes=True):
    """
    Render a spline scene. Uses tube meshes if available, falls back to points.

    Args:
        control_points: (N, K, 3) control points
        Returns: (H, W, 3) numpy uint8 image
    """
    curve_points = evaluate_bspline(control_points, samples_per_curve)  # (N, M, 3)

    if use_tubes:
        try:
            verts, faces = curves_to_tube_mesh(curve_points, tube_radius=tube_radius)
            img = render_tube_mesh(verts, faces, azimuth=azimuth,
                                    image_size=image_size, device=device)
            return (img.cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
        except Exception as e:
            print(f"    Tube rendering failed ({e}), falling back to points")

    # Fallback: point rendering
    from renderer import render_point_cloud
    pts_flat = curve_points.reshape(-1, 3)
    config = {"radius": 0.005, "image_size": image_size}
    img = render_point_cloud(pts_flat, azimuth=azimuth, config=config, device=device)
    return (img[..., :3].cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)


# =========================================================================
# Main Demo
# =========================================================================

def run_demo(args):
    device = args.device
    print(f"\n{'='*65}")
    print(f"  END-TO-END DEMO: Generative Spline Fields")
    print(f"  with Persistent Curve Memory")
    print(f"{'='*65}")

    os.makedirs(args.output_dir, exist_ok=True)
    render_dir = os.path.join(args.output_dir, "renders")
    os.makedirs(render_dir, exist_ok=True)

    # ==================================================================
    # STAGE 1: z ~ N(0,I) → Gθ(z) → initial spline scene
    # ==================================================================
    print(f"\n  --- Stage 1: Generate Scene via Gθ(z) ---")

    generator = load_generator(
        args.generator_path, device=device,
        latent_dim=args.latent_dim, num_curves=args.num_curves,
        K=args.K, hidden_dim=args.hidden_dim
    )

    # Load normalization
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
        generated_cp = (gen_cp_norm * data_std + data_mean).squeeze(0)  # (N, K, 3)

    N, K, D = generated_cp.shape
    print(f"  Generated: {N} curves, {K} CPs each")

    # ==================================================================
    # STAGE 2: Ground-truth scene
    # ==================================================================
    print(f"\n  --- Stage 2: Ground Truth ---")
    gt_cp = create_combined_scene(num_helix=20, num_wave=20, K=K, seed=args.gt_seed).to(device)
    gt_cp = gt_cp[:N]
    print(f"  GT: {gt_cp.shape}")

    # ==================================================================
    # STAGE 3: Render initial scenes (tube meshes)
    # ==================================================================
    print(f"\n  --- Stage 3: Render (tube meshes) ---")

    import imageio

    for az in [0.0, 90.0, 180.0, 270.0]:
        gen_img = render_scene(generated_cp, azimuth=az, image_size=args.image_size,
                                device=device, use_tubes=args.use_tubes)
        gt_img = render_scene(gt_cp, azimuth=az, image_size=args.image_size,
                               device=device, use_tubes=args.use_tubes)
        imageio.imwrite(os.path.join(render_dir, f"initial_gen_az{az:.0f}.png"), gen_img)
        imageio.imwrite(os.path.join(render_dir, f"gt_az{az:.0f}.png"), gt_img)
        # Side-by-side
        combined = np.concatenate([gt_img, gen_img], axis=1)
        imageio.imwrite(os.path.join(render_dir, f"compare_initial_az{az:.0f}.png"), combined)

    print(f"  Saved initial renders to {render_dir}/")

    # ==================================================================
    # STAGE 4: Initialize persistent memory
    # ==================================================================
    print(f"\n  --- Stage 4: Persistent Curve Memory ---")

    memory = SplineField(N, K).to(device)
    memory.control_points.data = generated_cp.clone()

    initial_drift = control_point_drift(gt_cp, memory.control_points.data).item()
    initial_curv = curvature_deviation(gt_cp.cpu(), memory.control_points.data.cpu()).item()
    print(f"  Initial drift (gen → GT): {initial_drift:.4f}")
    print(f"  Initial curvature dev:    {initial_curv:.4f}")

    anchor_cp = memory.control_points.data.clone().detach()

    # ==================================================================
    # STAGE 5: Sequential 360° orbit
    # ==================================================================
    print(f"\n  --- Stage 5: Sequential Optimization ---")

    azimuths = torch.linspace(0, 360, args.num_views + 1)[:-1]
    gt_curve_pts = evaluate_bspline(gt_cp, args.samples_per_curve)

    optimizer = torch.optim.Adam(memory.parameters(), lr=args.lr)

    view_drifts = []
    view_losses = []
    curvature_devs = []
    t_start = time.time()

    for vi in range(args.num_views):
        az = azimuths[vi].item()
        window = [azimuths[max(0, vi - w)].item() for w in range(args.view_window)]

        for step in range(args.steps_per_view):
            optimizer.zero_grad()
            pred_pts = evaluate_bspline(memory.control_points, args.samples_per_curve)

            loss_r = reprojection_loss(gt_curve_pts.detach(), pred_pts,
                                        azimuths=window, device=device)
            loss_a = anchor_loss(memory.control_points, anchor_cp)
            loss_s = smoothness_loss(memory.control_points)
            loss = loss_r + args.anchor_weight * loss_a + args.smooth_weight * loss_s

            loss.backward()
            optimizer.step()

        with torch.no_grad():
            anchor_cp = (1 - args.anchor_momentum) * anchor_cp + \
                         args.anchor_momentum * memory.control_points.data.clone()
            drift = control_point_drift(gt_cp, memory.control_points.data).item()
            curv = curvature_deviation(gt_cp.cpu(), memory.control_points.data.cpu()).item()

        view_drifts.append(drift)
        view_losses.append(loss_r.item())
        curvature_devs.append(curv)

        if vi % 6 == 0 or vi == args.num_views - 1:
            elapsed = time.time() - t_start
            print(f"  View {vi+1:>3d}/{args.num_views}  az={az:>6.1f}°  "
                  f"drift={drift:.4f}  [{elapsed:.1f}s]")

    # ==================================================================
    # STAGE 6: Revisitation
    # ==================================================================
    print(f"\n  --- Stage 6: Revisitation ---")
    revisit = {}
    with torch.no_grad():
        pred_pts = evaluate_bspline(memory.control_points, args.samples_per_curve)
        for test_az in [0.0, 90.0, 180.0, 270.0]:
            rl = reprojection_loss(gt_curve_pts, pred_pts,
                                    azimuths=[test_az], device=device).item()
            revisit[test_az] = rl
            print(f"  {test_az:>5.0f}°: {rl:.6f}")

    # ==================================================================
    # STAGE 7: Render AFTER optimization (shows improvement)
    # ==================================================================
    print(f"\n  --- Stage 7: Final Renders ---")

    final_cp = memory.control_points.data
    for az in [0.0, 90.0, 180.0, 270.0]:
        final_img = render_scene(final_cp, azimuth=az, image_size=args.image_size,
                                  device=device, use_tubes=args.use_tubes)
        gt_img = render_scene(gt_cp, azimuth=az, image_size=args.image_size,
                               device=device, use_tubes=args.use_tubes)
        imageio.imwrite(os.path.join(render_dir, f"final_az{az:.0f}.png"), final_img)

        # Three-panel: GT | Initial | Final
        init_img = render_scene(generated_cp, azimuth=az, image_size=args.image_size,
                                 device=device, use_tubes=args.use_tubes)
        triptych = np.concatenate([gt_img, init_img, final_img], axis=1)
        imageio.imwrite(os.path.join(render_dir, f"triptych_az{az:.0f}.png"), triptych)

    print(f"  Saved final + triptych renders to {render_dir}/")

    # ==================================================================
    # RESULTS
    # ==================================================================
    fm = compute_all_metrics(gt_cp.cpu(), final_cp.cpu(), num_samples=64)
    drift_pct = (1 - fm['cp_drift'] / initial_drift) * 100
    avg_revisit = sum(revisit.values()) / len(revisit)

    half = len(view_drifts) // 2
    avg1 = sum(view_drifts[:half]) / half
    avg2 = sum(view_drifts[half:]) / (len(view_drifts) - half)
    stable = "STABLE" if avg2 <= avg1 * 1.05 else "DRIFTING"

    print(f"\n  {'='*60}")
    print(f"  END-TO-END RESULTS")
    print(f"  {'='*60}")
    print(f"  Pipeline: z ~ N(0,I) → Gθ(z) → memory → 360° refine")
    print(f"")
    print(f"  Generated (Gθ output):    drift={initial_drift:.4f}  curv={initial_curv:.4f}")
    print(f"  After persistent memory:  drift={fm['cp_drift']:.4f}  curv={fm['curvature_deviation']:.4f}")
    print(f"  Drift improvement:        {drift_pct:+.1f}%")
    print(f"  Revisitation loss:        {avg_revisit:.6f}")
    print(f"  Memory:                   {stable} (1st={avg1:.4f}, 2nd={avg2:.4f})")
    print(f"  {'='*60}")

    # Save results
    results = {
        "initial_drift": initial_drift, "initial_curv": initial_curv,
        "final_metrics": fm, "view_drifts": view_drifts,
        "view_losses": view_losses, "curvature_devs": curvature_devs,
        "revisit_losses": revisit, "drift_pct": drift_pct,
        "azimuths": azimuths.tolist(), "args": vars(args),
    }
    torch.save(results, os.path.join(args.output_dir, "demo_results.pt"))

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        axes[0].plot(azimuths.numpy(), view_drifts, "b-o", markersize=3, label="With memory")
        axes[0].axhline(y=initial_drift, color="r", linestyle="--",
                         label=f"Gθ output: {initial_drift:.3f}", alpha=0.7)
        axes[0].set_xlabel("Azimuth (°)")
        axes[0].set_ylabel("CP Drift")
        axes[0].set_title("CP Drift: Generated → Refined")
        axes[0].legend()

        axes[1].plot(azimuths.numpy(), view_losses, "g-o", markersize=3)
        axes[1].set_xlabel("Azimuth (°)")
        axes[1].set_ylabel("Reprojection Loss")
        axes[1].set_title("Reconstruction Loss per View")

        axes[2].plot(azimuths.numpy(), curvature_devs, "m-o", markersize=3)
        axes[2].axhline(y=initial_curv, color="r", linestyle="--",
                         label=f"Initial: {initial_curv:.2f}", alpha=0.7)
        axes[2].set_xlabel("Azimuth (°)")
        axes[2].set_ylabel("Curvature Dev")
        axes[2].set_title("Curvature During Orbit")
        axes[2].legend()

        plt.suptitle("End-to-End: z → Gθ(z) → Persistent Curve Memory → 360° Refinement",
                      fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "demo_results.png"), dpi=150)
        plt.close()
        print(f"  Saved plot to {args.output_dir}/demo_results.png")
    except Exception as e:
        print(f"  Could not save plot: {e}")

    print(f"\n  All results saved to {args.output_dir}/")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Demo")
    parser.add_argument("--generator-path", type=str, required=True)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-curves", type=int, default=40)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gt-seed", type=int, default=42)
    # Rendering
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--samples-per-curve", type=int, default=128)
    parser.add_argument("--use-tubes", action="store_true", default=True,
                        help="Use tube mesh rendering (smooth strands)")
    parser.add_argument("--no-tubes", dest="use_tubes", action="store_false",
                        help="Fallback to point rendering")
    # Optimization
    parser.add_argument("--num-views", type=int, default=36)
    parser.add_argument("--steps-per-view", type=int, default=100)
    parser.add_argument("--view-window", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--anchor-weight", type=float, default=0.05)
    parser.add_argument("--smooth-weight", type=float, default=0.001)
    parser.add_argument("--anchor-momentum", type=float, default=0.3)
    # Output
    parser.add_argument("--output-dir", type=str, default="outputs/demo")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_demo(args)
