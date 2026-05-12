"""
render_utils.py — Tube mesh rendering for splines + point rendering for point clouds + export.

Drop this alongside run_world_model.py. It provides:
  - render_spline_tubes(): renders spline CPs as lit 3D tube meshes
  - render_point_dots(): renders points as dots (for point cloud baseline)
  - export_tubes_obj(): exports spline tubes as .OBJ mesh with vertex colors
  - export_points_ply(): exports point cloud as .PLY with colors

Usage in run_world_model.py:
    from render_utils import render_spline_tubes, render_point_dots
    from render_utils import export_tubes_obj, export_points_ply
"""

import math
import os
import numpy as np
import torch
import torch.nn.functional as F


# ── Vectorized tube mesh (fast, no Python loops) ──────────────

def build_tube_mesh(curve_points, radius=0.003, n_sides=4):
    """
    Build tube mesh from (N, M, 3) curve sample points.
    Fully vectorized — handles 2000+ curves instantly.

    Returns:
        verts: (V, 3) mesh vertices
        faces: (F, 3) triangle indices
    """
    N, M, D = curve_points.shape
    device = curve_points.device
    S = n_sides

    # Tangents
    tang = torch.zeros_like(curve_points)
    tang[:, 1:-1] = curve_points[:, 2:] - curve_points[:, :-2]
    tang[:, 0] = curve_points[:, 1] - curve_points[:, 0]
    tang[:, -1] = curve_points[:, -1] - curve_points[:, -2]
    tang = F.normalize(tang, dim=-1)

    # Local frame
    up = torch.zeros_like(tang); up[..., 1] = 1.0
    dot = (tang * up).sum(-1, keepdim=True).abs()
    alt = torch.zeros_like(tang); alt[..., 0] = 1.0
    ref = torch.where(dot > 0.9, alt, up)
    normal = F.normalize(torch.cross(tang, ref, dim=-1), dim=-1)
    binormal = F.normalize(torch.cross(tang, normal, dim=-1), dim=-1)

    # Ring vertices
    angles = torch.linspace(0, 2 * math.pi, S + 1, device=device)[:-1]
    offsets = (
        angles.cos().view(1, 1, S, 1) * normal.unsqueeze(2) * radius +
        angles.sin().view(1, 1, S, 1) * binormal.unsqueeze(2) * radius
    )
    verts = (curve_points.unsqueeze(2) + offsets).reshape(-1, 3)

    # Faces (vectorized)
    co = torch.arange(N, device=device).view(N, 1, 1) * (M * S)
    ro = torch.arange(M - 1, device=device).view(1, M - 1, 1) * S
    so = torch.arange(S, device=device).view(1, 1, S)
    ns = (so + 1) % S
    base = co + ro
    v0 = (base + so).reshape(-1)
    v1 = (base + ns).reshape(-1)
    v2 = (base + S + so).reshape(-1)
    v3 = (base + S + ns).reshape(-1)
    faces = torch.cat([torch.stack([v0, v2, v1], 1),
                       torch.stack([v1, v2, v3], 1)], 0)

    return verts, faces


def make_tube_colors(num_curves, pts_per_curve, n_sides, seed=42, color_type="hair"):
    """Per-vertex colors for tube mesh.
    
    Uses position-based brightness jitter (±10%) instead of random hue.
    Reads as natural variation rather than synthetic stripes.
    """
    rng = np.random.RandomState(seed)

    if color_type == "hair":
        root = np.array([0.55, 0.42, 0.22])
        mid  = np.array([0.78, 0.64, 0.35])
        tip  = np.array([0.88, 0.78, 0.48])
    else:
        root = np.array([0.18, 0.32, 0.10])
        mid  = np.array([0.28, 0.48, 0.16])
        tip  = np.array([0.40, 0.58, 0.22])

    all_colors = []
    for c in range(num_curves):
        # Subtle brightness jitter ±10% — NOT hue randomization
        brightness = rng.uniform(0.90, 1.10)
        # Very small warm/cool shift (5° hue, not 360°)
        warm_shift = rng.uniform(-0.03, 0.03)

        t = np.linspace(0, 1, pts_per_curve)
        colors = np.zeros((pts_per_curve, 3))
        for ch in range(3):
            base = (1-t)**2 * root[ch] + 2*(1-t)*t * mid[ch] + t**2 * tip[ch]
            colors[:, ch] = base * brightness + (warm_shift if ch == 0 else -warm_shift * 0.3)

        expanded = np.repeat(np.clip(colors, 0.04, 1.0), n_sides, axis=0)
        all_colors.append(expanded)

    return torch.tensor(np.clip(np.concatenate(all_colors), 0.04, 1), dtype=torch.float32)


def make_point_colors(num_points, seed=42, color_type="hair"):
    """Per-point colors for point cloud."""
    rng = np.random.RandomState(seed)
    if color_type == "hair":
        base = np.array([0.78, 0.62, 0.35])
    else:
        base = np.array([0.28, 0.48, 0.15])
    j = rng.uniform(-0.05, 0.05, size=(num_points, 3))
    return torch.tensor(np.clip(base + j, 0.05, 1), dtype=torch.float32)


BLONDE = (0.82, 0.72, 0.42)


def blonde_colors(num_points_per_curve_list, base=BLONDE):
    """Per-point blonde colors with subtle strand variation and root-to-tip
    darkening. Accepts either a list of per-curve point counts or a (N, M, 3)
    tensor (uses its second dim as the uniform point count)."""
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


# ── Renderers ─────────────────────────────────────────────────

BG = (0.13, 0.13, 0.17)


def render_spline_tubes(cp, num_samples, az, image_size, device,
                        tube_radius=0.003, n_sides=4,
                        elev=25.0, dist=3.0, seed=42):
    """
    Render spline control points as lit 3D tube meshes.

    Args:
        cp: (N, K, 3) control points (already oriented)
        num_samples: samples per curve for tube construction
        az: camera azimuth
        image_size: render resolution
    """
    from spline import evaluate_bspline
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        look_at_view_transform, FoVPerspectiveCameras,
        RasterizationSettings, MeshRenderer, MeshRasterizer,
        SoftPhongShader, PointLights, TexturesVertex,
    )

    N, K, _ = cp.shape
    tr = tube_radius * (500 / N) ** 0.2

    with torch.no_grad():
        curve_pts = evaluate_bspline(cp, num_samples)

    verts, faces = build_tube_mesh(curve_pts, radius=tr, n_sides=n_sides)
    vcols = make_tube_colors(N, num_samples, n_sides, seed).to(device)

    R, T = look_at_view_transform(dist=dist, elev=elev, azim=az)
    cam = FoVPerspectiveCameras(device=device, R=R, T=T)
    rset = RasterizationSettings(image_size=image_size, blur_radius=0.0,
                                 faces_per_pixel=2, bin_size=0)
    lights = PointLights(
        device=device, location=[[3., 5., 3.]],
        ambient_color=[[0.58, 0.53, 0.46]],
        diffuse_color=[[0.62, 0.56, 0.46]],
        specular_color=[[0.22, 0.20, 0.16]])
    rend = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cam, raster_settings=rset),
        shader=SoftPhongShader(device=device, cameras=cam, lights=lights))
    tex = TexturesVertex(verts_features=[vcols])
    mesh = Meshes(verts=[verts.to(device)], faces=[faces.to(device)], textures=tex)
    res = rend(mesh)
    img = res[0, ..., :3].cpu().numpy()
    alpha = res[0, ..., 3:4].cpu().numpy()
    return np.clip(img * alpha + np.array(BG).reshape(1, 1, 3) * (1 - alpha), 0, 1)


def render_point_dots(points, az, image_size, device,
                      radius=0.008, elev=25.0, dist=3.0, seed=99):
    """Render points as dots."""
    from pytorch3d.renderer import (
        AlphaCompositor, FoVPerspectiveCameras,
        PointsRasterizationSettings, PointsRasterizer,
        PointsRenderer, look_at_view_transform)
    from pytorch3d.structures import Pointclouds

    colors = make_point_colors(points.shape[0], seed).to(device)
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=az)
    cam = FoVPerspectiveCameras(device=device, R=R, T=T)
    rset = PointsRasterizationSettings(image_size=image_size, radius=radius,
                                       points_per_pixel=12, bin_size=0)
    rend = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cam, raster_settings=rset),
        compositor=AlphaCompositor(background_color=BG))
    pc = Pointclouds(points=[points.to(device)], features=[colors])
    return np.clip(rend(pc)[0, ..., :3].cpu().numpy(), 0, 1)


# ── Export functions ──────────────────────────────────────────

def export_tubes_obj(cp, num_samples, output_path, n_sides=4,
                     tube_radius=0.003, seed=42, compact=True):
    """
    Export spline tubes as .OBJ with vertex colors.

    If compact=True (default), uses fewer samples for smaller file:
      32 samples/curve, 3 sides → ~15MB instead of ~95MB.
    """
    from spline import evaluate_bspline

    N, K, _ = cp.shape
    tr = tube_radius * (500 / N) ** 0.2

    if compact:
        num_samples = min(num_samples, 32)
        n_sides = min(n_sides, 3)

    with torch.no_grad():
        curve_pts = evaluate_bspline(cp, num_samples)

    verts, faces = build_tube_mesh(curve_pts, radius=tr, n_sides=n_sides)
    vcols = make_tube_colors(N, num_samples, n_sides, seed)

    verts_np = verts.cpu().numpy()
    faces_np = faces.cpu().numpy()
    cols_np = vcols.numpy()

    with open(output_path, "w") as f:
        f.write(f"# Spline tube mesh: {N} curves, {num_samples} samples/curve\n")
        f.write(f"# {verts_np.shape[0]:,} verts, {faces_np.shape[0]:,} faces\n")
        f.write(f"# Stored params: {N*K*3} ({N*K*3*4//1024}KB) | This mesh: visualization only\n\n")

        for i in range(verts_np.shape[0]):
            v = verts_np[i]; c = cols_np[i]
            f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f} {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}\n")

        f.write(f"\n")
        for i in range(faces_np.shape[0]):
            face = faces_np[i]
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Exported spline tubes: {output_path}")
    print(f"    {verts_np.shape[0]:,} verts, {faces_np.shape[0]:,} faces ({size_mb:.1f}MB)")
    return verts_np.shape[0], faces_np.shape[0]


def export_tubes_obj_full(cp, num_samples, output_path, n_sides=4,
                          tube_radius=0.003, seed=42):
    """Full-quality OBJ export (no compaction). Use for final renders."""
    return export_tubes_obj(cp, num_samples, output_path, n_sides,
                            tube_radius, seed, compact=False)


def export_points_ply(points, output_path, seed=99):
    """
    Export point cloud as .PLY with vertex colors.
    Loadable in MeshLab, Blender, Three.js, any 3D viewer.
    """
    pts_np = points.cpu().numpy()
    cols = make_point_colors(pts_np.shape[0], seed).numpy()
    cols_uint8 = (cols * 255).astype(np.uint8)

    with open(output_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {pts_np.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for i in range(pts_np.shape[0]):
            p = pts_np[i]
            c = cols_uint8[i]
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")

    print(f"  Exported point cloud: {output_path}")
    print(f"    {pts_np.shape[0]:,} points")


def export_control_points_json(cp, output_path):
    """Export raw control points as JSON for the live viewer.
    
    This is the EFFICIENT format: 288KB for 2000 curves.
    The viewer reconstructs tube meshes client-side from CPs.
    """
    import json
    N, K, _ = cp.shape
    data = {
        "num_curves": int(N),
        "K": int(K),
        "params": int(N * K * 3),
        "control_points": cp.cpu().numpy().tolist()
    }
    with open(output_path, "w") as f:
        json.dump(data, f)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  Exported CPs (JSON): {output_path} ({N} curves, K={K}, {size_kb:.0f}KB)")


def export_control_points_bin(cp, output_path):
    """Export control points as compact binary (.npz).
    
    2000 curves × 12 CPs × 3 floats × 4 bytes = 288KB.
    This is what a real system would store — NOT the tube mesh.
    
    Load with:
        data = np.load("control_points.npz")
        cp = torch.tensor(data["control_points"])  # (N, K, 3)
    """
    N, K, _ = cp.shape
    np.savez_compressed(output_path,
                        control_points=cp.cpu().numpy(),
                        num_curves=N, K=K)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  Exported CPs (binary): {output_path} ({N} curves, K={K}, {size_kb:.0f}KB)")
