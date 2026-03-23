"""
renderer.py — Differentiable point-cloud rendering via PyTorch3D.

The RADIUS parameter is the single most important hyperparameter.
- Too small: gradients vanish (points don't cover enough pixels)
- Too large: thin structures blur into blobs

Start with radius=0.02, then sweep [0.005, 0.01, 0.02, 0.03, 0.05].
"""

import torch
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor,
)


# =============================================================================
# Default rendering config — TUNE THESE
# =============================================================================
DEFAULT_RENDER_CONFIG = {
    "image_size": 256,
    "radius": 0.02,          # CRITICAL — start here, sweep if gradients are weak
    "points_per_pixel": 8,   # More = smoother but slower
    "dist": 4.0,             # Camera distance from origin
    "fov": 60.0,             # Field of view
    "compositor": "alpha",   # "alpha" or "norm_weighted"
}


def build_renderer(config: dict = None, device: str = "cuda"):
    """
    Build a PyTorch3D PointsRenderer. Does NOT bind to a specific camera
    — call render_from_pose() to render from a specific viewpoint.

    Returns:
        (raster_settings, compositor) tuple — we build the full renderer
        per-call since camera changes each time.
    """
    cfg = {**DEFAULT_RENDER_CONFIG, **(config or {})}

    raster_settings = PointsRasterizationSettings(
        image_size=cfg["image_size"],
        radius=cfg["radius"],
        points_per_pixel=cfg["points_per_pixel"],
    )

    if cfg["compositor"] == "alpha":
        compositor = AlphaCompositor()
    else:
        compositor = NormWeightedCompositor()

    return raster_settings, compositor, cfg


def make_cameras(azimuth: float = 0.0, elevation: float = 30.0,
                 dist: float = 4.0, device: str = "cuda") -> FoVPerspectiveCameras:
    """Create a camera looking at the origin from (azimuth, elevation, dist)."""
    R, T = look_at_view_transform(dist=dist, elev=elevation, azim=azimuth)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    return cameras


def render_point_cloud(points_3d: torch.Tensor,
                       features: torch.Tensor = None,
                       azimuth: float = 0.0,
                       elevation: float = 30.0,
                       config: dict = None,
                       device: str = "cuda") -> torch.Tensor:
    """
    Render a 3D point cloud from a given viewpoint.

    Args:
        points_3d: (P, 3) tensor of 3D points (on device)
        features: (P, 3) per-point RGB. Defaults to white.
        azimuth: camera azimuth in degrees
        elevation: camera elevation in degrees
        config: override rendering config
        device: CUDA device

    Returns:
        (H, W, 4) RGBA image tensor (differentiable w.r.t. points_3d)
    """
    cfg = {**DEFAULT_RENDER_CONFIG, **(config or {})}

    if features is None:
        features = torch.ones_like(points_3d)  # white

    cameras = make_cameras(azimuth, elevation, cfg["dist"], device)

    raster_settings = PointsRasterizationSettings(
        image_size=cfg["image_size"],
        radius=cfg["radius"],
        points_per_pixel=cfg["points_per_pixel"],
    )

    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    compositor = AlphaCompositor() if cfg["compositor"] == "alpha" else NormWeightedCompositor()
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor)

    point_cloud = Pointclouds(points=[points_3d], features=[features])
    images = renderer(point_cloud)  # (1, H, W, 4)
    return images[0]


def render_silhouette(points_3d: torch.Tensor,
                      azimuth: float = 0.0,
                      elevation: float = 30.0,
                      config: dict = None,
                      device: str = "cuda") -> torch.Tensor:
    """
    Render just the alpha/silhouette channel. Useful as a fallback
    loss if RGB gradients are too noisy.

    Returns:
        (H, W) silhouette image
    """
    image = render_point_cloud(points_3d, azimuth=azimuth, elevation=elevation,
                               config=config, device=device)
    return image[..., 3]  # alpha channel


# =============================================================================
# Radius sweep utility — run this first to find your sweet spot
# =============================================================================
def sweep_radius(points_3d: torch.Tensor, radii: list = None, device: str = "cuda"):
    """
    Render the same point cloud at different radii and report
    gradient norms. Use this to find the best radius.

    Prints a table: radius | grad_norm | nonzero_pixels | image_mean

    Args:
        points_3d: (P, 3) detached point cloud
        radii: list of radius values to test
    """
    if radii is None:
        radii = [0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.1]

    print(f"\n{'='*65}")
    print(f"  RADIUS SWEEP — Finding optimal rendering radius")
    print(f"  Points: {points_3d.shape[0]}")
    print(f"{'='*65}")
    print(f"  {'Radius':>8}  {'Grad Norm':>12}  {'NonZero Px':>12}  {'Img Mean':>10}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*10}")

    for r in radii:
        pts = points_3d.clone().detach().requires_grad_(True)
        cfg = {"radius": r}
        img = render_point_cloud(pts, azimuth=45.0, config=cfg, device=device)
        loss = img[..., :3].sum()
        loss.backward()

        grad_norm = pts.grad.norm().item()
        nonzero = (img[..., 3] > 0.01).sum().item()
        img_mean = img[..., :3].mean().item()

        print(f"  {r:>8.4f}  {grad_norm:>12.4f}  {nonzero:>12d}  {img_mean:>10.6f}")

    print(f"{'='*65}")
    print("  Pick the radius where grad_norm is large AND nonzero_px is reasonable.")
    print("  Too many nonzero pixels = blurry. Too few = vanishing gradients.\n")
