"""
dataset.py — Generate synthetic strand scenes with known ground-truth control points.

Scenes consist of helix-like and wave-like 3D curves that mimic thin structures
(hair strands, wires, fibers). Ground-truth control points are saved so we can
measure control-point drift after optimization.
"""

import torch
import numpy as np
import os
import json
from spline import SplineField, evaluate_bspline
from renderer import render_point_cloud


def create_helix_strands(num_curves: int = 30, K: int = 8, seed: int = 42,
                          spread: float = 0.8, helix_turns: float = 3.0) -> torch.Tensor:
    """
    Create a ground-truth scene of helix-like 3D strands.

    Args:
        num_curves: number of curves
        K: control points per curve
        seed: random seed for reproducibility
        spread: how spread out the curves are (spatial extent)
        helix_turns: number of full helix rotations

    Returns:
        (num_curves, K, 3) control points tensor
    """
    torch.manual_seed(seed)
    control_points = []

    for i in range(num_curves):
        # Random helix parameters
        radius = 0.05 + 0.2 * torch.rand(1).item()
        center_x = spread * (torch.rand(1).item() - 0.5)
        center_y = spread * (torch.rand(1).item() - 0.5)
        angle_offset = 2 * np.pi * torch.rand(1).item()

        t_vals = torch.linspace(0, 1, K)
        angles = 2 * np.pi * helix_turns * t_vals + angle_offset

        x = center_x + radius * torch.cos(angles)
        y = center_y + radius * torch.sin(angles)
        z = torch.linspace(-0.8, 0.8, K) + 0.05 * torch.randn(K)

        cp = torch.stack([x, y, z], dim=-1)  # (K, 3)
        control_points.append(cp)

    return torch.stack(control_points)  # (N, K, 3)


def create_wave_strands(num_curves: int = 20, K: int = 8, seed: int = 123) -> torch.Tensor:
    """
    Create a ground-truth scene of sinusoidal wave strands.
    Complements helix strands with a different topology.
    """
    torch.manual_seed(seed)
    control_points = []

    for i in range(num_curves):
        freq = 1.0 + 3.0 * torch.rand(1).item()
        amp = 0.05 + 0.15 * torch.rand(1).item()
        phase = 2 * np.pi * torch.rand(1).item()
        offset_y = 0.8 * (torch.rand(1).item() - 0.5)

        t_vals = torch.linspace(0, 1, K)
        x = torch.linspace(-0.8, 0.8, K)
        y = offset_y + amp * torch.sin(2 * np.pi * freq * t_vals + phase)
        z = 0.05 * torch.randn(K)

        cp = torch.stack([x, y, z], dim=-1)
        control_points.append(cp)

    return torch.stack(control_points)


def create_combined_scene(num_helix: int = 20, num_wave: int = 15, K: int = 8,
                           seed: int = 42) -> torch.Tensor:
    """Create a mixed scene with both helix and wave strands."""
    helix = create_helix_strands(num_helix, K, seed=seed)
    wave = create_wave_strands(num_wave, K, seed=seed + 100)
    return torch.cat([helix, wave], dim=0)


def render_360_dataset(gt_control_points: torch.Tensor,
                        num_views: int = 36,
                        samples_per_curve: int = 128,
                        image_size: int = 256,
                        radius: float = 0.02,
                        device: str = "cuda",
                        save_dir: str = None) -> tuple:
    """
    Render ground-truth splines from 360 degrees of azimuth.

    Args:
        gt_control_points: (N, K, 3) ground-truth control points
        num_views: number of camera viewpoints
        samples_per_curve: points sampled along each curve for rendering
        image_size: rendered image resolution
        radius: rendering radius
        device: compute device
        save_dir: if provided, save images as PNGs

    Returns:
        images: (num_views, H, W, 4) RGBA images
        azimuths: (num_views,) azimuth angles in degrees
        gt_points: (P, 3) the ground-truth point cloud
    """
    N, K, D = gt_control_points.shape

    # Evaluate GT curves into point cloud
    gt_points = evaluate_bspline(gt_control_points.to(device), samples_per_curve)
    gt_points_flat = gt_points.reshape(-1, 3)

    config = {"image_size": image_size, "radius": radius}
    azimuths = torch.linspace(0, 360, num_views + 1)[:-1]

    images = []
    for i, az in enumerate(azimuths):
        img = render_point_cloud(gt_points_flat, azimuth=az.item(),
                                  config=config, device=device)
        images.append(img.detach().cpu())

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            import imageio
            img_np = (img.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(save_dir, f"view_{i:03d}_az{az.item():.0f}.png"),
                           img_np)

    images = torch.stack(images)
    return images, azimuths, gt_points_flat.detach().cpu()


def save_scene(gt_control_points: torch.Tensor, path: str):
    """Save ground-truth control points."""
    torch.save(gt_control_points, path)
    print(f"Saved GT control points: {gt_control_points.shape} -> {path}")


def load_scene(path: str) -> torch.Tensor:
    """Load ground-truth control points."""
    cp = torch.load(path, weights_only=True)
    print(f"Loaded GT control points: {cp.shape} <- {path}")
    return cp


# =============================================================================
# Quick dataset generation script
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic strand dataset")
    parser.add_argument("--num-helix", type=int, default=25)
    parser.add_argument("--num-wave", type=int, default=15)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--num-views", type=int, default=36)
    parser.add_argument("--samples-per-curve", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--radius", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/dataset")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("Creating combined scene...")
    gt_cp = create_combined_scene(args.num_helix, args.num_wave, args.K, args.seed)
    print(f"  {gt_cp.shape[0]} curves, {args.K} control points each")

    os.makedirs(args.output_dir, exist_ok=True)
    save_scene(gt_cp, os.path.join(args.output_dir, "gt_control_points.pt"))

    print(f"\nRendering {args.num_views} views...")
    images, azimuths, gt_points = render_360_dataset(
        gt_cp, args.num_views, args.samples_per_curve,
        args.image_size, args.radius, args.device,
        save_dir=os.path.join(args.output_dir, "images")
    )
    print(f"  Saved {len(images)} images to {args.output_dir}/images/")

    # Save metadata
    meta = {
        "num_curves": gt_cp.shape[0],
        "K": args.K,
        "num_views": args.num_views,
        "samples_per_curve": args.samples_per_curve,
        "image_size": args.image_size,
        "radius": args.radius,
        "seed": args.seed,
        "azimuths": azimuths.tolist(),
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved metadata to {args.output_dir}/meta.json")
    print("Done!")
