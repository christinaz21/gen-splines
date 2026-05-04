"""
generate_training_data.py — Create training data for the generative spline prior.

Matches the paper's approach:
  - Generate diverse synthetic spline scenes (control points)
  - Render each scene from multiple viewpoints via PyTorch3D
  - Both control points AND rendered images are saved as training targets

The generator Gθ(z) will be trained to produce control points whose
renderings match these target images.

Usage:
    python generate_training_data.py --num-scenes 2000 --output-dir outputs/training_data
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm


def generate_scene(seed: int, N: int = 40, K: int = 8) -> torch.Tensor:
    """
    Generate a single random scene with N curves, K control points each.

    Returns:
        (N, K, 3) control points
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    r = torch.rand(1, generator=rng).item()
    if r < 0.33:
        n_helix, n_wave, n_spiral = N // 2, N // 4, N - N // 2 - N // 4
    elif r < 0.66:
        n_helix, n_wave, n_spiral = N // 4, N // 2, N - N // 4 - N // 2
    else:
        n_helix, n_wave, n_spiral = N // 3, N // 3, N - 2 * (N // 3)

    global_spread = 0.3 + 0.7 * torch.rand(1, generator=rng).item()
    global_height = 0.5 + 0.5 * torch.rand(1, generator=rng).item()
    noise_scale = 0.02 + 0.08 * torch.rand(1, generator=rng).item()

    curves = []

    # Helix curves
    for i in range(n_helix):
        radius = 0.03 + 0.25 * torch.rand(1, generator=rng).item()
        turns = 1.0 + 5.0 * torch.rand(1, generator=rng).item()
        cx = global_spread * (torch.rand(1, generator=rng).item() - 0.5)
        cy = global_spread * (torch.rand(1, generator=rng).item() - 0.5)
        angle_off = 2 * np.pi * torch.rand(1, generator=rng).item()

        t = torch.linspace(0, 1, K)
        angles = 2 * np.pi * turns * t + angle_off
        x = cx + radius * torch.cos(angles)
        y = cy + radius * torch.sin(angles)
        z = torch.linspace(-global_height, global_height, K)
        z = z + noise_scale * torch.randn(K, generator=rng)
        curves.append(torch.stack([x, y, z], dim=-1))

    # Wave curves
    for i in range(n_wave):
        freq = 0.5 + 4.0 * torch.rand(1, generator=rng).item()
        amp = 0.03 + 0.2 * torch.rand(1, generator=rng).item()
        phase = 2 * np.pi * torch.rand(1, generator=rng).item()
        offset = global_spread * (torch.rand(1, generator=rng).item() - 0.5)
        axis = int(torch.randint(0, 3, (1,), generator=rng).item())

        t = torch.linspace(0, 1, K)
        primary = torch.linspace(-global_height, global_height, K)
        secondary = offset + amp * torch.sin(2 * np.pi * freq * t + phase)
        tertiary = noise_scale * torch.randn(K, generator=rng)

        if axis == 0:
            curves.append(torch.stack([primary, secondary, tertiary], dim=-1))
        elif axis == 1:
            curves.append(torch.stack([secondary, primary, tertiary], dim=-1))
        else:
            curves.append(torch.stack([tertiary, secondary, primary], dim=-1))

    # Spiral curves
    for i in range(n_spiral):
        r_start = 0.02 + 0.1 * torch.rand(1, generator=rng).item()
        r_end = r_start + 0.05 + 0.2 * torch.rand(1, generator=rng).item()
        turns = 1.5 + 3.0 * torch.rand(1, generator=rng).item()
        cx = global_spread * (torch.rand(1, generator=rng).item() - 0.5)
        cy = global_spread * (torch.rand(1, generator=rng).item() - 0.5)

        t = torch.linspace(0, 1, K)
        radii = r_start + (r_end - r_start) * t
        angles = 2 * np.pi * turns * t
        x = cx + radii * torch.cos(angles)
        y = cy + radii * torch.sin(angles)
        z = torch.linspace(-global_height * 0.5, global_height * 0.5, K)
        z = z + noise_scale * torch.randn(K, generator=rng)
        curves.append(torch.stack([x, y, z], dim=-1))

    cp = torch.stack(curves)

    # Random global rotation
    angle = 0.3 * torch.randn(1, generator=rng).item()
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]],
                        dtype=torch.float32)
    cp = torch.einsum('nkd,dD->nkD', cp, rot)
    return cp


def generate_dataset(num_scenes: int, N: int = 40, K: int = 8,
                      seed_offset: int = 0) -> torch.Tensor:
    """
    Generate a full dataset of diverse spline scenes.

    Returns:
        (num_scenes, N, K, 3) control points
    """
    scenes = []
    for i in tqdm(range(num_scenes), desc="Generating scenes"):
        scene = generate_scene(seed=seed_offset + i, N=N, K=K)
        scenes.append(scene)
    return torch.stack(scenes)


def render_dataset_views(all_cp: torch.Tensor, num_views: int = 8,
                          samples_per_curve: int = 128, radius: float = 0.005,
                          image_size: int = 128, device: str = "cuda") -> torch.Tensor:
    """
    Render each scene from multiple viewpoints.

    As per the paper: "We utilize PyTorch3D to... differentiably render the
    spline field from camera pose πm into a 2D feature map."

    Args:
        all_cp: (S, N, K, 3) all scenes' control points
        num_views: viewpoints per scene
        samples_per_curve: B-spline sampling density
        radius: rendering radius
        image_size: render resolution (smaller for training efficiency)
        device: GPU

    Returns:
        (S, num_views, H, W, 3) rendered images
    """
    from spline import evaluate_bspline
    from renderer import render_point_cloud

    S = all_cp.shape[0]
    azimuths = torch.linspace(0, 360, num_views + 1)[:-1]
    config = {"radius": radius, "image_size": image_size}

    all_images = []
    for i in tqdm(range(S), desc="Rendering scenes"):
        cp = all_cp[i].to(device)  # (N, K, 3)
        pts = evaluate_bspline(cp.unsqueeze(0), samples_per_curve).reshape(-1, 3)

        scene_images = []
        for az in azimuths:
            with torch.no_grad():
                img = render_point_cloud(pts, azimuth=az.item(),
                                          config=config, device=device)[..., :3]
                scene_images.append(img.cpu())

        all_images.append(torch.stack(scene_images))

    return torch.stack(all_images)  # (S, V, H, W, 3)


def compute_dataset_statistics(data: torch.Tensor) -> dict:
    """Compute normalization statistics."""
    flat = data.reshape(-1, 3)
    return {
        "mean": flat.mean(dim=0),
        "std": flat.std(dim=0),
        "min": flat.min(dim=0).values,
        "max": flat.max(dim=0).values,
        "global_std": flat.std().item(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-train", type=int, default=2000)
    parser.add_argument("--num-val", type=int, default=200)
    parser.add_argument("--num-test", type=int, default=100)
    parser.add_argument("--N", type=int, default=40)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--num-views", type=int, default=8,
                        help="Viewpoints per scene for rendering targets")
    parser.add_argument("--render-size", type=int, default=128,
                        help="Render resolution (smaller = faster training)")
    parser.add_argument("--radius", type=float, default=0.005)
    parser.add_argument("--samples-per-curve", type=int, default=128)
    parser.add_argument("--render", action="store_true",
                        help="Also render images (slower but needed for rendering loss)")
    parser.add_argument("--output-dir", type=str, default="outputs/training_data")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating {args.num_train} train + {args.num_val} val + "
          f"{args.num_test} test scenes")
    print(f"  Each: {args.N} curves x {args.K} CP x 3D")

    train_cp = generate_dataset(args.num_train, args.N, args.K, seed_offset=0)
    val_cp = generate_dataset(args.num_val, args.N, args.K, seed_offset=100000)
    test_cp = generate_dataset(args.num_test, args.N, args.K, seed_offset=200000)

    print(f"\n  Train: {train_cp.shape}")
    print(f"  Val:   {val_cp.shape}")
    print(f"  Test:  {test_cp.shape}")

    stats = compute_dataset_statistics(train_cp)
    print(f"\n  Stats — mean: {stats['mean']}, std: {stats['std']}, "
          f"global_std: {stats['global_std']:.4f}")

    torch.save(train_cp, os.path.join(args.output_dir, "train_cp.pt"))
    torch.save(val_cp, os.path.join(args.output_dir, "val_cp.pt"))
    torch.save(test_cp, os.path.join(args.output_dir, "test_cp.pt"))
    torch.save(stats, os.path.join(args.output_dir, "stats.pt"))

    # Render target images if requested
    if args.render:
        print(f"\n  Rendering {args.num_views} views per scene at {args.render_size}x{args.render_size}...")
        train_imgs = render_dataset_views(
            train_cp, args.num_views, args.samples_per_curve,
            args.radius, args.render_size, args.device
        )
        val_imgs = render_dataset_views(
            val_cp, args.num_views, args.samples_per_curve,
            args.radius, args.render_size, args.device
        )
        torch.save(train_imgs, os.path.join(args.output_dir, "train_images.pt"))
        torch.save(val_imgs, os.path.join(args.output_dir, "val_images.pt"))
        print(f"  Train images: {train_imgs.shape}")
        print(f"  Val images:   {val_imgs.shape}")

        # Save render config for training
        render_config = {
            "num_views": args.num_views,
            "render_size": args.render_size,
            "radius": args.radius,
            "samples_per_curve": args.samples_per_curve,
            "azimuths": torch.linspace(0, 360, args.num_views + 1)[:-1].tolist(),
        }
        torch.save(render_config, os.path.join(args.output_dir, "render_config.pt"))

    print(f"\n  Saved to {args.output_dir}/")
    print("Done!")
