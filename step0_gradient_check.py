"""
step0_gradient_check.py — THE FIRST THING YOU RUN.

This script verifies that:
  1. PyTorch3D is installed and working
  2. Gradients flow from rendered pixels back to control points
  3. Which radius values produce useful gradient signal

If this script fails, STOP and fix your setup before proceeding.
Expected runtime: ~30 seconds on a single GPU.
"""

import sys
import torch
import time

def check_imports():
    """Check all critical imports."""
    print("=" * 60)
    print("  STEP 0a: Import Check")
    print("=" * 60)

    checks = []

    try:
        import pytorch3d
        print(f"  [OK] PyTorch3D {pytorch3d.__version__}")
        checks.append(True)
    except ImportError as e:
        print(f"  [FAIL] PyTorch3D: {e}")
        checks.append(False)

    try:
        import torch
        print(f"  [OK] PyTorch {torch.__version__}")
        print(f"  [OK] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  [OK] GPU: {torch.cuda.get_device_name(0)}")
            print(f"  [OK] CUDA version: {torch.version.cuda}")
        checks.append(True)
    except Exception as e:
        print(f"  [FAIL] PyTorch: {e}")
        checks.append(False)

    try:
        from pytorch3d.renderer import (
            PointsRasterizationSettings, PointsRenderer,
            PointsRasterizer, AlphaCompositor,
            look_at_view_transform, FoVPerspectiveCameras,
        )
        from pytorch3d.structures import Pointclouds
        print("  [OK] PyTorch3D renderer components")
        checks.append(True)
    except ImportError as e:
        print(f"  [FAIL] PyTorch3D renderer: {e}")
        checks.append(False)

    if not all(checks):
        print("\n  CRITICAL: Some imports failed. Fix before proceeding.")
        sys.exit(1)
    print()


def gradient_flow_test(device="cuda"):
    """
    Core test: can we backpropagate from a rendered image to 3D control points?
    """
    print("=" * 60)
    print("  STEP 0b: Gradient Flow Test")
    print("=" * 60)

    from spline import SplineField
    from renderer import render_point_cloud

    # Create a small spline field
    field = SplineField(num_curves=10, control_points_per_curve=8).to(device)
    print(f"  Created SplineField: {field.num_curves} curves, {field.K} CP each")
    print(f"  Control points shape: {field.control_points.shape}")
    print(f"  Control points requires_grad: {field.control_points.requires_grad}")

    # Sample points
    points = field(num_samples_per_curve=64)
    print(f"  Sampled point cloud: {points.shape}")

    # Render
    image = render_point_cloud(points, azimuth=45.0, device=device)
    print(f"  Rendered image: {image.shape}")
    print(f"  Image range: [{image.min().item():.4f}, {image.max().item():.4f}]")
    # Detect nonzero pixels (works whether output is RGB or RGBA)
    if image.shape[-1] == 4:
        nonzero_px = (image[..., 3] > 0.01).sum().item()
    else:
        nonzero_px = (image.sum(dim=-1) > 0.01).sum().item()
    print(f"  Nonzero pixels: {nonzero_px}")

    # Backward pass
    loss = image[..., :3].mean()
    loss.backward()

    grad = field.control_points.grad
    if grad is None:
        print("\n  [FAIL] Gradients are None! The rendering pipeline is NOT differentiable.")
        print("         This is a showstopper — check PyTorch3D installation.")
        sys.exit(1)

    grad_norm = grad.norm().item()
    grad_nonzero = (grad.abs() > 1e-10).sum().item()
    total_params = grad.numel()

    print(f"\n  Gradient norm: {grad_norm:.6f}")
    print(f"  Nonzero gradient entries: {grad_nonzero}/{total_params} "
          f"({100*grad_nonzero/total_params:.1f}%)")

    if grad_norm < 1e-8:
        print("\n  [WARNING] Gradient norm is extremely small!")
        print("  Possible causes:")
        print("    - Radius too small (points not visible)")
        print("    - Points are outside camera frustum")
        print("    - Numerical issues")
        print("  -> Run the radius sweep below to diagnose.")
    elif grad_norm > 0:
        print(f"\n  [OK] Gradients are flowing! Norm = {grad_norm:.6f}")
    print()

    return grad_norm > 1e-8


def radius_sweep_test(device="cuda"):
    """
    Sweep the rendering radius and report gradient quality.
    This finds the sweet spot for your specific scene.
    """
    print("=" * 60)
    print("  STEP 0c: Radius Sweep")
    print("=" * 60)

    from spline import SplineField
    from renderer import render_point_cloud

    # Create reference scene
    field = SplineField(num_curves=20, control_points_per_curve=8).to(device)
    with torch.no_grad():
        ref_points = field(64)

    radii = [0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.08, 0.1]

    print(f"\n  {'Radius':>8}  {'GradNorm':>10}  {'GradNZ%':>8}  "
          f"{'AlphaPx':>8}  {'ImgMean':>8}  {'Verdict':>10}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")

    best_radius = None
    best_score = -1

    for r in radii:
        # Fresh points with gradients
        pts = ref_points.clone().detach().requires_grad_(True)
        cfg = {"radius": r, "image_size": 256}
        img = render_point_cloud(pts, azimuth=45.0, config=cfg, device=device)
        loss = img[..., :3].mean()
        loss.backward()

        gn = pts.grad.norm().item()
        gnz = (pts.grad.abs() > 1e-10).sum().item() / pts.grad.numel() * 100
        if img.shape[-1] == 4:
            alpha_px = (img[..., 3] > 0.01).sum().item()
        else:
            alpha_px = (img.sum(dim=-1) > 0.01).sum().item()
        img_mean = img[..., :3].mean().item()

        # Score: we want reasonable gradient norm without too much blur
        # Heuristic: grad_norm * (1 - alpha_coverage)
        coverage = alpha_px / (256 * 256)
        score = gn * max(0, 1.0 - coverage * 2)  # penalize heavy coverage

        if gn < 1e-6:
            verdict = "TOO SMALL"
        elif coverage > 0.7:
            verdict = "TOO BIG"
        elif gn > 0.001:
            verdict = "GOOD"
            if score > best_score:
                best_score = score
                best_radius = r
        else:
            verdict = "WEAK"

        print(f"  {r:>8.4f}  {gn:>10.6f}  {gnz:>7.1f}%  "
              f"{alpha_px:>8d}  {img_mean:>8.5f}  {verdict:>10}")

    if best_radius is not None:
        print(f"\n  >>> RECOMMENDED RADIUS: {best_radius}")
        print(f"  Use this in your rendering config and optimization runs.")
    else:
        print(f"\n  [WARNING] No good radius found! Check that your scene is visible.")
    print()
    return best_radius


def optimization_sanity_check(device="cuda"):
    """
    Final check: can we actually reduce a loss by optimizing control points?
    Runs 50 steps of gradient descent and checks that loss decreases.
    """
    print("=" * 60)
    print("  STEP 0d: Optimization Sanity Check (50 steps)")
    print("=" * 60)

    from spline import SplineField
    from renderer import render_point_cloud

    # Create GT and render a target
    gt_field = SplineField(num_curves=10, control_points_per_curve=8).to(device)
    with torch.no_grad():
        gt_points = gt_field(64)
        target = render_point_cloud(gt_points, azimuth=0.0, device=device)[..., :3].detach()

    # Create perturbed prediction
    pred_field = SplineField(num_curves=10, control_points_per_curve=8).to(device)
    pred_field.control_points.data = gt_field.control_points.data + 0.15 * torch.randn_like(
        gt_field.control_points.data
    )
    initial_drift = (pred_field.control_points.data - gt_field.control_points.data
                     ).norm(dim=-1).mean().item()
    print(f"  Initial CP drift: {initial_drift:.4f}")

    optimizer = torch.optim.Adam(pred_field.parameters(), lr=1e-3)

    losses = []
    for step in range(50):
        optimizer.zero_grad()
        pred_points = pred_field(64)
        pred_image = render_point_cloud(pred_points, azimuth=0.0, device=device)[..., :3]
        loss = torch.nn.functional.mse_loss(pred_image, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    final_drift = (pred_field.control_points.data - gt_field.control_points.data
                   ).norm(dim=-1).mean().item()

    print(f"  Loss: {losses[0]:.6f} -> {losses[-1]:.6f} "
          f"({'DECREASED' if losses[-1] < losses[0] else 'DID NOT DECREASE'})")
    print(f"  CP drift: {initial_drift:.4f} -> {final_drift:.4f} "
          f"({'DECREASED' if final_drift < initial_drift else 'DID NOT DECREASE'})")

    if losses[-1] < losses[0] and final_drift < initial_drift:
        print("\n  [OK] Optimization is working! Control points are being refined.")
        print("       You are clear to proceed with the full project.")
    elif losses[-1] < losses[0]:
        print("\n  [PARTIAL] Loss decreased but CP drift didn't.")
        print("  The renderer may be finding a different local minimum.")
        print("  This is expected — proceed with caution.")
    else:
        print("\n  [FAIL] Optimization is not reducing loss.")
        print("  Check radius, learning rate, and scene setup.")

    print()
    return losses[-1] < losses[0]


# =============================================================================
# Main — run all checks in sequence
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  GENERATIVE SPLINE FIELDS — PRE-FLIGHT CHECK")
    print("  Run this BEFORE anything else in the project.")
    print("=" * 60 + "\n")

    t_start = time.time()

    # Step 0a: imports
    check_imports()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("  [WARNING] Running on CPU — this will be slow but should work.\n")

    # Step 0b: gradient flow
    grad_ok = gradient_flow_test(device)

    # Step 0c: radius sweep
    best_radius = radius_sweep_test(device)

    # Step 0d: optimization sanity check
    opt_ok = optimization_sanity_check(device)

    # Summary
    elapsed = time.time() - t_start
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Gradients flowing:  {'YES' if grad_ok else 'NO'}")
    print(f"  Best radius:        {best_radius}")
    print(f"  Optimization works: {'YES' if opt_ok else 'NO'}")
    print(f"  Total time:         {elapsed:.1f}s")
    print()

    if grad_ok and opt_ok:
        print("  ALL CHECKS PASSED. You are clear to proceed.")
        print("  Next steps:")
        print("    1. python dataset.py            # Generate synthetic data")
        print("    2. python optimize.py            # Single-view optimization")
        print("    3. python optimize_sequential.py  # Multi-view persistent memory")
    else:
        print("  SOME CHECKS FAILED. Fix issues before proceeding.")
    print()
