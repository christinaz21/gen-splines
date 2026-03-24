"""
train_generator.py — Train Gθ(z) following the paper's pipeline.

Slide 11 pipeline:
  1. Latent Vector z
  2. → Spline Generator (Produce 3D Curves)
  3. → Rendered Image via PyTorch3D
  4. → Reconstruction Loss (Silhouette or RGB Error)
  5. → Backpropagate Gradients
  6. → Refine Control Points (Update Spline Parameters)
  7. → Fixed Topology Memory Updates (Preserve Curve Structure)

Training uses auto-decoder (like cited DeepSDF):
  - Per-scene latent codes z_i, optimized jointly with Gθ
  - Gθ generates control points per-curve (CHARM-compatible)
  - Loss: CP reconstruction + smoothness + latent regularization
  - Rendering via PyTorch3D used for evaluation visualization

After training:
  - z ~ N(0, I) → Gθ(z) → novel valid spline scene
  - Generated scenes can initialize persistent curve memory

Usage:
    python generate_training_data.py --num-train 2000 --output-dir outputs/training_data
    python train_generator.py --data-dir outputs/training_data --epochs 500 --output-dir outputs/generator
"""

import os
import time
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from generator import SplineGenerator, LatentCodes
from spline import evaluate_bspline
from losses import smoothness_loss


def train(args):
    device = args.device
    print(f"\n{'='*65}")
    print(f"  Training Gθ(z) — Per-Curve Spline Generator")
    print(f"  Pipeline: z → Gθ → curves → [render] → loss → backprop")
    print(f"{'='*65}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"\n  Loading data from {args.data_dir}...")
    train_cp = torch.load(os.path.join(args.data_dir, "train_cp.pt"), weights_only=True)
    val_cp = torch.load(os.path.join(args.data_dir, "val_cp.pt"), weights_only=True)
    test_cp = torch.load(os.path.join(args.data_dir, "test_cp.pt"), weights_only=True)

    S_train, N, K, D = train_cp.shape
    print(f"  Train: {train_cp.shape} ({S_train} scenes, {N} curves, {K} CP)")
    print(f"  Val:   {val_cp.shape}")

    # Normalize
    stats = torch.load(os.path.join(args.data_dir, "stats.pt"), weights_only=True)
    data_mean = stats["mean"].to(device)
    data_std = stats["std"].to(device).clamp(min=1e-6)

    def normalize(x):
        return (x.to(device) - data_mean) / data_std

    def denormalize(x):
        return x * data_std + data_mean

    train_norm = normalize(train_cp).cpu()
    val_norm = normalize(val_cp).cpu()
    test_norm = normalize(test_cp).cpu()

    # ------------------------------------------------------------------
    # 2. Model: Per-curve generator + per-scene latent codes
    # ------------------------------------------------------------------
    generator = SplineGenerator(
        latent_dim=args.latent_dim, num_curves=N, K=K, hidden_dim=args.hidden_dim
    ).to(device)

    latent_codes = LatentCodes(S_train, args.latent_dim).to(device)

    n_params_g = sum(p.numel() for p in generator.parameters())
    print(f"\n  Gθ: {n_params_g:,} parameters (per-curve decoder, shared weights)")
    print(f"  Latent codes: {S_train} × {args.latent_dim}")
    print(f"  Curve embeddings: {N} × {args.hidden_dim}")

    # Separate optimizers (higher LR for latent codes, like DeepSDF)
    opt_g = torch.optim.AdamW(generator.parameters(), lr=args.lr, weight_decay=1e-5)
    opt_z = torch.optim.Adam(latent_codes.parameters(), lr=args.lr_latent)
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=args.epochs)
    sched_z = torch.optim.lr_scheduler.CosineAnnealingLR(opt_z, T_max=args.epochs)

    # ------------------------------------------------------------------
    # 3. Training loop
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    train_loader = DataLoader(
        TensorDataset(torch.arange(S_train)),
        batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    best_val = float("inf")
    history = {"train_total": [], "train_cp": [], "train_smooth": [],
               "train_reg": [], "val_cp": [], "val_per_curve": []}

    t_start = time.time()

    for epoch in range(args.epochs):
        generator.train()
        ep = {"total": 0, "cp": 0, "smooth": 0, "reg": 0, "n": 0}

        for (batch_idx,) in train_loader:
            opt_g.zero_grad()
            opt_z.zero_grad()

            batch_idx = batch_idx.to(device)
            z = latent_codes(batch_idx)                    # (B, latent_dim)
            pred_cp = generator(z)                          # (B, N, K, 3)
            gt_cp = train_norm[batch_idx.cpu()].to(device)  # (B, N, K, 3)

            # === Reconstruction loss: per-curve CP matching ===
            # This is the core of the training pipeline.
            # Each predicted curve i should match GT curve i.
            loss_cp = F.mse_loss(pred_cp, gt_cp)

            # === Smoothness: valid B-spline curves ===
            # Penalize non-smooth control point sequences
            B_sz = pred_cp.shape[0]
            loss_smooth = torch.tensor(0.0, device=device)
            for b in range(B_sz):
                loss_smooth = loss_smooth + smoothness_loss(pred_cp[b])
            loss_smooth = loss_smooth / B_sz

            # === Latent regularization: z_i → N(0, I) ===
            loss_reg = latent_codes.regularization()

            # === Total ===
            loss = loss_cp + args.smooth_weight * loss_smooth + args.reg_weight * loss_reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(latent_codes.parameters(), 1.0)
            opt_g.step()
            opt_z.step()

            ep["total"] += loss.item()
            ep["cp"] += loss_cp.item()
            ep["smooth"] += loss_smooth.item()
            ep["reg"] += loss_reg.item()
            ep["n"] += 1

        sched_g.step()
        sched_z.step()

        for k in ["total", "cp", "smooth", "reg"]:
            ep[k] /= max(ep["n"], 1)

        history["train_total"].append(ep["total"])
        history["train_cp"].append(ep["cp"])
        history["train_smooth"].append(ep["smooth"])
        history["train_reg"].append(ep["reg"])

        # Validation: test-time latent optimization
        val_cp_loss, val_per_curve = validate(generator, val_norm, args, device)
        history["val_cp"].append(val_cp_loss)
        history["val_per_curve"].append(val_per_curve)

        if val_cp_loss < best_val:
            best_val = val_cp_loss
            torch.save(generator.state_dict(),
                       os.path.join(args.output_dir, "best_generator.pt"))
            torch.save(latent_codes.state_dict(),
                       os.path.join(args.output_dir, "best_latent_codes.pt"))

        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            z_norm = latent_codes.codes.data.norm(dim=-1).mean().item()
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch:>4d}/{args.epochs}  "
                  f"cp={ep['cp']:.5f} smooth={ep['smooth']:.5f}  "
                  f"val={val_cp_loss:.5f} val_curve={val_per_curve:.4f}  "
                  f"|z|={z_norm:.3f}  [{elapsed:.1f}s]")

    # Save final + normalization
    torch.save(generator.state_dict(), os.path.join(args.output_dir, "final_generator.pt"))
    torch.save(latent_codes.state_dict(), os.path.join(args.output_dir, "final_latent_codes.pt"))
    torch.save({"mean": data_mean.cpu(), "std": data_std.cpu()},
               os.path.join(args.output_dir, "normalization.pt"))

    # ------------------------------------------------------------------
    # 4. Evaluate
    # ------------------------------------------------------------------
    print(f"\n  --- Evaluation ---")
    generator.load_state_dict(
        torch.load(os.path.join(args.output_dir, "best_generator.pt"), weights_only=True)
    )
    generator.eval()

    results = evaluate(generator, test_norm, data_mean, data_std, args, device)

    print(f"  Test recon CP drift:       {results['test_cp_drift']:.4f}")
    print(f"  Test per-curve drift:      {results['test_per_curve_drift']:.4f}")
    print(f"  Gen spatial extent:        {results['gen_extent']:.4f} "
          f"(real: {results['real_extent']:.4f})")
    print(f"  Gen smoothness:            {results['gen_smoothness']:.6f} "
          f"(real: {results['real_smoothness']:.6f})")
    print(f"  Interpolation smoothness:  {results['interp_smoothness']:.4f}")

    # Save
    torch.save(history, os.path.join(args.output_dir, "history.pt"))
    torch.save(results, os.path.join(args.output_dir, "eval_results.pt"))

    gen_norm = generator.generate(50, device=device)
    gen_denorm = denormalize(gen_norm).cpu()
    torch.save(gen_denorm, os.path.join(args.output_dir, "generated_samples.pt"))

    save_plots(history, args.output_dir)
    render_samples(generator, data_std, data_mean, args.output_dir, device)

    total_time = time.time() - t_start
    print(f"\n  {'='*55}")
    print(f"  TRAINING COMPLETE")
    print(f"  {'='*55}")
    print(f"  Best val loss:      {best_val:.6f}")
    print(f"  Test CP drift:      {results['test_cp_drift']:.4f}")
    print(f"  Test curve drift:   {results['test_per_curve_drift']:.4f}")
    print(f"  Time:               {total_time/60:.1f} min")
    print(f"  Model:              {args.output_dir}/best_generator.pt")
    print()


# =========================================================================
# Validation & Evaluation
# =========================================================================

def validate(generator, val_norm, args, device):
    """Test-time latent optimization on val scenes (like DeepSDF inference)."""
    generator.eval()
    B = min(16, len(val_norm))
    targets = val_norm[:B].to(device)

    z = torch.randn(B, args.latent_dim, device=device, requires_grad=True)
    opt = torch.optim.Adam([z], lr=0.01)

    for _ in range(100):
        opt.zero_grad()
        pred = generator(z)
        loss = F.mse_loss(pred, targets)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred_final = generator(z)
        cp_loss = F.mse_loss(pred_final, targets).item()
        per_curve = (pred_final - targets).norm(dim=-1).mean().item()

    generator.train()
    return cp_loss, per_curve


def evaluate(generator, test_norm, data_mean, data_std, args, device):
    """Full evaluation on test set."""
    generator.eval()
    results = {}

    # 1. Test reconstruction via latent optimization (300 steps)
    target = test_norm[0:1].to(device)
    z = torch.randn(1, args.latent_dim, device=device, requires_grad=True)
    opt = torch.optim.Adam([z], lr=0.01)
    for _ in range(300):
        opt.zero_grad()
        loss = F.mse_loss(generator(z), target)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = generator(z)
        pred_real = pred * data_std + data_mean
        target_real = target * data_std + data_mean
        results["test_cp_drift"] = (pred_real - target_real).norm(dim=-1).mean().item()
        # Per-curve: mean drift for each curve
        per_curve_drift = (pred_real - target_real).norm(dim=-1).mean(dim=-1)  # (1, N)
        results["test_per_curve_drift"] = per_curve_drift.mean().item()
        results["test_worst_curve_drift"] = per_curve_drift.max().item()
        results["test_best_curve_drift"] = per_curve_drift.min().item()

    # 2. Generated scene quality
    with torch.no_grad():
        gen = generator.generate(50, device=device)
        gen_real = gen * data_std + data_mean

        gen_extent = gen_real.reshape(50, -1, 3).std(dim=1).mean().item()
        real_data = test_norm[:50].to(device) * data_std + data_mean
        real_extent = real_data.reshape(min(50, len(test_norm)), -1, 3).std(dim=1).mean().item()
        results["gen_extent"] = gen_extent
        results["real_extent"] = real_extent

        d1 = gen_real[:, :, 1:] - gen_real[:, :, :-1]
        d2 = d1[:, :, 1:] - d1[:, :, :-1]
        results["gen_smoothness"] = (d2 ** 2).mean().item()

        d1r = real_data[:, :, 1:] - real_data[:, :, :-1]
        d2r = d1r[:, :, 1:] - d1r[:, :, :-1]
        results["real_smoothness"] = (d2r ** 2).mean().item()

    # 3. Interpolation
    with torch.no_grad():
        z1 = torch.randn(1, args.latent_dim, device=device)
        z2 = torch.randn(1, args.latent_dim, device=device)
        interp = torch.cat([generator((1 - a) * z1 + a * z2)
                            for a in torch.linspace(0, 1, 5, device=device)], dim=0)
        diffs = [(interp[i+1] - interp[i]).norm(dim=-1).mean().item()
                 for i in range(len(interp) - 1)]
        results["interp_smoothness"] = sum(diffs) / len(diffs)

    return results


# =========================================================================
# Plotting & Rendering
# =========================================================================

def save_plots(history, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(history["train_cp"], label="Train CP", alpha=0.8)
        axes[0].plot(history["val_cp"], label="Val CP (test-time opt)", alpha=0.8)
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE")
        axes[0].set_title("CP Reconstruction"); axes[0].set_yscale("log"); axes[0].legend()

        axes[1].plot(history["val_per_curve"], label="Val per-curve drift", alpha=0.8)
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("L2 Drift")
        axes[1].set_title("Per-Curve Drift"); axes[1].legend()

        axes[2].plot(history["train_reg"], label="Latent |z|²", alpha=0.8)
        axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Reg")
        axes[2].set_title("Latent Regularization"); axes[2].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
        plt.close()
        print(f"  Saved plots to {output_dir}/training_curves.png")
    except Exception as e:
        print(f"  Could not save plots: {e}")


def render_samples(generator, data_std, data_mean, output_dir, device):
    """Render generated scenes via PyTorch3D (Slide 11: render step)."""
    try:
        from renderer import render_point_cloud
        import imageio
        import numpy as np

        config = {"radius": 0.005, "image_size": 256}
        render_dir = os.path.join(output_dir, "renders")
        os.makedirs(render_dir, exist_ok=True)

        gen = generator.generate(8, device=device)
        gen_real = gen * data_std + data_mean

        for i in range(len(gen_real)):
            cp = gen_real[i]  # (N, K, 3)
            pts = evaluate_bspline(cp, 64).reshape(-1, 3)
            img = render_point_cloud(pts, azimuth=45.0, config=config, device=device)
            img_np = (img[..., :3].cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(render_dir, f"gen_{i:03d}.png"), img_np)

        print(f"  Saved renders to {render_dir}/")
    except Exception as e:
        print(f"  Could not render: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Gθ(z) — per-curve generator")
    parser.add_argument("--data-dir", type=str, default="outputs/training_data")
    # Model
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    # Training
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-latent", type=float, default=1e-2)
    # Loss
    parser.add_argument("--smooth-weight", type=float, default=0.001)
    parser.add_argument("--reg-weight", type=float, default=0.001)
    # Output
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--output-dir", type=str, default="outputs/generator")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    train(args)
