"""
evaluate_external_targets.py — Fairer evaluation against shared external targets.

Why this exists:
  The point-cloud baseline can look very strong when evaluated against the exact
  sampled points it optimizes. This script evaluates both spline and point-cloud
  outputs against the same external references:

  1) Approximate Chamfer to dense raw strand points from .hair data.
  2) Render MSE on held-out (offset) camera azimuths.

Inputs:
  - Spline run results: run_dense.py output opt_results.pt
  - Point baseline results: run_pointcloud_baseline.py output point_baseline_results.pt
  - Hair model name / data dir for loading raw strands

Outputs:
  - external_eval.json
  - external_eval.md
  - external_eval_curves.png
"""

import argparse
import json
import os
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

from hair_loader import load_hair_file, subsample_strands
from renderer import render_point_cloud
from spline import evaluate_bspline


def load_torch(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def orient_pts(points):
    out = points.clone()
    new_y = out[..., 2].clone()
    new_z = -out[..., 1].clone()
    out[..., 1] = new_y
    out[..., 2] = new_z
    return out


def normalize_points(points):
    points = points.clone()
    c = points.mean(dim=0)
    points = points - c
    md = points.norm(dim=-1).max()
    if md > 1e-6:
        points = points / md
    return points


def dense_raw_hair_points(hair_path, num_strands=3000, max_pts_per_strand=48, seed=99):
    strands = load_hair_file(hair_path)
    sel = subsample_strands(
        strands, num_curves=num_strands, strategy="random", seed=seed, min_length=5
    )
    pts = []
    for s in sel:
        m = len(s)
        if m > max_pts_per_strand:
            idx = np.linspace(0, m - 1, max_pts_per_strand, dtype=int)
            s = s[idx]
        pts.append(s)
    raw = torch.tensor(np.concatenate(pts, axis=0), dtype=torch.float32)
    raw = normalize_points(raw)
    raw = orient_pts(raw)
    return raw


def sample_for_chamfer(points, max_points, seed):
    if points.shape[0] <= max_points:
        return points
    g = torch.Generator(device=points.device)
    g.manual_seed(seed)
    idx = torch.randperm(points.shape[0], generator=g, device=points.device)[:max_points]
    return points[idx]


def approx_chamfer(a, b):
    # Symmetric Chamfer on sampled sets.
    d = torch.cdist(a.unsqueeze(0), b.unsqueeze(0), p=2)[0]
    return (d.min(dim=1).values.mean() + d.min(dim=0).values.mean()).item()


def heldout_azimuths(num_train_views, num_eval):
    # Offset half-step from train grid to create interleaved held-out views.
    step = 360.0 / num_train_views
    base = np.linspace(0, 360.0, num_eval, endpoint=False)
    return (base + step * 0.5).tolist()


def render_mse_vs_raw(
    raw_points,
    pred_points,
    azimuths,
    image_size,
    radius,
    device,
    bin_size=0,
    max_points_per_bin=None,
):
    cfg = {
        "image_size": image_size,
        "radius": radius,
        "points_per_pixel": 8,
        "dist": 4.0,
        "fov": 60.0,
        "compositor": "alpha",
        "bin_size": bin_size,
    }
    if max_points_per_bin is not None:
        cfg["max_points_per_bin"] = max_points_per_bin
    losses = []
    with torch.no_grad():
        for az in azimuths:
            gt = render_point_cloud(raw_points, azimuth=float(az), elevation=30.0, config=cfg, device=device)[
                ..., :3
            ]
            pr = render_point_cloud(pred_points, azimuth=float(az), elevation=30.0, config=cfg, device=device)[
                ..., :3
            ]
            losses.append(F.mse_loss(pr, gt).item())
    return float(np.mean(losses)), losses


def render_visual_triptych(raw_points, spline_points, point_points, azimuth, args):
    cfg = {
        "image_size": args.visual_image_size,
        "radius": args.visual_radius,
        "points_per_pixel": args.visual_points_per_pixel,
        "dist": 4.0,
        "fov": 60.0,
        "compositor": "alpha",
        "bin_size": args.bin_size,
    }
    if args.max_points_per_bin is not None:
        cfg["max_points_per_bin"] = args.max_points_per_bin

    with torch.no_grad():
        gt = render_point_cloud(raw_points, azimuth=float(azimuth), elevation=30.0, config=cfg, device=args.device)[
            ..., :3
        ].cpu().numpy()
        sp = render_point_cloud(
            spline_points, azimuth=float(azimuth), elevation=30.0, config=cfg, device=args.device
        )[..., :3].cpu().numpy()
        pc = render_point_cloud(
            point_points, azimuth=float(azimuth), elevation=30.0, config=cfg, device=args.device
        )[..., :3].cpu().numpy()
    return gt, sp, pc


def main():
    p = argparse.ArgumentParser(description="Evaluate spline vs point baseline on external targets")
    p.add_argument("--spline-results", default="outputs/my_dense_run/opt_results.pt")
    p.add_argument("--point-results", default="outputs/pointcloud_baseline/point_baseline_results.pt")
    p.add_argument("--model-name", default="wStraight")
    p.add_argument("--data-dir", default="data/hairmodels")
    p.add_argument("--spline-eval-samples", type=int, default=96)
    p.add_argument("--raw-num-strands", type=int, default=3000)
    p.add_argument("--raw-max-pts-per-strand", type=int, default=48)
    p.add_argument("--chamfer-max-points", type=int, default=12000)
    p.add_argument("--num-heldout-views", type=int, default=36)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--radius", type=float, default=0.02)
    p.add_argument(
        "--bin-size",
        type=int,
        default=0,
        help="PyTorch3D raster bin size (0 uses naive rasterization, safer for dense clouds).",
    )
    p.add_argument(
        "--max-points-per-bin",
        type=int,
        default=None,
        help="Optional PyTorch3D coarse raster cap; only used when provided.",
    )
    p.add_argument("--visual-azimuth", type=float, default=30.0)
    p.add_argument("--visual-image-size", type=int, default=512)
    p.add_argument("--visual-radius", type=float, default=0.008)
    p.add_argument("--visual-points-per-pixel", type=int, default=10)
    p.add_argument("--visual-num-frames", type=int, default=72)
    p.add_argument("--visual-fps", type=int, default=12)
    p.add_argument("--visual-dpi", type=int, default=120)
    p.add_argument("--skip-visual-video", action="store_true")
    p.add_argument("--keep-visual-frames", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default="outputs/external_eval")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    spline = load_torch(args.spline_results)
    point = load_torch(args.point_results)

    spline_cp = spline["final_cp"].to(args.device)
    with torch.no_grad():
        spline_points = evaluate_bspline(spline_cp, args.spline_eval_samples).reshape(-1, 3)
    point_points = point["final_points"].to(args.device)

    hair_path = os.path.join(args.data_dir, f"{args.model_name}.hair")
    raw_points = dense_raw_hair_points(
        hair_path, num_strands=args.raw_num_strands, max_pts_per_strand=args.raw_max_pts_per_strand
    ).to(args.device)

    # Approximate Chamfer on sampled subsets for tractability.
    raw_c = sample_for_chamfer(raw_points, args.chamfer_max_points, seed=11)
    spline_c = sample_for_chamfer(spline_points, args.chamfer_max_points, seed=13)
    point_c = sample_for_chamfer(point_points, args.chamfer_max_points, seed=17)

    chamfer_spline = approx_chamfer(spline_c, raw_c)
    chamfer_point = approx_chamfer(point_c, raw_c)

    num_train_views = len(spline.get("azimuths", [])) or 72
    az = heldout_azimuths(num_train_views=num_train_views, num_eval=args.num_heldout_views)
    mse_spline, mse_spline_per_view = render_mse_vs_raw(
        raw_points,
        spline_points,
        az,
        args.image_size,
        args.radius,
        args.device,
        bin_size=args.bin_size,
        max_points_per_bin=args.max_points_per_bin,
    )
    mse_point, mse_point_per_view = render_mse_vs_raw(
        raw_points,
        point_points,
        az,
        args.image_size,
        args.radius,
        args.device,
        bin_size=args.bin_size,
        max_points_per_bin=args.max_points_per_bin,
    )

    winner_chamfer = "spline" if chamfer_spline < chamfer_point else "pointcloud"
    if abs(chamfer_spline - chamfer_point) < 1e-12:
        winner_chamfer = "tie"
    winner_heldout = "spline" if mse_spline < mse_point else "pointcloud"
    if abs(mse_spline - mse_point) < 1e-12:
        winner_heldout = "tie"

    out = {
        "model_name": args.model_name,
        "raw_target": {
            "hair_path": hair_path,
            "num_points": int(raw_points.shape[0]),
            "num_strands_sampled": int(args.raw_num_strands),
        },
        "spline": {
            "num_points_eval": int(spline_points.shape[0]),
            "approx_chamfer_to_raw": chamfer_spline,
            "heldout_render_mse_mean": mse_spline,
            "heldout_render_mse_per_view": mse_spline_per_view,
        },
        "pointcloud": {
            "num_points_eval": int(point_points.shape[0]),
            "approx_chamfer_to_raw": chamfer_point,
            "heldout_render_mse_mean": mse_point,
            "heldout_render_mse_per_view": mse_point_per_view,
        },
        "winners": {
            "by_approx_chamfer_to_raw": winner_chamfer,
            "by_heldout_render_mse": winner_heldout,
        },
        "notes": {
            "chamfer_is_approximate": True,
            "chamfer_subset_size": int(args.chamfer_max_points),
            "heldout_views_are_offset_from_training_grid": True,
            "raster_bin_size": args.bin_size,
            "max_points_per_bin": args.max_points_per_bin,
        },
    }

    json_path = os.path.join(args.output_dir, "external_eval.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    md_lines = [
        "# External Target Evaluation",
        "",
        f"- Model: `{args.model_name}`",
        f"- Raw target points: `{out['raw_target']['num_points']}` from `{hair_path}`",
        f"- Spline eval points: `{out['spline']['num_points_eval']}`",
        f"- Point eval points: `{out['pointcloud']['num_points_eval']}`",
        "",
        "## Metrics",
        "",
        "| Method | Approx Chamfer to Raw (lower better) | Held-out Render MSE (lower better) |",
        "|---|---:|---:|",
        f"| Spline | {chamfer_spline:.6f} | {mse_spline:.6f} |",
        f"| Point cloud | {chamfer_point:.6f} | {mse_point:.6f} |",
        "",
        "## Winners",
        "",
        f"- By approx Chamfer: `{winner_chamfer}`",
        f"- By held-out render MSE: `{winner_heldout}`",
        "",
        "## Notes",
        "",
        f"- Chamfer uses random subsets capped at `{args.chamfer_max_points}` points for tractability.",
        "- Held-out views are interleaved (half-step offset) relative to training azimuth grid.",
        "",
    ]
    md_path = os.path.join(args.output_dir, "external_eval.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # Plot summary figure (similar spirit to baseline_comparison_curves.png).
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = np.arange(len(az))
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        axes[0].plot(x, mse_spline_per_view, "b-", label="Spline")
        axes[0].plot(x, mse_point_per_view, "m-", label="Point cloud")
        axes[0].set_title("Held-out Render MSE per View")
        axes[0].set_xlabel("Held-out view index")
        axes[0].set_ylabel("MSE")
        axes[0].legend()

        labels = ["Spline", "Point cloud"]
        chamfer_vals = [chamfer_spline, chamfer_point]
        mse_vals = [mse_spline, mse_point]

        axes[1].bar(labels, chamfer_vals, color=["#4C72B0", "#C44E52"])
        axes[1].set_title("Approx Chamfer to Raw")
        axes[1].set_ylabel("Lower is better")

        axes[2].bar(labels, mse_vals, color=["#4C72B0", "#C44E52"])
        axes[2].set_title("Held-out Render MSE (mean)")
        axes[2].set_ylabel("Lower is better")

        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, "external_eval_curves.png")
        plt.savefig(plot_path, dpi=160)
        plt.close()
    except Exception as exc:
        print(f"  Plot skipped: {exc}")
        plot_path = None

    # Qualitative visual outputs: still + optional rotating video.
    visual_still_path = None
    visual_video_path = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        gt_im, sp_im, pc_im = render_visual_triptych(
            raw_points, spline_points, point_points, azimuth=args.visual_azimuth, args=args
        )
        fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor="#080810")
        fig.subplots_adjust(wspace=0.02, left=0.01, right=0.99, top=0.84, bottom=0.08)
        for ax, (im, title) in zip(
            axes,
            [
                (gt_im, "External Raw Target (Dense Hair)"),
                (sp_im, "Spline Final"),
                (pc_im, "Point-Cloud Final"),
            ],
        ):
            ax.imshow(np.clip(im, 0, 1))
            ax.axis("off")
            ax.set_title(title, color="#e0d0a0", fontsize=12, fontweight="bold", pad=10)

        fig.suptitle(
            f"External Visual Evaluation — {args.model_name} | Azimuth {args.visual_azimuth:.1f}°",
            color="white",
            fontsize=14,
            fontweight="bold",
            y=0.94,
        )
        fig.text(
            0.5,
            0.03,
            f"Chamfer: spline={chamfer_spline:.5f}, point={chamfer_point:.5f} | "
            f"Held-out MSE: spline={mse_spline:.5f}, point={mse_point:.5f}",
            color="#888",
            fontsize=9,
            ha="center",
        )
        visual_still_path = os.path.join(args.output_dir, "external_eval_visual.png")
        plt.savefig(visual_still_path, dpi=args.visual_dpi, facecolor="#080810", edgecolor="none")
        plt.close()

        if not args.skip_visual_video:
            frames_dir = os.path.join(args.output_dir, "external_eval_frames")
            os.makedirs(frames_dir, exist_ok=True)
            azimuths_vis = np.linspace(0, 360, args.visual_num_frames, endpoint=False)
            for i, az_vis in enumerate(azimuths_vis):
                g, s, p_im = render_visual_triptych(
                    raw_points, spline_points, point_points, azimuth=az_vis, args=args
                )
                fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor="#080810")
                fig.subplots_adjust(wspace=0.02, left=0.01, right=0.99, top=0.84, bottom=0.08)
                for ax, (im, title) in zip(
                    axes,
                    [
                        (g, "External Raw Target (Dense Hair)"),
                        (s, "Spline Final"),
                        (p_im, "Point-Cloud Final"),
                    ],
                ):
                    ax.imshow(np.clip(im, 0, 1))
                    ax.axis("off")
                    ax.set_title(title, color="#e0d0a0", fontsize=12, fontweight="bold", pad=10)
                fig.suptitle(
                    f"External Visual Evaluation — {args.model_name} | Azimuth {az_vis:.1f}°",
                    color="white",
                    fontsize=14,
                    fontweight="bold",
                    y=0.94,
                )
                fig.text(
                    0.5,
                    0.03,
                    f"Chamfer: spline={chamfer_spline:.5f}, point={chamfer_point:.5f} | "
                    f"Held-out MSE: spline={mse_spline:.5f}, point={mse_point:.5f}",
                    color="#888",
                    fontsize=9,
                    ha="center",
                )
                plt.savefig(
                    os.path.join(frames_dir, f"frame_{i:04d}.png"),
                    dpi=args.visual_dpi,
                    facecolor="#080810",
                    edgecolor="none",
                )
                plt.close()

            visual_video_path = os.path.join(args.output_dir, "external_eval_visual.mp4")
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(args.visual_fps),
                "-i",
                os.path.join(frames_dir, "frame_%04d.png"),
                "-c:v",
                "libopenh264",
                "-b:v",
                "6M",
                "-pix_fmt",
                "yuv420p",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                visual_video_path,
            ]
            proc = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"  Visual video failed: {proc.stderr.strip()[:400]}")
                visual_video_path = None
            elif not args.keep_visual_frames:
                import shutil

                shutil.rmtree(frames_dir)
    except Exception as exc:
        print(f"  Visual generation skipped: {exc}")

    print("External evaluation complete.")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")
    if plot_path is not None:
        print(f"  Plot: {plot_path}")
    if visual_still_path is not None:
        print(f"  Visual still: {visual_still_path}")
    if visual_video_path is not None:
        print(f"  Visual video: {visual_video_path}")
    print(f"  Winner by Chamfer: {winner_chamfer}")
    print(f"  Winner by held-out render MSE: {winner_heldout}")


if __name__ == "__main__":
    main()
