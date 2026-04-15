"""
run_full_baseline_pipeline.py — One-command spline vs point-cloud evaluation pipeline.

This orchestrates the full workflow so you do not have to run multiple files manually:
  1) run_dense.py
  2) run_pointcloud_baseline.py
  3) compare_baselines.py
  4) evaluate_external_targets.py

Outputs are grouped under a single root directory.
"""

import argparse
import os
import shlex
import subprocess
import sys
import time


def run_cmd(cmd, cwd):
    print(f"\n[RUN] {cmd}", flush=True)
    proc = subprocess.run(cmd, shell=True, cwd=cwd)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {cmd}")


def q(s):
    return shlex.quote(str(s))


def main():
    p = argparse.ArgumentParser(description="Run full spline vs point-cloud baseline pipeline")
    p.add_argument("--model-name", default="wStraight")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-root", default="outputs/full_pipeline")

    # Shared optimization knobs
    p.add_argument("--num-curves", type=int, default=500)
    p.add_argument("--K", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-views", type=int, default=72)
    p.add_argument("--steps-per-view", type=int, default=80)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--init-noise", type=float, default=0.35)
    p.add_argument("--view-buffer", type=int, default=5)
    p.add_argument("--ema-decay", type=float, default=0.8)

    # Spline run settings
    p.add_argument("--spline-samples-per-curve", type=int, default=96)
    p.add_argument("--render-weight", type=float, default=0.5)
    p.add_argument("--reproj-weight", type=float, default=1.5)
    p.add_argument("--tangent-weight", type=float, default=0.1)
    p.add_argument("--anchor-weight", type=float, default=0.02)
    p.add_argument("--opt-image-size", type=int, default=256)
    p.add_argument("--opt-radius", type=float, default=0.02)
    p.add_argument("--skip-video", action="store_true", help="Skip heavy video rendering stage in run_dense.")

    # Point baseline settings
    p.add_argument("--pc-points-per-curve", type=int, default=12)
    p.add_argument("--pc-image-size", type=int, default=256)
    p.add_argument("--pc-radius", type=float, default=0.02)
    p.add_argument("--pc-points-per-pixel", type=int, default=8)

    # External eval settings
    p.add_argument("--external-raw-num-strands", type=int, default=3000)
    p.add_argument("--external-raw-max-pts-per-strand", type=int, default=48)
    p.add_argument("--external-chamfer-max-points", type=int, default=12000)
    p.add_argument("--external-num-heldout-views", type=int, default=36)
    p.add_argument("--external-image-size", type=int, default=256)
    p.add_argument("--external-radius", type=float, default=0.02)
    p.add_argument("--external-spline-eval-samples", type=int, default=96)
    p.add_argument("--external-bin-size", type=int, default=0)
    p.add_argument("--external-max-points-per-bin", type=int, default=None)

    # Flow control
    p.add_argument("--quick", action="store_true", help="Use quicker settings for both methods.")
    p.add_argument("--skip-spline", action="store_true", help="Reuse existing spline outputs if present.")
    p.add_argument("--skip-point", action="store_true", help="Reuse existing point outputs if present.")
    p.add_argument("--skip-compare", action="store_true")
    p.add_argument("--skip-external", action="store_true")
    args = p.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    python = q(sys.executable)

    if args.quick:
        args.num_views = 36
        args.steps_per_view = 40

    # Output layout
    out_root = args.output_root
    spline_out = os.path.join(out_root, "spline")
    point_out = os.path.join(out_root, "pointcloud")
    compare_out = os.path.join(out_root, "compare")
    external_out = os.path.join(out_root, "external_eval")
    os.makedirs(out_root, exist_ok=True)

    print("\n" + "=" * 72)
    print("Full Baseline Pipeline")
    print(f"  Model: {args.model_name}")
    print(f"  Output root: {out_root}")
    print("=" * 72, flush=True)

    t0 = time.time()

    if not args.skip_spline:
        cmd = (
            f"{python} run_dense.py "
            f"--model-name {q(args.model_name)} "
            f"--data-dir {q(args.data_dir)} "
            f"--num-curves {args.num_curves} "
            f"--K {args.K} "
            f"--seed {args.seed} "
            f"--num-views {args.num_views} "
            f"--steps-per-view {args.steps_per_view} "
            f"--lr {args.lr} "
            f"--init-noise {args.init_noise} "
            f"--samples-per-curve {args.spline_samples_per_curve} "
            f"--render-weight {args.render_weight} "
            f"--reproj-weight {args.reproj_weight} "
            f"--tangent-weight {args.tangent_weight} "
            f"--anchor-weight {args.anchor_weight} "
            f"--view-buffer {args.view_buffer} "
            f"--ema-decay {args.ema_decay} "
            f"--opt-image-size {args.opt_image_size} "
            f"--opt-radius {args.opt_radius} "
            f"--device {q(args.device)} "
            f"--output-dir {q(spline_out)} "
        )
        if args.quick:
            cmd += "--quick "
        if args.skip_video:
            # Keep runtime lower by rendering minimal video footprint.
            cmd += "--num-video-frames 1 --fps 1 --keep-frames "
        run_cmd(cmd, repo_root)
    else:
        print("\n[SKIP] spline stage", flush=True)

    if not args.skip_point:
        cmd = (
            f"{python} run_pointcloud_baseline.py "
            f"--model-name {q(args.model_name)} "
            f"--data-dir {q(args.data_dir)} "
            f"--num-curves {args.num_curves} "
            f"--K {args.K} "
            f"--seed {args.seed} "
            f"--pc-points-per-curve {args.pc_points_per_curve} "
            f"--num-views {args.num_views} "
            f"--steps-per-view {args.steps_per_view} "
            f"--lr {args.lr} "
            f"--init-noise {args.init_noise} "
            f"--render-weight {args.render_weight} "
            f"--reproj-weight {args.reproj_weight} "
            f"--anchor-weight {args.anchor_weight} "
            f"--view-buffer {args.view_buffer} "
            f"--ema-decay {args.ema_decay} "
            f"--image-size {args.pc_image_size} "
            f"--radius {args.pc_radius} "
            f"--points-per-pixel {args.pc_points_per_pixel} "
            f"--device {q(args.device)} "
            f"--output-dir {q(point_out)} "
        )
        if args.quick:
            cmd += "--quick "
        run_cmd(cmd, repo_root)
    else:
        print("\n[SKIP] point baseline stage", flush=True)

    spline_results = os.path.join(spline_out, "opt_results.pt")
    point_results_pt = os.path.join(point_out, "point_baseline_results.pt")

    if not args.skip_compare:
        cmd = (
            f"{python} compare_baselines.py "
            f"--spline-results {q(spline_results)} "
            f"--point-results {q(point_results_pt)} "
            f"--output-dir {q(compare_out)}"
        )
        run_cmd(cmd, repo_root)
    else:
        print("\n[SKIP] internal compare stage", flush=True)

    if not args.skip_external:
        cmd = (
            f"{python} evaluate_external_targets.py "
            f"--spline-results {q(spline_results)} "
            f"--point-results {q(point_results_pt)} "
            f"--model-name {q(args.model_name)} "
            f"--data-dir {q(os.path.join(args.data_dir, 'hairmodels'))} "
            f"--spline-eval-samples {args.external_spline_eval_samples} "
            f"--raw-num-strands {args.external_raw_num_strands} "
            f"--raw-max-pts-per-strand {args.external_raw_max_pts_per_strand} "
            f"--chamfer-max-points {args.external_chamfer_max_points} "
            f"--num-heldout-views {args.external_num_heldout_views} "
            f"--image-size {args.external_image_size} "
            f"--radius {args.external_radius} "
            f"--bin-size {args.external_bin_size} "
            f"--device {q(args.device)} "
            f"--output-dir {q(external_out)} "
        )
        if args.external_max_points_per_bin is not None:
            cmd += f"--max-points-per-bin {args.external_max_points_per_bin} "
        run_cmd(cmd, repo_root)
    else:
        print("\n[SKIP] external evaluation stage", flush=True)

    total = time.time() - t0
    print("\n" + "=" * 72)
    print("Pipeline Complete")
    print(f"  Output root: {out_root}")
    print(f"  Spline:      {spline_out}")
    print(f"  Point cloud: {point_out}")
    print(f"  Compare:     {compare_out}")
    print(f"  External:    {external_out}")
    print(f"  Total time:  {total:.1f}s")
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
