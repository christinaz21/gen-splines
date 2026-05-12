"""
export_gt.py — Convert ground truth data to OBJ for the live viewer.

Supports two sources:
  1. gt_cp.pt from world_model.py output (already fitted B-spline CPs)
  2. Raw .hair file from Cem Yuksel dataset

Both produce an OBJ tube mesh that loads in the viewer alongside
spline_tubes.obj and pointcloud.ply.

Run on Amarel:
    # From world model output (recommended — same fitting as your optimization):
    python export_gt.py --from-pt outputs/wm_clean/gt_cp.pt \
        --output outputs/wm_clean/gt_tubes.obj

    # From raw .hair file:
    python export_gt.py --from-hair --model-name wStraight --num-curves 2000 \
        --output outputs/wm_clean/gt_tubes.obj

    # Also export GT as point cloud (for comparison):
    python export_gt.py --from-pt outputs/wm_clean/gt_cp.pt \
        --output-ply outputs/wm_clean/gt_points.ply
"""

import argparse, os, time
import numpy as np
import torch


def orient(pts):
    out = pts.clone()
    ny, nz = out[..., 2].clone(), -out[..., 1].clone()
    out[..., 1], out[..., 2] = ny, nz
    return out


def main():
    p = argparse.ArgumentParser()

    # Source selection
    p.add_argument("--from-pt", type=str, default=None,
                   help="Path to gt_cp.pt (control points from world_model.py)")
    p.add_argument("--from-hair", action="store_true",
                   help="Load from raw .hair file instead")
    p.add_argument("--model-name", default="wStraight")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--num-curves", type=int, default=2000)
    p.add_argument("--K", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)

    # Output
    p.add_argument("--output", default="gt_tubes.obj", help="OBJ output path")
    p.add_argument("--output-ply", default=None, help="Also export as PLY point cloud")

    # Tube settings
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--tube-radius", type=float, default=0.0015)
    p.add_argument("--n-sides", type=int, default=4)

    args = p.parse_args()

    # Load control points
    if args.from_pt:
        print(f"Loading CPs from {args.from_pt} ...")
        data = torch.load(args.from_pt, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            cp = data.get("gt_cp", data.get("control_points", None))
            if cp is None:
                for k, v in data.items():
                    if isinstance(v, torch.Tensor) and v.dim() == 3:
                        cp = v
                        print(f"  Using key '{k}': {v.shape}")
                        break
        else:
            cp = data
        if cp is None:
            raise ValueError(f"No (N, K, 3) tensor found in {args.from_pt}")

    elif args.from_hair:
        print(f"Loading from {args.model_name}.hair ...")
        from hair_loader import (download_yuksel_hair, load_hair_file,
                                      hair_to_spline_field)
        hp = download_yuksel_hair(args.model_name, save_dir=args.data_dir)
        strands = load_hair_file(hp)
        cp = hair_to_spline_field(strands, num_curves=args.num_curves,
                                  K=args.K, seed=args.seed, strategy="diverse")
        cp = orient(cp)
    else:
        raise ValueError("Provide --from-pt <path> or --from-hair")

    N, K, _ = cp.shape
    print(f"  GT: {N} curves, K={K}, {N*K*3:,} params")

    # Export OBJ
    from render_utils import export_tubes_obj
    print(f"\nExporting tubes OBJ ...")
    export_tubes_obj(cp, args.num_samples, args.output,
                     n_sides=args.n_sides, tube_radius=args.tube_radius,
                     seed=args.seed, compact=False)

    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"  {args.output}: {size_mb:.1f} MB")

    # Optional PLY export
    if args.output_ply:
        from spline import evaluate_bspline
        from render_utils import export_points_ply
        with torch.no_grad():
            pts = evaluate_bspline(cp, args.num_samples).reshape(-1, 3)
        export_points_ply(pts, args.output_ply, seed=args.seed)

    print(f"\nDone. Load {args.output} as 'GT .OBJ' in the viewer.")


if __name__ == "__main__":
    main()
