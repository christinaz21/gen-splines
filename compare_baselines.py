"""
compare_baselines.py — Compare spline-memory run vs baseline representations.
"""

import argparse
import json
import os

import numpy as np
import torch


def _safe_float(v):
    fv = float(v)
    if not np.isfinite(fv):
        return float("inf")
    return fv


def load_torch(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_results(path):
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return load_torch(path)


def normalize_method_dict(data, key_ppc=None):
    return {
        "initial_drift": _safe_float(data["initial_drift"]),
        "final_drift": _safe_float(data["final_drift"]),
        "drift_reduction_pct": _safe_float(data["drift_reduction_pct"]),
        "time_seconds": _safe_float(data.get("time_seconds", float("nan"))),
        "view_drifts": data.get("view_drifts", []),
        "azimuths": data.get("azimuths", []),
        "initial_chamfer": _safe_float(data["initial_chamfer"]) if data.get("initial_chamfer") is not None else None,
        "final_chamfer": _safe_float(data["final_chamfer"]) if data.get("final_chamfer") is not None else None,
        key_ppc: data.get(key_ppc, None) if key_ppc is not None else None,
    }


def normalize_spline_dict(spline_data):
    norm = {
        "initial_drift": float(spline_data["initial_drift"]),
        "final_drift": float(spline_data["final_drift"]),
        "time_seconds": float(spline_data.get("time_seconds", float("nan"))),
        "view_drifts": spline_data.get("view_drifts", []),
        "azimuths": spline_data.get("azimuths", []),
    }
    if "drift_reduction" in spline_data:
        norm["drift_reduction_pct"] = float(spline_data["drift_reduction"])
    else:
        norm["drift_reduction_pct"] = (1.0 - norm["final_drift"] / norm["initial_drift"]) * 100.0
    return norm


def to_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return list(x)


def winner_lower(vals):
    best_name, best_val, tie = None, None, False
    for n, v in vals.items():
        if best_val is None or v < best_val:
            best_name, best_val, tie = n, v, False
        elif abs(v - best_val) <= 1e-9:
            tie = True
    return "tie" if tie else best_name


def winner_higher(vals):
    best_name, best_val, tie = None, None, False
    for n, v in vals.items():
        if best_val is None or v > best_val:
            best_name, best_val, tie = n, v, False
        elif abs(v - best_val) <= 1e-9:
            tie = True
    return "tie" if tie else best_name


def build_summary(spline, point, gaussian=None):
    methods = {"spline": spline, "pointcloud": point}
    if gaussian is not None:
        methods["gaussian"] = gaussian

    by_drift = winner_higher({k: v["drift_reduction_pct"] for k, v in methods.items()})
    by_final_drift = winner_lower({k: v["final_drift"] for k, v in methods.items()})
    by_runtime = winner_lower({k: v["time_seconds"] for k, v in methods.items()})

    score = {k: 0 for k in methods.keys()}
    for w in (by_drift, by_final_drift, by_runtime):
        if w in score:
            score[w] += 1
    overall = max(score, key=score.get)
    if len(set(score.values())) == 1:
        overall = "tie"

    summary = {
        "spline": {
            "initial_drift": spline["initial_drift"],
            "final_drift": spline["final_drift"],
            "drift_reduction_pct": spline["drift_reduction_pct"],
            "time_seconds": spline["time_seconds"],
        },
        "pointcloud": {
            "initial_drift": point["initial_drift"],
            "final_drift": point["final_drift"],
            "drift_reduction_pct": point["drift_reduction_pct"],
            "time_seconds": point["time_seconds"],
            "initial_chamfer": point.get("initial_chamfer"),
            "final_chamfer": point.get("final_chamfer"),
        },
        "winner_by_drift_reduction": by_drift,
        "winner_multi_metric": {
            "by_drift_reduction_pct": by_drift,
            "by_final_drift": by_final_drift,
            "by_runtime": by_runtime,
            "score": score,
            "overall_majority_vote": overall,
        },
    }
    if gaussian is not None:
        summary["gaussian"] = {
            "initial_drift": gaussian["initial_drift"],
            "final_drift": gaussian["final_drift"],
            "drift_reduction_pct": gaussian["drift_reduction_pct"],
            "time_seconds": gaussian["time_seconds"],
            "initial_chamfer": gaussian.get("initial_chamfer"),
            "final_chamfer": gaussian.get("final_chamfer"),
        }
    return summary


def write_markdown(summary, output_path, spline_path, point_path, gaussian_path=None):
    s = summary["spline"]
    p = summary["pointcloud"]
    g = summary.get("gaussian")
    multi = summary["winner_multi_metric"]

    lines = [
        "# Baseline Comparison",
        "",
        f"- Spline results: `{spline_path}`",
        f"- Point-cloud results: `{point_path}`",
        f"- Gaussian results: `{gaussian_path}`" if gaussian_path else "- Gaussian results: `(not provided)`",
        "",
        "## Metrics",
        "",
        "| Method | Initial Drift | Final Drift | Drift Reduction (%) | Time (s) |",
        "|---|---:|---:|---:|---:|",
        f"| Spline memory | {s['initial_drift']:.4f} | {s['final_drift']:.4f} | {s['drift_reduction_pct']:.2f} | {s['time_seconds']:.1f} |",
        f"| Point-cloud baseline | {p['initial_drift']:.4f} | {p['final_drift']:.4f} | {p['drift_reduction_pct']:.2f} | {p['time_seconds']:.1f} |",
    ]
    if g is not None:
        lines.append(
            f"| Gaussian splat baseline | {g['initial_drift']:.4f} | {g['final_drift']:.4f} | {g['drift_reduction_pct']:.2f} | {g['time_seconds']:.1f} |"
        )
    lines.extend(
        [
            "",
            f"**Winner by drift reduction:** `{summary['winner_by_drift_reduction']}`",
            "",
            "## Multi-Metric Verdict",
            "",
            f"- By drift reduction: `{multi.get('by_drift_reduction_pct', 'n/a')}`",
            f"- By final drift: `{multi.get('by_final_drift', 'n/a')}`",
            f"- By runtime: `{multi.get('by_runtime', 'n/a')}`",
            f"- Majority vote: `{multi.get('overall_majority_vote', 'n/a')}`",
            "",
        ]
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_curves(spline, point, output_path, gaussian=None):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False, "matplotlib unavailable"

    spline_drifts = to_list(spline["view_drifts"])
    point_drifts = to_list(point["view_drifts"])
    gaussian_drifts = to_list(gaussian["view_drifts"]) if gaussian is not None else []

    if len(spline_drifts) == 0 and len(point_drifts) == 0 and len(gaussian_drifts) == 0:
        return False, "no view trajectories in inputs"

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    if len(spline_drifts) > 0:
        axes[0].plot(np.arange(len(spline_drifts)), spline_drifts, "b-", label="Spline")
    if len(point_drifts) > 0:
        axes[0].plot(np.arange(len(point_drifts)), point_drifts, "m-", label="Point cloud")
    if len(gaussian_drifts) > 0:
        axes[0].plot(np.arange(len(gaussian_drifts)), gaussian_drifts, color="#2ca02c", label="Gaussian")
    axes[0].set_title("Drift per View")
    axes[0].set_xlabel("View index")
    axes[0].set_ylabel("Drift")
    axes[0].legend()

    labels = ["Spline", "Point cloud"]
    final_drifts = [spline["final_drift"], point["final_drift"]]
    reductions = [spline["drift_reduction_pct"], point["drift_reduction_pct"]]
    colors = ["#4C72B0", "#C44E52"]
    if gaussian is not None:
        labels.append("Gaussian")
        final_drifts.append(gaussian["final_drift"])
        reductions.append(gaussian["drift_reduction_pct"])
        colors.append("#2ca02c")

    axes[1].bar(labels, final_drifts, color=colors)
    axes[1].set_title("Final Drift")
    axes[1].set_ylabel("Drift")
    axes[2].bar(labels, reductions, color=colors)
    axes[2].set_title("Drift Reduction (%)")
    axes[2].set_ylabel("%")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return True, None


def main():
    parser = argparse.ArgumentParser(description="Compare spline vs baseline runs")
    parser.add_argument("--spline-results", default="outputs/my_dense_run/opt_results.pt")
    parser.add_argument("--point-results", default="outputs/pointcloud_baseline/point_baseline_results.pt")
    parser.add_argument("--gaussian-results", default=None)
    parser.add_argument("--output-dir", default="outputs/baseline_compare")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    spline = normalize_spline_dict(load_torch(args.spline_results))
    point = normalize_method_dict(load_results(args.point_results), key_ppc="pc_points_per_curve")
    gaussian = None
    if args.gaussian_results is not None and os.path.exists(args.gaussian_results):
        gaussian = normalize_method_dict(load_results(args.gaussian_results), key_ppc="gs_points_per_curve")

    summary = build_summary(spline, point, gaussian=gaussian)
    json_path = os.path.join(args.output_dir, "baseline_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md_path = os.path.join(args.output_dir, "baseline_comparison.md")
    write_markdown(summary, md_path, args.spline_results, args.point_results, args.gaussian_results)
    plot_path = os.path.join(args.output_dir, "baseline_comparison_curves.png")
    ok, reason = plot_curves(spline, point, plot_path, gaussian=gaussian)

    print("\nComparison complete.")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")
    if ok:
        print(f"  Plot: {plot_path}")
    else:
        print(f"  Plot skipped: {reason}")
    print(f"  Winner (drift reduction): {summary['winner_by_drift_reduction']}")
    print(
        f"  Winner (multi-metric): "
        f"{summary.get('winner_multi_metric', {}).get('overall_majority_vote', 'n/a')}"
    )


if __name__ == "__main__":
    main()
