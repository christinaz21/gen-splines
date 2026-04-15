"""
compare_baselines.py — Compare spline-memory run vs point-cloud baseline.

Inputs:
  - Spline results file from run_dense.py (opt_results.pt)
  - Point-cloud results file from run_pointcloud_baseline.py
    (point_baseline_results.pt or point_baseline_results.json)

Outputs (in --output-dir):
  - baseline_comparison.json
  - baseline_comparison.md
  - baseline_comparison_curves.png
"""

import argparse
import json
import os

import numpy as np
import torch


def load_torch(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_point_results(path):
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return load_torch(path)


def normalize_point_dict(point_data):
    """
    Accept either PT dict (rich) or JSON dict (summary) and normalize keys.
    """
    norm = {}
    norm["initial_drift"] = float(point_data["initial_drift"])
    norm["final_drift"] = float(point_data["final_drift"])
    norm["drift_reduction_pct"] = float(point_data["drift_reduction_pct"])
    norm["time_seconds"] = float(point_data.get("time_seconds", float("nan")))
    norm["view_drifts"] = point_data.get("view_drifts", [])
    norm["azimuths"] = point_data.get("azimuths", [])
    norm["initial_chamfer"] = point_data.get("initial_chamfer", None)
    norm["final_chamfer"] = point_data.get("final_chamfer", None)
    norm["view_chamfers"] = point_data.get("view_chamfers", [])
    norm["view_losses"] = point_data.get("view_losses", [])
    norm["num_points"] = point_data.get("num_points", None)
    norm["num_curves"] = point_data.get("num_curves", None)
    norm["K"] = point_data.get("K", None)
    norm["pc_points_per_curve"] = point_data.get("pc_points_per_curve", None)
    return norm


def normalize_spline_dict(spline_data):
    norm = {}
    norm["initial_drift"] = float(spline_data["initial_drift"])
    norm["final_drift"] = float(spline_data["final_drift"])
    if "drift_reduction" in spline_data:
        norm["drift_reduction_pct"] = float(spline_data["drift_reduction"])
    else:
        norm["drift_reduction_pct"] = (1.0 - norm["final_drift"] / norm["initial_drift"]) * 100.0
    norm["time_seconds"] = float(spline_data.get("time_seconds", float("nan")))
    norm["view_drifts"] = spline_data.get("view_drifts", [])
    norm["azimuths"] = spline_data.get("azimuths", [])
    if "gt_cp" in spline_data and isinstance(spline_data["gt_cp"], torch.Tensor):
        gt_cp = spline_data["gt_cp"]
        if gt_cp.ndim == 3 and gt_cp.shape[-1] == 3:
            norm["num_curves"] = int(gt_cp.shape[0])
            norm["K"] = int(gt_cp.shape[1])
        else:
            norm["num_curves"] = None
            norm["K"] = None
    else:
        norm["num_curves"] = None
        norm["K"] = None
    return norm


def to_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return list(x)


def safe_last(seq, default=float("nan")):
    if seq is None or len(seq) == 0:
        return default
    return float(seq[-1])


def choose_winner_higher_better(a, b, eps=1e-9):
    if abs(a - b) <= eps:
        return "tie"
    return "spline" if a > b else "pointcloud"


def choose_winner_lower_better(a, b, eps=1e-9):
    if abs(a - b) <= eps:
        return "tie"
    return "spline" if a < b else "pointcloud"


def build_fairness_checks(spline, point):
    spline_num_curves = spline.get("num_curves")
    spline_k = spline.get("K")
    point_num_curves = point.get("num_curves")
    point_k = point.get("K")
    point_ppc = point.get("pc_points_per_curve")

    spline_param_count = None
    point_param_count = None
    if spline_num_curves is not None and spline_k is not None:
        spline_param_count = int(spline_num_curves * spline_k * 3)
    if point_num_curves is not None and point_ppc is not None:
        point_param_count = int(point_num_curves * point_ppc * 3)

    budget_match = None
    if spline_param_count is not None and point_param_count is not None:
        budget_match = spline_param_count == point_param_count

    curve_count_match = None
    if spline_num_curves is not None and point_num_curves is not None:
        curve_count_match = int(spline_num_curves) == int(point_num_curves)

    # "K" matching means same spline CP count and point baseline source spline K.
    source_k_match = None
    if spline_k is not None and point_k is not None:
        source_k_match = int(spline_k) == int(point_k)

    # Representation granularity match compares K vs points-per-curve.
    granularity_match = None
    if spline_k is not None and point_ppc is not None:
        granularity_match = int(spline_k) == int(point_ppc)

    warnings = []
    if budget_match is False:
        warnings.append("Parameter budget mismatch (learnable 3D coordinates differ).")
    if curve_count_match is False:
        warnings.append("num_curves mismatch between runs.")
    if source_k_match is False:
        warnings.append("Spline K mismatch between runs.")
    if granularity_match is False:
        warnings.append("Spline K != point points-per-curve (representation granularity mismatch).")
    if budget_match is None:
        warnings.append("Could not fully verify parameter budget from provided files.")

    return {
        "spline_num_curves": spline_num_curves,
        "spline_K": spline_k,
        "point_num_curves": point_num_curves,
        "point_K": point_k,
        "point_points_per_curve": point_ppc,
        "spline_param_count_xyz": spline_param_count,
        "point_param_count_xyz": point_param_count,
        "budget_match": budget_match,
        "curve_count_match": curve_count_match,
        "source_k_match": source_k_match,
        "granularity_match": granularity_match,
        "warnings": warnings,
    }


def build_multi_metric_verdict(spline, point):
    verdict = {
        "by_drift_reduction_pct": choose_winner_higher_better(
            spline["drift_reduction_pct"], point["drift_reduction_pct"]
        ),
        "by_final_drift": choose_winner_lower_better(spline["final_drift"], point["final_drift"]),
        "by_runtime": choose_winner_lower_better(spline["time_seconds"], point["time_seconds"]),
    }

    if point.get("final_chamfer") is not None:
        # Spline run does not currently report chamfer in this pipeline.
        verdict["by_final_chamfer"] = "pointcloud_only_metric"

    score = {"spline": 0, "pointcloud": 0}
    for key in ("by_drift_reduction_pct", "by_final_drift", "by_runtime"):
        winner = verdict[key]
        if winner in score:
            score[winner] += 1

    if score["spline"] > score["pointcloud"]:
        overall = "spline"
    elif score["pointcloud"] > score["spline"]:
        overall = "pointcloud"
    else:
        overall = "tie"

    verdict["score"] = score
    verdict["overall_majority_vote"] = overall
    return verdict


def build_summary(spline, point):
    spline_reduction = spline["drift_reduction_pct"]
    point_reduction = point["drift_reduction_pct"]
    winner = choose_winner_higher_better(spline_reduction, point_reduction)
    fairness = build_fairness_checks(spline, point)
    multi_verdict = build_multi_metric_verdict(spline, point)

    return {
        "spline": {
            "initial_drift": spline["initial_drift"],
            "final_drift": spline["final_drift"],
            "drift_reduction_pct": spline_reduction,
            "time_seconds": spline["time_seconds"],
            "final_view_drift": safe_last(spline["view_drifts"], spline["final_drift"]),
        },
        "pointcloud": {
            "initial_drift": point["initial_drift"],
            "final_drift": point["final_drift"],
            "drift_reduction_pct": point_reduction,
            "time_seconds": point["time_seconds"],
            "final_view_drift": safe_last(point["view_drifts"], point["final_drift"]),
            "initial_chamfer": point["initial_chamfer"],
            "final_chamfer": point["final_chamfer"],
            "final_view_chamfer": safe_last(point["view_chamfers"], point["final_chamfer"])
            if point["final_chamfer"] is not None
            else None,
        },
        "delta": {
            "drift_reduction_pct_spline_minus_pointcloud": spline_reduction - point_reduction,
            "final_drift_pointcloud_minus_spline": point["final_drift"] - spline["final_drift"],
            "time_seconds_pointcloud_minus_spline": point["time_seconds"] - spline["time_seconds"],
        },
        "fairness_checks": fairness,
        "winner_multi_metric": multi_verdict,
        "winner_by_drift_reduction": winner,
    }


def write_markdown(summary, output_path, spline_path, point_path):
    s = summary["spline"]
    p = summary["pointcloud"]
    d = summary["delta"]
    fairness = summary.get("fairness_checks", {})
    multi = summary.get("winner_multi_metric", {})

    lines = [
        "# Baseline Comparison",
        "",
        f"- Spline results: `{spline_path}`",
        f"- Point-cloud results: `{point_path}`",
        "",
        "## Metrics",
        "",
        "| Method | Initial Drift | Final Drift | Drift Reduction (%) | Time (s) |",
        "|---|---:|---:|---:|---:|",
        f"| Spline memory | {s['initial_drift']:.4f} | {s['final_drift']:.4f} | {s['drift_reduction_pct']:.2f} | {s['time_seconds']:.1f} |",
        f"| Point-cloud baseline | {p['initial_drift']:.4f} | {p['final_drift']:.4f} | {p['drift_reduction_pct']:.2f} | {p['time_seconds']:.1f} |",
        "",
    ]

    if p["initial_chamfer"] is not None and p["final_chamfer"] is not None:
        lines.extend(
            [
                "## Point-Cloud Chamfer",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Initial Chamfer | {float(p['initial_chamfer']):.4f} |",
                f"| Final Chamfer | {float(p['final_chamfer']):.4f} |",
                "",
            ]
        )

    lines.extend(
        [
            "## Delta (Spline - Point Cloud)",
            "",
            f"- Drift reduction delta: `{d['drift_reduction_pct_spline_minus_pointcloud']:+.2f}%`",
            f"- Final drift delta (point - spline): `{d['final_drift_pointcloud_minus_spline']:+.4f}`",
            f"- Runtime delta (point - spline): `{d['time_seconds_pointcloud_minus_spline']:+.1f}s`",
            "",
            f"**Winner by drift reduction:** `{summary['winner_by_drift_reduction']}`",
            "",
        ]
    )

    lines.extend(
        [
            "## Multi-Metric Verdict",
            "",
            f"- By drift reduction: `{multi.get('by_drift_reduction_pct', 'n/a')}`",
            f"- By final drift: `{multi.get('by_final_drift', 'n/a')}`",
            f"- By runtime: `{multi.get('by_runtime', 'n/a')}`",
            f"- Majority vote: `{multi.get('overall_majority_vote', 'n/a')}`",
            "",
            "## Fairness Checks",
            "",
            f"- Parameter budget match: `{fairness.get('budget_match', 'unknown')}`",
            f"- Curve count match: `{fairness.get('curve_count_match', 'unknown')}`",
            f"- Source K match: `{fairness.get('source_k_match', 'unknown')}`",
            f"- Granularity match (spline K vs point ppc): `{fairness.get('granularity_match', 'unknown')}`",
            "",
        ]
    )

    warnings = fairness.get("warnings", [])
    if warnings:
        lines.append("### Fairness Warnings")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_curves(spline, point, output_path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False, "matplotlib unavailable"

    spline_drifts = to_list(spline["view_drifts"])
    point_drifts = to_list(point["view_drifts"])

    if len(spline_drifts) == 0 and len(point_drifts) == 0:
        return False, "no view trajectories in inputs"

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Drift trajectories
    if len(spline_drifts) > 0:
        x = np.arange(len(spline_drifts))
        axes[0].plot(x, spline_drifts, "b-", label="Spline")
    if len(point_drifts) > 0:
        x = np.arange(len(point_drifts))
        axes[0].plot(x, point_drifts, "m-", label="Point cloud")
    axes[0].set_title("Drift per View")
    axes[0].set_xlabel("View index")
    axes[0].set_ylabel("Drift")
    axes[0].legend()

    # Final metric bars
    labels = ["Spline", "Point cloud"]
    final_drifts = [spline["final_drift"], point["final_drift"]]
    reductions = [spline["drift_reduction_pct"], point["drift_reduction_pct"]]
    axes[1].bar(labels, final_drifts, color=["#4C72B0", "#C44E52"])
    axes[1].set_title("Final Drift")
    axes[1].set_ylabel("Drift")
    axes[2].bar(labels, reductions, color=["#4C72B0", "#C44E52"])
    axes[2].set_title("Drift Reduction (%)")
    axes[2].set_ylabel("%")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return True, None


def main():
    parser = argparse.ArgumentParser(description="Compare spline vs point-cloud baseline runs")
    parser.add_argument(
        "--spline-results",
        default="outputs/my_dense_run/opt_results.pt",
        help="Path to run_dense.py opt_results.pt",
    )
    parser.add_argument(
        "--point-results",
        default="outputs/pointcloud_baseline/point_baseline_results.pt",
        help="Path to point baseline results (.pt or .json)",
    )
    parser.add_argument("--output-dir", default="outputs/baseline_compare")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    spline_raw = load_torch(args.spline_results)
    point_raw = load_point_results(args.point_results)
    spline = normalize_spline_dict(spline_raw)
    point = normalize_point_dict(point_raw)

    summary = build_summary(spline, point)

    json_path = os.path.join(args.output_dir, "baseline_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md_path = os.path.join(args.output_dir, "baseline_comparison.md")
    write_markdown(summary, md_path, args.spline_results, args.point_results)

    plot_path = os.path.join(args.output_dir, "baseline_comparison_curves.png")
    ok, reason = plot_curves(spline, point, plot_path)

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
