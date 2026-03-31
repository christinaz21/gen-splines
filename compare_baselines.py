"""
compare_baselines.py — Compare spline sequential results vs point-cloud baseline.

Usage:
    python compare_baselines.py \
        --spline outputs/sequential/sequential_results.pt \
        --pointcloud outputs/pointcloud_baseline/pointcloud_baseline_results.pt \
        --output-dir outputs/comparison
"""

import os
import argparse
import torch


def safe_get(d: dict, key: str, default=None):
    return d[key] if key in d else default


def summarize_spline(results: dict) -> dict:
    final_metrics = safe_get(results, "final_metrics", {})
    view_drifts = safe_get(results, "view_drifts", [])
    view_losses = safe_get(results, "view_losses", [])
    revisit = safe_get(results, "revisit_losses", {})
    initial = safe_get(results, "initial_drift", None)
    final_cp = safe_get(final_metrics, "cp_drift", None)
    curvature = safe_get(final_metrics, "curvature_deviation", None)

    avg_revisit = None
    if isinstance(revisit, dict) and len(revisit) > 0:
        avg_revisit = sum(float(v) for v in revisit.values()) / len(revisit)

    return {
        "name": "Spline (persistent curves)",
        "initial_drift": float(initial) if initial is not None else None,
        "final_drift": float(final_cp) if final_cp is not None else None,
        "avg_reproj_per_view": float(sum(view_losses) / len(view_losses)) if view_losses else None,
        "avg_revisit_reproj": float(avg_revisit) if avg_revisit is not None else None,
        "curvature_dev": float(curvature) if curvature is not None else None,
        "trajectory": [float(v) for v in view_drifts],
    }


def summarize_pointcloud(results: dict) -> dict:
    view_drift = safe_get(results, "view_point_drift", [])
    view_reproj = safe_get(results, "view_reproj_losses", [])
    initial = safe_get(results, "initial_point_drift", None)
    final_drift = safe_get(results, "final_point_drift", None)
    avg_revisit = safe_get(results, "avg_revisit", None)
    final_chamfer = safe_get(results, "final_chamfer", None)

    return {
        "name": "Point cloud (unconstrained)",
        "initial_drift": float(initial) if initial is not None else None,
        "final_drift": float(final_drift) if final_drift is not None else None,
        "avg_reproj_per_view": float(sum(view_reproj) / len(view_reproj)) if view_reproj else None,
        "avg_revisit_reproj": float(avg_revisit) if avg_revisit is not None else None,
        "curvature_dev": None,
        "final_chamfer": float(final_chamfer) if final_chamfer is not None else None,
        "trajectory": [float(v) for v in view_drift],
    }


def percent_change(initial, final):
    if initial is None or final is None or abs(initial) < 1e-12:
        return None
    return (1.0 - final / initial) * 100.0


def fmt(v, digits=4):
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def print_table(spline: dict, pointcloud: dict):
    headers = [
        ("Metric", 30),
        (spline["name"], 30),
        (pointcloud["name"], 30),
    ]
    sep = "+" + "+".join("-" * w for _, w in headers) + "+"
    row_fmt = "|" + "|".join("{:<" + str(w) + "}" for _, w in headers) + "|"

    print("\n" + sep)
    print(row_fmt.format(*[h for h, _ in headers]))
    print(sep)

    metrics = [
        ("Initial drift", fmt(spline["initial_drift"]), fmt(pointcloud["initial_drift"])),
        ("Final drift", fmt(spline["final_drift"]), fmt(pointcloud["final_drift"])),
        (
            "Drift improvement (%)",
            fmt(percent_change(spline["initial_drift"], spline["final_drift"]), 2),
            fmt(percent_change(pointcloud["initial_drift"], pointcloud["final_drift"]), 2),
        ),
        ("Avg reproj / view", fmt(spline["avg_reproj_per_view"], 6), fmt(pointcloud["avg_reproj_per_view"], 6)),
        ("Avg revisit reproj", fmt(spline["avg_revisit_reproj"], 6), fmt(pointcloud["avg_revisit_reproj"], 6)),
        ("Curvature deviation", fmt(spline["curvature_dev"], 6), "n/a"),
        ("Final Chamfer", "n/a", fmt(pointcloud.get("final_chamfer"), 6)),
    ]

    for m in metrics:
        print(row_fmt.format(*m))
    print(sep + "\n")


def make_plot(spline: dict, pointcloud: dict, out_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plot skipped (matplotlib unavailable): {exc}")
        return

    s = spline["trajectory"]
    p = pointcloud["trajectory"]
    n = max(len(s), len(p))
    x = list(range(1, n + 1))

    plt.figure(figsize=(8, 4))
    if s:
        plt.plot(range(1, len(s) + 1), s, "b-o", markersize=3, label="Spline CP drift")
    if p:
        plt.plot(range(1, len(p) + 1), p, "m-o", markersize=3, label="Point-cloud drift")
    plt.xlabel("View index")
    plt.ylabel("Drift")
    plt.title("Sequential Memory Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved comparison plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare spline and point-cloud baseline results")
    parser.add_argument("--spline", type=str, default="outputs/sequential/sequential_results.pt")
    parser.add_argument("--pointcloud", type=str, default="outputs/pointcloud_baseline/pointcloud_baseline_results.pt")
    parser.add_argument("--output-dir", type=str, default="outputs/comparison")
    args = parser.parse_args()

    spline_raw = torch.load(args.spline, map_location="cpu")
    pointcloud_raw = torch.load(args.pointcloud, map_location="cpu")

    spline = summarize_spline(spline_raw)
    pointcloud = summarize_pointcloud(pointcloud_raw)

    print_table(spline, pointcloud)

    os.makedirs(args.output_dir, exist_ok=True)
    out_plot = os.path.join(args.output_dir, "baseline_comparison.png")
    make_plot(spline, pointcloud, out_plot)

    summary = {
        "spline": spline,
        "pointcloud": pointcloud,
        "drift_improvement_percent": {
            "spline": percent_change(spline["initial_drift"], spline["final_drift"]),
            "pointcloud": percent_change(pointcloud["initial_drift"], pointcloud["final_drift"]),
        },
    }
    torch.save(summary, os.path.join(args.output_dir, "baseline_comparison_summary.pt"))
    print(f"Saved summary: {os.path.join(args.output_dir, 'baseline_comparison_summary.pt')}")


if __name__ == "__main__":
    main()
