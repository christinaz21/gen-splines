# Baseline Comparison

- Spline results: `outputs/my_dense_run/opt_results.pt`
- Point-cloud results: `outputs/pointcloud_baseline/point_baseline_results.json`

## Metrics

| Method | Initial Drift | Final Drift | Drift Reduction (%) | Time (s) |
|---|---:|---:|---:|---:|
| Spline memory | 0.5588 | 0.0953 | 82.94 | 142.0 |
| Point-cloud baseline | 0.5551 | 0.0033 | 99.41 | 78.5 |

## Point-Cloud Chamfer

| Metric | Value |
|---|---:|
| Initial Chamfer | 0.2549 |
| Final Chamfer | 0.0024 |

## Delta (Spline - Point Cloud)

- Drift reduction delta: `-16.48%`
- Final drift delta (point - spline): `-0.0921`
- Runtime delta (point - spline): `-63.4s`

**Winner by drift reduction:** `pointcloud`

## Multi-Metric Verdict

- By drift reduction: `pointcloud`
- By final drift: `pointcloud`
- By runtime: `pointcloud`
- Majority vote: `pointcloud`

## Fairness Checks

- Parameter budget match: `True`
- Curve count match: `True`
- Source K match: `True`
- Granularity match (spline K vs point ppc): `True`
