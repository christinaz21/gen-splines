# Baseline Comparison

- Spline results: `outputs/full_pipeline_wStraight_2/spline/opt_results.pt`
- Point-cloud results: `outputs/full_pipeline_wStraight_2/pointcloud/point_baseline_results.pt`

## Metrics

| Method | Initial Drift | Final Drift | Drift Reduction (%) | Time (s) |
|---|---:|---:|---:|---:|
| Spline memory | 0.5597 | 0.0944 | 83.14 | 147.8 |
| Point-cloud baseline | 0.5551 | 0.0033 | 99.41 | 78.1 |

## Point-Cloud Chamfer

| Metric | Value |
|---|---:|
| Initial Chamfer | 0.2549 |
| Final Chamfer | 0.0024 |

## Delta (Spline - Point Cloud)

- Drift reduction delta: `-16.27%`
- Final drift delta (point - spline): `-0.0911`
- Runtime delta (point - spline): `-69.7s`

**Winner by drift reduction:** `pointcloud`

## Multi-Metric Verdict

- By drift reduction: `pointcloud`
- By final drift: `pointcloud`
- By runtime: `pointcloud`
- Majority vote: `pointcloud`

## Fairness Checks

- Parameter budget match: `None`
- Curve count match: `None`
- Source K match: `None`
- Granularity match (spline K vs point ppc): `None`

### Fairness Warnings

- Could not fully verify parameter budget from provided files.
