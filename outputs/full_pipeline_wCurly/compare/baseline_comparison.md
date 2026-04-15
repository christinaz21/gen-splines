# Baseline Comparison

- Spline results: `outputs/full_pipeline_wCurly/spline/opt_results.pt`
- Point-cloud results: `outputs/full_pipeline_wCurly/pointcloud/point_baseline_results.pt`

## Metrics

| Method | Initial Drift | Final Drift | Drift Reduction (%) | Time (s) |
|---|---:|---:|---:|---:|
| Spline memory | 0.5607 | 0.0974 | 82.62 | 145.3 |
| Point-cloud baseline | 0.5551 | 0.0032 | 99.43 | 82.4 |

## Point-Cloud Chamfer

| Metric | Value |
|---|---:|
| Initial Chamfer | 0.2532 |
| Final Chamfer | 0.0022 |

## Delta (Spline - Point Cloud)

- Drift reduction delta: `-16.80%`
- Final drift delta (point - spline): `-0.0943`
- Runtime delta (point - spline): `-62.8s`

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
