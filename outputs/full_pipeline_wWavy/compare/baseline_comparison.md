# Baseline Comparison

- Spline results: `outputs/full_pipeline_wWavy/spline/opt_results.pt`
- Point-cloud results: `outputs/full_pipeline_wWavy/pointcloud/point_baseline_results.pt`

## Metrics

| Method | Initial Drift | Final Drift | Drift Reduction (%) | Time (s) |
|---|---:|---:|---:|---:|
| Spline memory | 0.5603 | 0.0940 | 83.23 | 140.7 |
| Point-cloud baseline | 0.5551 | 0.0033 | 99.41 | 78.2 |

## Point-Cloud Chamfer

| Metric | Value |
|---|---:|
| Initial Chamfer | 0.2539 |
| Final Chamfer | 0.0024 |

## Delta (Spline - Point Cloud)

- Drift reduction delta: `-16.18%`
- Final drift delta (point - spline): `-0.0907`
- Runtime delta (point - spline): `-62.5s`

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
