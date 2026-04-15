# Baseline Comparison

- Spline results: `outputs/full_pipeline_wWavyThin/spline/opt_results.pt`
- Point-cloud results: `outputs/full_pipeline_wWavyThin/pointcloud/point_baseline_results.pt`

## Metrics

| Method | Initial Drift | Final Drift | Drift Reduction (%) | Time (s) |
|---|---:|---:|---:|---:|
| Spline memory | 0.5599 | 0.0962 | 82.82 | 142.3 |
| Point-cloud baseline | 0.5551 | 0.0033 | 99.40 | 80.4 |

## Point-Cloud Chamfer

| Metric | Value |
|---|---:|
| Initial Chamfer | 0.2632 |
| Final Chamfer | 0.0021 |

## Delta (Spline - Point Cloud)

- Drift reduction delta: `-16.59%`
- Final drift delta (point - spline): `-0.0929`
- Runtime delta (point - spline): `-61.9s`

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
