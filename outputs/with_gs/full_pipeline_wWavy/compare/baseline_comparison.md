# Baseline Comparison

- Spline results: `outputs/with_gs/full_pipeline_wWavy/spline/opt_results.pt`
- Point-cloud results: `outputs/with_gs/full_pipeline_wWavy/pointcloud/point_baseline_results.pt`
- Gaussian results: `outputs/with_gs/full_pipeline_wWavy/gaussian/gaussian_baseline_results.pt`

## Metrics

| Method | Initial Drift | Final Drift | Drift Reduction (%) | Time (s) |
|---|---:|---:|---:|---:|
| Spline memory | 0.5581 | 0.0987 | 82.32 | 155.2 |
| Point-cloud baseline | 0.5551 | 0.0033 | 99.41 | 95.4 |
| Gaussian splat baseline | 0.5551 | 0.0032 | 99.43 | 2981.2 |

**Winner by drift reduction:** `gaussian`

## Multi-Metric Verdict

- By drift reduction: `gaussian`
- By final drift: `gaussian`
- By runtime: `pointcloud`
- Majority vote: `gaussian`
