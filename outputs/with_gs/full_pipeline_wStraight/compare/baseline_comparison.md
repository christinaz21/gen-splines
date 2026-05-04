# Baseline Comparison

- Spline results: `outputs/with_gs/full_pipeline_wStraight/spline/opt_results.pt`
- Point-cloud results: `outputs/with_gs/full_pipeline_wStraight/pointcloud/point_baseline_results.pt`
- Gaussian results: `outputs/with_gs/full_pipeline_wStraight/gaussian/gaussian_baseline_results.pt`

## Metrics

| Method | Initial Drift | Final Drift | Drift Reduction (%) | Time (s) |
|---|---:|---:|---:|---:|
| Spline memory | 0.5554 | 0.0894 | 83.91 | 155.6 |
| Point-cloud baseline | 0.5551 | 0.0033 | 99.41 | 93.8 |
| Gaussian splat baseline | 0.5551 | 0.0031 | 99.44 | 2957.0 |

**Winner by drift reduction:** `gaussian`

## Multi-Metric Verdict

- By drift reduction: `gaussian`
- By final drift: `gaussian`
- By runtime: `pointcloud`
- Majority vote: `gaussian`
