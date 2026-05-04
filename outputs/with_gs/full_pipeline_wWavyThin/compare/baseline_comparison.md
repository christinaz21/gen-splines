# Baseline Comparison

- Spline results: `outputs/with_gs/full_pipeline_wWavyThin/spline/opt_results.pt`
- Point-cloud results: `outputs/with_gs/full_pipeline_wWavyThin/pointcloud/point_baseline_results.pt`
- Gaussian results: `outputs/with_gs/full_pipeline_wWavyThin/gaussian/gaussian_baseline_results.pt`

## Metrics

| Method | Initial Drift | Final Drift | Drift Reduction (%) | Time (s) |
|---|---:|---:|---:|---:|
| Spline memory | 0.5558 | 0.0946 | 82.99 | 140.1 |
| Point-cloud baseline | 0.5551 | 0.0033 | 99.40 | 79.5 |
| Gaussian splat baseline | 0.5551 | 0.0032 | 99.42 | 3024.1 |

**Winner by drift reduction:** `gaussian`

## Multi-Metric Verdict

- By drift reduction: `gaussian`
- By final drift: `gaussian`
- By runtime: `pointcloud`
- Majority vote: `gaussian`
