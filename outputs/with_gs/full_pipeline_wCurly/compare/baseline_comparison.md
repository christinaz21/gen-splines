# Baseline Comparison

- Spline results: `outputs/with_gs/full_pipeline_wCurly/spline/opt_results.pt`
- Point-cloud results: `outputs/with_gs/full_pipeline_wCurly/pointcloud/point_baseline_results.pt`
- Gaussian results: `outputs/with_gs/full_pipeline_wCurly/gaussian/gaussian_baseline_results.pt`

## Metrics

| Method | Initial Drift | Final Drift | Drift Reduction (%) | Time (s) |
|---|---:|---:|---:|---:|
| Spline memory | 0.5626 | 0.0931 | 83.46 | 143.1 |
| Point-cloud baseline | 0.5551 | 0.0032 | 99.43 | 82.2 |
| Gaussian splat baseline | 0.5551 | 0.0031 | 99.44 | 3025.2 |

**Winner by drift reduction:** `gaussian`

## Multi-Metric Verdict

- By drift reduction: `gaussian`
- By final drift: `gaussian`
- By runtime: `pointcloud`
- Majority vote: `gaussian`
