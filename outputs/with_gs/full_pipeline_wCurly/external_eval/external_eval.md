# External Target Evaluation

- Model: `wCurly`
- Raw target points: `144000` from `data/hairmodels/wCurly.hair`
- Spline eval points: `6000`
- Point eval points: `6000`
- Gaussian eval points: `6000`

## Metrics

| Method | Approx Chamfer to Raw (lower better) | Held-out Render MSE (lower better) |
|---|---:|---:|
| Spline | 0.068556 | 0.014247 |
| Point cloud | 0.069744 | 0.014630 |
| Gaussian splat | 0.069744 | 0.014589 |

## Winners

- By approx Chamfer: `spline`
- By held-out render MSE: `spline`

## Notes

- Chamfer uses random subsets capped at `12000` points for tractability.
- Held-out views are interleaved (half-step offset) relative to training azimuth grid.
