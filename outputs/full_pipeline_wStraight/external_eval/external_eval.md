# External Target Evaluation

- Model: `wStraight`
- Raw target points: `75000` from `data/hairmodels/wStraight.hair`
- Spline eval points: `6000`
- Point eval points: `6000`

## Metrics

| Method | Approx Chamfer to Raw (lower better) | Held-out Render MSE (lower better) |
|---|---:|---:|
| Spline | 0.075729 | 0.018137 |
| Point cloud | 0.078038 | 0.018915 |

## Winners

- By approx Chamfer: `spline`
- By held-out render MSE: `spline`

## Notes

- Chamfer uses random subsets capped at `12000` points for tractability.
- Held-out views are interleaved (half-step offset) relative to training azimuth grid.
