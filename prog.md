# Progress Report: Generative Spline Fields with Persistent Curve Memory

**Project:** COS526/ECE576 — Neural Rendering  
**Authors:** Vineal Sunkara, Christina Zhang  
**Last Updated:** March 22, 2026

---

## Executive Summary

We are building a structured generative world model that represents thin 3D geometry
as cubic B-splines with persistent curve memory, aiming to reduce geometric drift
during sequential viewpoint updates. We have completed the spline parameterization,
differentiable optimization pipeline, and persistent memory evaluation, demonstrating
that anchor-regularized curve memory reduces control-point drift by 57.4% vs 48.1%
for the no-memory baseline.

**Status vs. Proposal Timeline:**

| Week | Planned | Status |
|------|---------|--------|
| 1-2 | PyTorch3D setup, spline parameterization, synthetic dataset | ✅ Complete |
| 3-4 | Train generative spline prior Gθ | ❌ Not started |
| 5-6 | Render-update loop, persistent memory | ✅ Complete (done early) |
| 7-8 | Revisitation benchmark, point-cloud baseline comparison | ⚠️ Partial (no baseline yet) |
| 8-9 | Analysis, evaluation, report | ⚠️ In progress |

**Critical gap:** The generative prior and point-cloud baseline comparison are not yet
implemented. These are core deliverables from the proposal.

---

## Phase 1: Environment Setup (Amarel HPC)

### Approach
Set up a conda environment on Rutgers Amarel cluster (CentOS 7, glibc 2.17) with
PyTorch + PyTorch3D for differentiable point-cloud rendering of B-spline curves.

### Issues Encountered

1. **glibc 2.17 incompatibility (CentOS 7):** Modern PyTorch (≥2.1) requires glibc 2.27+.
   - *Fix:* Cloned existing `ddpm` environment which had a working PyTorch 2.5.1+cu121.

2. **PyTorch3D — no git on GPU nodes:** `pip install git+...pytorch3d.git` failed.
   - *Fix:* Prebuilt wheel: `pip install pytorch3d -f https://...py310_cu121_pyt251/...`

3. **CUDA version mismatch:** Initially installed cu118 wheel, but env had cu121 PyTorch.
   - *Fix:* Installed matching cu121 PyTorch3D wheel.

4. **Storage quota exhaustion:** Conda caches filled home directory.
   - *Fix:* `conda clean --all && pip cache purge`; switched to pip-only installs.

### Final Working Environment
- NVIDIA A100-PCIE-40GB, CentOS 7.9, glibc 2.17
- Python 3.10, PyTorch 2.5.1+cu121, PyTorch3D 0.7.8

---

## Phase 2: Pre-Flight Gradient Check

### Approach
Created `step0_gradient_check.py` with 4 sub-checks: import verification, gradient flow
test, radius sweep, and optimization sanity check.

### Issues
- **PyTorch3D 0.7.8 returns 3-channel RGB, not 4-channel RGBA.** Code indexed `[..., 3]`.
  - *Fix:* Channel-count guards with `image.shape[-1] == 4` checks.

### Results
- Recommended radius: **0.002**
- Gradient flow confirmed, optimization reduced loss in 50 steps
- All checks passed in 25.3s

---

## Phase 3: Rendered Image Loss (Failed Approach)

### Approach v1 — Single-view rendered MSE loss
Pipeline: control points → B-spline eval → flat (N×M, 3) point cloud → PyTorch3D
point splatting → image → MSE loss vs GT rendered image.

### Results
| Experiment | Radius | Views | Init Noise | CP Drift Change |
|---|---|---|---|---|
| Baseline | 0.002 | 1 | 0.15 | 0.2465 → 0.2471 (**-0.2%**, worse) |
| Larger radius | 0.015 | 1 | 0.15 | 0.2465 → 0.2777 (**-12.7%**, worse) |
| Dense samples | 0.005 | 1 | 0.15 | 0.2465 → 0.2501 (**-1.5%**, worse) |
| Easy init | 0.005 | 1 | 0.05 | 0.0822 → 0.0925 (**-12.5%**, worse) |

**Every experiment: loss decreased but drift increased.** The optimizer found different
3D configurations producing similar 2D images (depth ambiguity).

### Approach v2 — Multi-view rendered loss + anchor regularization
Added: 6 simultaneous viewpoints, anchor regularization, smoothness regularization.

### Results
| Views | Anchor Weight | CP Drift Change |
|---|---|---|
| 6 | 0.10 | +3.0% (improved, marginal) |
| 6 | 0.05 | +3.1% (improved, marginal) |
| Sequential 36 views | 0.10 | +6.1% (improved, marginal) |
| Sequential 36 views | 0.05 | +7.0% (improved, marginal) |

### Root Cause Analysis
PyTorch3D's point renderer **destroys curve structure**: it flattens all curve points
into an unstructured point cloud, rendering each point as an independent circle. The
renderer has no knowledge that points belong to ordered curves. Gradients are sparse
(~1-4 pixels per point at radius 0.002) and lack curve coherence. No amount of
multi-view rendering or regularization can fix this fundamental information loss.

---

## Phase 4: Reprojection Loss (Working Approach)

### Approach
Replaced rendered image loss with **multi-view reprojection loss**:
- Project GT and predicted curve points to 2D screen coords via camera projection
- Loss = L2 between corresponding 2D points across multiple viewpoints
- Preserves curve identity (curve i ↔ curve i, point j ↔ point j)
- Every control point gets gradient signal every step
- Multiple views eliminate depth ambiguity

New file `losses.py` implements: `reprojection_loss()`, `anchor_loss()`,
`smoothness_loss()`, `curve_length_regularization()`.

### Multi-view Simultaneous Results
| Views | Steps | Init Noise | CP Drift Change |
|---|---|---|---|
| 8 | 1000 | 0.15 | 0.2465 → 0.1150 (**+53.3%**, major improvement) |
| 8 | 500 | 0.05 | 0.0822 → 0.0990 (-20.5%, overshoot — LR too high) |

### Ablation: Anchor Weight (Sequential, 36 views, 100 steps/view)

| Anchor Weight | Final CP Drift | Drift Reduction | 2nd Half Drift | Curvature Dev |
|---|---|---|---|---|
| 0.00 (no memory) | 0.1279 | 48.1% | 0.1291 | 2.160 |
| 0.01 | 0.1165 | 52.7% | 0.1136 | 2.107 |
| 0.02 | 0.1082 | 56.1% | 0.1067 | 2.088 |
| **0.05** | **0.1050** | **57.4%** | **0.1056** | **2.019** |
| 0.10 | 0.1066 | 56.8% | 0.1079 | 1.920 |
| 0.20 | 0.1097 | 55.5% | 0.1169 | 1.744 |

**Optimal anchor weight: ~0.05** — best drift reduction with good stability.
Diminishing returns / slight degradation above 0.1.

### High-Fidelity Run (72 views, 200 steps/view)
- CP drift: 0.2465 → 0.1067 (+56.7%)
- Stalled around view 46, converged to ~0.1062 floor
- 14,400 total steps, memory stable throughout

### Key Finding
Persistent curve memory (anchor_weight=0.05) outperforms no-memory baseline:
- **57.4% vs 48.1%** drift reduction (9.3 percentage point gap)
- Better 2nd-half stability (0.1056 vs 0.1291)
- All configurations report stable memory (no late-orbit degradation)

---

## Honest Assessment

### What Works
- Clean ablation with clear optimal anchor weight
- Persistent memory demonstrably improves over no-memory baseline
- Smooth experimental methodology, reproducible results

### Known Limitations (Must Address)

1. **Oracle correspondences:** The reprojection loss compares curve i point j of the
   prediction against curve i point j of GT. In a real system, these correspondences
   are unknown. This makes the current evaluation a proof-of-concept under ideal
   conditions, not a demonstration of the full pipeline. A reviewer would flag this.

2. **No generative prior:** The proposal promises Gθ(z) → control points, trained on
   synthetic data. This is the "generative" in "Generative Spline Fields" and it is
   completely unimplemented.

3. **No point-cloud baseline comparison:** The proposal explicitly promises comparison
   against point-cloud memory (Spatia-style). Current results only compare anchor
   weights within the spline framework.

4. **Convergence floor at ~0.105:** The optimization stalls at roughly 57% drift
   reduction regardless of steps/views. This may be fundamental (the B-spline
   parameterization can't perfectly represent the GT curves with K=8 control points)
   or may indicate a local minimum.

5. **Synthetic data only:** All experiments use procedurally generated helixes and
   waves. No real thin-structure data (hair, wires, fibers).

---

## Next Steps (Prioritized)

### Must-Do (to match proposal)

1. **Point-cloud baseline (1-2 hours):** Implement the same sequential optimization
   but with raw 3D point positions instead of spline control points. Compare drift.
   This is the most important missing piece — it directly tests the paper's hypothesis
   that structured curve memory outperforms unstructured point memory.

2. **Generative prior Gθ (3-5 hours):** Train the SplineGenerator MLP to map latent
   codes to control point configurations. Then show that the generator can produce
   novel valid spline scenes AND that persistent memory works when initialized from
   the generator output.

### Should-Do (strengthen the paper)

3. **Correspondence-free evaluation:** Add a Chamfer distance metric between predicted
   and GT point clouds (order-agnostic). This addresses the oracle correspondence
   limitation.

4. **Ablation on K (control points per curve):** Test K=4, 6, 8, 12 to see if the
   convergence floor is related to spline expressiveness.

5. **Comparison figures:** Generate side-by-side rendered images and drift-over-orbit
   plots overlaying all configurations.

### Nice-to-Have (stretch goals)

6. Real thin-structure datasets (hair strands from CHARM dataset)
7. Video generation integration
8. Dense curve fields (100+ curves)

---

## File Structure

```
spline_fields/
├── spline.py                  # B-spline math + SplineField + SplineGenerator
├── renderer.py                # PyTorch3D rendering + radius sweep
├── losses.py                  # Reprojection, anchor, smoothness losses
├── dataset.py                 # Synthetic strand scene generation
├── metrics.py                 # CP drift, curvature deviation
├── step0_gradient_check.py    # Pre-flight gradient verification
├── optimize.py                # Multi-view reprojection optimization
├── optimize_sequential.py     # Sequential persistent memory experiment
├── PROGRESS_REPORT.md         # This file
├── scripts/
│   ├── find_conda.sh
│   ├── setup_env.sh
│   ├── setup.slurm
│   └── run_pipeline.slurm
└── outputs/
    ├── reproj_test/           # Multi-view reprojection (53.3% improvement)
    ├── seq_no_memory/         # Baseline: no anchor (48.1%)
    ├── seq_memory/            # Persistent memory (57.4%)
    ├── ablation_anchor_*/     # Anchor weight sweep (0.0 to 0.2)
    └── seq_hifi/              # High-fidelity 72-view run
```
