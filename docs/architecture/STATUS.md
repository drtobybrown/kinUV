---
generation: 13
phase: s1-blocked-consultant-decision
code_freeze: true
next_role: consultant-astra
board: s1-renderer-convergence-escalated
build_licensed: false
pending:
  - nonrotation-null-bootstrap
  - stage-b-smoothness-recalibration
  - s1-dual-code-review
  - s1-renderer-convergence-decision
  - s2-geometry-covariance
  - exact-workflow-posterior-calibration
last_propose: docs/decisions/DEC-KINUV-CROSSDOMAIN-RECOVERY.md
last_review: docs/reviews/2026-09-06-code-review-b-crossdomain-s1.md
last_review_a: docs/reviews/2026-09-06-code-review-a-crossdomain-s1.md
last_review_b: docs/reviews/2026-09-06-code-review-b-crossdomain-s1.md
user_review: docs/reviews/artifacts/2026-09-05-kgas007-nuts/
open_questions:
  - s1-kinms-antialias-deposition-contract
  - s2-inclination-prior-provenance
  - s2-covariance-bmaj-and-bound-freeze
deadlocks:
  - s1-target-path-resolution-gate
canon_generation: 13
---

## Agent Run Status

* **Phase:** S1 bounded review repairs are complete at `0c95240`, but full target-path resolution convergence failed and requires an Astra renderer decision
* **Last Action:** Published a sealed failed-gate dossier after independently doubling the image, radial, azimuthal, dispersion, and spectral integration axes
* **Decisions Made:** `chi2_blank` detects emission; rotation requires `chi2_nonrot` with matched brightness/nuisance fitting. `Omega=|Delta2 V|/|Delta v_chan|` is dimensionless. The historical `Omega<0.3` applies only to the 20-mock KGAS066 exact-family residual-omega calibration and is not a universal production threshold.
* **Blockers / Gates:** KinMS nearest-cell/Gauss-Hermite rendering is not converged at `|Delta chi2| <= 0.1`; S2 also lacks an approved inclination-prior uncertainty/source and KGAS007 grouping metadata
* **Next Step:** Astra resolves [`ESC-KINUV-S1-RENDERER-CONVERGENCE`](../decisions/ESC-KINUV-S1-RENDERER-CONVERGENCE.md); a revised transform specification requires two fresh reviews before implementation resumes

## S1 operator/comparator closure

The first dossier at `results/validation/crossdomain-recovery-s1-20260906/`
is retained as pre-review evidence. The revised read-only dossier is
`results/validation/crossdomain-recovery-s1-20260906-r2/`; it is a failed-gate
record and is not promoted. No fit, bootstrap, posterior, NUTS, or G4 campaign
was run.

| Gate | KGAS066 | KGAS007 | Limit |
|---|---:|---:|---:|
| Analytic complex-visibility relative L2 | 4.70e-14 | 4.70e-14 | <=1e-6 |
| Analytic noise-normalized component RMS | 1.04e-12 | 1.04e-12 | <=0.001 |
| Zero-baseline flux relative error | 3.70e-15 | 3.70e-15 | <=0.1% |
| Native centroid error (channel) | <1e-9 | <1e-9 | <=0.02 |
| Doubled-sampling delta chi2 | 2.40e-5 | 2.40e-5 | <=0.1 |
| Nominal/high render RMS (thermal SD) | 6.73e-5 | 1.58e-5 | <=0.1 |
| Independent high-repeat RMS (thermal SD) | 4.58e-5 | 1.13e-5 | <=0.1 |
| Nominal/high absolute chi2 change | 0.01901 | 0.003332 | <=0.1 |
| Independent/high absolute chi2 change | 0.02640 | 0.000458 | <=0.1 |
| Maximum per-axis doubled-sampling absolute chi2 change | **896.724** | **143.100** | <=0.1; **FAIL** |
| Signed worker PA error | 0.001397 deg | 0.001397 deg | <=3 deg |
| S0 baseline replay absolute chi2 error | 0.0 | 0.0 | <=0.1 |

## Corrected prospective rotation accounting

| Quantity | Definition | Threshold/status |
|---|---|---|
| `chi2_blank` | Data against zero complex visibilities | Emission-detection diagnostic only |
| `chi2_nonrot` | Best-fit emitting disk with circular speed fixed to zero; flux, systemic velocity, dispersion, and center refitted | Required comparator |
| `chi2_rot` | Best-fit rotating visibility model, prior excluded | Required comparator |
| `delta_chi2_blank` | `chi2_blank - chi2_rot` | No rotation claim |
| `delta_chi2_nonrot` | `chi2_nonrot - chi2_rot` | At least 25 prospectively |
| Null bootstrap | `(k+1)/(M+1)` after complete refits with actual covariance | At least 199 trials and `p <= 0.01` |
| Stage B `max_omega_dimensionless` | `max(abs(Delta2(V_ring - V_reference)) / abs(Delta v_chan))` | Blocked until target/regime-specific mock calibration is registered |

The S0 MAP-only accounting check used the audited Ico template noise from the
official propagated error map, unchanged target physical parameters and
optimizer budgets, the version-2 gate schema, and both PA starts. These values
are prospective diagnostics,
not promoted rotation detections:

| Target | `chi2_blank` | fitted `chi2_nonrot` | refitted `chi2_rot` | `delta_chi2_nonrot` | Status |
|---|---:|---:|---:|---:|---|
| KGAS066 | 204228.248 | 200023.444 | 168526.073 | 31497.371 | Delta gate passes; bootstrap pending |
| KGAS007 | 128282.392 | 126567.540 | 122144.496 | 4423.045 | Delta gate passes; bootstrap pending |

Both non-rotating fits reached the allowed 50 km/s dispersion ceiling;
KGAS066 also reached the lower systemic-velocity bound and the +2 arcsec
declination-offset bound. This boundary pressure is retained as diagnostic
evidence and does not authorize wider
bounds. The durable run record and reproducibility manifest are in
`results/validation/crossdomain-recovery-s0-20260906/`.

## Current products

| Target | Product | Status |
|---|---|---|
| KGAS066 | `results/production/KGAS066/kinuv-KGAS066-4c1bc4-milestone1/` | Sealed historical engineering baseline; reported delta chi2 is versus blank complex signal and intervals are conditional/uncalibrated |
| KGAS007 | `results/production/KGAS007/kinuv-KGAS007-e1ee1a-milestone1/` | Sealed historical engineering baseline; delta chi2 6211.629 is versus blank complex signal, not non-rotation; intervals are conditional/uncalibrated |

The authoritative artifact index is `/arc/projects/KILOGAS/analysis/toby_sandbox/results/MANIFEST.md`.

# Architecture mailbox

Closed review and experiment history is synthesized in [`../PRODUCTION_RECORD.md`](../PRODUCTION_RECORD.md). Add only current state here; completed cards should be incorporated into the production record and removed from `docs/reviews/` after closure.
