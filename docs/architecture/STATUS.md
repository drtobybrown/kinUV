---
generation: 12
phase: s1-required-changes
code_freeze: false
next_role: implementer-s1-repair
board: s1-code-review-required-changes
build_licensed: true
pending:
  - nonrotation-null-bootstrap
  - stage-b-smoothness-recalibration
  - s1-dual-code-review
  - s1-pa-contract-and-convergence-repair
  - s2-geometry-covariance
  - exact-workflow-posterior-calibration
last_propose: docs/decisions/DEC-KINUV-CROSSDOMAIN-RECOVERY.md
last_review: docs/reviews/2026-09-06-code-review-b-crossdomain-s1.md
last_review_a: docs/reviews/2026-09-06-code-review-a-crossdomain-s1.md
last_review_b: docs/reviews/2026-09-06-code-review-b-crossdomain-s1.md
user_review: docs/reviews/artifacts/2026-09-05-kgas007-nuts/
open_questions: []
deadlocks: []
canon_generation: 12
---

## Agent Run Status

* **Phase:** S1 exact commit `1b629a1` received dual `accept-with-required-changes`; bounded repairs are active and S2 has not begun
* **Last Action:** Reviewer A falsified the KinMS/kinUV PA convention and convergence coverage; Reviewer B falsified the fail-closed metadata and dossier-publication contracts
* **Decisions Made:** `chi2_blank` detects emission; rotation requires `chi2_nonrot` with matched brightness/nuisance fitting. `Omega=|Delta2 V|/|Delta v_chan|` is dimensionless. The historical `Omega<0.3` applies only to the 20-mock KGAS066 exact-family residual-omega calibration and is not a universal production threshold.
* **Blockers / Gates:** at least 199 complete non-rotating null refits for a rotation claim; a newly registered mock-calibrated Stage B criterion; legacy KGAS007 grouping may block valid real holdout; posterior intervals remain uncalibrated
* **Next Step:** repair all seven required findings, regenerate the S1 dossier under unchanged thresholds, and obtain two fresh exact-commit verdicts

## S1 operator/comparator closure

The durable dossier is
`results/validation/crossdomain-recovery-s1-20260906/`. No fit, bootstrap,
posterior, NUTS, or G4 campaign was run. The external KinMS process emitted
intrinsic native-channel cubes with no restoring beam, primary beam, or
spectral response. kinUV then applied its production measurement operator.

| Gate | KGAS066 | KGAS007 | Limit |
|---|---:|---:|---:|
| Analytic complex-visibility relative L2 | 4.70e-14 | 4.70e-14 | <=1e-6 |
| Analytic noise-normalized component RMS | 1.04e-12 | 1.04e-12 | <=0.001 |
| Zero-baseline flux relative error | 3.70e-15 | 3.70e-15 | <=0.1% |
| Native centroid error (channel) | <1e-9 | <1e-9 | <=0.02 |
| Doubled-sampling delta chi2 | 2.40e-5 | 2.40e-5 | <=0.1 |
| Nominal/high render RMS (thermal SD) | 1.34e-4 | 3.35e-5 | <=0.1 |
| Independent high-repeat RMS (thermal SD) | 6.76e-5 | 1.63e-5 | <=0.1 |
| High-cloud absolute chi2 change | 0.04294 | 0.02802 | <=0.1 |
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
