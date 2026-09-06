---
generation: 9
phase: s0-implementation-review
code_freeze: false
next_role: independent-code-reviewers
board: s0-dual-review-pending
build_licensed: true
pending:
  - s0-independent-code-reviews
  - nonrotation-null-bootstrap
  - stage-b-smoothness-recalibration
  - s1-fair-comparator
  - exact-workflow-posterior-calibration
last_propose: docs/decisions/DEC-KINUV-CROSSDOMAIN-RECOVERY.md
last_review: null
last_review_a: null
last_review_b: null
user_review: docs/reviews/artifacts/2026-09-05-kgas007-nuts/
open_questions: []
deadlocks: []
canon_generation: 9
---

## Agent Run Status

* **Phase:** S0 scientific-accounting implementation is licensed and awaiting dual code review; MILESTONE-001 remains a sealed historical engineering baseline
* **Last Action:** Implemented explicit blank and fitted non-rotating likelihood paths, prospective rotation-test accounting, and dimensionally correct Stage B omega provenance
* **Decisions Made:** `chi2_blank` detects emission; rotation requires `chi2_nonrot` with matched brightness/nuisance fitting. `Omega=|Delta2 V|/|Delta v_chan|` is dimensionless. The historical `Omega<0.3` applies only to the 20-mock KGAS066 exact-family residual-omega calibration and is not a universal production threshold.
* **Blockers / Gates:** Reviewer A/B S0 verdicts; at least 199 complete non-rotating null refits for a rotation claim; a newly registered mock-calibrated Stage B criterion; legacy KGAS007 grouping may block valid real holdout; posterior intervals remain uncalibrated
* **Next Step:** close S0 only after dual review, then execute S1 intrinsic KinMS operator-parity closure before any new campaign

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
optimizer budgets, the version-2 gate schema, and both PA starts. These values are prospective diagnostics,
not promoted rotation detections:

| Target | `chi2_blank` | fitted `chi2_nonrot` | refitted `chi2_rot` | `delta_chi2_nonrot` | Status |
|---|---:|---:|---:|---:|---|
| KGAS066 | 204228.248 | 200023.444 | 168526.073 | 31497.371 | Delta gate passes; bootstrap pending |
| KGAS007 | 128282.392 | 126567.540 | 122144.496 | 4423.045 | Delta gate passes; bootstrap pending |

Both non-rotating fits reached the allowed 50 km/s dispersion ceiling;
KGAS066 also reached the lower systemic-velocity bound and the +2 arcsec
declination-offset bound. This boundary
pressure is retained as diagnostic evidence and does not authorize wider
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
