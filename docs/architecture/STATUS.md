---
generation: 7
phase: milestone-001-promoted
code_freeze: false
next_role: consultant
board: idle
build_licensed: false
pending:
  - exact-workflow-posterior-calibration
last_propose: null
last_review: null
last_review_a: null
last_review_b: null
user_review: docs/reviews/artifacts/2026-09-05-kgas007-nuts/
open_questions: []
deadlocks: []
canon_generation: 7
---

## Agent Run Status

* **Phase:** MILESTONE-001 promoted for KGAS066 and KGAS007
* **Last Action:** Ran clean two-start Stage A and Stage B visibility fits, rendered the complete image and visibility diagnostic suites, embedded KinMS comparisons, revalidated retained mixed NUTS draws, and promoted one immutable product per target
* **Decisions Made:** KGAS066 selects Stage B; KGAS007 selects Stage A because its Stage B fit hits the zero-speed bound and fails the oscillation gate; target values and site paths now live in versioned target configurations
* **Blockers / Gates:** posterior intervals remain uncalibrated; KGAS066 visibility residuals remain structured with velocity; no real-data inner slope may be quoted
* **Next Step:** run exact-workflow simulation-based calibration before promoting calibrated credible intervals

## Current products

| Target | Product | Status |
|---|---|---|
| KGAS066 | `results/production/KGAS066/kinuv-KGAS066-07b714-milestone1/` | Accepted MILESTONE-001 bundle; Stage B chi2 167302.963, Stage A delta chi2 versus V=0 35551.583, retained posterior max Rhat 1.00369 and min ESS 889 |
| KGAS007 | `results/production/KGAS007/kinuv-KGAS007-e1ee1a-milestone1/` | Accepted MILESTONE-001 bundle; Stage A chi2 122070.763 and delta chi2 versus V=0 6211.629; Stage B rejected for bound pressure and oscillation; retained posterior max Rhat 1.00214 and min ESS 1093 |

The authoritative artifact index is `/arc/projects/KILOGAS/analysis/toby_sandbox/results/MANIFEST.md`.

# Architecture mailbox

Closed review and experiment history is synthesized in [`../PRODUCTION_RECORD.md`](../PRODUCTION_RECORD.md). Add only current state here; completed cards should be incorporated into the production record and removed from `docs/reviews/` after closure.
