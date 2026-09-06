---
generation: 6
phase: visibility-hot-path-decoupling
code_freeze: false
next_role: senior-registrar
board: idle
build_licensed: false
pending:
  - target-config-migration
  - campaign-config-validation
last_propose: null
last_review: null
last_review_a: null
last_review_b: null
user_review: docs/reviews/artifacts/2026-09-05-kgas007-nuts/
open_questions: []
deadlocks: []
canon_generation: 6
---

## Agent Run Status

* **Phase:** Visibility-likelihood hot-path decoupling and downstream comparator automation
* **Last Action:** Removed physical-scale conversion from core constants, opened the forward model to caller-supplied kinematic profiles, added mass-decomposition dependency guards, and implemented the two-target downstream KinMS benchmark
* **Decisions Made:** visibility chi2 is the sole scientific likelihood; baryonic/halo decomposition and cosmology are downstream-only; kinUV is the sole production fitter; `ms2kinuv` is the separate CASA ETL companion; KGAS066 official MAP remains read-only; G4 is not licensed
* **Blockers / Gates:** posterior intervals are not calibrated; no real-data inner slope may be quoted; another production campaign is not licensed until target metadata and site paths leave Python entry points and validated configuration is recorded in run manifests
* **Next Step:** validate the canonical KinMS comparison in an environment containing KinMS, then migrate remaining inference target metadata into validated configuration before licensing another production campaign

## Current products

| Target | Product | Status |
|---|---|---|
| KGAS066 | `results/production/KGAS066/kinuv-KGAS066-uvsign-map/` | Official Stage A MAP; PA 199.730 deg, chi2 168675.596, delta chi2 versus V=0 35552.652 |
| KGAS066 | `results/production/KGAS066/kinuv-KGAS066-uvsign-map/stage_b_map.json` | Stage B N=7, lambda=0; chi2 167302.187; improvement over Stage A 1373.409 |
| KGAS066 | `results/production/KGAS066/kinuv-KGAS066-3de838-nuts/` | Mixed; max Rhat 1.004, min ESS 889; intervals uncalibrated |
| KGAS007 | `results/production/KGAS007/kinuv-KGAS007-stage-a-map/` | Stage A MAP; chi2 122070.763; delta chi2 versus V=0 6211.629 |
| KGAS007 | `results/production/KGAS007/kinuv-KGAS007-32cbbd-nuts/` | Mixed; max Rhat 1.00214, min ESS 1093; intervals uncalibrated |

The authoritative artifact index is `/arc/projects/KILOGAS/analysis/toby_sandbox/results/MANIFEST.md`.

# Architecture mailbox

Closed review and experiment history is synthesized in [`../PRODUCTION_RECORD.md`](../PRODUCTION_RECORD.md). Add only current state here; completed cards should be incorporated into the production record and removed from `docs/reviews/` after closure.
