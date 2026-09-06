---
generation: 5
phase: operational-generalization
code_freeze: false
next_role: senior-registrar
board: idle
build_licensed: false
pending:
  - target-config-migration
  - campaign-config-validation
  - runner-path-decoupling
last_propose: null
last_review: null
last_review_a: null
last_review_b: null
user_review: docs/reviews/artifacts/2026-09-05-kgas007-nuts/
open_questions: []
deadlocks: []
canon_generation: 5
---

## Agent Run Status

* **Phase:** Data-agnostic production governance and configuration migration
* **Last Action:** Curated all legacy result and run directories into the indexed `results/production/` and `results/archive/` layout; removed the separate `kinuv_runs/` tree
* **Decisions Made:** kinUV is the sole production fitter; `ms2kinuv` is the separate CASA ETL companion; KGAS066 official MAP remains read-only; 30 km/s Ico remains locked; receding NUTS is the sole KGAS066 posterior; approaching mode is terminated; G4 is not licensed
* **Blockers / Gates:** posterior intervals are not calibrated; no real-data inner slope may be quoted; another production campaign is not licensed until target metadata and site paths leave Python entry points and validated configuration is recorded in run manifests
* **Next Step:** migrate remaining target metadata and site paths from Python entry points into validated `configs/targets/` and `configs/campaigns/` records before licensing another production campaign

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
