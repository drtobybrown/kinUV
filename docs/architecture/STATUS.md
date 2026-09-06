---
generation: 4
phase: production-closeout
code_freeze: false
next_role: implementer
board: idle
build_licensed: true
pending: []
last_propose: null
last_review: null
last_review_a: null
last_review_b: null
user_review: docs/reviews/artifacts/2026-09-05-kgas007-nuts/
open_questions: []
deadlocks: []
canon_generation: 4
---

## Agent Run Status

* **Phase:** KGAS066 and KGAS007 production closeout
* **Last Action:** kinUV was decoupled from its archived predecessors; KGAS007 target metadata and visibility compatibility loading are now kinUV-owned
* **Decisions Made:** kinUV is the sole production fitter; `ms2kinuv` is the separate CASA ETL companion; KGAS066 official MAP remains read-only; 30 km/s Ico remains locked; receding NUTS is the sole KGAS066 posterior; approaching mode is terminated; G4 is not licensed
* **Blockers / Gates:** posterior intervals are not calibrated; no real-data inner slope may be quoted
* **Next Step:** run exact-workflow SBC, then re-export KGAS007 through the canonical `ms2kinuv-npz-v1` schema when its source Measurement Set is available

## Current products

| Target | Product | Status |
|---|---|---|
| KGAS066 | `results/KILOGAS066/kinuv-KGAS066-uvsign-map/` | Official Stage A MAP; PA 199.730 deg, chi2 168675.596, delta chi2 versus V=0 35552.652 |
| KGAS066 | Stage B N=7, lambda=0 | chi2 167302.187; improvement over Stage A 1373.409 |
| KGAS066 | NUTS `sd3ckpf2` | Mixed; max Rhat 1.004, min ESS 889; intervals uncalibrated |
| KGAS007 | Stage A MAP | chi2 122070.763; delta chi2 versus V=0 6211.629 |
| KGAS007 | merged NUTS | Mixed; max Rhat 1.00214, min ESS 1093; intervals uncalibrated |

# Architecture mailbox

Closed review and experiment history is synthesized in [`../PRODUCTION_RECORD.md`](../PRODUCTION_RECORD.md). Add only current state here; completed cards should be incorporated into the production record and removed from `docs/reviews/` after closure.
