---
generation: 4
phase: 066-12
code_freeze: false
next_role: proposer
pending: []
last_propose: docs/reviews/2026-08-21-handoff-orchestrator-pickup.md
last_review: docs/reviews/2026-08-18-review-g4.md
open_questions: []
deadlocks: []
canon_generation: 4
---

# Architecture mailbox

**Pickup 2026-08-21.** Orchestrator build in progress. Residual-Ω code + tests are in-tree; residual campaign **not** on disk yet. Handoff: `docs/reviews/2026-08-21-handoff-orchestrator-pickup.md`. Plan: `docs/reviews/2026-08-21-orchestrator-residual-omega.md`. Official product remains Stage A MAP. No NUTS. No new DEC id. Do not resume `kinuv-KGAS066-f47bc9-lambda/campaign.json` (absolute Ω).

Native preview: `⟨w|V|²⟩≈2.59`, `s≈0.77`. **Fit array (066-6):** `n_row=881`, `n_chan=95`, `Δv=5.080 km/s`, `N=4`, `s=0.514`. Replica was 881×125 (wider buffer).

## 066 npz (local inventory)

- Local: `/Users/thbrown/kilogas/DR1/visibilities/KILOGAS066.npz` (native 43240×1920)
- CANFAR: `/arc/projects/KILOGAS/analysis/toby_sandbox/visibilities/KILOGAS066.npz`
- Ico / cube on CANFAR: `/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/` (`KGAS66_Ico_K_kms-1.fits`, sibling clipped cube). Laptop `kinms_test` path is absent on `/arc`.
- YAML `obs_freq_range` clips the receding side — do not use it as the trim
