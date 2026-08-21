---
generation: 4
phase: 066-12
code_freeze: false
next_role: proposer
pending: []
last_propose: docs/reviews/2026-08-21-report-gate4-stage-b.md
last_review: docs/reviews/2026-08-18-review-g4.md
open_questions: []
deadlocks: []
canon_generation: 4
---

# Architecture mailbox

**User dispatch 2026-08-21.** Gate 4 finished on CANFAR. Full report: `docs/reviews/2026-08-21-report-gate4-stage-b.md` (+ bundled JSON under `docs/reviews/artifacts/2026-08-21-gate4/`). Resume completed λ=100 mocks 15–19 at N_rings=7; `select_lambda_reg` → **None**. ADR N_rings=8 full 20×5 also → **None** (`chosen.json`). OSCMETRIC conflict: only λ=100 passes Ω<0.3, but arctan recovery is high (V₀≈217 vs 200). **No Stage B on real 066.** Official kinematic product remains Stage A MAP (`kinuv-KGAS066-f47bc9-map`, Δχ²=+26213). **No NUTS.** CPU only.

Native preview: `⟨w|V|²⟩≈2.59`, `s≈0.77`. **Fit array (066-6):** `n_row=881`, `n_chan=95`, `Δv=5.080 km/s`, `N=4`, `s=0.514`. Replica was 881×125 (wider buffer).

## 066 npz (local inventory)

- Local: `/Users/thbrown/kilogas/DR1/visibilities/KILOGAS066.npz` (native 43240×1920)
- CANFAR: `/arc/projects/KILOGAS/analysis/toby_sandbox/visibilities/KILOGAS066.npz`
- Ico / cube on CANFAR: `/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/` (`KGAS66_Ico_K_kms-1.fits`, sibling clipped cube). Laptop `kinms_test` path is absent on `/arc`.
- YAML `obs_freq_range` clips the receding side — do not use it as the trim
