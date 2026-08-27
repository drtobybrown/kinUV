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

**2026-08-27.** Image-plane Stage B vs 10 km/s cube: `docs/diagnostics/stage-b-vs-imaging.md`, runner `scripts/plot_stage_b_vs_imaging.py`, figures `docs/reviews/artifacts/2026-08-27-stage-b-imaging/`. Not a new fit. No NUTS. No new DEC id.

**2026-08-24.** User: best vis fit; Ω 0.3 not a veto. **Stage B on real 066 is done** (`kinuv-KGAS066-f47bc9-map/stage_b_map.json`): N=7, λ=0, χ²=176879 vs Stage A 178016 (Δ=+1136), AIC prefers B. Grid: `.../lambda-resid/vis_fit/`. Stage A MAP kept as the arctan product. No NUTS. No new DEC id.

Native preview: `⟨w|V|²⟩≈2.59`, `s≈0.77`. **Fit array (066-6):** `n_row=881`, `n_chan=95`, `Δv=5.080 km/s`, `N=4`, `s=0.514`. Replica was 881×125 (wider buffer).

## 066 npz (local inventory)

- Local: `/Users/thbrown/kilogas/DR1/visibilities/KILOGAS066.npz` (native 43240×1920)
- CANFAR: `/arc/projects/KILOGAS/analysis/toby_sandbox/visibilities/KILOGAS066.npz`
- Ico / vis-trim: `/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/` (`KGAS66_Ico_K_kms-1.fits`, sibling clipped cube). Laptop `kinms_test` path is absent on `/arc`.
- Image-plane Stage B diagnostics: `.../KGAS66/10kms/` (`KGAS66_clipped_cube.fits`, `KGAS66_mask_cube.fits`).
- YAML `obs_freq_range` clips the receding side — do not use it as the trim
