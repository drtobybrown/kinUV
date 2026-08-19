---
generation: 4
phase: 066-8
code_freeze: false
next_role: proposer
pending: []
last_propose: docs/reviews/2026-08-19-handoff-canfar-map.md
last_review: docs/reviews/2026-08-18-review-g4.md
open_questions: []
deadlocks: []
canon_generation: 4
---

# Architecture mailbox

**User dispatch 2026-08-19.** Stage A MAP is **done on the laptop** (Δχ²=+26213, interior V_0). Next official artifact: CANFAR **CPU** `kinuv-KGAS066-{sha6}-map` after **git clone** `origin/dev` — same Stage A, `/arc` paths. **Not GPU.** **No NUTS.** **No rings.** Handoff: `docs/reviews/2026-08-19-handoff-canfar-map.md`. Gate 4 (`λ_reg` 20×5) still needs code after that job.

Native preview: `⟨w|V|²⟩≈2.59`, `s≈0.77`. **Fit array (066-6):** `n_row=881`, `n_chan=95`, `Δv=5.080 km/s`, `N=4`, `s=0.514`. Replica was 881×125 (wider buffer).

## 066 npz (local inventory)

- Local: `/Users/thbrown/kilogas/DR1/visibilities/KILOGAS066.npz` (native 43240×1920)
- CANFAR: same bytes (`Content-Length` 997305244)
- Ico / cube: `/Users/thbrown/kilogas/analysis/kinms_test/kgas066/KGAS66_Ico_K_kms-1.fits`, `KGAS66_clipped_cube.fits` (VOPT 8034–8536 km/s, 17×30 km/s)
- YAML `obs_freq_range` clips the receding side — do not use it as the trim
