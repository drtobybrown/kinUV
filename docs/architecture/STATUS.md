---
generation: 4
phase: 066-8
code_freeze: false
next_role: proposer
pending: []
last_propose: docs/reviews/2026-08-19-fix-066-8-vsys-pa.md
last_review: docs/reviews/2026-08-18-review-g4.md
open_questions: []
deadlocks: []
canon_generation: 4
---

# Architecture mailbox

**User dispatch 2026-08-19.** `dev`. 066-8 Stage A **interior** after radio vsys + PA ±180° two-start. Δχ² vs V=0 **+26213** (was blob +4341). vsys=8098.7 km/s (radio), PA=381.86°≡21.9°, V_0=268.4 km/s, σ=11.7 km/s, (dx,dy)=(−0.10″, −0.06″), r_t still 0.50″ floor, flux=60.6 Jy. Eval 0.98 s. No NUTS. No rings (`λ_reg` uncalibrated). No 066-9/10/11.

Native preview: `⟨w|V|²⟩≈2.59`, `s≈0.77`. **Fit array (066-6):** `n_row=881`, `n_chan=95`, `Δv=5.080 km/s`, `N=4`, `s=0.514`. Replica was 881×125 (wider buffer).

## 066 npz (local inventory)

- Local: `/Users/thbrown/kilogas/DR1/visibilities/KILOGAS066.npz` (native 43240×1920)
- CANFAR: same bytes (`Content-Length` 997305244)
- Ico / cube: `/Users/thbrown/kilogas/analysis/kinms_test/kgas066/KGAS66_Ico_K_kms-1.fits`, `KGAS66_clipped_cube.fits` (VOPT 8034–8536 km/s, 17×30 km/s)
- YAML `obs_freq_range` clips the receding side — do not use it as the trim
