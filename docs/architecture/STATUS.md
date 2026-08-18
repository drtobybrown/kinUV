---
generation: 4
phase: 066-8
code_freeze: false
next_role: proposer
pending: []
last_propose: docs/reviews/2026-08-18-merge-066-8.md
last_review: docs/reviews/2026-08-18-review-g4.md
open_questions: []
deadlocks: []
canon_generation: 4
---

# Architecture mailbox

**User dispatch 2026-08-18.** `dev` at `7f9f125` (`feat(066-8)` fast-forward). Wave 1 + 066-6 + 066-7 + **066-8** Stage A MAP. Gate 3: Δχ² vs V=0 is **+4341** (χ²_MAP=1.999×10⁵, χ²_zero=2.042×10⁵); eval 0.708 s. Fit array 881×95, N=4, s=0.514. Stage A vector **sat on the box** (PA=175.2°, σ=50 km/s, dy=+2″, V_0=0, r_t=0.5″, flux=47.29 Jy) — detection of a shifted Gaussian disk, not a kinematics start. No rings on real vis. No NUTS. No 066-9/10/11.

Native preview: `⟨w|V|²⟩≈2.59`, `s≈0.77`. **Fit array (066-6):** `n_row=881`, `n_chan=95`, `Δv=5.080 km/s`, `N=4`, `s=0.514`. Replica was 881×125 (wider buffer).

## 066 npz (local inventory)

- Local: `/Users/thbrown/kilogas/DR1/visibilities/KILOGAS066.npz` (native 43240×1920)
- CANFAR: same bytes (`Content-Length` 997305244)
- Ico / cube: `/Users/thbrown/kilogas/analysis/kinms_test/kgas066/KGAS66_Ico_K_kms-1.fits`, `KGAS66_clipped_cube.fits` (VOPT 8034–8536 km/s, 17×30 km/s)
- YAML `obs_freq_range` clips the receding side — do not use it as the trim
