---
generation: 4
phase: 066-7
code_freeze: false
next_role: proposer
pending: []
last_propose: docs/reviews/2026-08-18-handoff-new-chat.md
last_review: docs/reviews/2026-08-18-review-g4.md
open_questions: []
deadlocks: []
canon_generation: 4
---

# Architecture mailbox

**User dispatch 2026-08-18.** `dev`. Wave 1 + **066-6** (`1017602`) + **066-7** (`8b06c2b`) merged. Path to MAP: native npz + replica operator N=4, local CPU. Mock recovery (noise-free, native-window χ²): flux 0.9995, PA 205.199°, vsys 8299.554, (dx,dy)=(0.3003″, 0.3002″); eval 0.417 s. **066-8 next** (real MAP through Hann+bin). No NUTS.

Native preview: `⟨w|V|²⟩≈2.59`, `s≈0.77`. **Fit array (066-6):** `n_row=881`, `n_chan=95`, `Δv=5.080 km/s`, `N=4`, `s=0.514`. Replica was 881×125 (wider buffer).

## 066 npz (local inventory)

- Local: `/Users/thbrown/kilogas/DR1/visibilities/KILOGAS066.npz` (native 43240×1920)
- CANFAR: same bytes (`Content-Length` 997305244)
- Ico / cube: `/Users/thbrown/kilogas/analysis/kinms_test/kgas066/KGAS66_Ico_K_kms-1.fits`, `KGAS66_clipped_cube.fits` (VOPT 8034–8536 km/s, 17×30 km/s)
- YAML `obs_freq_range` clips the receding side — do not use it as the trim
