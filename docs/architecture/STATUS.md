---
generation: 4
phase: 066-2
code_freeze: false
next_role: proposer
pending: []
last_propose: docs/reviews/2026-08-18-dispatch-066-2.md
last_review: docs/reviews/2026-08-18-review-g4.md
open_questions:
  - native-npz-vs-replica-bin
deadlocks: []
canon_generation: 4
---

# Architecture mailbox

**User dispatch 2026-08-18.** Development branch is `dev`. Wave 1 on `dev`: 066-2 (`45d7ea8`), 066-3 (`0bbdb2c`), 066-4 (`b976d19`), 066-5 (`3c34344`). No fitter, no MAP, no 066-6 until bin N is chosen.

## 066 npz (local inventory)

- Local: `/Users/thbrown/kilogas/DR1/visibilities/KILOGAS066.npz`
- CANFAR: `/arc/projects/KILOGAS/analysis/toby_sandbox/visibilities/KILOGAS066.npz`
- Ico: `/Users/thbrown/kilogas/analysis/kinms_test/kgas066/KGAS66_Ico_K_kms-1.fits`
- Measured: `n_row=43240`, `n_chan=1920`, `Δν=0.9765625 MHz`, `|Δv|=1.270 km/s` (radio vs rest CO), software bin **N=1** (native SPW). `max |b|≈404 m` ≈ **302 kλ** at 224.6 GHz. Keys: `u_m,v_m,vis,weights,freqs,phase_dir_rad`. **Not** the forensic replica 881×125 / bin-4.

## Rank (DEC-066-INDEX)

1. `docs/decisions/DEC-*.md`
2. `field-guide/index.md`
3. this file
4. `docs/reviews/`
5. `PLAN.md`
