---
role: proposer
date: 2026-09-05
agent: parent
canon_generation: 4
ids:
  - DEC-066-INFER
  - DEC-066-PA
  - DEC-066-TARGET
  - DEC-066-VC
  - DEC-066-INC
  - DEC-066-ZEROMODEL
  - DEC-066-SHIFT
  - DEC-067-RUNNER
verdict: propose
---

# KGAS066 closure, external cube S3, KGAS007 Stage A MAP

## Scope

Approaching search is **closed**. Receding NUTS `sd3ckpf2` is the 066 sampling product. Official MAP `kinuv-KGAS066-uvsign-map` stays read-only. Existing DEC ids only. **Agents do not write a new `DEC-*`.** User is asked to add a TARGET stub if 007 MAP is to be a licensed second galaxy; this card treats 007 MAP as user-licensed 2026-09-05 (MAP + leftover only).

Two tracks if the board accepts. No approaching NUTS. No 007 NUTS. No G4. No G5. No GPU. Do not quote S2 16/50/84 or inner `dV/dr`. Do not label leftover as `s_1` / `c_3`. Leftover gate stays **SB-dominated**. `quote_inner_slope: false`. `intervals_calibrated: false`.

Canon 066 numbers (not chat):

| Product | PA (deg) | V_0 (km/s) | r_t (arcsec) | chi2 |
|---|---|---|---|---|
| Stage A MAP | 199.73 | 267.7 | 0.5 (L-BFGS box) | 168675.6 |
| Receding NUTS mean | 200.05 | **255** | **0.224** | 167486.8 |
| Stage B N=7 λ=0 | — | — | rings | 167302.2 |
| Approaching catalogue L-BFGS | 25.2 (stuck) | 0 (box) | 15 (box) | 199968 (Δχ² vs V=0 = 4260) |

Human review surface: `docs/reviews/artifacts/2026-09-05-kgas066-s3-image-benchmark/` plus 007 leftover if Track B finds vis.

## Track A — image-plane S3 (066)

Cube: `/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/10kms/`. Vis `chi2 = s * sum w |ΔV|^2` stays the fit. CLEAN cubes are the comparator.

| Tool | How | Code home |
|---|---|---|
| 3DBarolo | CLI via `subprocess` (`BBarolo` / `3dbarolo`) | `scripts/run_s3_barolo.py` — **no** `import kinms` |
| KinMS | standalone runner | `external/kinms_kgas66.py` (outside `src/kinuv/` and `scripts/*.py` so `test_no_uvkin_or_kinms_import` stays green) |

Ingest both outputs into `docs/reviews/artifacts/2026-09-05-kgas066-s3-image-benchmark/`. S3 table (comparators, not calibrated posteriors):

1. **Beam-smearing:** cube-fit `r_t` or inner slope vs S1 (truth 0.25″, vis 0.254″, CLEAN M1 94.7 vs 236.7 km/s/arcsec) **and** vs NUTS median `r_t=0.224″`. Do not quote inner `dV/dr` as a 066 science number.
2. **Correlated-beam errors:** Barolo/KinMS formal errors vs NUTS ESS widths; vis side stays `intervals_calibrated: false`.
3. **Geometry soak-up:** if the cube fitter floats `i` or warp, report Δ`i` / residual PV. kinUV **freezes `i`** (DEC-066-INC). Do not call leftover `s_1`.

If `BBarolo` or KinMS is missing on PATH, STATUS one-liner and ship S3 from S1 plus whichever tool ran. Do not `pip install` into `kinuv-venv-recovery`. uvkin notebook is not executed from kinUV.

## Track B — KGAS007 Stage A MAP only

User 2026-09-05: MAP + leftover, **not NUTS**. Recommend a user TARGET stub (007 MAP licensed; NUTS later). Implementer does not write that DEC.

New tree only: `/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS007/kinuv-KGAS007-stage-a-map/`. Never write `kinuv-KGAS066-uvsign-map`. Do not steal `KGAS066-latest`.

**Inventory gate:** find continuum-subtracted vis + Ico on `/arc`. If missing, stop Track B with a STATUS line. Do not invent an npz.

If present: Hann+bin, empirical `s`, two-start PA, freeze `i` from 007 catalogue, Δχ² vs V=0. Leftover PNG/JSON on that tree. Headless flexible CPU if the MAP is long (DEC-067-RUNNER). **No 007 NUTS** until MAP beats V=0 and a later propose.

## Rejected alternatives

- Relaunch approaching NUTS — `pa25/failure.md` closed it.
- KinMS/`from kinms` inside `src/kinuv` or `scripts/` — tests forbid it; leftover card already rejected a cube likelihood.
- 007 NUTS this card — MAP first (DEC-066-INFER).
- G4 Talts SBC — leftover structured; 16/50/84 stay uncalibrated.
- Quote V_0 = 353 km/s as the NUTS mean — that number is wrong; mean is 255 km/s.
- Label leftover as minor-axis `s_1` — SB-dominated; harmonics need a user DEC.

## Residual risks

1. 3DBarolo / KinMS absent on CANFAR/astroml PATH. S3 then restates S1 only.
2. 007 vis not staged on `/arc`. Track B stops; 066 S3 still ships.
3. TARGET without a user stub: reviewers may accept 007 MAP as a **new tree diagnostic** only. Official 066 product unchanged.
4. Cube-fit `r_t` vs 0.224″ will be copied as a science inner scale. README must say NUTS median, uncalibrated, `quote_inner_slope: false`.
5. External KinMS script can still be mistaken for a kinUV likelihood. Artifact README first sentence: vis χ² is the fit.

## Execute if accepted

1. Ledger already patched this wave (STATUS / AGENTS / field-guide / CHANGELOG). Keep mailbox honest.
2. Track A: PATH check; run Barolo subprocess and/or `external/kinms_kgas66.py`; write S3 JSON + moment/PV overlays; README with S1 + NUTS comparator rows. Point methodology `user_review` at the S3 folder.
3. Track B: inventory 007 vis/Ico. If found, Stage A MAP to the new 007 tree + leftover. If not, STATUS one-liner.
4. Tests: `test_no_uvkin_or_kinms_import` still passes; no `from kinms` under `src/` or `scripts/`.
5. Commit and push after propose, after tally, after S3 artifacts, after 007 MAP (if any). Conventional subject. Official MAP unchanged. Do not start G4.

## STATUS updates required

- `next_role: board`
- `board: open`
- `last_propose:` this file
