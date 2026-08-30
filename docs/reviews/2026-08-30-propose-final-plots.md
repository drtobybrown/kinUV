---
role: proposer
date: 2026-08-30
agent: parent
canon_generation: 4
ids:
  - DEC-066-AGENTS
  - DEC-066-SPECRESP
  - DEC-066-ZEROMODEL
  - DEC-066-INFER
  - DEC-066-TARGET
verdict: propose
---

# Final-fit plot handoff (066)

## Scope

User 2026-08-30: not babysitting; review **plots of the final fits**. This card licenses execute to a dated plot folder. Existing ids only. Official MAP stays `kinuv-KGAS066-uvsign-map` (do not overwrite in place).

## Gate calls (implementer)

1. Gates 1–3 already passed (analytic, Gate 2 Hann+bin mock, MAP `Delta_chi2` vs V=0 = +35553). **Proceed.**
2. Gate 4 / S1: vis recovered sub-beam `r_t` and `gas_sigma`; cube did not. **Proceed.** Production `r_t` floor 0.5 arcsec on real 066 is a product fact, not a stop.
3. Gate 5 / S2: Laplace SBC failed 68/95. **Proceed to plots anyway.** Do not quote Laplace intervals as calibrated. Do not run a new multi-hour MCMC this card. Do not label `laplace_mh` as NUTS.
4. SPECRESP: keep `hann_then_bin`. If a script is wrong, fix it and continue.
5. SB leftover vs velocity: plot it (already the S1 leftover figure). Do not restore Ico as intrinsic SB this card.
6. SHIFT/PB: keep image-plane shift then PB. Do not freeze `(dx,dy)` at 0.
7. BLOATED: split files; do not stop.
8. TARGET: 066 only. No survey runner.

## What changed / what was checked

- Handshake now says the user reviews final plots, not gates (`DEC-066-AGENTS` amended today).
- Official Stage B vs 10 km/s moments already exist (`2026-08-28-stage-b-imaging/`). This card **regenerates** leftover + imaging suite into one handoff folder so the user has a single place to look.
- Model cubes already on disk: `stage_a_model_cube.fits`, `stage_b_model_cube.fits`, `stage_b_model_on_10kms.fits`.

## Rejected alternatives

- New MAP or Stage B refit this card — not needed for a plot review; would overwrite product risk.
- Autodiff NUTS — still a rewrite; S2 already showed Laplace is uncalibrated.
- Waiting for a user ACK at leftover vs velocity — that is babysitting.

## Residual risks

1. Regenerated leftover `chi2` is a ~few-minute JAX/FINUFFT eval; imaging plots need the 10 km/s cube on `/arc`.
2. Structured M0 residual (spirals) will still be there; that is SB misspecification, not a failed fit gate.
3. Stage A-only moment maps are not a separate product; the D/M/R grid is Stage B vs the 10 km/s cube.

## Execute if accepted

1. Run leftover `chi2` of official Stage A (`scripts/plot_leftover_chi2.py`) into `docs/reviews/artifacts/2026-08-30-final-fit/`.
2. Run `scripts/plot_stage_b_vs_imaging.py` into the same folder (moments, spectra, PV). Do not write a new matched FITS over the official MAP tree; use `--matched-fits` under the artifact dir or the existing `stage_b_model_on_10kms.fits` as input only.
3. Write `README.md` in that folder: four plot paths, one paragraph on what "works" means (PA/vsys match, M1 residual not ~V_rot, leftover vs uv not a bowl).
4. Point `docs/methodology.md` and STATUS `user_review` at that folder.
5. Commit and push.

## STATUS updates required

- `next_role: board`
- `board: open`
- `build_licensed: true`
- `last_propose:` this file
