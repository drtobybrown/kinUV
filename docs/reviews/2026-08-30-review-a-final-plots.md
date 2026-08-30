---
role: reviewer
seat: a
date: 2026-08-30
agent: review-a
canon_generation: 4
ids:
  - DEC-066-AGENTS
  - DEC-066-SPECRESP
  - DEC-066-ZEROMODEL
  - DEC-066-INFER
  - DEC-066-TARGET
verdict: accept
severity: major
propose: docs/reviews/2026-08-30-propose-final-plots.md
---

# Review a: final-fit plot handoff (066)

Do not read the other seat's review file. Do not implement.

Scope check: regenerate leftover + Stage B D/M/R into `docs/reviews/artifacts/2026-08-30-final-fit/`. Official MAP stays `kinuv-KGAS066-uvsign-map`. No new MAP, no MCMC, no NUTS label, SPECRESP stays `hann_then_bin`. That is accept-eligible. The card still omits a write-path gate that the named scripts will hit if executed as written.

## Attacks / bounds

1. **`--matched-fits` default writes inside the official MAP tree.** `scripts/plot_stage_b_vs_imaging.py` defaults `--matched-fits` to `kinuv-KGAS066-uvsign-map/stage_b_model_on_10kms.fits` and always `writeto(..., overwrite=True)`. The 2026-08-28 run already did that (`summary.json` `matched_fits` is that path). The propose's second fork ("use the existing `stage_b_model_on_10kms.fits` as input only") is false: there is no read-only path. Omitting the flag, or passing the official path, overwrites a file under `kinuv-KGAS066-uvsign-map`. **Bound (execute, not optional):** pass `--matched-fits` under the handoff dir (e.g. `docs/reviews/artifacts/2026-08-30-final-fit/model_on_10kms.fits`). Do not omit the flag. Do not point it at the MAP tree. `scripts/plot_fit_diagnostics.py --imaging` already does this (`--matched-fits` -> `--out/model_on_10kms.fits`) and also calls `assert_hann_bin_operator`. Prefer that runner.

2. **Leftover script defaults `--out` to the S1 folder.** `scripts/plot_leftover_chi2.py` default `ARTIFACT` is `docs/reviews/artifacts/2026-08-29-s1-mock`. A bare invoke overwrites S1 leftover png/json/npz. **Bound:** `--out` must be the 2026-08-30 handoff dir.

3. **Missing leftover-identity gate.** S1 leftover of this MAP is `chi2_sum = 168675.6` matching `chi2_map_json`. The leftover script prints both and continues on mismatch. A wrong kernel, wrong `--map-dir`, or Stage B params would still write a pretty leftover vs uv. **Bound:** after the leftover eval, require `|chi2_sum - 168675.6| < 1` and `n_row=881`, `n_chan=95`, `pol=XX`. Record `pipeline_kernel: hann_then_bin` (call `assert_hann_bin_operator` before `predict_binned`; the leftover script does not). If the bound fails, fix SPECRESP / MAP path and re-run; do not hand the user a leftover of a different model.

4. **Handoff mixes Stage A leftover with Stage B imaging.** Leftover is Stage A (`stage_a_map.json`, `chi2 ~ 168676`). D/M/R is Stage B (`stage_b_model_cube.fits`, `chi2 = 167302`, Delta vs A = +1373). README "four plot paths" is also wrong: the two scripts write five pngs (`leftover_chi2.png`, `moments.png`, `spectra.png`, `pv_major.png`, `pv_minor.png`). **Bound:** README must name all five, label leftover as Stage A and D/M/R as Stage B, and must not quote Laplace intervals, `R_hat`, or `ESS` as calibrated 68/95 (S2 SBC failed 68/95). `r_t` on the 0.5 arcsec floor is a product fact on the leftover, not a recovered inner slope.

## Comments

1. `major` — Execute must not use `plot_stage_b_vs_imaging.py` defaults. `--matched-fits` under the artifact dir only. The propose "or existing as input only" fork is a write.

2. `major` — Leftover `--out` must be the new folder; leftover `chi2_sum` must match official Stage A `168675.6` within 1; assert Hann+bin before the eval.

3. `minor` — README: five plot paths; Stage A leftover vs Stage B D/M/R; no Laplace CI table. "What works" stays qualitative (PA/vsys match, M1 residual not ~V_rot, leftover vs uv not a bowl) plus the leftover `chi2` identity bound above.

4. `minor` — Pointing `docs/methodology.md` and STATUS `user_review` at the new folder is the mailbox change. Do not rewrite science claims. Keep `2026-08-28-stage-b-imaging/` as history.

## Residual risks

1. Structured M0 spirals remain SB misspecification (frozen Wiener Ico). Not a failed leftover gate if `chi2_sum` matches 168675.6.

2. Imaging still needs the 10 km/s cube on `/arc`. If that path is missing, stop and record; do not silently skip D/M/R.

3. Production `r_t` floor 0.5 arcsec is on the leftover Stage A product. Cube M1 will not show the S1 vis-recovered sub-beam slope. User review of plots is not a re-run of Gate 4.

4. Laplace CIs stay uncalibrated (S2). This card must not grow a CI overlay.

5. Regenerated leftover is a JAX/FINUFFT re-eval; imaging rematch can rewrite a FITS. The write must stay under the artifact dir (attack 1).

## STATUS updates required

- `verdict` and `severity` as in the header (`accept`, `major`)
- `last_review_a`: this file
- Do not set `board: accepted` (parent tallies)
