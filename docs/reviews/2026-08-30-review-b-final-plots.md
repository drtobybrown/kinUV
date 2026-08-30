---
role: reviewer
seat: b
date: 2026-08-30
agent: review-b
canon_generation: 4
ids:
  - DEC-066-SPECRESP
  - DEC-066-ZEROMODEL
  - DEC-066-AGENTS
  - DEC-066-TARGET
verdict: accept
severity: major
propose: docs/reviews/2026-08-30-propose-final-plots.md
---

# Review b: final-fit plot handoff (066)

Do not read the other seat's review file. Do not implement.

Scope of this card is plots only. It does not refit, does not call `native_diagonal`, and it names the official tree as read-only (`kinuv-KGAS066-uvsign-map`). That is why this is accept, not reject. The execute list is still loose enough that a default script invocation writes a FITS into that tree and a leftover run can skip the Hann+bin gate. Those are implementer-must-fix comments, not a new DEC.

## Attacks / bounds

1. **`--matched-fits` default overwrites the official MAP tree.** `scripts/plot_stage_b_vs_imaging.py` always `writeto(..., overwrite=True)` to `--matched-fits`. The argparse default is `MAP_DIR / "stage_b_model_on_10kms.fits"` (under `kinuv-KGAS066-uvsign-map`). The propose says do not write a new matched FITS over the official MAP tree, then offers "the existing `stage_b_model_on_10kms.fits` as input only." That second option is false: the script has no read-only path. Passing the official path as `--matched-fits` overwrites a file in `kinuv-KGAS066-uvsign-map`. **Bound:** always pass `--matched-fits` under `docs/reviews/artifacts/2026-08-30-final-fit/` (same pattern as `plot_fit_diagnostics.py --imaging`). Never point `--matched-fits`, `--out-dir`, or a model write at the official MAP directory. Read `stage_b_model_cube.fits` and `stage_a_map.json` from the official tree; do not write anything back.

2. **Leftover runner does not assert Hann+bin.** Execute step 1 names `scripts/plot_leftover_chi2.py`. That script never calls `assert_hann_bin_operator()`. The wrapper `scripts/plot_fit_diagnostics.py` does, then delegates. Gate 4 ("keep `hann_then_bin`; if a script is wrong, fix it and continue") is not a gate: leftover can plot a `chi2` without proving the operator. DEC-066-SPECRESP forbids Hann-on-binned and `native_diagonal`. **Bound:** leftover must call `assert_hann_bin_operator()` before `predict_binned`, or leftover must be launched only via `plot_fit_diagnostics.py`. Abort if the operator is not `kinuv.response.spectral.hann_then_bin`. Do not use `native_diagonal`.

3. **Imaging cube must be 10 km/s, not 30 km/s.** Field guide and `docs/diagnostics/stage-b-vs-imaging.md`: Ico / vis-trim / SB live in `KGAS66/30kms/`; Stage B D/M/R lives in `KGAS66/10kms/`. Leftover correctly uses the 30 km/s cube only as the vis window. `gas_sigma` is ~12 km/s and the fit array is `dv = 5.080` km/s; a 30 km/s channel is wider than both, so M2 and PV become channel-width, not kinematics. The propose lists "need the 10 km/s cube on `/arc`" as a residual risk, not as a refuse-30kms rule. **Bound:** `--data-cube` and `--mask-cube` must be `.../KGAS66/10kms/KGAS66_clipped_cube.fits` and `.../KGAS66/10kms/KGAS66_mask_cube.fits`. Refuse `30kms/` for moments / spectra / PV.

4. **Missing leftover-identity gate.** Official Stage A is `chi2 = 168676` (`Delta_chi2` vs V=0 = +35553). S1 leftover already matched that number. The propose treats leftover as a few-minute eval with no accept/reject on the sum. A plot of a different MAP is not a handoff. **Bound:** write `leftover_chi2.json`; require `chi2_sum` within 1 of `stage_a_map.json` `chi2_map` (168676). If it fails, stop and STATUS one line; do not hand the user a leftover from another tree.

5. **Recipe the README must lock (propose omits).** Plotting guide, not optional cosmetics: M1 displayed as `v - vsys` (optical, colourbar centred on 0), not absolute velocity; east left / north up (`xlim` descending); `apply_style()`; no viridis / inferno / magma. Approaching / receding labels follow the fitted PA (199.73 deg), not `f47bc9-map` (PA=21.9 deg). Do not quote Laplace intervals on these figures (S2 failed 68/95). Do not write matched FITS into the official MAP directory (attack 1).

## Comments

1. **major.** Execute imaging only with `--matched-fits` under `2026-08-30-final-fit/`. The "existing FITS as input only" clause is invalid against the current writer. This is the overwrite trap; fix it during execute, do not reject the card.

2. **major.** Leftover must assert Hann+bin (`assert_hann_bin_operator` or `plot_fit_diagnostics.py`) before `predict_binned`. `native_diagonal` is a reject condition; this card does not license it.

3. **major.** D/M/R against the 10 km/s cube only. Do not pass `30kms/` into `plot_stage_b_vs_imaging.py`.

4. **major.** Leftover `chi2_sum` must match official Stage A 168676 (within 1). Else the folder is not the official MAP.

5. **major.** README recipe: M1 as `v - vsys`; east left; fitted PA=199.73 deg; no Laplace CIs; no `f47bc9-map` as the product.

6. **minor.** Execute step 3 says "four plot paths." The handoff is leftover plus moments, spectra, `pv_major`, `pv_minor` (five). Name leftover explicitly so the user is not handed imaging-only.

7. **minor.** Script defaults write leftover into `2026-08-29-s1-mock` and imaging into `2026-08-28-stage-b-imaging`. Must pass `--out` / `--out-dir` to `2026-08-30-final-fit`. Do not overwrite those dated folders.

8. **minor.** Leftover is Stage A vis `chi2`; D/M/R is Stage B vs the 10 km/s cube. That split is already in methodology. README must not label leftover as Stage B.

## Residual risks

1. Structured M0 residual (spirals) remains SB misspecification; not a failed leftover gate (propose already has this).

2. If `/arc` .../`10kms/` is missing, imaging cannot run. Do not silently fall back to `30kms/`. STATUS one line and stop imaging; leftover can still ship if the `chi2` identity gate passes.

3. `plot_stage_b_vs_imaging.py` recomputes the matched cube every run. A numerical drift versus the on-disk `stage_b_model_on_10kms.fits` is possible. Do not "fix" it by writing back into the official MAP tree. If moments look unlike `2026-08-28-stage-b-imaging/`, compare against that folder; do not overwrite it.

4. Laplace SBC failed 68/95. These plots are not calibrated CIs. Propose already says do not quote them; README must not grow error bars from the Laplace Hessian.

5. Default leftover path still reads `CANFAR_CUBE_30` for the vis window. That is correct for SPECRESP/trim. The risk is a copy-paste of that path into the imaging argv (attack 3).

## STATUS updates required

- `verdict` and `severity` as in the header (`accept`, `major`)
- `last_review_b`: this file
- Do not set `board: accepted` (parent tallies)
