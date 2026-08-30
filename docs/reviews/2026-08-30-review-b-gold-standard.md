---
role: reviewer
seat: b
date: 2026-08-30
agent: review-b
canon_generation: 4
ids:
  - DEC-066-INC
  - DEC-066-TARGET
  - DEC-HIER-SELFUNC
  - DEC-066-INFER
verdict: accept
severity: major
propose: docs/reviews/2026-08-30-propose-gold-standard.md
---

# Review b: gold-standard inference sequence (066 kernel)

Do not read the other seat's review file. Do not implement.

Accept because the order is JAX `predict_binned` (CPU), then unconstrained chart, then NumPyro NUTS, then Talts SBC, then GPU, and this card's execute does not start NUTS, JAX, unfreeze `i`, or add `h_z`. Reject conditions (unfreeze `i` or add `h_z` without a user DEC stub; FD HMC labeled NUTS) are not in execute. The sequence still overclaims "gold-standard" while `DEC-HIER-SELFUNC` stays deferred, puts G0 flags in `infer/`, and treats the `r_t` L-BFGS box as a HMC prior. Those are implementer-must-fix bounds, not a re-propose.

## Attacks / bounds

1. **Hierarchical deferral vs user request.** The card is titled gold-standard and lists `DEC-HIER-SELFUNC`, then refuses hierarchical NUTS over 400 and keeps the id at Phase 5. That deferral is required: the DEC is `status: proposed`, generation 1, "DEFERRED until Phase 5. No 066 worker owns this id. Do not implement." `docs/diagnostics/survey-readiness.md` already says the same and locks code to 066 under `DEC-066-TARGET`. Implementing pooling now would contradict the DEC (no selection function; MAP plus failed Laplace widths are not exchangeable posteriors). The attack is the title, not the deferral: a user asking for gold-standard survey inference does not get a population product from this card. **Bound:** keep the deferral; do not create a HIER or TARGET stub. Roadmap and methodology must call this the **066 kernel** sequence (autodiff likelihood, NUTS, SBC on the exact mock kernel). Do not say gold-standard is complete while `DEC-HIER-SELFUNC` is still proposed. G0 flags on one Stage A JSON are not a 400-galaxy runner and not a substitute for TARGET.

2. **Bijection vs `r_t` floor.** Architect step 3: logit/softplus on `r_t` with Jacobian; "Production box `(0.5, 15)` stays the science prior; the bijection is the sampler chart." Residual risk 3 already admits a bound chart does not unstick a likelihood that wants 0.25 arcsec. The category error is calling `RT_BOUNDS_ARCSEC = (0.5, 15.0)` a science prior. `src/kinuv/infer/seeds.py` documents it as the L-BFGS-B box. Official Stage A MAP sits on the 0.5 arcsec endpoint (`docs/methodology.md`). S1 recovered inject `r_t=0.25` only with a script-local box `(0.05, 15)`. `src/kinuv/infer/posterior.py` already records that a Gaussian interval around a bound is not an interior CI (S2 real-066 `r_t` 68/95 hugs 0.5). Logit of a closed interval with the MAP at the endpoint sends the unconstrained coordinate to -inf; the Jacobian diverges; HMC cannot sit on the wall. **Bound:** G0 must flag `r_t_at_floor` from the official JSON (do not refit). G2 must not treat the production box as a prior. If `r_t` is on 0.5 arcsec, do not run NUTS in a logit chart of `[0.5, 15]`. Production box stays for MAP. Mock-only wider box stays licensed only where S1 already licensed it. Do not lower the production floor in this card.

3. **G0 flags belong in `diagnostics/`, not `infer/`.** Execute step 3: `kinuv.infer.flags.map_quality_flags(stage_a_json) -> dict`. `src/kinuv/infer/` is the fitter: `map.py` (L-BFGS, freeze `i`, `Delta_chi2` vs V=0, two-start PA 205.2/25.2), `posterior.py` (`sampler: laplace_mh`, not NUTS). Flags do not enter `map_objective` or `ln L`. `src/kinuv/diagnostics/` already owns leftover `chi2` vs uv and velocity (`figures.plot_leftover_chi2`). Architect step 1 names leftover-vs-velocity structure; execute step 3 drops it (only floor `r_t`, `delta_chi2`, PA vs 21.9 alias, i frozen). Putting taxonomy next to the optimiser invites a later survey to import flags as if they were the likelihood. **Bound:** implement G0 as `kinuv.diagnostics.flags.map_quality_flags` (or equivalent under `diagnostics/`). Read official Stage A JSON only; do not refit; do not import from `run_stage_a_map`. Include leftover-vs-velocity (or an explicit omit with a one-line reason). `i_frozen` is a restatement of `DEC-066-INC` (`i = arccos(0.721) = 43.9 deg`, freeze); it must not grow an `i` parameter. PA alias is vs 21.9 deg (`f47bc9-map` pre-sign), not a second fit start.

4. **PSIS-LOO unit is vis cells on 066, not galaxies.** Architect step 6: PSIS-LOO Stage A vs Stage B after NUTS. MAP AIC already prefers B (`Delta_chi2` vs A = +1373). The propose never names the leave-out unit. Pointwise PSIS-LOO needs observations. On one galaxy that is vis rows / binned vis-chan cells under `chi2 = s * sum w |d-m|^2` (`posterior.py`). Hann+bin correlates adjacent native channels, so iid pointwise LOO is already approximate. Galaxy-level LOO is the hierarchical object; `DEC-HIER-SELFUNC` and `DEC-066-TARGET` forbid it. **Bound:** G5 is Stage A vs Stage B on **066 vis cells after Hann+bin**, only after a NUTS posterior exists (after 4). Do not skip JAX because MAP AIC prefers B. Do not treat vis rows as iid without stating Hann dependence as residual risk. Do not compute leave-one-galaxy LOO. Do not start G5 in this card.

5. **GPU sequencing.** Architect: JAX `predict_binned` on CPU, tests on CPU, do not provision GPU yet; NumPyro NUTS on Stage A CPU; CANFAR GPU only after a 066 NUTS smoke passes on CPU. Rejected: FD HMC/NUTS on GPU (NumPy does not differentiate; tens of CPU-hours/chain already). NUFFT is already JAX; the remaining pole is sky + Hann+bin + `chi2` on XLA (`map.py` `predict_binned` is still NumPy `predict_vis` then `hann_then_bin`). Costing in `survey-readiness.md` (~12-17 h/galaxy at 1e5 evals) is not a license to GPU G1. Residual risk 4 (jax-finufft / CUDA images on Skaha) is independent of a correct CPU likelihood. **Bound:** G1 is full `predict_binned` on CPU XLA, not GPU NUFFT glued to NumPy `chi2`. Do not provision GPU, write a Skaha image, or label FD MH as NUTS. GPU only after a 066 CPU NUTS smoke (`R_hat` / `ESS`, `sampler: nuts`). This card must not start G1.

## Comments

1. **major.** Do not implement hierarchical pooling, a 400-galaxy runner, or a `DEC-HIER-SELFUNC` stub. Name the roadmap the 066 kernel sequence. Gold-standard population inference stays Phase 5.

2. **major.** Do not bijection the production `r_t` floor as a science prior. Flag `r_t_at_floor` on the official MAP. G2/NUTS on a logit of `[0.5, 15]` with MAP at 0.5 arcsec is a reject for that wave, not this one. Do not lower `RT_BOUNDS_ARCSEC`.

3. **major.** Place G0 in `kinuv.diagnostics`, not `kinuv.infer`. JSON in, dict out, unit test on official Stage A numbers, no refit. Execute must not omit leftover-vs-velocity after architect step 1 named it, unless the roadmap states the omit.

4. **major.** G2 chart, when a later card runs it, is the current Stage A `PARAM_NAMES` only (`flux`, `pa_deg`, `vsys_kms`, `gas_sigma_kms`, `dx_arcsec`, `dy_arcsec`, `v0_kms`, `r_t_arcsec`). Architect step 3 "inclination-if-unfrozen" must not add `i` or `h_z` without a user DEC stub (`DEC-066-INC` freeze stands). This card does not run G2.

5. **major.** PSIS-LOO is 066 vis cells after NUTS, not galaxies, not this card. Do not use MAP `Delta_chi2` vs A as a reason to skip JAX.

6. **major.** GPU after CPU NUTS smoke only. This execute: roadmap + methodology/survey-readiness pointers + G0 flags + commit/push. Do not start JAX or NUTS.

7. **minor.** `docs/diagnostics/gold-standard-roadmap.md` already exists (propose says write it on accept). Execute must rewrite it to this sequence plus these bounds. Do not ship a pre-board draft that puts GPU on G4 SBC as "maybe" before a CPU NUTS smoke.

8. **minor.** Flag `delta_chi2_vs_zero` on 066 will not fire (`Delta_chi2` = +35553). The taxonomy is for later hard targets; the unit test should still record the official number and `beats_zero`.

## Residual risks

1. JAX rewrite remains the long pole. Dual accept of this card does not finish autodiff, NUTS, or SBC (propose residual 1).

2. True NUTS CIs on real 066 stay overconfident if SB leftover is structured (frozen Wiener Ico). Mock SBC tests the exact kernel only (propose residual 2). Real-066 column must not claim calibration.

3. `r_t` on the production floor: a later unconstrained chart cannot recover 0.25 arcsec on real 066 without a licensed box change. S1 showed the engine can; the product MAP cannot.

4. GPU jax-finufft / CUDA on Skaha can fail after a correct CPU JAX likelihood (propose residual 4). CPU NUTS smoke is the gate, not a Skaha reservation.

5. Hann-correlated vis cells make PSIS-LOO effective sample size optimistic even after NUTS. Document when G5 is proposed; do not treat it as independent rows.

6. `DEC-HIER-SELFUNC` still has no selection function. Calibrated 066 posteriors do not license population `gamma`.

## STATUS updates required

- `verdict` and `severity` as in the header (`accept`, `major`)
- `last_review_b`: this file
- Do not set `board: accepted` (parent tallies)
