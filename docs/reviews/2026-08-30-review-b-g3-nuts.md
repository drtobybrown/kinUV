---
role: reviewer
seat: b
date: 2026-08-30
agent: review-b
canon_generation: 4
ids:
  - DEC-066-INFER
  - DEC-066-INC
  - DEC-066-SHIFT
  - DEC-066-SPECRESP
  - DEC-066-TARGET
  - DEC-066-WEIGHT
  - DEC-066-VC
  - DEC-HIER-SELFUNC
verdict: accept
severity: major
propose: docs/reviews/2026-08-30-propose-g3-nuts.md
---

# Review b: G3 autodiff `chi2(θ(z))` + CPU NumPyro NUTS (066 kernel)

Do not read the other seat's review file. Do not implement.

Accept because the selected path is the gold-standard G3 wave: a JAX potential on the landed G2 chart, then CPU NumPyro NUTS, existing ids only, official `kinuv-KGAS066-uvsign-map` read-only, frozen `i`, no `h_z`, no logit of `RT_BOUNDS_ARCSEC=(0.5, 15)`, host `log_prob_unconstrained` not autodiff, `(dx, dy)` frozen at MAP, two PA runs, `sampler: nuts` only after autodiff, no GPU. The card as written can still ship an 8-site NumPyro model that samples the shifts it claims to freeze, a potential that `float()`s sampled `gas_sigma`, a 6-column JSON that `plot_posterior_corner` refuses, mixing stats padded by constant columns, a `sampler: nuts` label on host `chi2_and_prior`, or a 066 overnight that never finishes while the tiny mock mixes. Those are implementer-must-fix bounds, not a re-propose.

## Attacks / bounds

1. **Execute step 1 misses sampled host bounces that Gate 1 will hit; `fourier_shift` / freqs are a freeze contract, not a delete-three-floats list.** Propose enumerates `arctan_vc` (`if r_t_arcsec <= 0` and `float(v0_kms)` / `float(r_t_arcsec)`), `los_velocity` `float(vsys_kms)`, `sky_cube` `float(i_rad)` / `float(dx_arcsec)` / `float(dy_arcsec)`, `shift_prior` `float(dx)` / `float(dy)`, `chi2_and_prior` without `xla=True` then `float(c)`, and dict-of-floats `predict_binned(..., xla=True)`. Residual 1 then shrugs “fourier_shift, NUFFT scale, or Hann weights.” Live path that Gate 1 (`jax.grad` of `U(z)` finite, `is_jax` on the jitted potential) actually traces, and that step 1 does **not** name:

   - `_gaussian_pdf` (`model.py`): `sig = float(sigma_kms)`. `gas_sigma_kms` is a **sampled** name (`PARAM_NAMES[3]`, log chart). `float` of a tracer is `ConcretizationTypeError` under `jax.grad` / `jit`. G1’s tiny `jax.grad` was flux-only (`test_g1_jax.py`); it does not prove `gas_sigma` is traceable.
   - `predict_binned` XLA (`map.py`): `pa_rad=np.radians(params["pa_deg"])`. `pa_deg` is sampled (identity chart). Host `np.radians` on a tracer is a bounce. Frozen `i` may stay `float(i_rad)` / `inclination_rad()`.
   - `arctan_vc`: Python `if r_t_arcsec <= 0.0: raise` on a traced `r_t` is `TracerBoolConversionError` under `jit` (eager `if` on a 0-d `jnp` scalar host-converts and will green-light a non-jitted smoke). `r_t` is sampled (`exp(z[7])`).
   - `fourier_shift` XLA: `float(dx_arcsec)` and `float(dy_arcsec)` on the phase ramp. Legal **only if** `(dx, dy)` are host Python floats (the freeze). If the potential traces `z[4]`, `z[5]`, this bounce remains and the freeze is a lie.
   - `sky_cube`: `if is_jax(freqs_hz): raise TypeError("sky_cube freqs_hz must be NumPy; dv is a host scalar")`. `channel_width_kms` returns `float(np.median(np.abs(np.diff(vel))))`. `dv` as a host scalar times a JAX `phi` is a constant multiplier and is **not** a kinematic bounce. Do not `jnp.asarray` `data.freqs_native` into `sky_cube`. NUFFT already does `jnp.asarray(np.asarray(freqs_hz))` on the host copy (`nufft.py` `_nufft2_degrid_xla`). Hann has no `float()` on the vis path (`spectral.py`).

   **Bound:** Gate 1 fail-if any **sampled** kinematic `float()` / Python `if` remains: `v0`, `r_t`, `vsys`, `gas_sigma`, `pa`, `flux`. Frozen `i` and host `dv` / `nu_mid` / `NPZ_UV_SIGN` / `eps` / `cell_arcsec` may stay Python floats. Do not JAX-ify `freqs_hz` into `sky_cube`. `fourier_shift` and `sky_cube` offset `float(dx/dy)` stay **if and only if** those two are injected as official-MAP host floats (see attack 2). `tests/test_g3_nuts.py` autodiff smoke uses the same skip as `tests/test_g1_jax.py` (`_require_jax_finufft`: `pytest.importorskip("jax")` then skip unless `BACKEND == "jax-finufft"`). Finite `jax.grad` of `U` at official-like `z` on **all six sampled axes**, not flux-only. `is_jax` on the jitted potential output. Do not `jax.grad` G2 `log_prob_unconstrained`. Do not JIT that host helper. Live jax in `kinuv-venv-recovery` is **0.11.1** / jaxlib 0.11.1 / jax-finufft 1.3.1.

2. **Freeze `(dx, dy)` means they are not in `U(z)` as live coordinates; `shift_prior` is a constant 0.0345, not a sampled prior.** DEC-066-SHIFT: after MAP, freeze for NUTS only if both are <1σ of 0. Official `stage_a_map.json`: `dx_arcsec=0.09104737371760792`, `dy_arcsec=0.018566961155444102`, σ = 0.5″ (`SHIFT_PRIOR_SIGMA_ARCSEC`). Ratios 0.182 and 0.037. `shift_prior = (dx/σ)²+(dy/σ)² = 0.034537`. `chi2_map=168675.59555208942`. Architect `U(z) = 0.5 (chi2 + shift_prior) - log|det J|` and execute `U = -log p(θ(z)) - log|J|` agree when `log p = -0.5 (chi2 + shift_prior)`, and they **disagree with freeze** if `dx`, `dy` still enter `shift_prior` as tracers. A constant addend does not change `∂U/∂z_sampled`. A live `shift_prior(θ_dx, θ_dy)` gives `∂U/∂dx = dx/σ² ≠ 0` — that is sampling the shifts, not freezing them. Double-count is: that term **plus** `numpyro.sample("dx_arcsec", dist.Normal(0, 0.5))` (or a second addend). Omit-from-U and also-omit-NumPyro-prior is the freeze. Architect “length-8 unconstrained vector” plus “NUTS samples the other six names” plus execute “make `sky_cube` offsets and `shift_prior` traceable” is three instructions. Following step 1 literally unfreezes.

   **Bound:** NumPyro sample sites are the **six** names `flux`, `pa_deg`, `vsys_kms`, `gas_sigma_kms`, `v0_kms`, `r_t_arcsec`. Inject official MAP `(dx, dy)` as Python `float`s into `predict_binned` / `fourier_shift` / `sky_cube` offsets. Do not `numpyro.sample` those two. Do not put `Normal(0, 0.5)` in the NumPyro model. `U(z_6) = 0.5 * chi2(θ(z_6); dx_MAP, dy_MAP) - log|J_6|`. `log|J|` on the identity shift axes is 0, so `log|J_6|` equals the 8-vector sum. `shift_prior` is omitted or added once as the host constant `0.034537…`. Assert `∂U/∂` the two shift slots is identically 0 (they are not inputs). `chi2_and_prior` / `SAMPLER_NAME="laplace_mh"` stay the MH path; do not wrap them in NumPyro.

3. **6-vector model vs `plot_posterior_corner` 8-col is a write contract, not a hope.** Live `_as_nuts_draws`: exact `sampler == "nuts"` and `arr.shape[1] == 8` after flattening `(n_chain, n_draw, 8)` → `(n_draw, 8)`. `STAGE_A_NAMES` / `PARAM_NAMES` order: `flux`, `pa_deg`, `vsys_kms`, `gas_sigma_kms`, `dx_arcsec`, `dy_arcsec`, `v0_kms`, `r_t_arcsec` (indices 4, 5 are the freeze). A 6-column write raises. An 8-column write with shifts in the wrong slots labels the corner. `tests/test_posterior_corner.py` already refuses `laplace_mh` and interval-only `sampler: nuts`.

   **Bound:** stitch is host-only: NumPyro trace (n_chain, n_draw, 6) → (n_chain, n_draw, 8) with columns 4, 5 equal to official MAP `dx`, `dy` (atol 0). Product JSON `sampler == "nuts"` only after Gate 1. `plot_posterior_corner` accepts that JSON and still raises on S2 `docs/reviews/artifacts/2026-08-29-s2/s2_mock_mcmc.json`. Chart source has no `numpyro` / `logit` / `RT_BOUNDS`. No NumPyro import in `chart.py`. `posterior.SAMPLER_NAME` stays `laplace_mh`.

4. **`split_rhat` / `ess_bulk` on constant columns are not mixing.** Live `split_rhat`: `w = mean var` is 0 on a constant, `rhat = sqrt(var_hat / w)` → NaN (errstate already swallows). Live `ess_bulk`: zero-mean constant → `ac[0]==0` → `ac /= 1`, `tau=1`, **ESS = n_chain * n_draw** (looks like 4800 on a delta). Gate 5 already says exclude frozen names. Residual 2 repeats it. There is no test. A report that `nanmean`s R_hat or quotes min-ESS including the two constants will pass a broken 6-name run.

   **Bound:** mixing gate is R_hat < 1.01 and ESS > 200 on the **six sampled names only**. Assert R_hat on columns 4, 5 is non-finite (or skip those indices in the helper). Do not use ESS on those columns. Do not require finite R_hat on the 8-vector. Tiny-mock gate 4 uses the same six-name rule.

5. **Tiny-mock mix does not bound 066 leapfrog cost; G1 3.01 eval/s is not a NUTS rate.** G1 timing: `eval_per_s=3.0119`, `seconds_post_warmup=0.332` on the 881×95 array (`docs/reviews/artifacts/2026-08-30-g1-jax/timing.json`). Reverse-mode `jax.grad` is ~2–3× a forward. NumPyro NUTS default `max_tree_depth=10` allows 1024 leapfrog steps per draw. At mean tree depth 6: ~64 grads × ~0.8 s ≈ 50 s/draw; 4 chains × 1000 (warmup+draw) ≈ **50–70 h per PA run**, two runs ~5 days. Tiny mock in `test_g1_jax._tiny_data` is 6 rows × 2 fit channels on a small grid — it will mix in minutes and does not measure 066 wall. Propose residual 5 talks about first-JIT minutes and “cost miss is STATUS + continue if autodiff + mixing hold.” Mixing on 066 **is** Gate 5. That residual does not license a `sampler: nuts` 066 JSON that never reached R_hat / ESS, and it does not license GPU.

   **Bound:** tiny-mock records post-warmup **forward** eval/s **and** mean NumPyro `num_steps` / potential evals per draw. Project 066 wall as `(n_chain * (n_warmup + n_draw) * mean_num_steps * t_grad_066)` with `t_grad_066` from a 066 `jax.grad(U)` after warmup, not 1/3.01. Cap the 066 attempt (document the cap on STATUS). If the projection exceeds the cap, **do not** write 066 `sampler: nuts`; land autodiff + tiny-mock `sampler: nuts` if gates 1–4 hold; do not start a GPU session. If 066 finishes, Gate 5 mixing is on the six names, both PA runs separate (do not pool 199.73° and 25.2°). Receding init is official MAP `pa_deg=199.7298` (roadmap seed 205.2 is the L-BFGS start; MAP is the mode). Approaching init `25.2°` with other θ at the receding MAP is allowed to fail to mix — report it; do not flip PA in the product without saying so. Post-warmup eval/s vs G1 3.01 and S2 FD 0.329 is a note, not a substitute for leapfrog counts. `JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`, scratch under `/scratch/kinuv-$USER` else `/tmp`. No vis I/O over `/arc` in the loop. Official MAP tree untouched.

6. **Unpinned `nuts = ["numpyro"]` can replace jax 0.11.1 / jax-finufft 1.3.1.** Live recovery venv: jax 0.11.1, numpyro **absent**. `pyproject.toml` has `nufft = ["jax>=0.4.30", "jax-finufft>=1.3"]` and no `nuts` extra. PyPI NumPyro 0.21.0 depends on `jax>=0.7`. Extra `numpyro[cpu]` installs **a** CPU jax, not necessarily 0.11.1 + jax-finufft 1.3.1. Residual 6 names the pin risk; the extra as written does not pin. Identity `|chi2-168675.6|<1` after install is necessary and not sufficient if `import jax_finufft` dies.

   **Bound:** `nuts = ["numpyro"]` with a version pin that installs **without** upgrading jax/jaxlib/jax-finufft. Install with `--no-deps` on numpyro if needed, then verify `jax.__version__ == "0.11.1"` and `BACKEND == "jax-finufft"`. Do not `pip install 'numpyro[cpu]'`. Do not add `emcee`. Identity `|chi2-168675.6|<1` at official θ, `s=0.5136098555284736`, after the install. Fail the extra if jax moved.

7. **`sampler: nuts` is a provenance bit, not a module default.** `plot_posterior_corner` already treats `sampler == "nuts"` as “this is a NUTS posterior” and draws 16/50/84 on every column. Propose: label only if autodiff gates pass; reject wrapping NumPy `chi2_and_prior` in NumPyro. Gate 3 then requires the product JSON to have `sampler == "nuts"`. A stub written before Gate 1, or a tiny-mock file copied onto a 066 filename, becomes the human corner. G0 `map_quality_flags` hardcodes `nuts_absent: True` — that dict is about the official MAP, not a license to flip the MAP tree, and not proof the G3 artifact is NUTS.

   **Bound:** writer sets `sampler: "nuts"` only after Gate 1 is finite `jax.grad` + `is_jax`. Tiny-mock JSON may carry it if gates 1–4 pass. 066 JSON may carry it only if Gate 5 mixing holds on that run. A host-wrapped `chi2_and_prior` path must not be able to set the string. `tests/test_g3_nuts.py` includes a source/API assert that the NUTS potential is not `log_prob_unconstrained` / default `predict_binned`. Do not write `kinuv-KGAS066-uvsign-map`. Do not change `nuts_absent` on the official MAP.

8. **Leftover-vs-velocity is still on; real-066 NUTS 16/50/84 are not calibrated.** Official leftover `docs/reviews/artifacts/2026-08-30-final-fit/leftover_chi2.npz`: uv-binned span 0.115, velocity span 0.355 → `leftover_chi2_structured` True. `r_t_arcsec=0.5` → `r_t_at_floor` True; `quote_inner_slope` False. S2 Laplace SBC n=20 already failed 68/95 on the exact mock. NUTS samples the same misspecified likelihood. `plot_posterior_corner` will draw 16/50/84 on all eight columns, including delta-function frozen shifts and kinematics the leftover already says the model misses. Gate 6 asks for caveats “in the product, not in a later excuse.” A README sentence next to a corner that looks like a calibrated 8-param posterior is the later excuse.

   **Bound:** each G3 product JSON records `leftover_chi2_structured: true` (066), `r_t_at_floor: true` when the init/MAP is on the floor, and `intervals_calibrated: false`. 066 corner title must say not calibrated / leftover structured / not S2 Laplace. Do not quote S2 16/50/84. Do not quote inner `dV/dr`. Tiny-mock (exact kernel) must not copy “calibrated” language onto the 066 JSON. `log(r_t)` may walk below 0.5″; that does not clear the leftover flag.

## Comments

1. **major.** Gate 1 must fail if sampled `float()` / Python `if` remains on `gas_sigma` (`_gaussian_pdf`), `v0`/`r_t` (`arctan_vc`), `vsys` (`los_velocity`), `pa` (`np.radians` in `predict_binned`). Do not JAX-ify `sky_cube` freqs; `channel_width_kms` stays a host `dv`. `fourier_shift` `float(dx/dy)` stays only because those two are host MAP floats. Same jax-finufft skip as G1. Finite `jax.grad(U)` on all six sampled axes. No `jax.grad` of `log_prob_unconstrained`.

2. **major.** Six NumPyro sample sites. Official MAP `(dx, dy)` injected as Python floats. `U` omits live `shift_prior` (constant 0.034537, or add once). No `Normal(0, 0.5)` site. Live `shift_prior(θ)` in `U` is not a freeze (DEC-066-SHIFT).

3. **major.** Written draws are `(n_chain, n_draw, 8)` in `PARAM_NAMES` order; columns 4, 5 constant at official `dx`, `dy`. `plot_posterior_corner` accepts; S2 `laplace_mh` still raises. No `numpyro` in `chart.py`. `SAMPLER_NAME` stays `laplace_mh`.

4. **major.** R_hat < 1.01 and ESS > 200 on the six sampled names only. Constant-column ESS (`= n_tot`) must not enter the gate. `split_rhat` NaN on constants is expected.

5. **major.** Tiny-mock records mean leapfrog / n_eval per draw. 066 wall is projected from that × a 066 `jax.grad` time, not G1 3.01 eval/s. Over cap: no 066 `sampler: nuts`, no GPU. Two PA runs unpooled. `JAX_PLATFORMS=cpu`.

6. **major.** Pin `numpyro` so jax stays 0.11.1 and jax-finufft 1.3.1. Do not install `numpyro[cpu]`. Identity `|chi2-168675.6|<1` after install with frozen `s=0.5136098555284736`.

7. **major.** `sampler: nuts` only after autodiff Gate 1; 066 JSON only after Gate 5. Do not wrap `chi2_and_prior`. Do not write the official MAP tree. Do not flip G0 `nuts_absent` on that tree.

8. **major.** 066 product JSON: `intervals_calibrated: false`, leftover and `r_t_at_floor` recorded. Corner title not calibrated. Do not quote S2 intervals or inner `dV/dr`. No logit of `RT_BOUNDS_ARCSEC=(0.5, 15)`. No new `DEC-*`. No G4/G5/GPU/`h_z`/unfreeze `i`.

9. **minor.** Official identity remains `|chi2-168675.6|<1` at official `z` after `unconstrained_to_physical`, `predict_binned(..., xla=True)`, no refit. 066 tests skip if npz/MAP json missing (same spirit as `test_g1_jax.test_official_066_chi2_identity`). Artifacts under `docs/reviews/artifacts/2026-08-30-g3-nuts/`. Human plot folder stays `docs/reviews/artifacts/2026-08-30-final-fit/` unless a NUTS corner PNG is dropped in the G3 dir (not S2). Checkpoints on `/scratch`, not vis dumps.

10. **minor.** Image-plane vsys vs the 10 km/s cube stays vis-weighted MAP vs CLEAN, not a WCS bug. Do not freeze `vsys` to catalogue 8299.563 optical. Identity PA does not wrap.

## Residual risks

1. Removing `float()` from sampled kinematics can still leave a host bounce in NUFFT `eps`, Hann weights, or a dict pack inside the jitted `U`. Gate 1 (`is_jax` + six-axis finite `jax.grad`) is the reject-if-fail.

2. Frozen `(dx, dy)` 8-column draws fake ESS (`ess_bulk` → n_tot) and NaN `R_hat`. Six-name mixing gate is mandatory.

3. `log(r_t)` lets NUTS walk below 0.5″. That does not unstick leftover-vs-velocity (0.355 vs 0.115) and does not license inner `dV/dr` while `r_t_at_floor` fires.

4. Real-066 NUTS 16/50/84 will look tight on a leftover-structured likelihood. They are not calibrated. S2 SBC already failed on the exact mock. G4 is the calibration wave.

5. 066 NUTS at ~0.3 s/forward × leapfrog is multi-day at default `max_tree_depth`. Tiny-mock mix will pass first. A cost miss is STATUS + tiny-mock `sampler: nuts` only; not GPU; not a partial 066 JSON labeled `nuts`.

6. NumPyro 0.21 + jax 0.11.1 can still import-fail (pmap / provenance). Identity χ² and `BACKEND == "jax-finufft"` after install are the compatibility gates. Do not let pip move jax.

7. Approaching-PA (25.2°) with receding MAP kinematics may not mix. That is a 180° mode result, not a license to average runs or silently flip PA.

8. G0 `nuts_absent: True` stays true on the official MAP. Do not treat that flag as G3 provenance.

## STATUS updates required

- `verdict` and `severity` as in the header (`accept`, `major`)
- `last_review_b`: this file
- Do not set `board: accepted` (parent tallies)
