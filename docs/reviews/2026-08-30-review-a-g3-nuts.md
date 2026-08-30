---
role: reviewer
seat: a
date: 2026-08-30
agent: review-a
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

# Review a: G3 autodiff `chi2(θ(z))` + CPU NumPyro NUTS (066 kernel)

Do not read the other seat's review file. Do not implement.

Scope check: autodiff potential on the G2 eight-name chart plus CPU NumPyro NUTS. Existing ids only. No new `DEC-*`. Official MAP `kinuv-KGAS066-uvsign-map` stays read-only (live `stage_a_map.json`: `pa_deg=199.72980072503037`, `v0_kms=267.6703121989014`, `r_t_arcsec=0.5`, `dx_arcsec=0.09104737371760792`, `dy_arcsec=0.018566961155444102`, `chi2_map=168675.59555208942`, `s=0.5136098555284736`). Frozen `i`. No `h_z`. Do not logit `RT_BOUNDS_ARCSEC=(0.5, 15)`. Host `log_prob_unconstrained` is not autodiff (G2 dual-accept; live function is `float(log_prob(...)) + float(log_abs_det_jacobian)`). Freeze `(dx, dy)` at those MAP values (`σ=0.5″` → `0.182σ` and `0.037σ`, both `<1σ`; DEC-066-SHIFT). Two PA runs, 4 chains. `sampler: nuts` only after autodiff. No GPU. `DEC-HIER-SELFUNC` defer-only. `DEC-066-TARGET` still 066. That selected path is accept-eligible. Execute as typed can still install Gate 1's `2U` as the NumPyro potential, copy G1's flux-only `jax.grad`, or leave sampled-name `float()` / `np.radians` on the vis path.

## Attacks / bounds

1. **Gate 1 writes `2U` and calls it equivalent to `U`; that is a temperature error if it becomes `potential_fn`.** Live convention (`kinuv.infer.posterior`): `ln L = -chi2/2 - prior/2`, `chi2_and_prior = chi2 + shift_prior`, `log_prob = -0.5 * chi2_and_prior`. G2 host `log_prob_unconstrained = log_prob(θ(z)) + log|J|` (`tests/test_g2_chart.py` `test_log_prob_unconstrained_is_host_sum`, abs `1e-9`). Architect `U(z) = 0.5 (chi2 + shift_prior) - log|det J|` equals `-log_prob_unconstrained`. Execute item 2 `U = -log p(θ(z)) - log|J|` is the same `U`. Gate 1 instead names `jax.grad` of `chi2(θ(z)) + shift_prior - 2 log|J|`. That scalar is `2U`. Finite-grad is shared; the NUTS potential is not. Installing `2U` as NumPyro `potential_fn` (or `numpyro.factor(-2U)`) makes the posterior twice too cold: posterior variance `/2`, 16/50/84 widths `/√2`, while `R_hat` / `ESS` can still pass. Residual risk 4 already says leftover intervals will look tight; a factor-of-2 bug would look like calibration. **Bound:** NumPyro energy is `U = 0.5 (chi2 + shift_prior) - log|J|` = `-log_prob_unconstrained`, never `chi2 + prior - 2 log|J|`. Gate 1 may smoke-test finite `jax.grad(U)` or `jax.grad(2U)`, but `potential_fn` / `numpyro.factor` must use `U`. Always-on tiny-grid test (G1 jax-finufft skip, no 066 npz): `|U_jax(z) + log_prob_unconstrained(data, z, ...)| < 1e-6` at an official-like `z` (host G2 helper vs jitted JAX `U`; same `t=1`). Fail if the NUTS module calls `chi2_and_prior` or `log_prob_unconstrained` as the traced potential. Do not `jax.grad` the host helper (G2; `float` + `unconstrained_to_params` is not a tape). Do not fold `log|J|` into `chi2`.

2. **Execute item 1's `float()` list is incomplete; G1's flux-only grad will not catch the misses.** Live sampled-name host conversions on the XLA vis path, propose-listed and not:

   - `arctan_vc`: Python `if r_t_arcsec <= 0.0` and `float(v0_kms)` / `float(r_t_arcsec)` (`profiles/rotation.py`). Listed.
   - `los_velocity`: `float(vsys_kms)`. Listed.
   - `sky_cube`: `float(dx_arcsec)` / `float(dy_arcsec)` on `xe - …`, `yn - …`; `float(i_rad)` (frozen `i` may stay). Listed for offsets.
   - `shift_prior`: `float(dx)` / `float(dy)` (`infer/map.py`). Listed; constant after the freeze, still illegal if those names are tracers in an 8-vector `U(z)`.
   - `chi2_and_prior`: `predict_binned` **without** `xla=True`, then `float(c)`. Listed. Wrapping this (or host `log_prob`) as NumPyro and writing `sampler: nuts` is already on the reject-this-wave list; keep it there.
   - **Missed, sampled:** `_gaussian_pdf` does `sig = float(sigma_kms)` (`forward/model.py`). `gas_sigma_kms` is a live NUTS name (`exp(z_gs)`).
   - **Missed, sampled:** `predict_binned(..., xla=True)` does `pa_rad=np.radians(params["pa_deg"])`. `np.radians` on a tracer is a host bounce. PA is identity-chart and sampled.
   - **Missed, freeze-sensitive:** `fourier_shift` JAX arm does `float(dx_arcsec)` / `float(dy_arcsec)` (`template/fourier_shift.py`). Residual risk 1 names this as leftover; Gate 1 text says fail if *any* kinematic `float()` remains. Those two sentences disagree. Frozen MAP `(dx, dy)` as **Python constants** may keep `float(dx)`; the same calls on identity-`z` tracers break `jax.grad` of an 8-vector `U`.

   G1 `test_tiny_numpy_vs_jax_vis_and_grad` only `jax.grad`s **flux** (`tests/test_g1_jax.py`). Flux is a multiplier after `phi`; `float(sigma)` / `float(v0)` / `float(vsys)` / `np.radians(pa)` never enter that tape. G1 identity `|chi2-168675.6|<1` also uses a dict of Python floats. Copying G1 as Gate 1 can go green with every kinematic `float()` still in place. **Bound:** Gate 1 is `jax.jit` + `jax.grad` of `U` (or `2U` smoke) w.r.t. **all six sampled names** (flux, PA, vsys, gas_sigma, `V_0`, `r_t`), each component finite **and** matching a tiny-grid FD (`step 1e-5`–`1e-6` in `z`; rel `1e-3` / abs `1e-4` as G1 flux). Fail if `_gaussian_pdf` still `float(sigma_kms)` or `predict_binned` still `np.radians(params["pa_deg"])` on the XLA path. Frozen `i`, host `freqs`/`cell`/`eps`/`NPZ_UV_SIGN`, and Python-constant MAP `(dx, dy)` may `float()`. If `U` is defined on the length-8 vector, remove `float(dx)` / `float(dy)` from `fourier_shift` / `sky_cube` / `shift_prior` as well. Same skip as G1: `_require_jax_finufft` (`pytest.importorskip("jax")` then skip unless `BACKEND == "jax-finufft"`). Do not treat eager `float()` on a concrete `jnp` scalar as a pass (G2: that path can succeed and zero a gradient).

3. **`predict_binned` dict-of-Python-floats plus `unconstrained_to_params` is the host wire, not the NUTS tape.** G2: JIT surface is the length-8 `PARAM_NAMES` vector; dict packing is host-only; `unconstrained_to_params` does `np.asarray` + `vec_to_params` → `float(x[i])`. Propose lists the dict unpack as a live autodiff break, then execute item 2 still says `predict_binned(..., xla=True)` without changing that API. A NUTS module that maps `z → unconstrained_to_params(z) → chi2_and_prior` is wrapping NumPy `chi2_and_prior`. **Bound:** jitted `U` calls `unconstrained_to_physical(z)` (vector), never `unconstrained_to_params` / `vec_to_params` / `log_prob` / `log_prob_unconstrained`. If `predict_binned` stays a dict, values on the XLA path are JAX scalars (or the potential unpacks the length-8 θ itself). `chart.py` still has no `numpyro` import (Gate 3). `SAMPLER_NAME` in `posterior.py` stays `laplace_mh`.

4. **Logit of `[0.5, 15]` is still a wall; `log(r_t)` walking below 0.5″ is not a license to quote `dV/dr`.** Official `r_t=0.5` is exactly `RT_BOUNDS_ARCSEC[0]` (`infer/seeds.py`). `logit((0.5-0.5)/(15-0.5))` is `-inf`; HMC cannot start. G2 rejected this; G0 `r_t_at_floor` / `quote_inner_slope` already say the floor is an L-BFGS box, not a science prior (`diagnostics/flags.py`, `RT_FLOOR_ARCSEC=0.5`). Chart `exp(z_rt)` is the right support. **Bound:** `chart.py` source still has no `logit`, `RT_BOUNDS`, or numeric `(0.5, 15)` clip (keep the G2 grep). Do not import `RT_BOUNDS_ARCSEC` to “keep HMC in the MAP box.” Product must keep `r_t_at_floor` and must not quote inner `dV/dr` while it fires — including if NUTS walks `r_t < 0.5`.

5. **`R_hat` / `ESS` on frozen `(dx, dy)` or on a stacked 199.73°+25.2° soup will lie.** Live `split_rhat`: `W=0` on a constant column → `sqrt(var_hat/W)` is `nan`; `ess_bulk` demeans a constant and divides by `ac[0]`. Gate 5 already excludes frozen names. Propose does not lock the JSON shape or forbid stacking the two PA runs for mixing. `plot_posterior_corner` will still histogram all 8 columns and draw 16/50/84 on the two delta spikes (`diagnostics/figures.py`). Roadmap G3 text still says PA starts 205.2 and 25.2; the receding start on this card is the official MAP `199.7298°` (seed 205.2 is history). **Bound:** mixing dict keys are exactly the six sampled names. Fail Gate 5 if `R_hat`/`ESS` are reported on a column with chain variance `0`, or if the two PA runs are concatenated for those stats. Two artifacts (or one JSON with two named runs); do not average; do not stack. Draw arrays stay 8 columns in `PARAM_NAMES` order with `dx`/`dy` constant at the official MAP floats above. JSON records `frozen_names: ["dx_arcsec", "dy_arcsec"]` and the actual `pa_init_deg` (`199.72980072503037` and `25.2`). Corner may plot the eight columns; interval tables must not present frozen 16/50/84 as sampled CIs. `shift_prior` is a constant and must not appear as a NumPyro `sample` on `(dx, dy)` (exactly six RVs).

6. **NumPyro is not in `kinuv-venv-recovery`; `importorskip` plus “gates 1–4 always” can skip the only NUTS test.** Live `pyproject.toml` has extras `test`, `nufft`, `io`, `plot` — no `numpyro`. Propose residual 6 (jax 0.11.x pin) is real. Execute item 5 says `tests/test_g3_nuts.py` covers gates 1–4 always. Gate 1 needs jax-finufft (G1 skip). Gate 4 needs NumPyro. A default `pytest` that `importorskip("numpyro")` goes green without a single NUTS step. **Bound:** add optional extra `nuts = ["numpyro"]` (no `emcee`). Install into the recovery venv on execute. After that install, re-run official `|chi2-168675.6|<1` with frozen `s=0.5136098555284736` (compatibility gate). In that venv Gate 4 must **run**, not skip; STATUS records ran/skipped explicitly. Tiny-mock: 4 chains, `R_hat < 1.01` and `ESS > 200` on sampled names only, before 066. Skip 066 if tiny-mock fails; do not write `sampler: nuts` on a 066 JSON in that case. Tests that do not need NUFFT (no-`logit` grep, no-`numpyro` in `chart.py`, `SAMPLER_NAME == "laplace_mh"`, `plot_posterior_corner` still raises on `laplace_mh`) run without the G1 skip.

7. **Leftover-vs-velocity plus failed S2 SBC means 066 16/50/84 are not calibrated; a README sentence is not a gate.** Official MAP still fires G0 `r_t_at_floor` and `leftover_chi2_structured`. S2 Laplace SBC n=20 failed 68/95. `plot_posterior_corner` draws 16/50/84 on every column, including kinematics that leftover-vs-velocity already says the model misses. Gate 6 asks for caveats “in the product, not in a later excuse.” That is prose unless the JSON is locked. `map_quality_flags` still hardcodes `nuts_absent: True` — do not call that dict as proof the G3 artifact is NUTS, and do not “fix” flags on the official MAP tree. **Bound:** each G3 product JSON has `intervals_calibrated: false` (or equivalent) and `sampler == "nuts"` only after Gate 1. Corner title / README in `docs/reviews/artifacts/2026-08-30-g3-nuts/` must say not calibrated / not S2 Laplace. Do not quote S2 16/50/84. Do not quote inner `dV/dr`. Do not treat leftover-vs-velocity as fixed because NUTS mixed.

## Comments

1. `major` -- NUTS potential is `U = 0.5(chi2 + shift_prior) - log|J|` = `-log_prob_unconstrained`. Tiny-grid `|U_jax + log_prob_unconstrained| < 1e-6`. Gate 1 finite-grad may use `U` or `2U`; `potential_fn` must not use `2U`. Do not `jax.grad` the host helper. Do not wrap `chi2_and_prior`.

2. `major` -- Remove sampled-name host conversions: `arctan_vc` (`if r_t<=0`, `float(v0)`, `float(r_t)`), `los_velocity` `float(vsys)`, `_gaussian_pdf` `float(sigma)`, `predict_binned` `np.radians(pa)`. Gate 1 = jitted `jax.grad(U)` on all six sampled names vs FD, same skip as `tests/test_g1_jax.py` `_require_jax_finufft`. Flux-only grad is not Gate 1.

3. `major` -- Jitted `U` uses `unconstrained_to_physical` + `predict_binned(..., xla=True)` with JAX-valued params, not `unconstrained_to_params` / dict of Python floats. `chart.py` has no `numpyro`. `SAMPLER_NAME` stays `laplace_mh`. Label `sampler: nuts` only after Gate 1.

4. `major` -- Do not logit `RT_BOUNDS`. Freeze `(dx, dy)` at official MAP (not at 0). Mixing stats on the six sampled names only; do not stack the two PA runs; JSON `frozen_names` + `intervals_calibrated: false`. Two runs: receding init `199.72980072503037`, approaching `25.2`, 4 chains each. `plot_posterior_corner` accepts that JSON and still raises on `laplace_mh`.

5. `minor` -- `nuts = ["numpyro"]` extra; install in recovery venv; identity χ² after install. Gate 4 must run there (not `importorskip` green). `JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`, scratch/JAX cache under `/scratch/kinuv-$USER` else `/tmp`. No GPU. Post-warmup eval/s vs G1 3.01 and S2 0.329; a cost miss is STATUS + continue only if autodiff + mixing hold. Frozen `s=0.5136098555284736`. Operator `hann_then_bin`; `NPZ_UV_SIGN=-1`. No second SPECRESP path. Official MAP tree untouched. No new `DEC-*`. No G4. CHANGELOG + G3 roadmap pointer (receding start is MAP 199.73, not seed 205.2).

## Residual risks

1. First JIT of the 066 potential can be minutes. Speed notes are post-warmup. Do not GPU this card.

2. NumPyro + live jax 0.11.x pin in the recovery venv. Identity `|chi2-168675.6|<1` after install is the compatibility gate.

3. `log(r_t)` lets NUTS walk below 0.5″. That does not unstick leftover-vs-velocity and does not license inner `dV/dr` while `r_t_at_floor` fires.

4. Real-066 NUTS 16/50/84 will look tight while leftover vs velocity is structured. They are not calibrated. S2 Laplace already failed SBC. JSON `intervals_calibrated: false` is the lock; a README sentence is not.

5. Approaching-PA (25.2°) may not mix toward the receding MAP. That is a 180° mode, not a license to flip PA in the product without saying so, and not a license to stack runs to “make ESS.”

6. Frozen `(dx, dy)` 8-column corners show delta spikes. Readers will quote those as CIs unless `frozen_names` is on the JSON and interval tables omit them.

7. `fourier_shift` / NUFFT `eps` / Hann weights can still host-bounce if treated as tracers. Gate 1 (`is_jax` on jitted `U`, finite six-name grad vs FD) is reject-if-fail, not “we deleted the listed floats.”

8. `is_jax` remains a module-prefix test (G2 carry-forward). Do not “fix” `xp.py` unless Gate 1 fails on live arrays.

9. G0 `map_quality_flags` still hardcodes `nuts_absent: True`. That flag is about the official MAP product, not the G3 artifact. Do not write the official MAP tree to flip it.

## STATUS updates required

- `verdict` and `severity` as in the header (`accept`, `major`)
- `last_review_a`: this file
- Do not set `board: accepted` (parent tallies)
