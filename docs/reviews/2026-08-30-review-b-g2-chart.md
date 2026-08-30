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
  - DEC-HIER-SELFUNC
verdict: accept
severity: major
propose: docs/reviews/2026-08-30-propose-g2-chart.md
---

# Review b: G2 unconstrained Stage A chart + Jacobian (066 kernel)

Do not read the other seat's review file. Do not implement.

Accept because execute is `kinuv.infer.chart` plus `tests/test_g2_chart.py` only: eight live `PARAM_NAMES`, log on `flux` / `gas_sigma_kms` / `r_t_arcsec`, stable softplus on `v0_kms`, identity on PA / vsys / `(dx, dy)`, no logit of `RT_BOUNDS_ARCSEC=(0.5, 15)`, no `i` / `h_z`, no NumPyro, `SAMPLER_NAME` stays `laplace_mh`, official `kinuv-KGAS066-uvsign-map` read-only. The card as written can still ship a Jacobian that passes a mixed `chi2` finite difference, a Python-`if` softplus that cannot `jit` (or an `xp.where` that overflows at `V_0=1e3`), and a `dict[str, float]` `log_prob_unconstrained` that host-bounces through NumPy `chi2_and_prior`. Those are implementer-must-fix bounds, not a re-propose.

## Attacks / bounds

1. **`chi2` FD is not the chart Jacobian; the 8-vector log-det is a sum of signed 1-D terms.** Propose: `log p(z) = log p(θ(z)) + Σ log |dθ_i / dz_i|` with independent 1-D maps; coded identities `log: ln|dp/dz|=z`, `softplus: -softplus(-z)`, `identity: 0`. That sum is correct only if `log_abs_det_jacobian` is the scalar sum of those 1-D terms on a diagonal map, not `slogdet` of a mixed `chi2` Jacobian. Gate 3 then says analytic `log|det J|` matches a numerical derivative of the coordinate transform per active dimension, *and* `jax.grad` of a tiny `chi2(θ(z))` when jax-finufft is present. Those are different objects. `chi2(θ(z))` never sees the density correction; matching `jax.grad(chi2 ∘ θ)` to an FD of `chi2 ∘ θ` passes a wrong `log_abs_det_jacobian` of all zeros. Official Stage A is a critical point of `chi2 + shift_prior` (`chi2_map=168675.59555208942`, `v0_kms=267.6703121989014`, `r_t_arcsec=0.5`): FD of `chi2` wrt `z_i` and wrt `θ_i` are both ~0, so a ratio `d chi2/dz / d chi2/dθ` is 0/0 and cannot recover `ln|dp/dz|`. At that MAP the softplus term is also invisible: `inv_softplus(267.7)` takes the `y>20` identity, `σ(z)≈1`, `ln|J|≈0`. A MAP-only FD would accept `log|J|=0` on `v0` and would hide a sign error on `r_t` (`z=ln(0.5)≈-0.693147`; `ln|J|=z` must stay **negative**). Live `PARAM_NAMES` order (`map.py` / `posterior.py`): `flux`, `pa_deg`, `vsys_kms`, `gas_sigma_kms`, `dx_arcsec`, `dy_arcsec`, `v0_kms`, `r_t_arcsec`. **Bound:** Jacobian FD is of `unconstrained_to_physical` wrt each `z_i`, not of `chi2` or `log_prob`. Central step `h ∈ [1e-6, 1e-4]` on unconstrained `z` (do not reuse `map.FD_STEP` / `posterior.FD_STEP = 1e-3` on `chi2`). Assert **each** active axis against the named formula, signed, not `|analytic|` vs `|numeric|`: `flux` log (`ln|J|=z=ln p`), `gas_sigma_kms` log, `r_t_arcsec` log at official `r_t=0.5` (`ln|J|<0`), `v0_kms` softplus at a point where `σ(z)` is not ~1 (required: `z=0` → `ln|J|=-ln 2`, or `v0=1e-4`), and one identity axis (`pa_deg` or `dx_arcsec`) analytic `0` with `|FD dθ/dz - 1| < 1e-8`. Then assert the 8-vector `log_abs_det_jacobian` equals the **sum** of those eight 1-D terms. `jax.grad` of tiny `chi2(θ(z))` stays a separate skip (same rule as `tests/test_g1_jax.py`: jax-finufft present). Do not let that skip stand in for the Jacobian gate. Jacobian tests run without jax-finufft.

2. **Stable softplus: Python `if` breaks `jit`; `xp.where` re-overflows; `v0=0` is `-inf`.** Propose branches at 20: `softplus(x)=x+ln(1+e^{-x})` if `x>20` else `ln(1+e^x)`; `inv_softplus(y)=y` if `y>20` else `ln(-expm1(-y))+y`. Gate 2 only roundtrips `V_0 ∈ [1e-4, 1e3]`. Official MAP `267.7` is the identity branch and will pass a naive `log1p(exp(x))`. Live `V0_BOUNDS_KM_S = (0.0, 400.0)` in `seeds.py`; L-BFGS may sit on `0`. `physical_to_unconstrained` of `v0=0`: `inv_softplus(0)=ln(-expm1(0))+0=ln(0)=-inf` (same wall `log` was accused of). That is not a green roundtrip. At `y=1e-4` the expm1 form is finite (`z≈-9.210`). At `y=1e3>20` the identity branch avoids `exp(1000)` overflow **only if the unused branch is not evaluated**. A Python `if x > 20` on a JAX tracer is `ConcretizationTypeError` under `jax.jit`. `xp.where` traces **both** sides: `where(x>20, x+log1p(exp(-x)), log1p(exp(x)))` still computes `exp(1000)` on the unused arm; `jax.grad` through that `inf` is nan. `is_jax` (`xp.py`) is `type(x).__module__.startswith("jax")` and will see a tracer, so dispatch is not the bug — the branch is. **Bound:** no Python `if` on the array value. Both arms must stay finite for `V_0 ∈ [1e-4, 1e3]` (e.g. `logaddexp(0, x)` or `max(x,0)+log1p(exp(-abs(x)))`, and clip any `exp` argument). `jax.jit` + `jax.grad` of `softplus` / `inv_softplus` on a vector that straddles 20 (`[1e-4, 21, 1e3]`). Grad at `y=1e-4` is finite (`~1/y`); do not require grad at `0`. Host `v0=0`: `-inf` or a documented `ValueError`; do not silently clip to `1e-4` (that is not a bijection on the L-BFGS wall). `v0=1e-4` and `v0=1e3` roundtrip. Do not treat official `267.7` as the softplus gate.

3. **JAX dispatch dies if the API is `dict[str, float]` or if `log_prob_unconstrained` wraps NumPy `chi2_and_prior`.** Gate 4: `jnp` in (including a `jax.jit` tracer) → `jnp` out via `numpy_or_jax`. Live Stage A API is already a host bounce: `vec_to_params` / `_unpack` (`map.py`) build `dict[str, float]` with `float()` on every slot; `chi2_and_prior` calls `predict_binned(...)` **without** `xla=True` then `return float(c) + shift_prior(...)`; `shift_prior` is `(float(dx)/0.5)**2+(float(dy)/0.5)**2`. `log_prob` is `-0.5 * chi2_and_prior` (`SAMPLER_NAME="laplace_mh"`). Execute still requires `log_prob_unconstrained`. If that function does `θ = unconstrained_to_physical(z)` → `vec_to_params` → `log_prob` + `log|J|`, then (i) `jax.jit` hits `float()` / `ConcretizationTypeError`, (ii) `jax.grad` of a Python-`float` `chi2` plus a JAX `log|J|` differentiates **only the Jacobian term** and would let G3 sample the chart measure without the likelihood, (iii) NumPy-only tests will miss it. Returning `dict[str, float]` from `unconstrained_to_physical` has the same host bounce; `is_jax` on a `dict` is false. `arctan_vc` still does `if r_t_arcsec <= 0.0: raise` and `float(v0_kms)` / `float(r_t_arcsec)` — a JAX `r_t` from the chart cannot enter that path this card. **Bound:** JIT path is an 8-vector in `PARAM_NAMES` order → 8-vector out (`unconstrained_to_physical`, `physical_to_unconstrained`) and a **scalar** `log_abs_det_jacobian` of the same module. `jax.jit` the 8-vector maps; `assert is_jax(out)`. Dict packing is host-only and not inside the jit. `log_prob_unconstrained` this card is either a caller-supplied `log p(θ)` plus the analytic sum, or a host-only convenience around `chi2_and_prior`; it is **not** jitted and is **not** autodiff. Do not import NumPyro. Do not pass `xla=True` from `chart.py` (G3 wires that). Do not reuse `map._unpack` / `_pack` / `_SCALES` — that `z` is affine L-BFGS, not log/softplus. Do not change `SAMPLER_NAME`.

Carry-forward (this execute must not reopen): do not logit `RT_BOUNDS_ARCSEC=(0.5, 15)` (`seeds.py` L-BFGS box; official MAP on the wall; G0 `r_t_at_floor`). Do not logit `GAS_SIGMA_BOUNDS_KM_S=(2, 50)` or `FLUX_BOUNDS_JY`. Do not lower the production floor. Do not unfreeze `i` (`inclination_rad()` stays). Do not add `h_z` (`h_z_in_model` stays false). Eight Stage A names only. Do not label `laplace_mh` as NUTS. Do not quote S2 16/50/84. GPU only after a 066 CPU NUTS smoke. `DEC-HIER-SELFUNC` / TARGET stay deferred. Hann+bin only. `NPZ_UV_SIGN=-1`. Frozen `s=0.5136098555284736`.

## Comments

1. **major.** Jacobian gate is FD of `unconstrained_to_physical` per axis, `h ∈ [1e-6, 1e-4]`, signed identities, individual asserts on `flux`, `gas_sigma_kms`, `r_t_arcsec` (negative at `0.5`), `v0_kms` away from `σ≈1`, and one identity zero. Scalar `log_abs_det_jacobian` is the sum. Do not mix `chi2` FD. Tiny `chi2(θ(z))` grad stays a jax-finufft skip, not the Jacobian test.

2. **major.** Softplus / `inv_softplus`: no Python `if` on the value; both `where` arms finite on `[1e-4, 1e3]`; `jax.jit` + `jax.grad` on a vector that straddles 20. `v0=0` is `-inf` or `ValueError`, not a silent clip. `v0=1e-4` and `v0=1e3` roundtrip. Official `267.7` is not the edge gate.

3. **major.** JIT path is 8-vector ↔ 8-vector plus scalar log-det, `jnp` in → `jnp` out under `jax.jit`. Not `dict[str, float]`. `log_prob_unconstrained` must not wrap NumPy `chi2_and_prior` / default `predict_binned` and claim autodiff. Do not reuse `map._unpack`. `SAMPLER_NAME` stays `laplace_mh`.

4. **major.** Source of `chart.py` must not logit `RT_BOUNDS_ARCSEC`, `GAS_SIGMA_BOUNDS_KM_S`, or `FLUX_BOUNDS_JY`. Do not lower `RT_BOUNDS_ARCSEC`. Do not add `i` or `h_z`. Official MAP read-only; no refit; no new `DEC-*`.

5. **major.** Do not start G3 NumPyro, G4 SBC, G5 PSIS-LOO, GPU, a 400-galaxy runner, or a second SPECRESP path. Commit subject is unconstrained chart / Jacobian, not NUTS. Human plot folder stays `docs/reviews/artifacts/2026-08-30-final-fit/`.

6. **minor.** Official `|chi2-168675.6|<1` after roundtrip uses leftover `s=0.5136098555284736` when the CANFAR npz exists (same skip spirit as G1). Always-on tests are (a)–(d) without jax-finufft except the type-preservation `jit` (needs `jax`, not jax-finufft). `conftest.py` already sets `JAX_PLATFORMS=cpu`, x64, and `/scratch` then `/tmp`; do not write a second cache policy.

7. **minor.** `DEC-066-SHIFT` Gaussian σ=0.5″ stays in `log p(θ)`, not in the chart Jacobian. Identity on `(dx, dy)` is the chart; the ±2″ box stays MAP-only. G3 may freeze shifts if both are <1σ of 0 (official `dx=0.091`, `dy=0.019`); not this card.

## Residual risks

1. `log(r_t)` lets later NUTS walk below 0.5″. That does not unstick leftover-vs-velocity and does not license quoting inner `dV/dr` while G0 `r_t_at_floor` fires (propose residual 1). Production box stays for MAP.

2. Identity PA/vsys do not wrap. G3 still uses two PA starts (205.2 and 25.2). Official MAP `pa_deg=199.73` is interior of the identity chart (propose residual 2).

3. A correct Jacobian is not a calibrated posterior. S2 Laplace SBC failed 68/95. Do not quote S2 16/50/84. No NUTS posterior exists (`laplace_mh`).

4. Host `log_prob` still calls NumPy `predict_binned` unless the caller passes `xla=True`. G3 wires NumPyro to JAX `chi2`. `arctan_vc` `float(r_t)` / Python `r_t<=0` still host-bounce a JAX scalar; that is a G3 path bug, not a G2 license to fork SPECRESP or rewrite `arctan_vc` this card.

5. `v0=0` remains `-inf` on this chart. Softplus lets `V_0→0+` but does not make the L-BFGS endpoint an interior unconstrained point. A later NUTS run that starts from a zero-V box edge will fail.

6. Leftover vis `chi2` vs velocity stays structured (frozen Wiener Ico). True NUTS CIs on real 066 stay overconfident after a correct chart. `DEC-HIER-SELFUNC` still has no selection function.

## STATUS updates required

- `verdict` and `severity` as in the header (`accept`, `major`)
- `last_review_b`: this file
- Do not set `board: accepted` (parent tallies)
