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
  - DEC-066-VC
  - DEC-066-WEIGHT
  - DEC-HIER-SELFUNC
verdict: accept
severity: major
propose: docs/reviews/2026-08-30-propose-g2-chart.md
---

# Review a: G2 unconstrained Stage A chart + Jacobian (066 kernel)

Do not read the other seat's review file. Do not implement.

Scope check: G2 chart + unit tests only. Existing ids. No new `DEC-*`. Official MAP `kinuv-KGAS066-uvsign-map` stays read-only (verified live: `r_t_arcsec=0.5` exactly, `chi2_map=168675.59555208942`, `s=0.5136098555284736`). Operator remains `hann_then_bin`; `native_diagonal` still raises. `NPZ_UV_SIGN=-1`. Frozen `i`. No `h_z`. `SAMPLER_NAME` stays `laplace_mh`. Do not quote S2 16/50/84. Do not logit `RT_BOUNDS_ARCSEC=(0.5, 15)` (L-BFGS box in `kinuv.infer.seeds`, not a prior; G0 `r_t_at_floor` already says this). No NumPyro, no NUTS label, no G3, no SBC, no MAP rewrite, no 400-galaxy runner. `DEC-HIER-SELFUNC` defer-only. `DEC-066-TARGET` still 066. Log flux / gas_sigma / r_t, stable softplus `V_0`, identity PA / vsys / dx / dy is the right eight-name bijection. That is accept-eligible. Execute as typed can still ship a Python `if x > 20` and a dict-of-floats chart that NumPy tests pass and `jax.jit` cannot trace.

## Attacks / bounds

1. **Python `if x > 20` is not JAX-traceable; the written softplus is a math note, not a `jax.numpy` body.** Propose: branch at 20 with `log1p`/`expm1`, then writes `softplus(x) = x + ln(1+e^{-x})` if `x > 20` else `ln(1+e^x)`, and `inv_softplus(y) = y` if `y > 20` else `ln(-expm1(-y)) + y`. Live jax 0.11.1 (`kinuv-venv-recovery`): `jax.jit` of that Python `if` raises `TracerBoolConversionError: Attempted boolean conversion of traced array with shape bool[]`. Eager `if` on a 0-d `jnp` scalar **succeeds** (host-converts the bool). `jax.grad` of the Python `if` also succeeds. A MAP-only or eager-JAX test at `V_0 ≈ 268` will green-light the formula G3 will JIT. Naive `log(1+exp(x))` is finite at 20–100 in float64 and becomes `inf` at `x=710`; the `[1e-4, 1e3]` gate hits `z ≈ 1000` only if both maps use the large branch. The inverse `log(exp(y)-1)` cancels as `y → 0`; `ln(-expm1(-y))+y` is the right small-`y` form. `jnp.where` evaluates both branches; unused `log1p(exp(1000))` is `+inf`, and live `where` still returns 710 at `x=710`, but a NaN in the unused branch would poison. **Bound:** implement on `xp = numpy_or_jax(x)` (resp. `y`), never a Python `if` on a possibly-traced scalar:

   `softplus(x) = xp.where(x > 20, x + xp.log1p(xp.exp(-x)), xp.log1p(xp.exp(x)))`

   `inv_softplus(y) = xp.where(y > 20, y, xp.log(-xp.expm1(-y)) + y)`

   `log1p` of `xp.exp(±x)`, not `xp.log(1 + xp.exp(...))`. Clip the unused `where` arm (`xp.minimum(x, 20)` on the small branch) or use `lax.select` if `jit(softplus)(1000)` is not finite. Gate: `V_0 ∈ {1e-4, 1e-3, 1.0, 20, 21, 267.6703121989014, 1e3}` physical→z→physical, max relative error `< 1e-10` on both NumPy and `jax.jit` paths (branch switch at 21 is the weak point; live where+log1p/expm1 was ~4e-11 there). Jitted `softplus` on `linspace(-40, 40)` and at `1000` must stay finite and return a JAX array. `log_abs_det_jacobian` for the softplus axis is `-softplus(-z)` using **that** stable `softplus`, not `log(1+exp(-z))`. `V_0=0` maps to `-inf`; that is not a finite-z gate (unlike official `r_t=0.5`). Do not logit `(0, 400)` either.

2. **`is_jax` on tracers is OK today; a dict-of-floats API and `log_prob_unconstrained` → live `chi2_and_prior` are not a JIT surface.** Live `kinuv.xp.is_jax` is `type(x).__module__.startswith("jax")`. On jax 0.11.1 a concrete array is `jaxlib._jax.ArrayImpl` (prefix matches because `jaxlib` starts with `jax`); a `jax.jit` tracer is `jax._src.interpreters.partial_eval.DynamicJaxprTracer` (`is_jax` True; `numpy_or_jax` returns `jax.numpy`). Gate 4 as written can still pass a false test: `np.asarray` of a jitted output is `numpy.ndarray`, `is_jax` False; `isinstance(..., jnp.ndarray)` after that asarray is a NumPy pass. `numpy_or_jax(*{"flux": 70.46, ...}.values())` returns **NumPy** — every value is a Python float. `float(z[i])` inside `unconstrained_to_physical` raises `ConcretizationTypeError` under `jit` (live). Live `shift_prior` is `(float(dx)/σ)² + (float(dy)/σ)²`. Live `chi2_and_prior` calls `predict_binned(..., xla=False)` then `float(c) + shift_prior(...)`. Live `log_prob` is `-0.5 *` that. Execute ships `log_prob_unconstrained` on top of "log p(θ) as today." Residual risk 4 only names NumPy `predict_binned`. Gate 4 names only `unconstrained_to_physical` and `log_abs_det_jacobian`. **Bound:** the JIT surface is a length-8 vector in `PARAM_NAMES` order (`posterior.params_to_vec` / `map.PARAM_NAMES`, same eight names). Those two maps plus `log_abs_det_jacobian` must not `float()` a coordinate. Dict-of-Python-float helpers, if any, are host-only and are **not** what gate 4 tests. Test must be:

   `z = jnp.ones(8); out = jax.jit(unconstrained_to_physical)(z); assert is_jax(out)`

   (or `is_jax` on each entry if the return is a dict of arrays) and the same for `log_abs_det_jacobian`. Fail the card if the test `np.asarray`s first, or only checks `isinstance` on a host copy. Chart JIT / FD-Jacobian / `V_0` roundtrip run whenever `jax` imports — do **not** hide them behind the G1 `BACKEND != "jax-finufft"` skip (`tests/test_g1_jax.py` `_require_jax_finufft`). Only the optional tiny `jax.grad` of `chi2(θ(z))` uses that skip. `log_prob_unconstrained` in G2 is host/NumPy: `log_prob(θ(z)) + log_abs_det_jacobian(z)`. Unit test that equality; the Jacobian is **added to `log_prob`**, not folded into `chi2_and_prior` inside the `1/2`. Do not `jax.jit` `log_prob_unconstrained` this wave. Do not pass `xla=True` from the chart. Do not claim this function is G3-ready.

3. **`DEC-066-VC` / live `arctan_vc` still host-converts; G2 must not pretend the likelihood is chart-JIT-complete.** `arctan_vc` does `if r_t_arcsec <= 0.0: raise` then `float(v0_kms) * (2/π) * arctan(r / float(r_t_arcsec))`. Live `jit` of that `if` and of `float(rt)` is the same `TracerBoolConversionError` / `ConcretizationTypeError`. Official `r_t=0.5` is finite in `log` (`log(0.5) ≈ -0.693147`, not `-inf`). The log chart is correct: `r_t > 0` is the VC requirement; `(0.5, 15)` is not. **Bound:** `chart.py` source must not contain `logit`, `RT_BOUNDS`, or a numeric `(0.5, 15)` clip. Grep test as gate 1. Do not import `RT_BOUNDS_ARCSEC` to "validate." Do not reuse `map._unpack` / `_SCALES` (that is the L-BFGS scaled box, a different `z`). Do not edit `arctan_vc`, `shift_prior`, `chi2_and_prior`, or `predict_binned` this wave. G3 owns making `chi2(θ(z))` traceable.

4. **Numerical Jacobian gate has no number; official χ² gate must not become a sampler wire.** Gate 3: analytic `log|det J|` vs FD of the coordinate transform per active dimension; optional `jax.grad` of tiny `chi2(θ(z))` when jax-finufft is present. No step, no tol. Gate 5: `|chi2 - 168675.6| < 1` after roundtrip, frozen `s=0.5136`, no refit. Live JSON `s` is `0.5136098555284736` (G1 already asserts that). `shift_prior` at official `(0.091047, 0.018567)` is ~0.033; do not compare `chi2_and_prior` to 168675.6 as if it were χ². **Bound:** FD step on `z` `1e-5`–`1e-6`; per-axis `|analytic - FD|` `< 1e-6` rel or `< 1e-8` abs on all eight names (identity axes: analytic `0`). Official χ² uses `predict_binned` on the round-tripped θ, `data.s` from the loaded vis / JSON, Hann+bin XX 881×95, `|chi2 - 168675.6| < 1` when the CANFAR npz exists. Skip official only if npz/JSON missing; chart tests still run. `SAMPLER_NAME` remains `laplace_mh`. No NumPyro import in `chart.py` or `test_g2_chart.py`. Do not write `sampler: nuts`. Do not quote S2 16/50/84.

## Comments

1. `major` -- Softplus / inv_softplus: `xp.where` + `log1p`/`expm1` as in attack 1. No Python `if` on tracers. Jacobian uses the same stable `softplus`. `[1e-4, 1e3]` + branch points `{20, 21}` + official `v0_kms=267.6703121989014`, rel `< 1e-10`, NumPy and `jax.jit`. `jit(softplus)(1000)` finite.

2. `major` -- JIT surface is the length-8 `PARAM_NAMES` vector. Gate 4 asserts `is_jax` on outputs of `jax.jit(...)` with no `np.asarray` first. Chart JIT tests are jax-only, not jax-finufft. `log_prob_unconstrained` is host `log_prob(θ(z)) + log|det J|`; equality test; do not JIT it; do not call `xla=True`.

3. `major` -- `chart.py` must not logit or mention `RT_BOUNDS_ARCSEC`. Official `r_t=0.5` → finite `z`. Do not reuse MAP `_SCALES` z. Do not touch `arctan_vc` / `shift_prior` / SPECRESP this wave.

4. `minor` -- Import `PARAM_NAMES` from one live module (`kinuv.infer.posterior` or `kinuv.infer.map`); do not mint a third copy. FD Jacobian tol as in attack 4. Official χ² is χ², not χ²+prior. Frozen `s` is the JSON / `VisData.s` value, not a recomputed `empirical_s`. `JAX_PLATFORMS=cpu`; cache under `/scratch/kinuv-$USER` if writable, else `/tmp`. `JAX_ENABLE_X64=1` before importing jax in tests that mix with G1 χ².

5. `minor` -- Gate 2 still: `hann_then_bin` only; `native_diagonal` still raises. No second SPECRESP path. `NPZ_UV_SIGN` stays `-1`. Frozen `i`. No `h_z`. CHANGELOG + G2 roadmap pointer + one STATUS line. Commit subject is unconstrained chart / Jacobian, not NUTS. Human plots stay `docs/reviews/artifacts/2026-08-30-final-fit/`.

## Residual risks

1. `log(r_t)` lets later NUTS walk below 0.5″. That does not unstick leftover-vs-velocity and does not license quoting inner `dV/dr` while G0 `r_t_at_floor` fires on the official MAP.

2. Identity PA/vsys do not wrap. G3 still uses two PA starts (205.2 and 25.2).

3. A correct Jacobian is not a calibrated posterior. S2 Laplace SBC already failed 68/95. Do not quote those intervals.

4. A JIT-safe chart does not make live `log_prob` / `chi2_and_prior` / `shift_prior` / `arctan_vc` traceable. G3 wires JAX `chi2`; G2 must not claim `log_prob_unconstrained` is that wire.

5. `is_jax` is a module-prefix test. It works on jax 0.11.1 tracers and `jaxlib._jax.ArrayImpl`. A future array type whose module does not start with `jax` would silently dispatch to NumPy. Do not "fix" `xp.py` on this card unless gate 4 fails on live arrays.

6. `DEC-066-SHIFT`: after MAP, freeze `(dx, dy)` for NUTS only if both are consistent with 0 at <1σ. Official `(0.091, 0.019)` / σ=0.5″ are both <1σ. That freeze is a G3 sampling call, not a reason to drop those two names from this chart.

7. Official Stage A χ² identity needs the CANFAR npz. Chart bijection / Jacobian / JIT type tests are the always-on gates.

## STATUS updates required

- `verdict` and `severity` as in the header (`accept`, `major`)
- `last_review_a`: this file
- Do not set `board: accepted` (parent tallies)
