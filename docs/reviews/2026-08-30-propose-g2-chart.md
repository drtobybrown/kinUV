---
role: proposer
date: 2026-08-30
agent: parent
canon_generation: 4
ids:
  - DEC-066-INFER
  - DEC-066-INC
  - DEC-066-SHIFT
  - DEC-066-SPECRESP
  - DEC-066-TARGET
  - DEC-HIER-SELFUNC
verdict: propose
---

# G2: unconstrained Stage A chart + Jacobian (066 kernel)

## Scope

Next wave of the dual-accepted 066 kernel sequence ([`gold-standard-roadmap.md`](../diagnostics/gold-standard-roadmap.md)). G0 flags and G1 JAX identity are landed. Existing ids only. No new `DEC-*`. Official MAP `kinuv-KGAS066-uvsign-map` stays read-only. This card is the unconstrained chart on current Stage A names, not NUTS, not NumPyro, not SBC, not a MAP rewrite, not a 400-galaxy runner.

`DEC-HIER-SELFUNC` is defer-only. `DEC-066-TARGET` still 066. `DEC-066-INC` freeze stands: no `i`, no `h_z`.

## Architect verdict (selected path)

**Do this card:** ship a bijection `z ∈ R^8 ↔ θ` plus `log |det dθ/dz|` so a later HMC card can sample unconstrained coordinates. Production L-BFGS boxes stay MAP-only. The chart is physics (positivity / identity) plus the existing Gaussian `(dx, dy)` prior (`DEC-066-SHIFT`, σ = 0.5″).

**Eight names only** (`PARAM_NAMES` in `kinuv.infer.map` / `kinuv.infer.posterior`):

| Parameter | Chart | Reason |
|---|---|---|
| `flux` | `log` | strictly positive; MAP 70.46 is interior of `(0, inf)` |
| `gas_sigma_kms` | `log` | strictly positive; do **not** logit `(2, 50)` |
| `r_t_arcsec` | `log` | strictly positive (`arctan_vc` requires `r_t > 0`); `log(0.5)` is finite. **Not** logit of `[0.5, 15]` |
| `v0_kms` | `softplus` | non-negative; MAP 267.7 is interior; `log` would forbid `V_0 → 0` |
| `pa_deg`, `vsys_kms`, `dx_arcsec`, `dy_arcsec` | identity | interior at the official MAP; two G3 PA starts remain the 180° handle; `(dx, dy)` prior stays Gaussian, not the ±2″ MAP box |

Unconstrained density: `log p(z) = log p(θ(z)) + Σ log |dθ_i / dz_i|` with `log p(θ) = -0.5 * (chi2 + shift_prior)` as today. Independent 1-D maps; Jacobian is a sum.

**Jacobian identities (coded, not implied):**

- log (`p = exp(z)`): `dp/dz = p`, so `ln |dp/dz| = z = ln p`
- softplus (`p = ln(1+e^z)`): `dp/dz = σ_logistic(z)`, so `ln |dp/dz| = -softplus(-z)`
- identity: `ln |dp/dz| = 0`

**Stable softplus / inverse (required):** naive `log(1+exp(x))` overflows for large `x`; `log(exp(y)-1)` cancels for `y → 0`. Implement with `log1p` / `expm1` and a branch at 20:

- `softplus(x) = x + ln(1+e^{-x})` if `x > 20`, else `ln(1+e^x)`
- `inv_softplus(y) = y` if `y > 20`, else `ln(-expm1(-y)) + y`

**Array-module dispatch under JIT:** `unconstrained_to_physical` and `log_abs_det_jacobian` preserve input type via `kinuv.xp.numpy_or_jax`. A `jnp.ndarray` (including a `jax.jit` tracer) must return `jnp.ndarray`. NumPy-in / JAX-out (or the reverse) is a G3-break that NumPy-only tests will miss.

**Do not** lower `RT_BOUNDS_ARCSEC`. Production MAP box unchanged. Sampling `r_t < 0.5` is allowed on this chart (0.5 is interior of `(0, inf)`); that is not a new MAP product. G0 `r_t_at_floor` still forbids quoting inner `dV/dr`. G3 decides whether to run NUTS on real 066 under that flag.

**Hard gates (implementer decides pass/fail, records on STATUS):**

1. Official MAP `r_t = 0.5` maps to a **finite** unconstrained coordinate. Source of `kinuv.infer.chart` must not logit `RT_BOUNDS_ARCSEC`.
2. Roundtrip `physical → z → physical` recovers official Stage A θ (and a tiny interior fixture). Softplus roundtrip for `V_0 ∈ [1e-4, 1e3]` km/s using the stable branches.
3. Analytic `log|det J|` matches a numerical derivative of the coordinate transform **per active dimension**. When jax-finufft is present, also `jax.grad` of a tiny `chi2(θ(z))` through the chart (same skip rule as G1).
4. `unconstrained_to_physical` and `log_abs_det_jacobian` preserve array module: `jnp` in (including under `jax.jit`) → `jnp` out. Fail the card if JIT traces collapse to NumPy.
5. Roundtrip at official θ still satisfies `|chi2 - 168675.6| < 1` when the CANFAR npz exists. Frozen `s = 0.5136`. No refit.
6. `SAMPLER_NAME` stays `laplace_mh`. No NumPyro import. Do not wire the chart into a sampler.

**Defer (already decided; do not reopen):**

- G3 NumPyro NUTS (two Stage A runs, PA 205.2 and 25.2, 4 chains each, `R_hat` < 1.01, `ESS` > 200, label `sampler: nuts` only after autodiff).
- G4 Talts SBC, G5 PSIS-LOO on 066 vis cells.
- `DEC-HIER-SELFUNC` Phase 5. Unfreeze `i`. Add `h_z`. Warp/strip classes. TARGET subset.
- GPU.

**Reject this wave:**

- Logit of `RT_BOUNDS_ARCSEC=(0.5, 15)` (MAP at the wall → `-inf`; Jacobian diverges).
- Naive `softplus` / `inv_softplus` without the `log1p`/`expm1` branch (passes `V_0 ≈ 268`, overflows at the edges G3 will hit).
- NumPyro / `sampler: nuts`. FD-HMC labeled NUTS. emcee.
- Quoting S2 Laplace 16/50/84.
- Unfreeze `i`. Add `h_z`. GPU. 400-galaxy runner. Edit `DEC-066-TARGET`. Overwrite the official MAP tree. New `DEC-*`.

## What changed / what was checked

- G1: `predict_binned(..., xla=True)` identity `|chi2-168675.6|<1`, post-warmup 3.01 eval/s vs S2 FD 0.329. Tiny `jax.grad` vs FD. Timing: `docs/reviews/artifacts/2026-08-30-g1-jax/timing.json`.
- Official Stage A: `r_t_arcsec = 0.5` exactly, `flux=70.46`, `pa_deg=199.73`, `v0_kms=267.7`, `gas_sigma=12.05`, `dx=0.091`, `dy=0.019`. G0 fires `r_t_at_floor` and `leftover_chi2_structured`.
- S2: `sampler: laplace_mh`. Laplace SBC n=20 failed 68/95. No NUTS posterior exists.
- `RT_BOUNDS_ARCSEC` is the L-BFGS box in `kinuv.infer.seeds`, not a science prior (`kinuv.diagnostics.flags` already states this).
- Live `log_prob` is `-0.5 * chi2_and_prior` in `kinuv.infer.posterior`. There is no chart module yet.

## Rejected alternatives

- "Logit the production floor so HMC stays in the MAP box" — MAP at 0.5 sends `z → -inf`; dual-accepted G0/G1 bound: do not treat `(0.5, 15)` as a prior.
- "Logit the S1 box `(0.05, 15)`" — still a box-as-prior; G2 does not lower or replace `RT_BOUNDS_ARCSEC`.
- "Freeze `r_t` at the floor" — a G3 sampling decision, not a chart. This card ships the bijection on all eight names.
- "Start NumPyro now that JAX exists" — G3; autodiff chart first; do not label `laplace_mh` as NUTS.
- "Softplus every positive parameter" — `flux`, `gas_sigma`, `r_t` are scale parameters; `log` is the conjugate chart. `V_0` may approach 0; softplus is for that one axis.

## Residual risks

1. `log(r_t)` lets later NUTS walk below 0.5″; that does not unstick leftover-vs-velocity and does not license quoting inner slope while G0 `r_t_at_floor` fires on the official MAP.
2. Identity charts on PA/vsys do not wrap; G3 still uses two PA starts (205.2 and 25.2).
3. A correct Jacobian is not a calibrated posterior. S2 Laplace SBC already failed 68/95.
4. `log_prob` today still calls NumPy `predict_binned` unless the caller passes `xla=True`. G2 ships the chart; G3 wires NumPyro to JAX `chi2`.
5. Naive `softplus` / `inv_softplus` will pass a MAP-only test (`V_0 ≈ 268`) and still overflow or cancel at the edges G3 will hit. The `[1e-4, 1e3]` roundtrip is the gate.
6. NumPy-only chart tests can hide a host bounce that breaks `jax.jit` in G3. Type preservation under JIT is the gate.

## Execute if accepted

Boundary: chart module + unit tests only. No NumPyro. No NUTS claim.

1. Add `src/kinuv/infer/chart.py`: `physical_to_unconstrained`, `unconstrained_to_physical`, `log_abs_det_jacobian`, `log_prob_unconstrained`, plus stable `softplus` / `inv_softplus`. Array-module dispatch via `kinuv.xp`. Do not add a second SPECRESP path. Do not import NumPyro. Do not claim `sampler: nuts`.
2. Tests: `tests/test_g2_chart.py` must include (a) official MAP `r_t=0.5` finite `z` and no logit of `RT_BOUNDS_ARCSEC` in source, (b) stable `V_0` softplus roundtrip on `[1e-4, 1e3]`, (c) per-dimension analytic vs numerical Jacobian, (d) `jax.jit` type preservation, (e) official `|chi2-168675.6|<1` after roundtrip when `/arc` npz exists. `JAX_PLATFORMS=cpu`. Cache under `/scratch/kinuv-$USER` if writable, else `/tmp`.
3. Point the G2 row in `docs/diagnostics/gold-standard-roadmap.md` at the landed chart. CHANGELOG + one STATUS line. Human plot folder stays `docs/reviews/artifacts/2026-08-30-final-fit/`.
4. Commit and push `origin/dev`. Conventional subject about unconstrained chart / Jacobian, not NUTS. Do not start G3. Do not refit. Do not write `kinuv-KGAS066-uvsign-map`.

## STATUS updates required

- `next_role: board`
- `board: open`
- `last_propose:` this file
