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
  - DEC-066-WEIGHT
  - DEC-066-VC
  - DEC-HIER-SELFUNC
verdict: propose
---

# G3: autodiff `chi2(θ(z))` + NumPyro NUTS (066 kernel)

## Scope

Next wave of the dual-accepted 066 kernel sequence ([`gold-standard-roadmap.md`](../diagnostics/gold-standard-roadmap.md)). G0 flags, G1 CPU JAX `predict_binned`, and G2 unconstrained chart are landed. Existing ids only. No new `DEC-*`. Official MAP `kinuv-KGAS066-uvsign-map` stays read-only. This card wires autodiff through the G2 chart **and** runs CPU NumPyro NUTS. It is not SBC, not PSIS-LOO, not GPU, not a MAP rewrite, not a 400-galaxy runner.

`DEC-HIER-SELFUNC` is defer-only. `DEC-066-TARGET` still 066. `DEC-066-INC` freeze stands: no `i`, no `h_z`.

G2 (`ee459af`) is already on `origin/dev`. Re-verified this turn: `tests/test_g2_chart.py` **17 passed**, including official `|chi2 - 168675.6| < 1` after roundtrip. Do not re-land the chart. Do not `jax.grad` the host `log_prob_unconstrained` (G2 dual-accept: that function is not autodiff).

## Architect verdict (selected path)

**Do this card in one wave:** make `U(z) = 0.5 (chi2 + shift_prior) - log|det J|` a JAX scalar of the length-8 unconstrained vector, then run NumPyro NUTS on CPU.

G1 `predict_binned(..., xla=True)` matches official `chi2` when **params are Python floats**. That is identity, not autodiff. Live host conversions that still break `jax.grad` of `chi2(θ(z))`:

- `arctan_vc`: Python `if r_t_arcsec <= 0` and `float(v0_kms)` / `float(r_t_arcsec)` (`profiles/rotation.py`).
- `los_velocity`: `float(vsys_kms)`.
- `sky_cube`: `float(i_rad)`, `float(dx_arcsec)`, `float(dy_arcsec)` on the sky offsets.
- `shift_prior`: `float(dx)` / `float(dy)`.
- `chi2_and_prior`: calls `predict_binned` **without** `xla=True`, then `float(c)`.
- `predict_binned(..., xla=True)` still unpacks a `dict` of Python floats.

G2 `log_prob_unconstrained` is host `log_prob(θ(z)) + log|J|` and must stay that convenience. G3 adds a **new** JAX potential (name it `potential_unconstrained` or similar in `kinuv.infer.chart` / a small `kinuv.infer.nuts` module). Do not JIT the host helper. Do not fold `log|J|` into `chi2`.

**Chart (do not reopen):** log flux / gas_sigma / `r_t`; stable softplus `V_0`; identity PA / vsys / `(dx, dy)`. **Do not logit** `RT_BOUNDS_ARCSEC=(0.5, 15)`. Official `r_t=0.5` is interior of `(0, inf)`. G0 `r_t_at_floor` still forbids quoting inner `dV/dr`.

**`(dx, dy)` freeze (DEC-066-SHIFT):** after MAP, freeze for NUTS only if both are consistent with 0 at <1σ. Official `dx=0.091″`, `dy=0.019″`, σ = 0.5″ are both <1σ. **Freeze them at the official MAP.** NUTS samples the other six names. Draw arrays written for `plot_posterior_corner` stay **8 columns** in `PARAM_NAMES` order with the two shift columns constant. The Gaussian shift prior is then a constant and must not be double-counted as a sampled prior.

**Two Stage A runs, not one 8-chain soup:**

| Run | Physical PA init | Other θ | Chains |
|---|---|---|---|
| receding | official MAP `199.73°` (near seed 205.2) | official MAP, `(dx, dy)` frozen | 4 |
| approaching | `25.2°` | official MAP otherwise, `(dx, dy)` frozen | 4 |

Identity PA does not wrap. Report both. Do not average them. Label the JSON `sampler: nuts` **only** if autodiff gates pass. `kinuv.infer.posterior.SAMPLER_NAME` stays `laplace_mh` (that module is still the MH path).

**NumPyro:** not installed in `kinuv-venv-recovery` today. Add optional extra `nuts = ["numpyro"]` next to `nufft` in `pyproject.toml`. Install into the recovery venv on execute. Do not add `emcee`. Do not call FD HMC NUTS.

**Scratch:** `JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`, TMP + JAX cache under `/scratch/kinuv-$USER` (else `/tmp`). Checkpoints are small JSON (last `z`, `chi2`, `nfev`, eval/s). Never loop vis I/O over `/arc`. Artifacts: `docs/reviews/artifacts/2026-08-30-g3-nuts/` (not the official MAP tree).

**Hard gates (implementer decides pass/fail, records on STATUS):**

1. **Autodiff smoke (tiny grid, jax-finufft present; same skip as G1 if missing):** `jax.grad` of `chi2(θ(z)) + shift_prior - 2 log|J|` (or equivalent `U(z)`) is finite at official-like `z`. Outputs of the jitted potential are JAX arrays (`is_jax`). Fail if any kinematic `float()` remains on the vis path used by that potential.
2. **Official identity:** `unconstrained_to_physical` of official `z`, then `predict_binned(..., xla=True)`, `|chi2 - 168675.6| < 1`, frozen `s = 0.5136098555284736`. No refit.
3. **Provenance:** product JSON has exact `sampler == "nuts"` and draws `shape[1] == 8`. `plot_posterior_corner` accepts it and still **raises** on S2 `laplace_mh`. No NumPyro import in `chart.py`. Host `log_prob_unconstrained` is not the NUTS potential.
4. **Tiny-mock NUTS:** 4 chains on a tiny vis array mix (`R_hat` < 1.01 on sampled names, `ESS` > 200) before 066. Skip 066 if this fails; do not label 066 `sampler: nuts`.
5. **066 CPU NUTS:** both PA runs, 4 chains each. Gates: `R_hat` < 1.01 and `ESS` > 200 on **sampled** names (frozen `(dx, dy)` excluded from mixing stats). `JAX_PLATFORMS=cpu`. No GPU session. Record post-warmup eval/s vs G1 3.01 and S2 FD 0.329.
6. **Physics caveats in the product, not in a later excuse:** G0 `r_t_at_floor` and `leftover_chi2_structured` still fire on the official MAP. Real-066 16/50/84 are **not** calibrated (S2 Laplace SBC already failed 68/95 on the exact mock; leftover vs velocity remains). Do not quote inner `dV/dr`. Do not quote S2 Laplace intervals.

**Defer (already decided; do not reopen):**

- G4 Talts SBC, G5 PSIS-LOO on 066 vis cells.
- `DEC-HIER-SELFUNC` Phase 5. Unfreeze `i`. Add `h_z`. Warp/strip classes. TARGET subset.
- GPU / Skaha CUDA image.

**Reject this wave:**

- Wrapping NumPy `chi2_and_prior` in NumPyro and writing `sampler: nuts`.
- `jax.grad` of G2 host `log_prob_unconstrained` as the autodiff gate.
- Logit of `RT_BOUNDS_ARCSEC=(0.5, 15)` (or of gas_sigma / flux boxes).
- FD-HMC labeled NUTS. emcee. One combined 8-chain run that mixes 199.73° and 25.2°.
- Unfreeze `i`. Add `h_z`. Sample Stage B rings.
- Overwrite `kinuv-KGAS066-uvsign-map`. New `DEC-*`.
- Quoting S2 16/50/84. Quoting inner `dV/dr` while `r_t` is at the floor.

## What changed / what was checked

- G2: `kinuv.infer.chart` 8-vector log/softplus/identity; `logaddexp` softplus (no Python `if`); official `r_t=0.5` → `z = ln 0.5`; JIT type preservation; per-axis FD Jacobian; host `log_prob_unconstrained`. Tests 17/17 this turn. Commit `ee459af`.
- G1: `predict_binned(..., xla=True)` identity `|chi2-168675.6|<1`, 3.01 eval/s vs S2 0.329. Params on that path are still host floats.
- S2: `sampler: laplace_mh`. Laplace SBC n=20 failed 68/95. Mixing gates `R_hat` < 1.01, `ESS` > 200 already existed; they do not distinguish MH from NUTS.
- NumPyro: **not** in `kinuv-venv-recovery`. `pyproject.toml` has no `numpyro` extra.
- `plot_posterior_corner` already requires `sampler == "nuts"` and an 8-column draw array.
- Official Stage A: PA=199.73, `V_0=267.7`, `r_t=0.5`, `dx=0.091`, `dy=0.019`, `chi2=168675.6`, `s=0.5136098555284736`. G0 fires `r_t_at_floor` and leftover-vs-velocity.
- Image-plane vsys offset vs the 10 km/s cube is vis-weighted MAP vs CLEAN weighting, **not** a WCS bug. G3 still samples vis `vsys` (identity chart). Do not freeze `vsys` to the catalogue 8299.563 optical.

## Rejected alternatives

- "NUTS on G2 `log_prob_unconstrained`" — host `float` `chi2`; dual-accepted as not autodiff.
- "Keep sampling `(dx, dy)`" — DEC-066-SHIFT already says freeze after MAP if both <1σ of 0; official offsets are. Sampling them is optional later, not this card's selected path.
- "Logit `[0.5, 15]` so HMC stays in the MAP box" — G2 rejected; MAP at the wall → `-inf`.
- "One NUTS run from 205.2 only" — gold-standard G3 is **two** PA starts.
- "GPU because 066 NUTS is slow" — GPU only after a CPU smoke with `sampler: nuts`, `R_hat` / `ESS`.
- "Skip tiny-mock NUTS and go straight to 066" — a broken potential will burn CPU on the 881×95 array.

## Residual risks

1. Removing `float()` from `arctan_vc` / `sky_cube` / `shift_prior` can still leave a host bounce in `fourier_shift`, NUFFT scale, or Hann weights. Gate 1 (`is_jax` + finite `jax.grad`) is the reject-if-fail, not "we deleted three floats."
2. Frozen `(dx, dy)` 8-column draws can fake mixing if `R_hat`/`ESS` are computed on constant columns. Exclude those two names from the mixing gate.
3. `log(r_t)` lets NUTS walk below 0.5″. That does not unstick leftover-vs-velocity and does not license inner `dV/dr`.
4. Real-066 NUTS intervals will look tight while leftover vs velocity is structured. Product README must say they are not calibrated.
5. First JIT of the 066 potential can be minutes. Speed notes are post-warmup. A cost miss is STATUS + continue only if autodiff + mixing gates hold; do not GPU this card.
6. NumPyro + jax 0.11.x pin risk in the recovery venv. Identity `|chi2-168675.6|<1` after the install is the compatibility gate.
7. Approaching-PA (25.2°) run may not mix toward the receding MAP; that is a result (180° mode), not a license to flip PA in the product without saying so.

## Execute if accepted

Boundary: autodiff potential + CPU NumPyro NUTS on Stage A eight-name chart (six sampled). No G4. No GPU. No MAP write.

1. Make kinematic parameters traceable on the G1 XLA vis path: `arctan_vc`, `los_velocity`, `sky_cube` offsets, `shift_prior`. Frozen `i` may stay a host scalar. Production operator remains `hann_then_bin`. `NPZ_UV_SIGN=-1`. Do not add a second SPECRESP path.
2. JAX potential `U(z)` = `-log p(θ(z)) - log|det J|` with `predict_binned(..., xla=True)` and `chi2` that returns a JAX scalar. Tests: tiny `jax.grad` finite; official `|chi2-168675.6|<1`; `is_jax` on the jitted potential. Do not JIT G2 `log_prob_unconstrained`.
3. `pyproject.toml` optional extra `nuts = ["numpyro"]`. Install in `kinuv-venv-recovery`. Identity χ² still holds after the install.
4. `kinuv.infer.nuts` (or equivalent): NumPyro model on unconstrained `z`, freeze `(dx, dy)` at MAP, two PA inits, 4 chains, CPU. Write `docs/reviews/artifacts/2026-08-30-g3-nuts/` with `sampler: nuts` JSON + `R_hat`/`ESS` + eval/s. `plot_posterior_corner` on that JSON. Checkpoints on `/scratch`, not vis dumps.
5. Tests: `tests/test_g3_nuts.py` cover gates 1–4 always (066 run skipped if npz missing). Chart source still has no `logit` / `RT_BOUNDS` / `numpyro`.
6. Point the G3 row in `gold-standard-roadmap.md` at the artifact dir. CHANGELOG + STATUS. Human plot folder stays `docs/reviews/artifacts/2026-08-30-final-fit/` unless the implementer also drops a NUTS corner PNG in the G3 artifact dir (not S2).
7. Commit and push `origin/dev`. Conventional subject about autodiff potential / NumPyro NUTS, not SBC. Do not start G4. Do not refit. Do not write `kinuv-KGAS066-uvsign-map`.

## STATUS updates required

- `next_role: board`
- `board: open`
- `last_propose:` this file
