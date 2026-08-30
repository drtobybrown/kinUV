---
role: reviewer
seat: b
date: 2026-08-30
agent: review-b
canon_generation: 4
ids:
  - DEC-066-SPECRESP
  - DEC-066-INFER
  - DEC-066-TARGET
  - DEC-066-INC
  - DEC-066-GRID
  - DEC-066-WEIGHT
  - DEC-HIER-SELFUNC
verdict: accept
severity: major
propose: docs/reviews/2026-08-30-propose-g1-jax.md
---

# Review b: G1 CPU JAX predict_binned (066 kernel)

Do not read the other seat's review file. Do not implement.

Accept because execute is G1 only: CPU JAX `predict_binned` (sky + Hann+bin + `chi2`), official MAP read-only, no G2 bijection, no G3 NUTS, no G4 SBC, no G5 PSIS-LOO, no GPU, no `i` / `h_z`, no 400-galaxy runner, `DEC-HIER-SELFUNC` stays Phase 5, and hygiene is not re-run. The card as written can still ship a numerically matching NumPy sandwich and call that G1. Identity `|chi2 - 168675.6| < 1` already holds on live NumPy `predict_binned`. Without an XLA/autodiff gate, a CANFAR path that actually runs, and a single SPECRESP operator, G3 still cannot start. Those are implementer-must-fix bounds, not a re-propose.

## Attacks / bounds

1. **Scalar `chi2` identity is not an XLA gate.** Live `predict_binned` (`src/kinuv/infer/map.py`) is NumPy `predict_vis` then `hann_then_bin` then `chi2`. That path already reproduces leftover `chi2_sum` 168675.59555208945 in `docs/reviews/artifacts/2026-08-30-final-fit/leftover_chi2.json`. `nufft2_degrid` imports jax-finufft then immediately host-bounces: `_t2_jax` returns `np.asarray(vis)`; the wrapper does `image = np.asarray(image)` and `return np.ascontiguousarray(np.asarray(vis, dtype=np.complex128).T)` (`src/kinuv/transforms/nufft.py`). `hann_then_bin` starts with `m = np.asarray(model_native)`. `chi2` accumulates via `np.asarray` (`src/kinuv/likelihood/chi2.py`). Execute names "no host bounce on the vis path used by `log_like`". There is no `log_like`. The live scalar is `chi2`; `log_prob` is `-0.5 * chi2_and_prior` in `src/kinuv/infer/posterior.py` (`SAMPLER_NAME = "laplace_mh"`). An implementer can wrap sky in JAX, call the existing host NUFFT/Hann/`chi2`, match 168675.6, and leave G3 undifferentiable. **Bound:** tiny-grid test must `jax.grad` of `chi2` w.r.t. `flux` (one scalar; Stage A names only) and match a CPU finite-difference to rtol 1e-4. `JAX_PLATFORMS=cpu`. That is not NUTS, not an 8-param Hessian, not `sampler: nuts`. If jax-finufft does not trace on CPU, record a STATUS one-liner and do not provision GPU. Do not add a NumPyro `log_like` this card. Keep NumPy as the identity reference; do not delete it.

2. **DEC-066-SPECRESP forbids a second Hann operator.** Propose: production operator stays `kinuv.response.spectral.hann_then_bin`; Gate 2 still asserts it before mock vis. That function is NumPy (kernel `[0.25, 0.5, 0.25]`, guards, weighted bin `N`). A parallel `hann_then_bin_jax` (or Hann inside `forward/model.py`) forks SPECRESP. Gate 2 (`tests/test_mock_recovery.py`) only inspects the mock import. JAX `predict_binned` could skip `spectral.hann_then_bin` and still pass Gate 2. **Bound:** JAX Hann+bin lives in `kinuv.response.spectral.hann_then_bin` (array-type dispatch is fine). Do not add a second module. Extend Gate 2 so the JAX vis path still calls that function before mock vis. `native_diagonal` already raises; do not re-purge (`331d787`). `sky_cube` includes `fourier_shift` (NumPy FFT today); the no-bounce vis path includes that shift, `los_velocity`, and PB, not only the Gaussian cube multiply.

3. **066 identity can skip on the machine that has the data.** `src/kinuv/io/vis.py` `DEFAULT_NPZ` is `/Users/thbrown/kilogas/DR1/visibilities/KILOGAS066.npz`. Several tests skip unless that Mac path exists (`tests/test_nufft.py`, `tests/test_map.py` `test_stage_a_map_on_real_hann_bin_066`). Propose: skip official identity if `/arc` npz is missing and keep a tiny numpy-vs-JAX test. On CANFAR the npz is `/arc/projects/KILOGAS/analysis/toby_sandbox/visibilities/KILOGAS066.npz` and Ico is `/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/KGAS66_Ico_K_kms-1.fits` (STATUS). A Mac-only skip plus tiny-grid identity would close G1 without ever hitting 881x95. Tiny-grid max-abs vis error does not imply `|Delta chi2| < 1` on 066 unless the tolerance is named. With `s = 0.5136098555284736`, `n_vis = 881*95 = 83695`, `chi2 ~ 1.7e5`, first-order `|Delta chi2|` is O(1e5) times max vis error if weights are order-1. **Bound:** when either Mac or CANFAR npz+Ico exist, run official Stage A vector, Hann+bin XX 881x95, same `s` as leftover json (not YAML 0.5, not `12/29`), `|chi2 - 168675.6| < 1`, and numpy-vs-JAX max abs vis `< 1e-6` on that array. Skip 066 only if both paths are absent. Always-on tiny identity: max abs vis `< 1e-6` (not a prose "small enough"). Do not treat tiny-grid error as a 066 `chi2` substitute when `/arc` is present. Load Stage A from official `kinuv-KGAS066-uvsign-map` read-only; do not refit; do not write that tree; do not regenerate `docs/reviews/artifacts/2026-08-30-final-fit/` leftover plots. Leftover npz is not in the committed plot folder (json only); leftover identity is that json (`chi2_sum`, `s`, `n_row`, `n_chan`, `pol=XX`, `pipeline_kernel=hann_then_bin`). If JAX leftover `chi2_chan` is computed, compare to NumPy in the same process; do not require a missing npz.

4. **CPU / x64 / cache are process env, not a STATUS wish.** `tests/conftest.py` does not set `JAX_PLATFORMS`, `JAX_ENABLE_X64`, or a compile cache. `nufft.py` `setdefault("JAX_ENABLE_X64", "1")` runs only when that module imports jax first. JAX default x32 on an 8e4-cell `chi2` sum can miss `|Delta chi2| < 1` (`DEC-066-WEIGHT` float64 accumulation; `tests/test_likelihood.py` `test_chi2_accumulates_float64`). Compile cache under NFS `/arc` stalls pytest. Propose residual 1 ("identity on whatever backend the session uses") is too weak for "no host bounce": python-finufft `_t2_cpu` is host by construction. **Bound:** G1 tests set `JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`, and `XDG_CACHE_HOME` or `JAX_COMPILATION_CACHE_DIR` under `/tmp` in `conftest.py` before jax import. XLA identity/`jax.grad` tests skip (do not pass) unless `nufft_backend() == "jax-finufft"`. Do not require GPU jax-finufft. Do not `canfar create --gpu`. Speed: one post-warmup `predict_binned` + `chi2` on the 066 fit array vs S2 0.329 eval/s; identity is reject-if-fail; speed miss is a STATUS one-liner only if identity holds. Do not reopen `DEC-066-GRID` 0.5 s as a G1 reject.

Carry-forward (this execute must not reopen): do not logit `RT_BOUNDS_ARCSEC=(0.5, 15)` as a prior (`seeds.py` documents an L-BFGS box; official MAP sits on 0.5 arcsec; G0 already flags `r_t_at_floor`). Do not unfreeze `i` (`inclination_rad()` default stays). Do not add `h_z`. `PARAM_NAMES` stays the eight Stage A names. Do not label FD MH as NUTS. GPU only after a 066 CPU NUTS smoke (`R_hat` < 1.01, `ESS` > 200, `sampler: nuts`). PSIS-LOO is 066 vis cells after Hann+bin, after NUTS, not galaxies. `DEC-HIER-SELFUNC` / TARGET stay deferred.

## Comments

1. **major.** Tiny-grid `jax.grad` of `chi2` w.r.t. `flux` vs FD, CPU, rtol 1e-4. Identity alone is not G1. Do not add `log_like`. Do not start NumPyro. Do not label this NUTS.

2. **major.** JAX Hann+bin stays `kinuv.response.spectral.hann_then_bin`. Gate 2 must cover the JAX vis path. `native_diagonal` still raises; no second purge. `fourier_shift` is on the sky path.

3. **major.** Official 066 identity uses CANFAR npz+Ico when present (also Mac `DEFAULT_NPZ`). `|chi2 - 168675.6| < 1`, same leftover json `s`, 881x95 XX, numpy-vs-JAX max abs vis `< 1e-6`. Tiny always-on cap is `1e-6`, not a 066 substitute. Official MAP read-only. Do not rewrite the plot folder.

4. **major.** `conftest.py`: `JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`, compile cache under `/tmp`. XLA tests skip unless backend is jax-finufft. No GPU session.

5. **major.** Do not start G2, G3, G4, G5, GPU, a 400-galaxy runner, unfreeze `i`, or add `h_z`. Do not bijection the production `r_t` floor. `DEC-HIER-SELFUNC` stays Phase 5. Commit subject is JAX identity, not NUTS.

6. **minor.** Keep NumPy `predict_binned` as the identity reference this card. Do not switch L-BFGS to autodiff jac. Do not refit.

7. **minor.** Timing JSON is eval/s after one warmup vs 0.329. Do not dump vis arrays into chat.

## Residual risks

1. jax-finufft CPU tracing / JIT of the full 066 native cube can fail or compile for minutes after a correct tiny-grid grad. Record on STATUS. Do not provision GPU to "debug" it (propose residual 1, tightened: fallback python-finufft is not G1 done).

2. First JIT can exceed 0.329 s; the speed gate is post-warmup (propose residual 2). A speed miss is not a physics stop if identity holds.

3. Official Stage A identity still needs the CANFAR npz + Wiener Ico (propose residual 3). Tiny-grid identity does not calibrate 066 leftover.

4. A correct JAX `chi2` does not calibrate leftover-vs-velocity (frozen Wiener Ico; propose residual 4). G0 `leftover_chi2_structured` still stands.

5. Hann-correlated vis cells remain. G5 PSIS-LOO on 066 vis cells after NUTS is still approximate; not this card.

6. `r_t` on the L-BFGS floor: G2 must not logit `[0.5, 15]` as a prior. This card does not run G2.

7. True NUTS CIs on real 066 stay overconfident if SB leftover is structured. No NUTS posterior exists (`laplace_mh`). Do not quote S2 Laplace 68/95.

## STATUS updates required

- `verdict` and `severity` as in the header (`accept`, `major`)
- `last_review_b`: this file
- Do not set `board: accepted` (parent tallies)
