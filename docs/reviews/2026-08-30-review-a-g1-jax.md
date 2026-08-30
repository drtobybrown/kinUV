---
role: reviewer
seat: a
date: 2026-08-30
agent: review-a
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

# Review a: G1 CPU JAX `predict_binned` (066 kernel)

Do not read the other seat's review file. Do not implement.

Scope check: G1 only. Official MAP `kinuv-KGAS066-uvsign-map` stays read-only (no refit, no write). No GPU, no `canfar create --gpu`, no CUDA image. No NumPyro NUTS, no FD-HMC labeled NUTS, no emcee, no G2 logit of `RT_BOUNDS_ARCSEC`, no unfreeze `i`, no `h_z`, no G3/G4/G5, no `DEC-HIER-SELFUNC` execute, `DEC-066-TARGET` still 066 only. G0 leftover-vs-velocity already landed (`map_quality_flags`); do not re-run G0. `native_diagonal` already raises (`331d787`); do not re-purge. Identity `|chi2 - 168675.6| < 1` and eval/s vs S2 0.329 are the named gates. That is accept-eligible. Execute as typed can still ship a numpy sandwich that matches 168675.6 and is not XLA.

## Attacks / bounds

1. **Identity is not a no-host-bounce gate.** Architect verdict: sky + Hann+bin + `chi2` onto XLA; NUFFT is already `jax-finufft` but `nufft2_degrid` host-bounces (`np.asarray` in `_t2_jax` and again on the return). Live `predict_binned` is NumPy `predict_vis` then `hann_then_bin`. The vis path is not one `np.asarray`: `sky_cube` / `fourier_shift` (`np.fft`) are host; `hann_then_bin` starts with `np.asarray`; `chi2` is `Sigma s w |d-m|^2` with `np.asarray` and float64 accumulation (`DEC-066-WEIGHT`). Execute item 1 says "no host bounce on the vis path used by `log_like`." There is no `log_like`. `log_prob` is `-0.5 * chi2_and_prior`, which calls NumPy `predict_binned` then NumPy `chi2`. An implementer can JAX only Hann+bin+`chi2`, leave `nufft2_degrid` returning numpy, or score JAX vis with `kinuv.likelihood.chi2.chi2` (that call is a host bounce). Numbers can still hit `|chi2 - 168675.6| < 1`. python-finufft fallback cannot sit in an XLA graph; residual risk 1 only requires identity on whatever backend the session imported. **Bound:** G1 exit is one CPU `predict_binned` + `chi2` path that stays JAX arrays from sky through NUFFT through Hann+bin through `chi2`. Keep NumPy `hann_then_bin` and NumPy `chi2` as the identity reference; do not score the JAX path with those `np.asarray` functions. Record `nufft.BACKEND`. G1 is done only if `BACKEND == "jax-finufft"` on that CPU session. python-finufft identity is extra, not the wave exit. Do not require GPU jax-finufft. Tiny-grid `jax.grad` of `chi2` wrt `flux` must be finite and match a one-sided FD (step ~1e-3, relative ~1e-3). That is not NUTS, not G2, not unconstrained `r_t`. If CPU jax-finufft has no VJP, G1 is not done: STATUS one line, do not start G2/G3, do not relabel `laplace_mh` as `sampler: nuts`.

2. **`DEC-066-WEIGHT` + x64 omitted from hard gates.** Gate 1 says "same `s`" and a tiny vis error "small enough that `chi2` cannot drift by 1 on the 066 array." It does not name the number or the dtype. Plot-folder leftover (`docs/reviews/artifacts/2026-08-30-final-fit/leftover_chi2.json`): `chi2_sum = 168675.59555208945`, `s = 0.5136098555284736`, `n_row=881`, `n_chan=95`, `pol=XX`, `pipeline_kernel: hann_then_bin`. `chi2.py` accumulates in float64. `s_theory(4) = 6/13 ~ 0.462`. Using that instead of `VisData.s` scales `chi2` by ~0.90 and misses 168675.6 by ~1.7e4. YAML 0.5 and `12/29` are already forbidden (`DEC-066-WEIGHT`). JAX default is float32; `nufft.py` sets `JAX_ENABLE_X64=1` only on the jax-finufft import path. Propose tests list `JAX_PLATFORMS=cpu` and `/tmp` cache, not x64. Float32 sum over `n_vis = 83695` of `s w |r|^2` can move `chi2` by more than 1. Linear identity: `|dchi2| <= 2 s max|dV| sum(w |r|) + s max|dV|^2 sum(w)`. **Bound:** freeze `data.s` from the loaded vis object (066: 0.5136); do not recompute `empirical_s`; do not use `s_theory`. Set `JAX_ENABLE_X64=1` before importing jax in tests. Tiny numpy-vs-JAX max abs vis error (real and imag) `< 1e-8` (NUFFT `eps=1e-8` scale). Official 066 when `/arc` exists: same official Stage A vector, Hann+bin XX 881x95, `|chi2 - 168675.6| < 1`. Skip official only if the npz/Ico is missing; tiny identity still runs. Do not dump vis arrays.

3. **`DEC-066-GRID` 0.5 s/eval vs frozen identity grid.** GRID: if one likelihood eval exceeds ~0.5 s on that M1, the grid is wrong. S2 FD was 0.329 eval/s (~3 s/eval) on the production Nyquist grid. Gate 3 says beat 0.329 after one warmup JIT; speed miss is STATUS + continue if identity holds. That is the right reject-if-fail split for G1, but it does not say the grid is frozen. Coarsening `image_grid_for_vis` to chase GRID 0.5 s or 0.329 eval/s would change the FINUFFT image and miss 168675.6. **Bound:** do not change the 066 image grid, cell, FOV, or NUFFT `eps` this wave. Identity uses the same grid as the leftover eval. Speed miss: STATUS one-liner with post-warmup eval/s, `BACKEND`, x64, platforms; continue only if identity holds. Do not treat beating 0.329 as GRID compliance. Do not provision GPU to win eval/s.

4. **`log_like` / `DEC-066-INFER` is not a NUTS license.** INFER already allows NUTS after MAP vs V=0 (066 `Delta_chi2` = +35553) and vsys/PA/flux mock recover (S1). This card correctly defers G3 and rejects FD-HMC labeled NUTS. Execute still talks `log_like`. Adding a NumPyro model, logit of `(0.5, 15)` on `r_t`, or two PA NUTS runs is G2/G3. Official MAP sits on `r_t` = 0.5 arcsec; G0 already flags `r_t_at_floor`. **Bound:** G1 ships JAX `predict_binned` + JAX `chi2` identity and the tiny grad smoke in attack 1. Do not add NumPyro. Do not logit `r_t`. Do not unfreeze `i`. Do not add `h_z`. Do not write `sampler: nuts`. Commit subject is JAX identity, not "Stage A NUTS engine". `DEC-HIER-SELFUNC` in the propose `ids` list is defer-only; this card does not execute Phase 5.

## Comments

1. `major` -- Host bounce is the wave, not a comment on `np.asarray` in `nufft.py` alone. JAX Hann+bin+`chi2` glued to a numpy NUFFT/sky, or JAX vis scored by NumPy `chi2`, is not G1. Tiny-grid `jax.grad` wrt `flux`; `BACKEND == "jax-finufft"` on the CPU session that claims done.

2. `major` -- Identity uses frozen `VisData.s` (0.5136 on 066), `JAX_ENABLE_X64=1`, tiny max abs vis error `< 1e-8`, and `|chi2 - 168675.6| < 1` on official Stage A when `/arc` paths exist. Same `s`, XX, 881x95, Hann+bin. Do not use `s_theory` / YAML 0.5 / `12/29`.

3. `major` -- Do not retune `DEC-066-GRID` this wave. Freeze the production image grid. Speed vs 0.329 is post-warmup; miss is STATUS, not a grid rewrite, not GPU.

4. `minor` -- Gate 2 still asserts `kinuv.response.spectral.hann_then_bin` before mock vis; `native_diagonal` still raises. Do not fork a second SPECRESP name. Numpy `hann_then_bin` stays the production operator and the identity reference.

5. `minor` -- Timing JSON: post-warmup eval/s vs 0.329, `chi2`, `s`, `BACKEND`, `JAX_ENABLE_X64`, `JAX_PLATFORMS`. CHANGELOG + one STATUS line. Do not start G2/G3. Do not refit. Do not write `kinuv-KGAS066-uvsign-map`. Do not regenerate leftover plots; G0 already has leftover-vs-velocity (0.355 vs 0.115). ntfy / Agent Run Status are process; keep fail-open and no secrets.

## Residual risks

1. A correct JAX `chi2` does not calibrate leftover-vs-velocity (frozen Wiener Ico). Official 066 already fires `leftover_chi2_structured` and `r_t_at_floor`. Do not quote inner `dV/dr`. Do not quote S2 Laplace 68/95.

2. jax-finufft CPU VJP can fail after value identity. That blocks G3 autodiff; it does not license FD-HMC or GPU.

3. JIT compile on NFS `/arc` or first-eval vs 0.329. Cache under `/tmp`. Speed gate is post-warmup.

4. Official Stage A identity needs CANFAR npz + Wiener Ico. Tiny-grid identity is the always-on test.

5. This accept does not retag `kinuv-KGAS066-uvsign-map`. Read the official Stage A vector; do not write the tree.

## STATUS updates required

- `verdict` and `severity` as in the header (`accept`, `major`)
- `last_review_a`: this file
- Do not set `board: accepted` (parent tallies)
