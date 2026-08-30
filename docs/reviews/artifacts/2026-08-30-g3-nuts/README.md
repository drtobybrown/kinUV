# G3 autodiff NUTS (2026-08-30)

Tiny-mock CPU NumPyro NUTS on the G2 chart. Not official 066. Not S2 Laplace.

## What this is

- Potential `U(z) = 0.5 (chi2 + shift_prior_const) - log|det J|` (same density as unconstrained `log p(z)`). Host `log_prob_unconstrained` is not autodiff.
- Six sampled names; `(dx, dy)` frozen at host MAP floats (`0.1″`, `-0.05″` on the mock). JSON draws are 8 columns in `PARAM_NAMES` order.
- Corner PNG shows the **six sampled** axes (serif, inward ticks, 16/50/84). Frozen shifts are omitted from the figure, not from the JSON.
- `sampler: nuts` after Gate 1 (finite six-axis `jax.grad(U)` vs FD) and tiny-mock mixing `R_hat < 1.01`, bulk ESS > 200.

## Mixing (sampled names only)

| name | R_hat | ESS bulk | ESS tail |
|---|---|---|---|
| flux | 1.001 | 725 | 741 |
| pa_deg | 1.000 | 825 | 886 |
| vsys_kms | 1.000 | 1090 | 801 |
| gas_sigma_kms | 1.005 | 704 | 692 |
| v0_kms | 1.003 | 847 | 790 |
| r_t_arcsec | 1.003 | 777 | 695 |

4 chains × 320 draws. Mean leapfrog steps 28.6. Tiny-grid post-warmup forward 64 eval/s (not a 066 rate).

## What this is not

- **Not calibrated.** 16/50/84 on the corner are not coverage-checked. Do not quote them as 066 intervals. Do not quote S2 Laplace 16/50/84.
- **Not 066.** Official Stage A still fires G0 `r_t_at_floor` and leftover-vs-velocity. Do not quote inner `dV/dr`.
- **No 066 `sampler: nuts`.** Projected wall `4 × (200+400) × 28.65 × 0.434 s ≈ 8.3 h` exceeds the 7200 s CPU cap. No GPU. No partial 066 JSON labeled `nuts`.
- Receding 066 init would have been MAP PA **199.73°**, not the L-BFGS seed 205.2°. Approaching 25.2° was not run.

## Files

- `tiny_mock_nuts.json` — 8-col draws `(4, 320, 8)`, `sampler: nuts`, `intervals_calibrated: false`
- `tiny_mock_corner.png` — 6D sampled corner
- `timing_projection.json` — tiny eval/s vs G1 3.01 and S2 0.329; 066 `jax.grad` 0.434 s; projection

Official MAP `kinuv-KGAS066-uvsign-map` was not written. `posterior.SAMPLER_NAME` stays `laplace_mh`.
