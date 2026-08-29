# S2 hybrid coverage (Stage A)

User-licensed hybrid after S1. Sampler is `laplace_mh`, **not** autodiff NUTS. Propose: [`docs/reviews/2026-08-29-propose-s2.md`](../reviews/2026-08-29-propose-s2.md). Artifacts: [`docs/reviews/artifacts/2026-08-29-s2/`](../reviews/artifacts/2026-08-29-s2/).

ASCII only: `chi2`, `chi2_red`, `gas_sigma`, `r_t`, `R_hat`, `ESS`, `T_dof`, `T_nvis`.

## Operator and likelihood

- `pipeline_kernel = Hann+bin` (`kinuv.response.spectral.hann_then_bin`) before inject.
- `chi2 = s * sum w |d-m|^2` with XX `s` from the real npz (`s = 0.5136`).
- `n_vis = n_row * n_chan = 881 * 95 = 83695`.
- `E[chi2] = 2 * n_vis` when the model is exact (`|z|^2` has two dof).
- `T_dof = chi2 / (2 n_vis)` is the product scale on real 066.
- `T_nvis = chi2 / n_vis` is a sensitivity column (wrong dof).
- Mocks use `T = 1`. `i` frozen; no `h_z`. Stage A only.

## Mock MCMC (`s2_mock_mcmc.json`)

Independence Metropolis with a Laplace proposal at the S1 MAP (inject seed 66, truth `r_t=0.25`, `gas_sigma=8`, `V_0=250`). Four chains, 300 warmup + 1200 draw each. Wall ~100 min including SBC and the real Hessian.

| | value |
|---|---|
| `sampler` | `laplace_mh` |
| accept | 0.613 |
| eval/s | 0.329 |
| `nfev` | 6005 |
| `R_hat` `pa_deg` / `vsys_kms` / `flux` | 1.003 / 1.001 / 1.003 |
| `ESS` `pa_deg` / `vsys_kms` / `flux` | 1394 / 1688 / 1487 |
| `R_hat` `gas_sigma` / `v0_kms` / `r_t` | 1.002 / 1.001 / 1.001 |
| `ESS` `gas_sigma` / `v0_kms` / `r_t` | 1757 / 876 / 902 |

Gates `R_hat < 1.01` and `ESS > 200` passed on `pa_deg`, `vsys_kms`, `flux`. All eight Stage A parameters have `R_hat < 1.01` and `ESS > 200`.

MH 68% / 95% intervals vs inject truth:

| param | truth | median | in 68% | in 95% |
|---|---:|---:|---|---|
| `pa_deg` | 199.73 | 199.75 | yes | yes |
| `vsys_kms` | 8075.98 | 8076.02 | yes | yes |
| `flux` | 70.00 | 69.61 | yes | yes |
| `gas_sigma` | 8.00 | 7.89 | no | yes |
| `v0_kms` | 250.00 | 250.36 | no | yes |
| `r_t` | 0.250 | 0.257 | no | yes |

One noise draw can sit just outside a 68% interval. That is not a coverage test; SBC is.

## Laplace SBC (`s2_sbc.json`)

20 XX-noise draws. Each: L-BFGS from **truth** (`maxiter` 20, mock `r_t` box) plus FD-Hessian Laplace 68/95% intervals. `T = 1`.

| param | rate68 | rate95 |
|---|---:|---:|
| `pa_deg` | 0.30 | 0.60 |
| `vsys_kms` | 0.45 | 0.55 |
| `flux` | 0.20 | 0.40 |
| `gas_sigma` | 0.25 | 0.30 |
| `v0_kms` | 0.10 | 0.35 |
| `r_t` | 0.10 | 0.55 |

Pass rule was binomial consistency with 0.68 / 0.95, not a point match. n=20 Clopper-Pearson 95% intervals: 6/20 is about 0.12-0.54 (excludes 0.68); 2/20 is about 0.01-0.32; 12/20 is about 0.36-0.81 (excludes 0.95). **Laplace SBC fails** on every reported parameter.

Two caveats, neither rescues the rates:

1. A few L-BFGS starts with `maxiter` 20 left the truth neighborhood (flux 46 and 58 Jy; one `gas_sigma` 11.8). Those are optimizer misses, not just thin Gaussians.
2. Draws that stayed near truth still miss 68% on `v0_kms` and `r_t` often. The FD Laplace width is too small for this likelihood even when the model is exact.

Do not treat Stage A Laplace CIs as calibrated 68/95% intervals. The MH chain on seed 66 mixed (`R_hat`, `ESS` ok) but uses the same Laplace proposal; it is not a substitute for a well-calibrated posterior width.

## Real-066 CI table (`s2_real_ci_table.md`)

Official Stage A MAP Hessian (`kinuv-KGAS066-uvsign-map`). `chi2_map = 168675.6`.

- `T_dof = 1.0077` (`chi2 / (2 n_vis)`). Width ratio vs unscaled = 1.004.
- `T_nvis = 2.0154` (`chi2 / n_vis`). Width ratio vs unscaled = 1.420 (`sqrt(T_nvis)`).

`T_dof ~ 1` means global XX `s` is fine. The leftover `chi2 / n_vis ~ 2` is the two-dof `|z|^2` accounting, not a factor-of-two noise bug.

| param | MAP | unscaled 68% | T_dof 68% | T_nvis 68% |
|---|---:|---|---|---|
| `pa_deg` | 199.730 | [199.636, 199.823] | [199.636, 199.824] | [199.597, 199.862] |
| `vsys_kms` | 8098.773 | [8098.614, 8098.932] | [8098.613, 8098.933] | [8098.547, 8098.999] |
| `flux` | 70.459 | [70.009, 70.910] | [70.008, 70.911] | [69.820, 71.099] |
| `gas_sigma` | 12.050 | [11.878, 12.222] | [11.877, 12.223] | [11.806, 12.294] |
| `v0_kms` | 267.67 | [267.11, 268.23] | [267.11, 268.23] | [266.87, 268.47] |
| `r_t` | 0.5000 | [0.4881, 0.5119] | [0.4880, 0.5120] | [0.4831, 0.5169] |

`r_t` MAP is the production floor 0.5 arcsec. A Gaussian interval around a bound is not an interior Laplace CI.

Mock coverage tested the exact-kernel case only and already failed. On real 066, structured leftover vs velocity (SB misspecification, frozen Wiener Ico) can still over-narrow intervals **after** `T_dof`. `T_nvis` is a ~1.42 stretch of the std; it does not model that leftover.

## Out of scope

NumPyro, Stage B, XX+YY, calling this NUTS. No real-066 MCMC this wave (mock `R_hat` / `ESS` already used the budget).
