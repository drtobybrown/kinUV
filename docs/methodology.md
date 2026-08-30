# kinUV methodology (human)

This is the science write-up. Agent process is in [`docs/reviews/BOARD.md`](reviews/BOARD.md) and [`DEC-066-AGENTS`](decisions/DEC-066-AGENTS.md). Official 066 product: **`kinuv-KGAS066-uvsign-map`**.

**Your job:** look at the latest Data | Model | Residual moments / spectra / PV and leftover `chi2` in the plot folder named on STATUS. You are not sitting gates. Agents decide those and keep going.

066 kernel sequence (autodiff likelihood, NUTS, SBC on the exact mock; not a 400-galaxy runner; `DEC-HIER-SELFUNC` stays Phase 5): [`docs/diagnostics/gold-standard-roadmap.md`](diagnostics/gold-standard-roadmap.md). G0 MAP flags live in `kinuv.diagnostics.flags`. G1 JAX `predict_binned(..., xla=True)` matches official Stage A `chi2=168675.6` on CPU (3.01 eval/s vs S2 0.329). Official 066 fires `r_t_at_floor` and leftover-vs-velocity (vs leftover-vs-uv). Do not quote inner `dV/dr` while `r_t` sits on the 0.5 arcsec L-BFGS floor. That box is not a science prior for later HMC.

## What we fit

Kinematics are inferred from **ALMA visibilities**, not from the CLEANed cube. The number we minimise is

`chi2 = s * sum w |data - model|^2`

on the XX fit array (066: 881 rows by 95 channels, `N=4`, `dv = 5.080` km/s, `s = 0.514`). Each visibility is complex, so a correct model has `E[chi2] = 2 * n_vis`. Official Stage A: `chi2 = 168676`, so `chi2_red = chi2 / (2 n_vis) ≈ 1.008`. The leftover `chi2 / n_vis ≈ 2` is that two-dof accounting, not a factor-of-two noise bug.

The spectral operator is **Hann then bin** on native channels with guards (`kinuv.response.spectral.hann_then_bin`). Hann on already-binned channels is invalid.

## Geometry we do not argue with (yet)

- Inclination frozen at the catalogue 43.86 deg. No disk thickness `h_z`.
- CASA vis Fourier sign: `NPZ_UV_SIGN = -1`.
- Ico maps with `CDELT1 < 0` are flipped to +x east before the Wiener SB template.
- Surface brightness is that frozen Wiener Ico, not a free light profile.
- Stage A rotation is an arctan (`V_0`, `r_t`). Stage B is N=7 rings with geometry frozen at the Stage A MAP; official B used `lambda = 0`.

## What 066 has shown

**Stage A MAP** (official): PA = 199.73 deg, `V_0` = 267.7 km/s, `r_t` sits on the 0.5 arcsec production floor, `gas_sigma` = 12.05 km/s, flux = 70.46 Jy. `Delta_chi2` vs a V=0 model = +35553.

**Stage B MAP**: `chi2 = 167302` (`Delta` vs A = +1373). AIC prefers B. The arctan Stage A vector is still the quoted rotation-curve product.

**S1 (inject on real 066 uv):** truth `r_t = 0.25` arcsec, `gas_sigma = 8` km/s, `V_0 = 250` km/s. Vis Stage A recovered the inject. The CLEAN-beam cube did not (M1 inner slope ~95 vs truth ~237 km/s/arcsec; M2 ~56 vs 8 km/s). That is the UV science claim: visibilities see sub-beam `dV/dr` and `gas_sigma` where the cube does not. Note: [`docs/diagnostics/s1-mock.md`](diagnostics/s1-mock.md).

**S2 (coverage):** independence Metropolis with a Laplace proposal mixed (`R_hat` ~1.00, `ESS` hundreds). Laplace SBC on 20 exact-model noise draws **failed** binomial 68/95% coverage. Do not treat Stage A Laplace intervals as calibrated. Real-066 `T_dof = chi2 / (2 n_vis) ≈ 1.008` means global `s` is fine; leftover vs velocity is still SB misspecification. Note: [`docs/diagnostics/s2-coverage.md`](diagnostics/s2-coverage.md). We do not call this sampler NUTS.

**Image-plane check:** Data | Model | Residual moments, spectra, and PV of Stage B vs the **10 km/s** cube (not 30 km/s). Figures: [`docs/reviews/artifacts/2026-08-28-stage-b-imaging/`](reviews/artifacts/2026-08-28-stage-b-imaging/). These plots are a check. Vis `chi2` is the fit.

The 27 Aug moment maps used the pre-sign PA=21.9 deg winner (`f47bc9-map`). Keep that folder as history; do not quote it as the product.

## Figures we expect after a fit

Style: [`docs/diagnostics/plotting.md`](diagnostics/plotting.md) (`kinuv.diagnostics.style`; not viridis).

1. Leftover `chi2` vs baseline and vs velocity (SB leftover vs a missing-flux bowl).
2. `chi2` slices on coupled parameters when affordable (PA–`gas_sigma`, `gas_sigma`–`i` scan, PA–`r_t`).
3. Moments / spectra / PV Data | Model | Residual.

Runner: `scripts/plot_fit_diagnostics.py`.

## What a "build" will do

The parent proposes a scope, two independent reviewers accept or reject on the board, then the parent executes every accepted stage, **chooses each gate**, and stops at a plot folder for you. KGAS066 stays the code target until you add a TARGET stub. uvkin is not the production vis MAP.

## Where to look

| Item | Path |
|---|---|
| 066 kernel sequence | [`docs/diagnostics/gold-standard-roadmap.md`](diagnostics/gold-standard-roadmap.md) |
| Official MAP | `/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS066/kinuv-KGAS066-uvsign-map/` |
| S1 artifacts | `docs/reviews/artifacts/2026-08-29-s1-mock/` |
| S2 artifacts | `docs/reviews/artifacts/2026-08-29-s2/` |
| **Your review folder** | [`docs/reviews/artifacts/2026-08-30-final-fit/`](reviews/artifacts/2026-08-30-final-fit/) |
| Moment maps | [`docs/reviews/artifacts/2026-08-30-final-fit/moments.png`](reviews/artifacts/2026-08-30-final-fit/moments.png) |
| Decisions | `docs/decisions/DEC-066-INDEX.md` |
