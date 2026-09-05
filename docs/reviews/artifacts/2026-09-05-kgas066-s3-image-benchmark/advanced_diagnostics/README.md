# Advanced mock diagnostics (S3)

Controlled arctan mock only. `quote_inner_slope: true` here; **false** on real 066. Vis χ² is the fit. Official MAP was not written. Do not start G4.

Runner: `scripts/run_s3_advanced_diagnostics.py`. Style: `docs/diagnostics/plotting.md`.

| Figure | File |
|---|---|
| A | `fig_dirty_channel_residuals.png` |
| B | `fig_radial_gradient_profiles.png` |
| C | `fig_pv_angle_fan.png` |
| D | `fig_parameter_degeneracies.png` |
| SKA yield | `fig_ska_survey_kinuv_impact.png` |

## Honesty constraints

- **Figure A, kinUV column** is a **script-local type-1 DFT adjoint** of `(V_data − V_kinUV)` on the 881×95 fit array (natural weights, no PB divide). Production NUFFT remains type-2 degrid only. This is not `numpy.fft` of the vis array and is not a CASA CLEAN residual. Units are adjoint arbitrary, not Kelvin.
- **Figure A, data column** is the beam-convolved mock cube, not a CASA CLEAN.
- **Figure D** contours are **χ² slices** (Δχ² = 2.30 / 5.99). They are **not MCMC**. kinUV inclination is frozen except the labelled i-scan panel. S2 Laplace SBC failed 68/95; intervals are not calibrated.

## Metrics (mock)

True `r_t` = 0.250″, kinUV 0.253″, KinMS 0.395″ (1.58×). Inner `dV/dr` 236.7 vs 237.1 vs 265.1 km/s/arcsec.

## SKA1 yield model

`scripts/analysis/simulate_ska_survey_yield.py` integrates the ALFALFA Schechter function over comoving \(dV_c\) for 20,000 deg². Analytic SNR and size cuts only. It is not an SKA visibility simulation and not a KGAS066 result. Ledger: `survey_yield_metrics.json`. A copy of the script is stored in this folder.
