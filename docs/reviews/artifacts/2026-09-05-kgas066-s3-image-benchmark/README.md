# 066 S3 image-plane benchmark

Vis χ² is the fit; quote_inner_slope: false. KinMS is an image-plane comparator, not a kinUV likelihood.

Official MAP `kinuv-KGAS066-uvsign-map` was not written. Two-way benchmark: **kinUV (vis)** vs **KinMS (cube fit)**. Leftover gate is **SB-dominated** on 066. `intervals_calibrated: false`. Real-066 `quote_inner_slope: false`. Figure D is vis/cube χ² slices, not MCMC/NUTS. Real moment colourbars are K km/s and km/s. SKA panel is an analytic volume integral. Figure A kinUV column is a script-local type-1 DFT adjoint (not Kelvin). Live annotations use official MAP θ, not NUTS mean. Do not start G4.

## Live fitters

- KinMS best (inClouds): `ran`
- KinMS legacy (Gaussian SB): `ran`
- Mock controlled: `ran`
- Init: kinUV Stage A catalogue seeds only (no kinUV posterior)
- Figures: `live_fitters/pv_comparison_real.png`, `moments_comparison_real.png`, `rotation_curves_real.png`, `mock_controlled/mock_benchmark.png`, `mock_controlled/mock_moments_comparison.png`, `advanced_diagnostics/`
- Geometry note: `docs/architecture/notes/2026-09-05-kinms-investigation-report.md` (sbProf; transpose `(2,1,0)`)
- Receipt: `live_fitters/kinms_best.json`

## S1 restated

| | truth | vis Stage A | CLEAN-beam cube |
|---|---|---|---|
| r_t (arcsec) | 0.25 | 0.254 | — |
| inner slope (km/s / arcsec) | 236.7 | 237.8 | M1 94.7 |
| σ / M2 (km/s) | 8 | 7.89 | 56.1 |
