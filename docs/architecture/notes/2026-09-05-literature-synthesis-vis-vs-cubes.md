# Rank

Not an ADR. If this file disagrees with a `DEC-*`, the DEC wins. Rank below `docs/architecture/STATUS.md`. Human science remains [`docs/methodology.md`](../../methodology.md). Do not paste this essay into the Field Guide.

# Vis versus cube fitters (synthesis)

## Interferometric deconvolution and CLEAN covariance

Briggs (1995) weights and a restoring beam turn independent visibility thermal noise into spatially correlated image-plane residuals. Formal pixel or ring errors on a CLEANed cube then understate uncertainty (often by a few) because neighbouring pixels share the synthesized beam. Later imaging notes (Cicone and others on ALMA cubes; Davis et al. on KinMS diagnostics) repeat the same point: image-plane χ² is not a sum of independent Gaussians. kinUV does not use that sum. The fit is `chi2 = s * sum w |ΔV|^2` on irregular `(u,v)` (066: 881×95, `s=0.5136`; 007 MAP: 956×66, `s=0.571`). CLEAN cubes and Barolo/KinMS products are comparators.

## Beam smearing in cube fitters

Di Teodoro & Fraternali (2015; 3D-Barolo) and Davis et al. (KinMS) recover `V_rot(R)` after convolving a model cube with the restoring beam. When the turnover `r_t` is ≲ θ_beam, beam convolution couples `V_0`, `i`, and `r_t`. That is a cube-likelihood degeneracy, not a visibility one. Landed S1 on real 066 uv: inject `r_t=0.25″`, vis Stage A 0.254″; CLEAN-beam M1 inner slope 94.7 vs truth 236.7 km/s/arcsec; M2 56 vs 8 km/s. 3DBarolo was not on PATH for S1. A live Barolo/KinMS row is a beam-smearing comparator, not a replacement `V_c`.

kinUV freezes 066 `i` (DEC-066-INC, `arccos(0.721)`). Cube fitters that float `i` or warp soak geometry into Δ`i` / residual PV. That is not an `s_1` / `c_3` detection (DEC-066-VC has no harmonics).

## Visibility forward modeling

Pearson (1999) and uv-plane fitting treat each visibility as an independent complex Gaussian (after the recorded weights and empirical `s`). Evaluating the model at the measured `(u,v)` avoids the restoring-beam covariance. That is why S1 recovered the injected inner scale on visibilities and the CLEAN cube did not.

Receding NUTS `sd3ckpf2` left the Stage A L-BFGS `r_t=0.5″` wall: **mean** `r_t=0.2239″`, `V_0` mean 255 km/s, χ² 167486.8 versus MAP 168675.6. `intervals_calibrated: false` (S2 Laplace SBC failed 68/95). `quote_inner_slope: false`. Do **not** treat 0.2239″ as a published 066 inner scale. Do not form `V_0/r_t`. Leftover on 066 is **SB-dominated** (vel span > uv span), not a harmonic detection.

## This card

Live Barolo/KinMS, if they run, write under `docs/reviews/artifacts/2026-09-05-kgas066-s3-image-benchmark/live_fitters/`. README first sentence: vis χ² is the fit. Official MAP `kinuv-KGAS066-uvsign-map` unchanged. Do not start G4.
