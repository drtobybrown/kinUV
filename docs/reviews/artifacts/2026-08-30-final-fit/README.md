# KGAS066 final-fit plots (your review)

Official MAP (read-only): `kinuv-KGAS066-uvsign-map`. Stage A leftover `chi2` plus Stage B vs the **10 km/s** cube. Dual-board accept 2026-08-30 (major comments applied). No Laplace CIs. Fitted PA = 199.73 deg. East left, north up. M1 is `v - vsys`.

## Look at these five

| File | What |
|---|---|
| [`leftover_chi2.png`](leftover_chi2.png) | **Stage A** vis leftover `chi2` vs uv-distance and vs velocity. Must sum to 168675.6. Flat-in-baseline + structured-in-velocity is SB leftover, not a missing-flux bowl. |
| [`moments.png`](moments.png) | **Stage B** Data \| Model \| Residual for M0, M1 (`v - vsys`), M2 |
| [`spectra.png`](spectra.png) | Mask and 1-beam apertures along fitted PA 199.73 deg |
| [`pv_major.png`](pv_major.png) | Major-axis PV, receding + |
| [`pv_minor.png`](pv_minor.png) | Minor-axis PV |

`model_on_10kms.fits` is a rematch written **here**, not under the official MAP tree.

What "works" means: PA/vsys match the cube (not the old 21.9 deg winner); M1 residual is not of order `V_rot`; leftover vs uv is not a short-baseline bowl. Spiral M0 residual is expected (frozen Wiener Ico).

Spectra are **not** expected to overlay at the CLEAN-cube centroid. MAP optical \(v_{\rm sys}\approx 8323.6\) km/s vs catalogue 8299.6 km/s. Panel \(\Delta v_{\rm M-D}\) is the flux-weighted (model − data) centroid after the radio→optical match. That offset is vis-weighted vsys vs image-plane weighting, not a Hann/`CRPIX3`/`RESTFRQ` bug. Do not fudge the model axis. Numbers: `vsys_shift.json`.
