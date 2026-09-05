Vis χ² is the fit; quote_inner_slope: false. KinMS/Barolo are image-plane comparators, not a kinUV likelihood.

# 066 S3 image-plane benchmark

Vis χ² `s * sum w |ΔV|^2` is the fit; KinMS/Barolo are image-plane comparators, not a kinUV likelihood.

Official MAP `kinuv-KGAS066-uvsign-map` was not written. Receding NUTS `sd3ckpf2` stays the 066 sampling product. Approaching search is closed (`pa25/failure.md`). Leftover gate is **SB-dominated**. That is not an s1 or c3 detection. `quote_inner_slope: false`. `intervals_calibrated: false`. Do not start G4.

## S1 restated (not a new Barolo run)

| | truth | vis Stage A | CLEAN-beam cube |
|---|---|---|---|
| r_t (arcsec) | 0.25 | 0.254 | — |
| inner slope (km/s / arcsec) | 236.7 | 237.8 | M1 94.7 |
| σ / M2 (km/s) | 8 | 7.89 | 56.1 |

3DBarolo was **not on PATH** for S1. The cube estimator was `sky_cube` → restoring beam → major-axis M1/M2.

## Receding NUTS mean (uncalibrated)

`r_t` **mean** 0.2239 arcsec (left the 0.5″ L-BFGS wall). V_0 mean 255.0 km/s. χ² 167486.8. This is not a quoted 066 inner scale. Do not form V_0/r_t.

## External tools this card

| Tool | status |
|---|---|
| 3DBarolo CLI | missing_on_path |
| KinMS (standalone under external/) | missing |

PATH miss does not license adding packages to the recovery venv. S3 still ships from S1.

Cube-fit PV/moment overlays were not produced (no Barolo/KinMS on PATH). Vis leftover D/M/R remains [`2026-09-02-kgas066-leftover-and-modes`](../2026-09-02-kgas066-leftover-and-modes/).

## Files

- `s3_table.json` — machine table
- `barolo.json` / `kinms.json` — tool receipts
## Live fitters (2026-09-05)

- 3D-Barolo: `missing_on_path`
- KinMS: `failed`
- Overlays: `live_fitters/pv_major_minor.png`, `live_fitters/moments_slices.png`
- Isolated env: `/arc/projects/KILOGAS/analysis/toby_sandbox/external_fitters`
- kinUV NUTS mean r_t = 0.224 arcsec (not a quoted inner scale)
