# Stage B sky model vs the 10 km/s imaging cube

This is an **image-plane check** of a fit that was performed in the visibility plane. It does not enter the likelihood and does not replace Stage A/B MAP. A graduate astronomer who already knows cubes, moments, and PV diagrams should be able to reproduce the figures from the paths and steps below. Figure encoding (crop, east-left, shared clim, v−vsys, colours) is [`plotting.md`](plotting.md); do not copy rcParams into a new script.

## Why compare in the image plane at all?

Stage B maximises a visibility-plane χ² (N=7 rings, λ=0, geometry frozen at Stage A). That χ² is the science figure of merit. The plots here answer a different question: *does that model look like the published CLEAN cube?* Disagreement can mean a real kinematic mismatch, or a known difference in how the two products are made (beam, mask, primary beam, velocity convention). The procedure below is designed so that remaining residuals are mostly the former.

## Which cube, and why 10 km/s

KILOGAS v1.3 ships both 10 km/s and 30 km/s cubes:

- **Ico / vis-trim / SB template:** `/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/` (`KGAS66_Ico_K_kms-1.fits`, `KGAS66_clipped_cube.fits`). The visibility spectral window was taken from this cube. The Wiener surface-brightness template is this Ico map.
- **This comparison:** `/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/10kms/` (`KGAS66_clipped_cube.fits`, `KGAS66_mask_cube.fits`).

The fitted gas dispersion is σ ≈ 12 km/s and the visibility fit array has Δv ≈ 5.1 km/s. A 30 km/s channel is wider than both, so moment 2 and PV diagrams would be dominated by the channel width. The 10 km/s cube (Δv ≈ 10.4 km/s, beam 1.04″ × 0.95″) is the coarsest product that still resolves the line.

The Stage B sky cube (native vis channels, ~1.27 km/s, Jy/pixel, no restoring beam) lives next to the MAP JSON:

`/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS066/kinuv-KGAS066-uvsign-map/stage_b_model_cube.fits`

`sky_cube` uses +x east. `scripts/write_model_cubes.py` flips NAXIS1 so the FITS `CDELT1 < 0` WCS is sky-true. Do not flip again on read. The Ico SB template is flipped the same way on ingest (`fits_image_east_north`) because the CASA map has `CDELT1<0`. The 066 npz Fourier kernel uses `NPZ_UV_SIGN = -1`: CASA visibilities match the WCS-true CLEAN cube with a `+2πi` kernel, not a conjugated export. `kinuv-KGAS066-f47bc9-map` is the historical vis-winner at PA=21.9° before that sign.

## What the two cubes actually are

| | Stage B `sky_cube` | v1.3 10 km/s cube |
|---|---|---|
| Units | Jy/pixel | K |
| Beam | none (intrinsic, Nyquist vs uv) | CLEAN restoring beam |
| Spectral axis | radio velocity from the vis frequencies | optical, `VOPT-W2W`, LSRK |
| Primary beam | multiplied on (what the visibilities see) | science cube, treated as pb-corrected |
| Grid | 0.287″, 92² | 0.30″, 180² |

A raw overlay of the two FITS files is not a model–data comparison. The model is put on the imaging grid **before** moments, spectra, or PV are formed.

## Matching procedure (code: `kinuv.diagnostics.imaging`)

Order is mandatory:

1. **Undo the primary beam** on the model (divide by the same 12 m Gaussian used in the forward model, floored at 0.05) so both cubes are in “true sky” units. Over this disk the correction is tens of percent at the outer knot, not a few percent.
2. **Radio → optical velocity** with \(v_{\rm opt} = v_{\rm rad}/(1 - v_{\rm rad}/c)\), the inverse of the vis-loader conversion. The imaging cube is optical; the fitter is radio. Do not mix them on one axis. Model cubes are written with `CTYPE3=VRAD`, `SPECSYS=LSRK`, and `RESTFRQ` = CO(2–1). Matching does not apply a barycentric correction from `SPECSYS`; the frequencies are the visibility frequencies already trimmed to this galaxy’s cube window.
3. **Spectral average** onto the 50 imaging channels (overlap-weighted mean of native channels inside each 10.4 km/s window). Average, do not sum: both products are brightness per channel.
4. **Convolve** with the 10 km/s restoring beam (elliptical Gaussian, BPA east of north).
5. **Jy/pixel → K** with the Rayleigh–Jeans factor already used for the Ico template (`k_to_jy_per_beam`), converting smoothed Jy/pixel to Jy/beam with \(\Omega_{\rm beam}/\Omega_{\rm pix}\) first.
6. **Regrid** bilinearly onto the imaging WCS (brightness-conserving; K is surface brightness).
7. **Same spatial clip, not the 3-D voxel mask on the model.** The 10 km/s cube is already blanked outside `KGAS66_mask_cube.fits`, so data moments with that 3-D mask match the catalogue Ico. The model is a smooth line at every pixel; intersecting it with the data’s velocity mask throws away flux that is merely at a slightly different \(v_{\rm los}\) and makes moment 0 look like a flux error. Moments, spectra, and the spatial footprint of the comparison therefore use the **2-D projection** of the official mask (pixels that are in the mask in any channel). PV slits are unmasked cuts through that same centre and PA.

Moments are the usual masked sums:

- \(M_0 = \sum T\,\Delta v\) (K km/s)
- \(M_1 = \sum T v / \sum T\) (km/s)
- \(M_2 = \sqrt{\sum T (v-M_1)^2 / \sum T}\) (km/s)

PV slits are 16″ long and one beam wide, centred on the Stage A kinematic centre (phase centre + \((d_x,d_y)\)). Position angle is the fitted receding-side PA (~202° after the npz `(u,v)` sign). Positive offset is the receding side. The minor-axis cut is PA+90° and is a vsys/PA check, not a rotation-curve product.

Spectra are converted to mJy with the same K→Jy/beam factor so they share an axis with the KILOGAS `KGAS66_spectrum.csv` convention. Apertures: the 2-D mask footprint, plus 1-beam circles at the centre and ±4″ along the major axis.

## What to look at

- **Moment 0 residual:** flux scale, beam, or SB-template mismatch. A bulk offset is \((d_x,d_y)\) or PA. Do not interpret a 3-D-mask moment-0 deficit as a flux error.
- **Moment 1 residual:** rotation-curve or PA/vsys error. A dipole along the major axis is \(V(r)\); a rotation of the zero-velocity line is PA.
- **Moment 2 residual:** the model is a single \(\sigma = 11.7\) km/s plus unresolved shear in the beam. Extra width in the data is beam smearing the data does not share, or a real dispersion residual.
- **Spectra:** total flux and the approaching/receding horns. A stretch of both horns is \(V_{\rm rot}\). If the ±4″ apertures swap horns relative to the data, the image-plane receding side is 180° from the vis-fitted PA (that was the 2026-08-27 figure set; `NPZ_UV_SIGN` is the fix). An apparent rigid redshift of the model vs the CLEAN cube is **not** a WCS or Hann bug — see below.
- **Major-axis PV:** the actual \(v_{\rm los}(r)\) the rings are trying to match, after the imaging beam. Positive offset is the fitted receding PA. Data high-velocity on the negative side is the same 180° flag.
- **Minor-axis PV:** should sit near systemic. A tilt is a PA error.

None of these plots are χ². The fit was to visibilities; a pretty image-plane residual is neither necessary nor sufficient.

## Apparent +30–50 km/s redshift of Stage B vs the CLEAN cube

By-eye the mask-integrated and 1-beam profiles look like the Stage B model is translated by tens of km/s to the red. That is **not** a coordinate-pipeline error. Locked checks:

| Suspect | Result |
|---|---|
| Radio vs optical | Matching converts model `VRAD` → optical with \(v_{\rm opt}=v_{\rm rad}/(1-v_{\rm rad}/c)\). Cube is `VOPT-W2W` LSRK. Mixing conventions would be ~230 km/s, not 30–50. |
| `RESTFRQ` | Cube and vis use CO(2–1) `230.538e9` Hz. \(\Delta v = c\,\Delta f_0/f_0\) is not the offset. |
| `CRPIX3` / channel centre | `spectral_axis_kms` uses FITS 1-indexed `CRPIX3`. Native vis frequencies vs 10 km/s optical→freq disagree by ≲0.15 km/s. |
| `hann_then_bin` phase | Kernel `[0.25, 0.5, 0.25]` is centred. An impulse stays in the same \(N=4\) bin; Hann leak moves the centroid by 0.25 native channels, not a 10 km/s imaging channel. |
| Visibility \(\chi^2\) | Integer channel-roll of `predict_binned` is **minimum at roll 0**. ±1 native bin (~5 km/s after \(N=4\)) raises \(\chi^2\) by thousands. MAP \(V_{\rm sys}\) is correct **in the vis plane**. |

What remains is a **different estimator of the same galaxy**:

- Catalogue / YAML seed is optical \(V_{\rm sys}=8299.563\) km/s (`VSYS_SEED_KM_S`). Converted to radio it seeds Stage A.
- Official Stage A MAP is radio \(V_{\rm sys}\approx 8098.77\) km/s → optical \(\approx 8323.6\) km/s (**+24 km/s** vs catalogue).
- Image-plane flux-weighted centroids (model − data, this run): mask-integrated **+15.4 km/s**, centre **+35.0 km/s**, receding **+36.3 km/s**, approaching **+12.7 km/s**. Not one rigid WCS translation.

That offset is the vis-weighted MAP vs the CLEAN-cube brightness-weighted line (frozen Wiener Ico, leftover-vs-velocity, \(r_t\) at the 0.5″ floor, 1-beam vs mask weighting). **Do not apply a silent velocity fudge** to the model cube or to `hann_then_bin` — that would raise vis \(\chi^2\). A catalogue-frozen \(V_{\rm sys}\) refit would be a **new MAP tree**, not an in-place overwrite of `kinuv-KGAS066-uvsign-map`.

The 4-panel spectra annotate \(\Delta v_{\rm M-D}\) per aperture and draw MAP \(v_{\rm sys}\) (dashed) plus catalogue \(v_{\rm sys}\) (dotted). Exact overlay of centroids is not expected and is not a pass/fail on the operator.

## How to run

Display only: moment 1 is shown as \(v - v_{\rm sys}\) (optical) so the colourbar is centred on 0. Moments themselves are still computed in absolute optical km/s.

```bash
cd /arc/projects/KILOGAS/analysis/toby_sandbox/kinUV
source ~/kinuv-venv-recovery/bin/activate
export PYTHONPATH=$PWD/src MPLBACKEND=Agg
python scripts/plot_stage_b_vs_imaging.py
```

Writes under `docs/reviews/artifacts/2026-08-30-final-fit/` (not the official MAP tree):

- `model_on_10kms.fits` (K, imaging WCS; gitignored)
- `{moments,spectra,pv_major,pv_minor}.png`
- `vsys_shift.json` (WCS dump + aperture centroids)

The 2026-08-27 artifact folder is the pre-sign inverted-PA comparison against `f47bc9-map`. Do not treat it as the current model. `2026-08-28-stage-b-imaging` is the first post-sign set.

Tests that do not need `/arc` FITS: `pytest tests/test_diagnostics_imaging.py tests/test_plot_style.py tests/test_spectral.py`.
