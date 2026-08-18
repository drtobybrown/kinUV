---
id: DEC-066-PB
status: accepted
generation: 3
date: 2026-08-18
owner: 066-2-template
supersedes: generation 2 (order of (dx,dy) vs A unspecified)
---
# Primary beam handling in the forward model

**Question:** How should the ALMA primary beam be handled when the morphology template is `.pbcor`?

## Answer

Re-apply primary-beam attenuation **in the image plane** before FINUFFT. Do not modify visibilities. This is **mandatory for KGAS066**, not survey-only.

`KGAS66_Ico_K_kms-1.fits` HISTORY has `pbcor = True`. The template is intrinsic (modulo the restoring beam in DEC-066-SB). Visibilities measure `F{A · I}`. Skipping `A` aliases 20%+ outer-disk flux into `V_c`.

## Decision

### 1. Do not modify visibilities

Never divide `V_data` by the primary beam in the uv-plane. That would amplify edge noise, break the weight structure, and invalidate the Gaussian likelihood.

### 2. Attenuate the model on the sky

Order is mandatory (DEC-066-SHIFT):

```
I_sky(x, y, ν)        = I_intrinsic(x − dx, y − dy, ν)   # galaxy moves; phase centre fixed
I_attenuated(x, y, ν) = A(x, y, ν) · I_sky(x, y, ν)     # A stays on the pointing
V_model(u, v, ν)      = FINUFFT_T2{ I_attenuated }
```

Do **not** apply `A` first and then a visibility phase ramp for `(dx, dy)`. That drags the PB with the galaxy. The analytic ramp is allowed only as an equivalent of a pure image translation when `A ≡ 1`.

`I_intrinsic` is the Wiener-deconvolved Ico template times the kinematic line profile (DEC-066-SB).

### 3. Parameterisation

- **Centre:** MS pointing / interferometric phase centre (`phase_dir_rad` in the npz), **not** the fitted kinematic offset `(dx, dy)`.
- **Profile (066 fallback):** ALMA 12 m Gaussian

  `FWHM_PB(ν) = 1.13 × λ / D = 1.13 × c / (ν D_ant)` with `D_ant = 12 m`.

  At 224.5 GHz: **FWHM_PB ≈ 25.9″**. Do not use `56.6″ / ν_GHz` (that is 0.25″ and is wrong).

  `A(x, y, ν) = exp(−4 ln 2 · r² / FWHM_PB(ν)²)`

  with `r² = (x − x_phase)² + (y − y_phase)²` in arcsec.

- **Preferred if present:** interpolate a CASA `.pb` image onto the model grid (Airy + blockage). Gaussian is the 066 fallback.

### 4. 066 numbers (FWHM = 25.9″)

- CO extent ~15″ vs FWHM_PB ~25.9″.
- At r = 7.5″ (disk edge): `A ≈ 0.79` (~21% suppression).
- At r = 10″: `A ≈ 0.66` (~34% suppression).
- Across the 066 line window 224.1–224.5 GHz, FWHM varies ~0.2%; one FWHM per channel is enough.

## Implementation notes

- PB multiply is `O(N_pix)` per channel, negligible vs NUFFT.
- T3 (later): multiply each quadrature node's flux by `A(x_node, y_node, ν)`.

## Validation

- Uniform disk to r = 15″: short-baseline amplitudes show the ~20% edge suppression vs a no-PB model.
- Regression: no-PB + pbcor template produces a radial residual-phase trend or a depressed outer SB; with PB that trend is gone.
- **Stationary-PB gate:** set `(dx, dy) = (1″, 1″)`. The emission centroid must move; the PB envelope must remain centred on the phase centre. A control with ramp-after-A must fail this test.
