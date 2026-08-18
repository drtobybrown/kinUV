---
id: DEC-066-SB
status: accepted
generation: 2
date: 2026-08-18
owner: 066-2-template
supersedes: generation 1 (K = σ²/S_peak had wrong units; Approach B double-beamed)
---
# Surface brightness template: Ico Wiener deconvolution

**Question:** How to extract the intrinsic SB template from the CLEAN-restored Ico map?

## Answer

Wiener deconvolution of the **restoring** Gaussian in the Fourier domain, with a dimensionless NSR and a hard taper against noise amplification. Then NUFFT. Never Fourier-transform the restored map as if it were intrinsic SB.

Source: `KGAS66_Ico_K_kms-1.fits`. HISTORY: `tclean` `restoration=True`, `restoringbeam=common`, `pbcor=True`, `width=30 km/s`. Restoring beam ≈ 1.30″ × 1.18″, BPA −18.3°. Finite mask: 1709 pixels.

## Protocol

1. **Unit conversion.** Ico (K km/s) → Jy/beam km/s using the Rayleigh–Jeans factor at the **observed** frequency of the cube (~224.3 GHz for 066), not rest 230.538 GHz (`S ∝ ν²`; rest vs observed is ~5.6%):

   `S_Jy/beam = T_K × (2 k ν_obs² / c²) × π θ_maj θ_min / (4 ln 2)`

   with `θ_maj`, `θ_min` from the FITS header (radians). Recompute in the unit test; do not hardcode 0.063 Jy/K. Free flux absorbs a global scale for the 066 MAP; catalogue-flux comparisons will not.

2. **Wiener deconvolution.** Normalise the restoring-beam FT so `|B̃(0)| = 1`. Then

   `Ĩ_intrinsic(u,v) = Ĩ_restored(u,v) × B̃*(u,v) / (|B̃(u,v)|² + K)`

   `B̃` is the analytic Gaussian from header `BMAJ`, `BMIN`, `BPA`. Undo only the restoring beam, never the dirty/synthesized beam.

3. **Wiener constant (dimensionless).** `K = (σ_empty / I_peak)²` where `σ_empty` is rms in emission-free (empty-corner) pixels of the Ico map and `I_peak` is the map peak. Generation 1's `K = σ² / S_peak` had units of Jy and is invalid inside `|B|² + K`.

   Restored Ico = `model ∗ B_rest + residual`. Wiener/`B_rest` over-amplifies dirty-beam residuals; K and the taper exist for that. A scalar K is adequate because frequencies with `|B̃|` small are zeroed anyway.

4. **Taper.** Where `|B̃(u,v)| < 0.05`, set deconvolved amplitudes to zero (gain cap 20× if K=0). Conservative: the template must not claim spatial frequencies the restored image does not constrain.

5. **Inverse FT** to the sky grid. Mask = the finite pixels of the Ico map. Do not inpaint CLEAN residuals.

6. **Normalisation.** Shape only; flux is a free parameter. Normalise the template to unit integral.

Primary-beam re-attenuation of this intrinsic template is **DEC-066-PB**, not this step.

## Alternatives (rejected for 066)

**Multiply model visibilities by `B̃`.** On a restored template this implements `FT(I_true) B̃²`. A scalar free flux cannot absorb a uv-dependent taper. Do not implement this.

**CLEAN `.model` components.** If a `.model` cube appears, it is already intrinsic and needs no Wiener. 066 products today are restored Ico only, so Wiener stays.

## Robustness

Exponential-SB run (free `r_scale`, free flux) is required. If Ico-template `V_c(r)` and exponential `V_c(r)` differ by more than posterior 1σ, the template is suspect and free-SB becomes the default.

## Unit tests

1. **Gaussian fake:** known SB → convolve with restoring beam → Wiener. Long-baseline FT must match unconvolved truth inside the taper, not the restored map.
2. **Exponential disk fake:** Re=7.4″, 066-like mask. Recovered scale length unbiased; no ringing at the taper/mask boundary.
3. **Correlated noise (load-bearing):** exponential into dirty-beam noise → CLEAN-like Ico → deconvolve. Recovered scale length still unbiased. Optional to skip this test is not allowed.
