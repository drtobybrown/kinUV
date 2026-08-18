---
id: DEC-066-SHIFT
status: accepted
generation: 4
date: 2026-08-18
owner: 066-8-map
supersedes: generation 3 (shift interpolator unspecified)
---
# Phase centre (dx, dy)

**Question:** Are (dx, dy) frozen at zero, and how are they applied once the PB is in the model?

## Answer

They are MAP parameters. Gaussian prior σ = 0.5″, support ±2″. YAML `[0,0]` is a seed. Mock inject 0.3″ and require recovery. After MAP, freeze for NUTS only if both are consistent with 0 at <1σ — that freeze is a result, not an input.

## Implementation (with DEC-066-PB)

Translate the **sky-plane** template / line-profile cube by `(dx, dy)` **before** multiplying by `A(x, y)` anchored at the interferometric phase centre, then NUFFT. See DEC-066-PB stationary-PB gate.

A visibility phase ramp `V ↦ V exp(−2πi (u dx + v dy))` after PB attenuation is **forbidden**: it moves the galaxy and the primary beam together. The ramp remains a valid identity only if `A ≡ 1` (unit tests without PB).

## Sub-pixel interpolator

Bilinear (or any first-order) image shift is **forbidden**: it low-pass filters SB and aliases into `r_scale` / outer `V_c`. Required: **Fourier phase shift on the padded sky grid** (same pad as DEC-066-SB Wiener FFT), or a cubic/quintic spline with documented equivalent.

**Broadening bound:** after a 1″ shift, the azimuthally averaged SB scale length of an exponential disk with 066 `r_scale` must change by **< 0.5%**. Measure on the padded grid before PB. Test at 0.3″ (mock inject) and at the prior edge 2″.

**Wrap-around margin:** the padded grid must extend **≥ 2.0″** (the `(dx, dy)` prior box, not 1.5″ = 3σ) past the outermost finite mask pixel, in **arcsec**, on the grid that is Fourier-shifted. Do not specify this as “15 pixels at 0.1″” (that re-couples to ImageGrid). 512² at 0.4″ already exceeds this; a cropped stamp does not.
