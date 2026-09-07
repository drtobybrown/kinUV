# kinUV plotting guide

Follow this for every figure. Cosmetics live in `kinuv.diagnostics.style`. Matching physics (mask, PA, cubes, rebin, beam) lives in `kinuv.diagnostics.imaging` and `docs/diagnostics/stage-b-vs-imaging.md`. Do not copy rcParams into a script. Do not invent a new palette.

## When to plot

- After a MAP or a cube match, to show the astronomer what the model *looks like*.
- Not as a likelihood. Vis `chi2` is the fit; these figures are a check.
- Do not add dashboard chrome, sparkline insets, or a novel title on every panel.

## Standard production suite

Use `scripts/generate_production_figures.py` for accepted products. It creates
the full `best_model/`, `plots/`, and `benchmarks/` hierarchy in one clean
output root. Supply `--recovery-root` pointing to the S4-bound directory with
`s3/` and `replay/`; `--source-root` supplies only the canonical KinMS fit
record. The renderer verifies selected parameter equality, cube/input hashes,
and WCS before plotting. It retrieves the exact fit config at the replay
commit. Never attach a historical posterior from another model to the selected
checkpoint. Astronomical east increases RA and receding PA is east of north. Preview-only fit diagnostics may still use
`kinuv.diagnostics.figures` and `scripts/plot_fit_diagnostics.py`.

1. **Residual breakdown** — `plot_leftover_chi2`: `chi2` vs uv-distance and vs velocity. Flat-in-baseline + structured-in-velocity is SB misspecification, not a missing-flux bowl. Official example: `docs/reviews/artifacts/2026-08-29-s1-mock/leftover_chi2.png`.
2. **Likelihood geometry** — `plot_chi2_slices`: 2-D `chi2` on `PA-gas_sigma`, `gas_sigma-i` (scan only), `PA-r_t`. S1 example: `s1_chi2_slices.png`. Expensive on the full fit array; do not run on every survey galaxy by default.
3. **Data | Model | Residual** — `scripts/plot_stage_b_vs_imaging.py` (moments, spectra, PV). Official: `docs/reviews/artifacts/2026-08-28-stage-b-imaging/`. `plot_fit_diagnostics.py --imaging` calls this after leftover.

Preview writes go to `docs/reviews/artifacts/fit-diagnostics/` (gitignored). Canon PNGs stay under dated `docs/reviews/artifacts/YYYY-MM-DD-<slug>/`.

4. **NUTS corner (later)** — `plot_posterior_corner`: 8 Stage A names, 16/50/84 from **draws**, only if `sampler == "nuts"`. Refuses `laplace_mh` and S2 interval tables. Not part of every MAP handoff.

## Required imports

```python
from kinuv.diagnostics.style import (
    COLOUR,
    CROP_ARCSEC,
    apply_style,
    beam_ellipse,
    cbar,
    data_model_residual_grid,
    format_sky_ax,
    imshow_masked,
    intensity_cmap,
    panel_letter,
    residual_cmap,
    save_fig,
    sequential_clim,
    sky_extent_arcsec,
    symmetric_clim,
    velocity_cmap,
    vsys_line,
)

apply_style()  # once per process, before any Figure
```

Run with `MPLBACKEND=Agg` on CANFAR. matplotlib only; do not add cmcrameri/cmasher/seaborn.

## ApJ geometry contract

`kinuv.diagnostics.style` carries the required formatting internally; do not
add `apj-formatter` as a package dependency. Its vendored geometry follows the
ApJ presets of 3.5 inches for one column and 7.1 inches for two columns, with a
9-inch maximum page height. Use `apj_dimensions`, `publication_figure`, or
`publication_subplots` when creating new panels, and `save_publication` to
write paired vector PDF and 300-dpi PNG products. The rcParams are adapted from
[`drtobybrown/apj-formatter`](https://github.com/drtobybrown/apj-formatter)
under its MIT license.

## Colour roles

Use the named tokens. Never `C0` / `C1` / `tab10` for science lines.

| Token / cmap | Hex or name | Role |
|---|---|---|
| `COLOUR["data"]` | `#1A1A1A` | Observed spectrum / trace |
| `COLOUR["model"]` | `#2A6F97` | Stage B / model trace |
| `COLOUR["vsys"]` | `#737373` | Systemic-velocity dashed line |
| `COLOUR["zero"]` | `#C8C8C8` | Zero-flux or zero-offset line |
| `COLOUR["mask"]` | `#FFFFFF` | Blanked pixels (not a mapped 0) |
| `intensity_cmap()` | matplotlib `magma` | M0, M2, PV brightness |
| `velocity_cmap()` | matplotlib `coolwarm` | Moment 1, **after** subtracting vsys |
| `residual_cmap()` | matplotlib `RdBu_r` | data − model. Not the M1 cmap. |

Do not use rainbow or `seismic` maps. Sequential maps must be dark at low
intensity so a white mask is visible. `apply_style()` sets inward ticks, white
frames, a Times/STIX-compatible serif stack, 12-point axis labels, 10-point
tick labels, and Type 42 font embedding for ApJ/AAS-compatible output.

## Sky recipe (moments / channel maps)

1. One figure-level title. Column headers **Data | Model | Residual** once (top row). Row labels **M0 / M1 / M2** once (left). Use `data_model_residual_grid` so Data|Model share a colourbar sitting *between* Model and Residual (do not stack two bars on the far right — labels will collide).
2. `extent = sky_extent_arcsec(header)` then `imshow_masked(...)`.
3. Determine the crop from finite moment-0 support above 5 percent of peak,
   then add at least one BMAJ. Cap at 12 arcsec. East is left and north is up.
4. Data and model share `vmin`/`vmax` per row (`sequential_clim` for M0/M2; `symmetric_clim` for M1). One colourbar for the pair, one for residual (`cbar(..., cax=...)`). Units on every colourbar label.
5. Plot M1 as `v − vsys` (optical). Colourbar centred on 0, label `v − vsys (km/s)`.
6. Residuals: `residual_cmap()`, symmetric, 95th-percentile clip. If an M1 residual colourbar is of order \(V_{\rm rot}\) (~200 km/s), that is a 180° PA flag — do not clip it to tens of km/s to make the map look quiet.
7. Masked pixels are NaN → white via `imshow_masked`. Do not `nan_to_num` for display.
8. Restoring beam: `beam_ellipse` on M0 data (or the whole data column), lower-left. `BPA` is east of north; the helper rotates in the east/north plane.
9. Tick labels only on the left column and bottom row. Physical ticks in arcsec.
10. `save_publication(fig, path)` writes paired vector PDF and 300-dpi PNG.

## Spectrum recipe

1. Shared x (optical km/s, LSRK). Y label once per column. Panel letters `(a)`–`(d)`.
2. Data = `COLOUR["data"]` solid; model = `COLOUR["model"]` solid. One legend for the figure.
3. `vsys_line(ax, vsys, orientation="v")`. Thin zero-flux line in `COLOUR["zero"]`. Annotate each panel with the flux-weighted \(\Delta v_{\rm M-D}\) (model − data). A common offset of tens of km/s vs the CLEAN cube is vis-weighted MAP vsys, not a plotting WCS error — do not slide the model spectrum to fake overlay.
4. Approaching / receding titles **must** say they are along the *fitted* PA (include the PA value). A horn swap is a 180° PA flag, not a plotting bug.

## PV recipe

1. Shared velocity limits (`sharey`). Taller velocity axis than a squat strip (≈6″ figure height, not 4.4″).
2. Data/model share `sequential_clim`. Residual uses `residual_cmap()` + `symmetric_clim`.
3. `vsys_line(ax, vsys, orientation="h")`. X label: `Offset (arcsec; receding +)`.
4. Panel letters. One figure title that states fitted PA (major) or PA+90° (minor).

## File naming

Accepted direct diagnostics go under `results/production/<target>/plots/`.
KinMS comparisons go under `results/production/<target>/benchmarks/`.
Selected parameters, posterior checkpoints, and model cubes go under
`results/production/<target>/best_model/`. Preview products remain under dated
review-artifact directories. Do not commit `/arc` science FITS.

## Checklist before merging a figure

- [ ] `apply_style()` used; no copied rcParams
- [ ] East left, north up; source-adaptive crop contains all detected emission plus one beam
- [ ] Shared clim for data and model; residual separate and diverging
- [ ] Units on every colourbar; M1 is `v − vsys` around 0
- [ ] Masked pixels white, not zero
- [ ] Beam ellipse on M0 data
- [ ] vsys dashed grey on spectra and PV
- [ ] Approaching/receding labelled along **fitted** PA
- [ ] One legend; no C0/C1; no rainbow; no per-panel novel
- [ ] paired PDF/PNG, dpi 300, white background, no overlapping labels
