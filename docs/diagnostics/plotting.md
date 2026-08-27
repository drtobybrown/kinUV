# kinUV plotting guide

Follow this for every figure. Cosmetics live in `kinuv.diagnostics.style`. Matching physics (mask, PA, cubes, rebin, beam) lives in `kinuv.diagnostics.imaging` and `docs/diagnostics/stage-b-vs-imaging.md`. Do not copy rcParams into a script. Do not invent a new palette.

## When to plot

- After a MAP or a cube match, to show the astronomer what the model *looks like*.
- Not as a likelihood. Vis χ² is the fit; these figures are a check.
- Do not add dashboard chrome, sparkline insets, or a novel title on every panel.

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

## Colour roles

Use the named tokens. Never `C0` / `C1` / `tab10` for science lines.

| Token / cmap | Hex or name | Role |
|---|---|---|
| `COLOUR["data"]` | `#1A1A1A` | Observed spectrum / trace |
| `COLOUR["model"]` | `#2A6F97` | Stage B / model trace |
| `COLOUR["vsys"]` | `#737373` | Systemic-velocity dashed line |
| `COLOUR["zero"]` | `#C8C8C8` | Zero-flux or zero-offset line |
| `COLOUR["mask"]` | `#FFFFFF` | Blanked pixels (not a mapped 0) |
| `intensity_cmap()` | in-repo teal–sand (`kinuv_intensity`; `mako` if present) | M0, M2, PV brightness |
| `velocity_cmap()` | matplotlib `coolwarm` | Moment 1, **after** subtracting vsys |
| `residual_cmap()` | matplotlib `RdBu_r` | data − model. Not the M1 cmap. |

Do not use `inferno`, `viridis`, or `seismic`. Sequential maps must be dark at low intensity so a white mask is visible.

## Sky recipe (moments / channel maps)

1. One figure-level title. Column headers **Data | Model | Residual** once (top row). Row labels **M0 / M1 / M2** once (left). Use `data_model_residual_grid` so Data|Model share a colourbar sitting *between* Model and Residual (do not stack two bars on the far right — labels will collide).
2. `extent = sky_extent_arcsec(header)` then `imshow_masked(...)`.
3. `format_sky_ax(ax, CROP_ARCSEC, centre=(dx, dy), xlabel=..., ylabel=...)`. Crop ~±12″ around the galaxy, not the full empty field. East left, north up (`xlim` must be descending).
4. Data and model share `vmin`/`vmax` per row (`sequential_clim` for M0/M2; `symmetric_clim` for M1). One colourbar for the pair, one for residual (`cbar(..., cax=...)`). Units on every colourbar label.
5. Plot M1 as `v − vsys` (optical). Colourbar centred on 0, label `v − vsys (km/s)`.
6. Residuals: `residual_cmap()`, symmetric, 95th-percentile clip. If an M1 residual colourbar is of order \(V_{\rm rot}\) (~200 km/s), that is a 180° PA flag — do not clip it to tens of km/s to make the map look quiet.
7. Masked pixels are NaN → white via `imshow_masked`. Do not `nan_to_num` for display.
8. Restoring beam: `beam_ellipse` on M0 data (or the whole data column), lower-left. `BPA` is east of north; the helper rotates in the east/north plane.
9. Tick labels only on the left column and bottom row. Physical ticks in arcsec.
10. `save_fig(fig, path)` — white PNG, dpi 200.

## Spectrum recipe

1. Shared x (optical km/s, LSRK). Y label once per column. Panel letters `(a)`–`(d)`.
2. Data = `COLOUR["data"]` solid; model = `COLOUR["model"]` solid. One legend for the figure.
3. `vsys_line(ax, vsys, orientation="v")`. Thin zero-flux line in `COLOUR["zero"]`.
4. Approaching / receding titles **must** say they are along the *fitted* PA (include the PA value). A horn swap is a 180° PA flag, not a plotting bug.

## PV recipe

1. Shared velocity limits (`sharey`). Taller velocity axis than a squat strip (≈6″ figure height, not 4.4″).
2. Data/model share `sequential_clim`. Residual uses `residual_cmap()` + `symmetric_clim`.
3. `vsys_line(ax, vsys, orientation="h")`. X label: `Offset (arcsec; receding +)`.
4. Panel letters. One figure title that states fitted PA (major) or PA+90° (minor).

## File naming

Write under `docs/reviews/artifacts/YYYY-MM-DD-<slug>/` with a `README.md` that links this guide and the science note. PNG names: `moments.png`, `spectra.png`, `pv_major.png`, `pv_minor.png` (or equally specific). Do not commit `/arc` science FITS.

## Checklist before merging a figure

- [ ] `apply_style()` used; no copied rcParams
- [ ] East left, north up; crop ~±12″, not the full FoV
- [ ] Shared clim for data and model; residual separate and diverging
- [ ] Units on every colourbar; M1 is `v − vsys` around 0
- [ ] Masked pixels white, not zero
- [ ] Beam ellipse on M0 data
- [ ] vsys dashed grey on spectra and PV
- [ ] Approaching/receding labelled along **fitted** PA
- [ ] One legend; no C0/C1; no rainbow; no per-panel novel
- [ ] dpi 200, white background, no overlapping labels
