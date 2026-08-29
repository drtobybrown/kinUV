# 2026-08-27 Stage B vs 10 km/s imaging

**Superseded 2026-08-28.** These figures used `kinuv-KGAS066-f47bc9-map` (PA=21.9°) before `NPZ_UV_SIGN = -1`. Moment 1 and major-axis PV are inverted 180° vs the cube. Current comparison: [`../2026-08-28-stage-b-imaging/`](../2026-08-28-stage-b-imaging/).

Figures from `python scripts/plot_stage_b_vs_imaging.py`. Matching physics: [`docs/diagnostics/stage-b-vs-imaging.md`](../../../diagnostics/stage-b-vs-imaging.md). Figure style: [`docs/diagnostics/plotting.md`](../../../diagnostics/plotting.md) (`kinuv.diagnostics.style`).

- `moments.png` — M0/M1/M2, Data | Model | Residual, same 2-D spatial mask; M1 as v − vsys; east left
- `spectra.png` — mask-integrated and 1-beam apertures along the fitted PA, mJy vs optical km/s
- `pv_major.png` / `pv_minor.png` — same slit geometry; receding +; vsys dashed
- `summary.json` — paths and mask-summed mom0
