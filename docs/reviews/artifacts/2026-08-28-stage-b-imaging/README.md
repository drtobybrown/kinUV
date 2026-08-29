# 2026-08-28 Stage B vs 10 km/s imaging (CASA uv-sign + sky-east Ico)

Figures from `python scripts/plot_stage_b_vs_imaging.py`. Matching physics: [`docs/diagnostics/stage-b-vs-imaging.md`](../../../diagnostics/stage-b-vs-imaging.md). Figure style: [`docs/diagnostics/plotting.md`](../../../diagnostics/plotting.md). MAP: `kinuv-KGAS066-uvsign-map` (PA=199.73°; Ico `CDELT1<0` flipped to +x east). The 2026-08-27 folder is the pre-sign inverted-PA comparison against `f47bc9-map`.

- `moments.png` — M0/M1/M2, Data | Model | Residual, same 2-D spatial mask; M1 as v − vsys; east left
- `spectra.png` — mask-integrated and 1-beam apertures along the fitted PA, mJy vs optical km/s
- `pv_major.png` / `pv_minor.png` — same slit geometry; receding +; vsys dashed
- `summary.json` — paths and mask-summed mom0
