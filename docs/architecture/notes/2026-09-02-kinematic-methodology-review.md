# Rank

Not an ADR. If this file disagrees with a `DEC-*`, the DEC wins. Rank below `docs/architecture/STATUS.md`. Human science remains [`docs/methodology.md`](../../methodology.md). INDEX is unchanged. Do not paste this essay into the Field Guide.

# Vis vs restoring-beam / CLEAN (landed)

## A. Interferometry

Not KinMS. These PNGs are CLEAN-matched cubes of vis models, not vis inversions.

The likelihood is `chi2 = s * sum w |ΔV|^2` on the 881×95 Hann+bin array (`s=0.5136`, `NPZ_UV_SIGN=-1`). ΔV is `data.vis − model` on irregular `(u,v)` samples. CLEAN cubes (Briggs weighting, restoring beam, image-plane covariance) are a diagnostic, not that sum. Type-1 NUFFT is not implemented; leftover Data|Model|Residual moments are CLEAN-matched cubes, not `F^{-1}{ΔV}`.

v1.3 imaging (path list only; not opened this card):

- `/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/10kms/` — Stage B vs imaging cube already plotted
- `/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/` — Ico / vis-trim

uvkin has a KinMS notebook at `/arc/projects/KILOGAS/analysis/toby_sandbox/uvkin/kinMS_kgas66_example.ipynb`; not a kinUV likelihood; not executed from kinUV.

## B. Chart and receding NUTS

G2 logs `r_t` (finite at the 0.5″ MAP); it does not logit `[0.5, 15]`. Receding NUTS `sd3ckpf2` mixed (`R_hat≤1.004`, ESS≥889). `V_0`–`r_t` correlation 0.87 is a funnel, not a license to quote inner slope. Approaching mixing is not in; this file does not invent a two-mode topology. 16/50/84 are not calibrated (S2 Laplace SBC failed 68/95; leftover structured). `intervals_calibrated: false`.

## C. Leftover vs harmonic terms

Franx (1994) and Schoenmakers (1997) name harmonic terms that are **not** in DEC-066-VC. Leftover-vs-velocity at Stage B is frozen Wiener Ico (uv span 0.093, vel span 0.335), not a harmonic detection and not `s_1`/`c_3`. Adding those terms needs a user DEC stub, not this notes file. Quoted `V_c` stays Stage A arctan. `rings_are_not_a_warp: true`.

## Landed comparison (not calibrated posteriors)

| Row | vis product | restoring-beam / CLEAN (S1) | not this table |
|---|---|---|---|
| S1 inject | vis `r_t=0.254″` vs truth `0.25″` | M1 slope 94.7 vs truth 236.7 km/s/arcsec; M2 56 vs 8 | not a cube-fitter posterior; 3DBarolo was not on PATH |
| Receding NUTS vs MAP | Δχ²=−1189; `r_t` mean 0.224″ left the 0.5″ L-BFGS wall; leftover structured | leftover D/M/R already plotted | no inner dV/dr; no 16/50/84 |
| Stage B vs NUTS-mean | vis gap +185; leftover vel span 0.335 vs uv 0.093 | same leftover PNGs | not missing circular `V_c`; gate **SB-dominated** |
| v1.3 `10kms/` / `30kms/` | vis χ² is the fit | path list; already used as CLEAN product | not opened; not a second likelihood |

Canon vis χ² (881×95, `s=0.5136`): Stage A MAP 168675.6; receding NUTS-mean 167486.8; Stage B 167302.2. `quote_inner_slope: false`. Official MAP `kinuv-KGAS066-uvsign-map` unchanged. Do not start G4.
