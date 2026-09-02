# KGAS066 leftover three-way (your review)

Official MAP `kinuv-KGAS066-uvsign-map` was not written. Quoted \(V_c\) stays Stage A arctan. `rings_are_not_a_warp: true`. `quote_inner_slope: false` (leftover vs velocity still structured). 16/50/84 not calibrated. Do not quote inner `dV/dr`. Not KinMS.

Vis leftover was recomputed on Hann+bin 881×95, `s=0.5136098555284736`, `NPZ_UV_SIGN=-1`. Stage B used `stage_b.predict_binned` + official knots, not recovered arctan (`r_t_recovered=0.5`).

## Look at these

| File | What |
|---|---|
| [comparison.json](comparison.json) | Identity χ² table + leftover uv/vel spans |
| [stage-a-map/leftover_chi2.png](stage-a-map/leftover_chi2.png) | Stage A MAP leftover (sum 168675.6) |
| [nuts-mean-receding/leftover_chi2.png](nuts-mean-receding/leftover_chi2.png) | Receding NUTS-mean leftover (sum 167486.8) |
| [stage-b-rings/leftover_chi2.png](stage-b-rings/leftover_chi2.png) | Stage B rings leftover (sum 167302.2) |
| [stage-a-map/moments.png](stage-a-map/moments.png) | Stage A MAP vs 10 km/s cube (CLEAN-matched, not vis inverse) |
| [nuts-mean-receding/moments.png](nuts-mean-receding/moments.png) | NUTS-mean Stage A vs 10 km/s cube |
| [stage-b-rings/moments.png](stage-b-rings/moments.png) | Stage B rings vs 10 km/s cube |
| [dirty-residuals/README.md](dirty-residuals/README.md) | Not KinMS. S1 restating. CLEAN-matched ≠ F^{-1}{ΔV} |

## Vis χ² identity (recomputed)

| Product | χ² | leftover-vs-velocity | uv span | vel span |
|---|---|---|---|---|
| Stage A MAP | 168675.596 | true | 0.115 | 0.355 |
| Receding NUTS mean | 167486.764 | true | 0.091 | 0.306 |
| Stage B N=7 λ=0 | 167302.187 | true | 0.093 | 0.335 |

Δ NUTS-mean vs MAP = −1188.8. Gap NUTS-mean vs Stage B = **+184.6**. Stage B leftover-vs-velocity is still True, so the post-leftover gate is **SB-dominated** (frozen Wiener I_CO). Do not add \(s_1\)/\(c_3\) this card. Do not land a new official MAP.

Image-plane aperture \(\Delta v_{M-D}\) at NUTS mean: approaching +12.04 vs receding +36.85 km/s (non-rigid). Catalogue offset remains ~+24 km/s vis-weighted vsys vs CLEAN. Do not fudge velocity.

Approaching PA 25.2° NUTS is a separate headless run (`nuts-pa25`); products go under `pa25/` and must not overwrite `docs/reviews/artifacts/2026-08-30-g3-nuts/`.
