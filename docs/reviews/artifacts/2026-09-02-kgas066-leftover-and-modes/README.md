# KGAS066 leftover (three vis points) + approaching-PA mode

Official MAP `kinuv-KGAS066-uvsign-map` was not written. 16/50/84 are not calibrated. Do not quote inner dV/dr. Quoted \(V_c\) stays Stage A arctan. `rings_are_not_a_warp: true`. Image-plane D/M/R is a CLEAN-matched cube, **not** an inverse FT of residual vis. S1 vis vs restoring-beam is `docs/diagnostics/s1-mock.md` (3DBarolo was not on PATH).

## Look at these

| File | What |
|---|---|
| [comparison.json](comparison.json) | Identity vis leftover at three points |
| [stage-a-map/leftover_chi2.png](stage-a-map/leftover_chi2.png) | Stage A MAP leftover (sum 168675.6) |
| [nuts-mean-receding/leftover_chi2.png](nuts-mean-receding/leftover_chi2.png) | Receding NUTS mean leftover (sum 167486.8) |
| [stage-b-rings/leftover_chi2.png](stage-b-rings/leftover_chi2.png) | Stage B rings leftover via `stage_b.predict_binned` (sum 167302.2) |
| [stage-a-map/moments.png](stage-a-map/moments.png) | Stage A MAP vs 10 km/s cube |
| [nuts-mean-receding/moments.png](nuts-mean-receding/moments.png) | NUTS-mean Stage A vs 10 km/s cube |
| [stage-b-rings/moments.png](stage-b-rings/moments.png) | Stage B rings vs 10 km/s cube |
| [stage-a-map/spectra.png](stage-a-map/spectra.png) | Aperture \(\Delta v_{M-D}\) (do not fudge) |

Approaching PA 25.2 NUTS products go in [pa25/](pa25/) when the headless job finishes. That job must not write `2026-08-30-g3-nuts/` or retarget `KGAS066-latest`.

## Identity leftover (881 x 95, s=0.5136, hann_then_bin)

| Model | chi2 | leftover vs vel more structured than uv? | uv span | vel span |
|---|---|---|---|---|
| Stage A MAP | 168675.6 | true | 0.115 | 0.355 |
| Receding NUTS mean | 167486.8 | true | 0.091 | 0.306 |
| Stage B N=7 \(\lambda=0\) | 167302.2 | true | 0.093 | 0.335 |

\(\Delta\chi^2\) NUTS-mean vs MAP = \(-1189\). Gap NUTS-mean vs Stage B = \(+185\). Leftover-vs-velocity stays True at Stage B, so the leftover gate is **SB-dominated** (frozen Wiener Ico). Do not add \(s_1\)/\(c_3\). Do not unfreeze \(i\). Do not quote inner dV/dr.

Aperture \(\Delta v_{M-D}\) (optical, no velocity nudge): Stage B approaching \(+12.71\) vs receding \(+36.31\) km/s; MAP−catalog \(+24.07\) km/s. Same root cause as `docs/reviews/artifacts/2026-08-30-final-fit/vsys_shift.json`.

## DEC-067 leaves (approaching)

- Artifact dest is this folder / `pa25/`, not `2026-08-30-g3-nuts/`.
- `KGAS066-latest` stays the receding run.
