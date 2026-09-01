# G3 NUTS (066 receding + tiny mock)

Official MAP `kinuv-KGAS066-uvsign-map` was not written. `posterior.SAMPLER_NAME` stays `laplace_mh`. 16/50/84 on the corner are not calibrated. Do not quote S2 Laplace intervals. Do not quote inner dV/dr.

Style: `docs/diagnostics/plotting.md`. MAP Stage B comparison remains `docs/reviews/artifacts/2026-08-30-final-fit/`.

## Look at these (066 NUTS mean)

| File | What |
|---|---|
| [corner.png](corner.png) | 6 sampled names from 4x600 draws. Not calibrated. Frozen dx, dy omitted. |
| [leftover_chi2.png](leftover_chi2.png) | vis leftover chi2 at the NUTS mean (sum 167487 vs MAP 168676). Still structured in velocity. |
| [moments.png](moments.png) | Data \| Model \| Residual M0, M1 (v - vsys), M2 at NUTS mean Stage A (not Stage B rings) |
| [spectra.png](spectra.png) | Mask and 1-beam apertures along NUTS-mean PA 200.05 deg |
| [pv_major.png](pv_major.png) | Major-axis PV, receding + |
| [pv_minor.png](pv_minor.png) | Minor-axis PV |

JSON: [kgas066_nuts.json](kgas066_nuts.json), [summary.json](summary.json), [nuts_mean_params.json](nuts_mean_params.json). FITS cubes stay in the run dir `kinuv_runs/KGAS066-20260831T194009Z-nuts/plots/` (not git).

## 066 mixing (sampled names)

| name | R_hat | ESS bulk | ESS tail |
|---|---|---|---|
| flux | 1.000 | 1426 | 1753 |
| pa_deg | 1.000 | 2400 | 1845 |
| vsys_kms | 0.999 | 2400 | 1608 |
| gas_sigma_kms | 1.001 | 1150 | 1285 |
| v0_kms | 1.004 | 899 | 1018 |
| r_t_arcsec | 1.004 | 889 | 985 |

Session `sd3ckpf2`. Wall 4.84 h. Mean leapfrog steps 10.0. Receding init MAP PA 199.73 deg. Approaching 25.2 deg was not run.

NUTS mean vs MAP: r_t 0.224 vs 0.5 arcsec (log-r_t chart, no logit of the 0.5-15 box); V_0 255 vs 268 km/s; chi2 167487 vs 168676 (Delta=-1189). leftover_chi2_structured still True. r_t_at_floor False on this posterior.

## Tiny mock (not 066)

- [tiny_mock_nuts.json](tiny_mock_nuts.json) — 8-col draws (4, 320, 8)
- [tiny_mock_corner.png](tiny_mock_corner.png)
- [timing_projection.json](timing_projection.json)
