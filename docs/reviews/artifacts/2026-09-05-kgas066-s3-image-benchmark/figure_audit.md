# S3 figure honesty audit (2026-09-05 closeout)

Ten PNGs under this tree. Re-render only for honesty/layout defects. No KinMS rerun. Official MAP read-only. `quote_inner_slope: false` on real 066.

| Figure | File | Honesty | Action |
|---|---|---|---|
| A mock moments | `mock_controlled/mock_moments_comparison.png` | Mock-only inner-slope quote is licensed | keep |
| A mock bench | `mock_controlled/mock_benchmark.png` | Mock | keep |
| B real moments | `live_fitters/moments_comparison_real.png` | Colourbar K km/s / km/s | keep units; annotation fix via live script |
| B real PV | `live_fitters/pv_comparison_real.png` | Had “kinUV (NUTS mean)” on a MAP cube | re-label official MAP θ |
| B real V(r) | `live_fitters/rotation_curves_real.png` | Same NUTS-mean box | re-label official MAP θ |
| C dirty | `advanced_diagnostics/fig_dirty_channel_residuals.png` | Type-1 adjoint, not Kelvin | keep (do not drop caption) |
| C radial | `advanced_diagnostics/fig_radial_gradient_profiles.png` | Mock | keep |
| C PV fan | `advanced_diagnostics/fig_pv_angle_fan.png` | Mock | keep |
| D degeneracies | `advanced_diagnostics/fig_parameter_degeneracies.png` | Laplace/vis-cube χ² slices, not MCMC | keep |
| SKA | `advanced_diagnostics/fig_ska_survey_kinuv_impact.png` | Analytic volume integral | keep |

Inward ticks and `cax` colourbars come from `apply_style()` / `save_fig`. No production `sb.py` or official MAP change.
