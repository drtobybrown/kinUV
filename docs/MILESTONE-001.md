# MILESTONE-001 — canonical post-refactor baseline

Promoted by Astra directive on 2026-09-06. This milestone establishes one accepted, immutable product bundle for each canonical target after the visibility-likelihood and dark-matter decoupling refactor.

## 2026-09-07 validation addendum

The immutable target bundles remain unchanged. Cross-domain recovery stages
S0--S5 subsequently closed under `DEC-PI-S4-STANDARD-USE-BENCHMARK`. The final
S4 dossier at `results/validation/crossdomain-recovery-s4-final-20260907/`
establishes positive grouped visibility-prediction bounds and at least 10
percent better Python-native matched-family projected-velocity recovery for
both target samplings. Reviewer A accepted at `98f7730`; Reviewer B accepted at
`0442542`. The S5 seal is
`results/validation/crossdomain-recovery-s5-20260907/`. This addendum validates
the milestone architecture and comparison boundary; it does not recalibrate
the retained posterior intervals or alter the production files.

An additive S5 diagnostic at
`results/validation/crossdomain-recovery-s5-subbeam-20260907/` demonstrates
sub-beam turnover recovery using the accepted fitted realizations. The
kinUV/KinMS mean absolute `R_turn` error ratios are 0.02493 for KGAS066 and
0.27665 for KGAS007; inner `r <= BMAJ` projected-velocity RMSE ratios are
0.01438 and 0.13749. It is explicitly non-gating and leaves the S4 and target
bundles immutable.

The production hierarchy was standardized at code commit `e3a54fc`. Each
target now contains `best_model/`, `plots/`, and `benchmarks/` directly beneath
`results/production/<target>/`. Five direct-diagnostic and five matched
kinUV/KinMS figure pairs cover moments 0/1/2, major/minor PVDs, integrated
spectra, conditional posterior covariance, real rotation profiles, and
synthetic sub-beam recovery. Source-adaptive moment-map framing retains all
detected emission plus at least one beam. No fit or sampler was rerun.

| Target | Production product | Selected model | Visibility result | Posterior status |
|---|---|---|---|---|
| KGAS066 | `results/production/KGAS066/best_model/` | Stage B, seven rings | chi2 167302.365796; Stage A delta chi2 versus blank visibilities 35552.645475 | Four retained chains; max Rhat 1.00369; min bulk ESS 889; intervals uncalibrated |
| KGAS007 | `results/production/KGAS007/best_model/` | Stage A arctan | chi2 122070.763375; delta chi2 versus blank visibilities 6211.628755 | Four retained chains; max Rhat 1.00214; min bulk ESS 1093; intervals uncalibrated |

Both two-start optimizations converged to the same physical PA mode, and both saved likelihoods reproduce exactly from the serialized selected model. KGAS007 Stage B reached a 0 km/s ring bound. Its historical `max_omega=76.329` is dimensionless (`|Delta2 V|/|Delta v_chan|`); the old 20 km/s label was invalid, so this value is retained only as rejected historical evidence and does not supply the accepted cube or rotation curve.

Each product contains the resolved target configuration, environment, input hashes, append-only state history, Stage A and Stage B records, native and 10 km/s matched cubes, moments 0/1/2, integrated and aperture spectra, rotation curves, channel maps, major and minor PVDs, visibility residuals, KinMS products, posterior provenance, `METRICS.md`, and `CHECKSUMS.sha256`.

The image-cube normalized RMSE is 0.8181 for kinUV and 0.5364 for KinMS on KGAS066, and 0.6706 for kinUV and 0.4271 for KinMS on KGAS007. KinMS is optimized in the image domain, so these values are diagnostic and do not replace the visibility likelihood. The S4 paired truth suite and grouped visibility audit provide the accepted recovery evidence.

The retained NUTS draws passed mixing but failed prior coverage calibration. Their quantiles are computational posterior summaries only. No real-data inner rotation-curve slope is promoted.
