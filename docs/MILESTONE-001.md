# MILESTONE-001 — canonical post-refactor baseline

Promoted by Astra directive on 2026-09-06. This milestone establishes one accepted, immutable product bundle for each canonical target after the visibility-likelihood and dark-matter decoupling refactor.

| Target | Production product | Selected model | Visibility result | Posterior status |
|---|---|---|---|---|
| KGAS066 | `results/production/KGAS066/kinuv-KGAS066-4c1bc4-milestone1/` | Stage B, seven rings | chi2 167302.365796; Stage A delta chi2 versus V=0 35552.645475 | Four retained chains; max Rhat 1.00369; min bulk ESS 889; intervals uncalibrated |
| KGAS007 | `results/production/KGAS007/kinuv-KGAS007-e1ee1a-milestone1/` | Stage A arctan | chi2 122070.763375; delta chi2 versus V=0 6211.628755 | Four retained chains; max Rhat 1.00214; min bulk ESS 1093; intervals uncalibrated |

Both two-start optimizations converged to the same physical PA mode, and both saved likelihoods reproduce exactly from the serialized selected model. KGAS007 Stage B reached a 0 km/s ring bound and `max_omega=76.329 km/s`, above the 20 km/s campaign threshold, so it is retained as rejected evidence and does not supply the accepted cube or rotation curve.

Each product contains the resolved target configuration, environment, input hashes, append-only state history, Stage A and Stage B records, native and 10 km/s matched cubes, moments 0/1/2, integrated and aperture spectra, rotation curves, channel maps, major and minor PVDs, visibility residuals, KinMS products, posterior provenance, `METRICS.md`, and `CHECKSUMS.sha256`.

The image-cube normalized RMSE is 0.8181 for kinUV and 0.5364 for KinMS on KGAS066, and 0.6706 for kinUV and 0.4271 for KinMS on KGAS007. KinMS is optimized in the image domain, so these values are diagnostic and do not replace the visibility likelihood. The controlled mock remains the evidence for sub-beam recovery.

The retained NUTS draws passed mixing but failed prior coverage calibration. Their quantiles are computational posterior summaries only. No real-data inner rotation-curve slope is promoted.
