# Changelog

Durable results, rejected alternatives, benchmark values, and retained artifact paths are recorded in [`docs/PRODUCTION_RECORD.md`](docs/PRODUCTION_RECORD.md). Git history and the compressed legacy-doc bundle preserve the original review exchanges.

## 2026-09-07 - cross-domain recovery S0-S5 closure

- Added a zero-dependency ApJ plotting contract with vendored 3.5/7.1-inch
  dimensions, Times/STIX typography, Type 42 fonts, inward ticks, and 300 dpi
  export helpers.
- Generated the four required publication figure pairs for KGAS066 and
  KGAS007 under their accepted production bundles. Recomputed restored-cube
  centroids with the repaired celestial-WCS coordinate contract, added hashed
  per-target provenance, and received Astra science-completeness acceptance.
- Closed S4 after kinUV beat the information-advantaged canonical KinMS
  comparator on every held-out visibility fold for KGAS066 and KGAS007.
- Added a CASA-free paired truth suite using Python-native analytic cubes and
  visibilities; kinUV/KinMS projected-velocity RMSE ratios are 0.01180 and
  0.04569 against the 0.90 gate.
- Repaired signed-coordinate, primary-beam-frequency, and spectral accounting;
  retained cube, moment, spectrum, channel-map, PVD, and flux products as
  supporting diagnostics.
- Added smooth positive emissivity, fold-safe visibility loading, authenticated
  external-fit checkpoints, complete source hashes, and behavioral stale-state
  rejection tests.
- Removed the abandoned CASA reimaging environments and logs. S4 validation no
  longer requires training-fold `tclean` products.
- Received independent Reviewer A and Reviewer B acceptance, verified the
  50-file S4 manifest, and sealed S5 without launching a posterior campaign.
- Added a non-blocking, checksum-bound S5 sub-beam diagnostic from the accepted
  synthetic fits. It reports direct turnover-radius error and inner
  `r <= BMAJ` projected-velocity RMSE, with per-target and combined figures,
  without reopening S4.

## 2026-09-06 — production closeout and documentation cleanup

- Replaced KinMS nearest-cell/Gauss-Hermite rendering with analytic Gaussian channel integration, cubic B-spline deposition, explicit Jy/Jy km/s accounting, and proportional validation. Composite radial quadrature subsequently closed the remaining refinement axis, and S1 advanced to the now-complete S2--S5 cascade.
- Streamlined the Field Guide so localized mathematical corrections use focused tests and empirical target closure without mandatory ADR, dossier, or proposal-review ceremony; full review remains for scientific-contract changes and production promotion.
- Registered Astra's continuum-adapter/S2 architecture at `e7a1e71`; independent Reviewer A/B verdicts both require executable metric, refinement, unit, campaign-configuration, covariance, and optimizer details before implementation is licensed.
- Closed S0 scientific accounting at `fb4a145`, distinguishing blank from fitted non-rotating emission and blocking universal use of the historical dimensionless omega threshold.
- Added the S1 intrinsic KinMS adapter and shared PB/NUFFT/Hann-bin operator, then repaired the independently discovered PA, fail-closed metadata, atomic-publication, input-lock, and provenance defects at `0c95240`.
- Sealed the revised S1 dossier as failed evidence after the newly complete per-axis target matrix exposed unconverged KinMS cloud deposition; no S2 work or scientific promotion followed.
- Promoted MILESTONE-001 as exactly one immutable production bundle for KGAS066 and KGAS007, each containing fresh visibility MAP fits, cubes, moments, spectra, PVDs, rotation curves, visibility residuals, a KinMS comparison, checksums, and retained mixed posterior draws.
- Fixed target inclination propagation through Stage B and model-cube export; KGAS007 now uses 28.9 degrees throughout its forward model and diagnostics.
- Added versioned target configurations and a clean-checkout production runner with two PA starts, likelihood-identity verification, environment and input hashes, bounded output, and immutable promotion receipts.
- Added a Stage B adequacy gate. KGAS066 selects its stable ring fit at chi2 167302.366; KGAS007 falls back to Stage A at chi2 122070.763 because its nominal Stage B solution reaches the zero-speed bound and exceeds the oscillation threshold.
- Corrected KGAS066's frozen inclination to the exact catalogue relation `acos(0.721) = 43.862896 deg` and reran its complete product after an identity test detected a 0.003 deg configuration discrepancy.
- Archived the superseded production products, two rejected milestone attempts, and the duplicate standalone KinMS benchmark after checksum verification.

- Decoupled cosmological scale conversion from the visibility hot path, added a generic kinematic velocity-profile interface, and added automated KGAS066/KGAS007 downstream KinMS comparison products.
- Added source-level guards that prohibit dark-matter, baryonic/halo, cosmology, and physical-scale dependencies in `kinuv.forward`, `kinuv.likelihood`, and `kinuv.infer`.
- Replaced the target-coupled field guide with a data-agnostic production standard covering Consultant/Registrar/Implementer authority, independent science and software reviews, precommitted scientific gates, immutable promotion, and scratch/durable storage tiers.
- Added target and campaign configuration templates and blocked new production campaigns until remaining target metadata and site paths are migrated from Python entry points.
- Removed the last legacy runtime coupling: KGAS007 metadata now lives in `kinuv.targets`, and canonical plus historical visibility tables share `kinuv.io.vis.load_visibility_table`.
- Established `ms2kinuv` as the separately installed CASA Measurement Set extractor and documented its versioned NPZ boundary.
- Archived the final uvkin and uvfit Git snapshots; neither repository remains in the active workspace or package path.
- Merged four KGAS007 CPU NUTS shards, including chain-2 relaunch `faoik171`; the product passed mixing with max Rhat 1.00214 and min ESS 1093.
- Confirmed KGAS066 receding CPU NUTS `sd3ckpf2` as the only accepted KGAS066 posterior; the PA 25.2 deg branch is terminated.
- Retained the official KGAS066 Stage A MAP and Stage B N=7 products unchanged.
- Consolidated closed proposals, reviews, architecture notes, superseded figures, and rejected experiments into the production record and `../archives/kinuv_docs_legacy_20260906.tar.gz`.
- Moved tests and the Stage B plotting default from the superseded 2026-08-30 final-fit bundle to the current 2026-09-02 comparison artifacts.
- Curated all legacy run artifacts into `results/production` and `results/archive`, verified merged NUTS shards and archive checksums, removed `kinuv_runs`, and rendered the missing KGAS007 production diagnostics.

## 2026-09-05 — second target and image-plane benchmark

- Added KGAS007 Stage A MAP and licensed KGAS007 CPU NUTS.
- Completed the KGAS066 kinUV-versus-KinMS real and controlled-mock benchmark.
- Corrected the KinMS geometry wrapper: face-on disk inputs, KinMS-owned projection, cube transpose `(2, 1, 0)`, and systemic `vOffset`.
- Kept the 30 km/s Ico surface-brightness template after the conditional 10-versus-30 km/s comparison.
- Rejected the optional m=2 surface-brightness model because it remained substantially worse than the production two-dimensional Ico template.

## 2026-09-02 to 2026-09-03 — posterior and compute path

- Completed KGAS066 receding CPU NUTS with max Rhat 1.004 and min ESS 889.
- Rejected GPU NUTS after the H100 MIG path ran about 5.5 times slower than CPU on the official likelihood.
- Established four independent one-chain CPU jobs plus host merge as the scalable headless path.
- Diagnosed the approaching PA branch as an unmixed, nonphysical search and closed it after recovery returned to the official receding solution.

## 2026-08-28 to 2026-08-30 — validated visibility kernel

- Corrected the uv-sign convention and established the official KGAS066 MAP at PA 199.73 deg.
- Completed Stage B N=7, lambda=0 with a chi2 improvement of 1373.4 over Stage A.
- Landed S1 real-uv injection recovery, S2 coverage diagnostics, the JAX likelihood, the unconstrained chart, and NumPyro NUTS.
- Recorded the S2 calibration failure; credible intervals remain uncalibrated.
