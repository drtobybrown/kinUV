# Changelog

Durable results, rejected alternatives, benchmark values, and retained artifact paths are recorded in [`docs/PRODUCTION_RECORD.md`](docs/PRODUCTION_RECORD.md). Git history and the compressed legacy-doc bundle preserve the original review exchanges.

## 2026-09-06 — production closeout and documentation cleanup

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
