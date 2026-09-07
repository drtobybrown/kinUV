---
role: reviewer
seat: science-numerics
phase: implementation
date: 2026-09-07
reviewer: review-a-s4-recovery
canon_generation: 27
campaign_id: crossdomain-recovery-s4-scientific-recovery
proposal: docs/decisions/DEC-PI-S4-STANDARD-USE-BENCHMARK.md
reviewed_commit: f0d3067562e3b65b07a6a748ad869ea6656f9c0e
reviewed_range: 3462efd05481aa8ae4c51d8e21c5cc5badf505c2..17dc9d7b576b2fede281a935d3b69a23e6d25fa8
verdict: accept
---
# Independent code review A: cross-domain S4 scientific recovery

I independently audited the S4 recovery implementation, the binding
`DEC-PI-S4-STANDARD-USE-BENCHMARK` contract, the review packet, the exact final
dossier at `results/validation/crossdomain-recovery-s4-final-20260907/`, the
grouped evidence at
`results/validation/crossdomain-recovery-s4-remediation-20260907-r2/grouped/`,
and the supporting repaired replay. I accept S4 under the Project-PI's
standard-use comparison. This verdict is limited to held-out visibility
prediction and matched-family projected-speed recovery; it does not convert
restored-cube diagnostics into truth or establish performance outside the
tested thin, axisymmetric arctan family.

## Findings

1. **Standard-use comparison — pass.** Stock KinMS remains frozen at its
   canonical full-cube fit and is rendered through the validated intrinsic
   continuum adapter onto the exact held-out visibility rows. kinUV is refit
   on the complementary training rows. Both predictions are scored on the same
   native frequencies, flags, weights, rows, Hann/bin response, Fourier
   operator, and selected C1 covariance. KinMS has seen the complete
   channelized science cube before scoring, so its fitted kinematics include
   the held-out observation. kinUV retains its normal frozen Ico brightness
   auxiliary, so the fold result is a conditional standard-use prediction
   test rather than an end-to-end re-estimation of every data-derived input.
   This is the comparison explicitly authorized by the PI.

2. **Grouped real-data advantage — pass.** Every one of five disjoint test
   folds favors kinUV for both targets. Per-real-component gains range from
   `0.02677` to `0.05655` for KGAS066 and `0.001597` to `0.006958` for
   KGAS007. Aggregate gains are `0.04265074` and `0.00456083`; the recorded
   20,000-draw fold-bootstrap lower 95% bounds are respectively
   `+0.03178592` and `+0.00290304`. All ten training fits are successful,
   interior, and satisfy the PI gradient gate; maximum normalized gradients
   are `2.80e-4` and `1.40e-4`. Because only five fold aggregates are
   resampled and their training sets overlap, the quoted interval is a
   conditional empirical uncertainty summary, not a general sampling-theory
   confidence claim. Its positive sign is nevertheless not fragile: all ten
   individual fold deltas are positive.

3. **Matched-family Python truth — pass.** The analytic cube and Fourier
   visibilities use one normalized projected exponential disk, the same thin
   arctan rotation law, declared geometry and dispersion, target-native
   sampling, and fixed seeds. kinUV receives noisy complex visibilities;
   stock KinMS receives a beam-restored noisy cube and the noise-free truth M0
   profile, which gives the image-plane comparator favorable brightness
   information. No CASA product enters the synthetic metrics. All six KinMS
   fits converge with 100,000-cloud polish, have successful extent checks, and
   report no renderer error. All six kinUV fits succeed with normalized
   gradients between `2.58e-5` and `8.92e-5`.

4. **Projected-speed RMSE — reproduced.** The metric evaluates
   `u(r)=v_c(r) sin(i)` at 24 common radii from `0.25` to `3.5 BMAJ`, weighted
   by the declared radial emissivity support. I independently recomputed every
   realization RMSE from the saved truth and fitted parameters. For KGAS066,
   RMS RMSE is `0.08836 km/s` for kinUV and `7.48616 km/s` for KinMS, ratio
   `0.01180`. For KGAS007 it is `0.39607` and `8.66773 km/s`, ratio `0.04569`.
   Each is far below the required `0.90`; every individual realization also
   passes. The use of projected speed correctly avoids promoting an intrinsic
   circular-speed claim from the unconstrained inclination decomposition.

5. **KGAS066 non-regression — pass.** In the grouped record, new kinUV has
   summed held-out chi-square `625913.97` versus `625703.89` for the frozen
   predecessor, a relative degradation of only `0.0336%`. A direct
   fold-bootstrap gives an upper 97.5th percentile of approximately `0.0621%`,
   far inside the previously registered 2% guard. This comparison is also
   conservative because the predecessor is a frozen full-data fit whereas the
   new candidate is refit without each test fold. The synthetic KGAS066
   projected-speed recovery passes by a much wider margin.

6. **Noise, frame, and provenance — pass for the stated claim.** Synthetic
   visibility noise uses target weights and the empirical line-free scale;
   cube noise uses
   the official target channel RMS. The two domains receive reproducible
   domain-appropriate draws rather than pretending that an image voxel is an
   independent visibility. Registered TOPO-to-LSRK reporting corrections and
   the common spectral response remain intact. I revalidated all 50 top-level
   dossier entries, all 49 synthetic-manifest entries, all 9 grouped-manifest
   entries, and all 83 repaired-replay entries against byte counts and SHA-256
   digests. The synthetic source is a clean exact `f0d3067` commit; later
   commits only finalize decision metadata, resumability, the review packet,
   and status.

## Claim limits

The synthetic sample has three fixed noise seeds per target and tests an
exact, thin, axisymmetric arctan family. Its very large margin supports the
PI's empirical gate on these two canonical samplings, but it is not a calibrated
population uncertainty statement and cannot be generalized to warps,
thickness, radial flows, non-circular disturbances, clumpy asymmetric
brightness, or other arrays without new tests. The white restored-cube noise
is an idealized standard-use proxy. The synthetic binned visibility draws are
also independent between channels rather than explicit C1 innovations, so the
synthetic suite is not a covariance-calibration test. Real C1 prediction and
correlated deconvolution residuals remain represented only by the real-data
comparison.

The earlier common restored-cube failure remains valid diagnostic history.
Under the binding PI decision, restored moments, PVDs, spectra, and cube
residuals do not veto a result that passes known-truth projected recovery and
held-out native visibility prediction. They should remain visible in the
production record rather than be described as superseded measurements.

## Verdict

**Accept.** Both canonical target samplings exceed the required 10% advantage
in matched-family projected-speed recovery, both have positive held-out
standard-use predictive bounds, and KGAS066 satisfies the quantitative
non-regression guard. The evidence is checksum-bound and the physical claim
is appropriately restricted. S4 may close after the independent software and
reproducibility seat also accepts; S5 may seal this evidence without expanding
the scientific claim or launching an unlicensed sampler campaign.
