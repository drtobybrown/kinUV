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
rereviewed_commit: f2f6a22b6749f46da70e55f57e599f919190c9f8
status_packet_commit: 879a04b896df6a22b72437e98bb937385b0a9e68
stable_evidence_commit: 354bbad8fb721871f82bb374ac9cfa62606067f4
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

## Authentication revision re-review

I re-reviewed the corrected scientific implementation and regenerated evidence
at exact clean commit
`f2f6a22b6749f46da70e55f57e599f919190c9f8`, together with the status and
review packet at `879a04b896df6a22b72437e98bb937385b0a9e68`.

The correction closes the checkpoint-provenance defect without changing the
science. A retained KinMS fit is now reusable only when its saved configuration
exactly matches the current configuration, including runner commit,
realization seed, mock-cube hash, mask hash, truth-cube hash, and worker hash.
Because the previous configurations lacked these fields, the corrected runner
necessarily removed and regenerated all six fit directories. Both target
summaries now also bind the covariance metrics, visibility table, diagnostic
cube, mask, error map, target configuration, runner, and worker by path and
SHA-256. I independently matched every recorded source digest to the current
file and verified all six per-realization provenance records. The grouped
bootstrap seed is explicitly recorded as `4404`.

The final dossier manifest identifies `f2f6a22`, contains 50 entries, and
passes complete byte-count and SHA-256 verification. Its nested synthetic
manifest identifies the same commit and all 49 entries verify. The regenerated
science values are exactly equal to those reviewed above: projected-speed RMSE
ratios remain `0.01180266879454277` for KGAS066 and
`0.04569431293329279` for KGAS007; real held-out gains remain
`0.04265073718804183` and `0.00456082894209484` chi-square per real
component, with lower bounds `0.03178592352114772` and
`0.0029030375555849555`. Every individual recovery ratio and optimizer status
also remains unchanged. The `f2f6a22..879a04b` diff contains only status and
review-packet documentation, so it does not alter those metrics.

**Final re-review verdict: Accept.** The authenticated rerun preserves the
previous scientific conclusion and removes the stale-checkpoint ambiguity.
The claim limits in this review remain binding.

## Stable-evidence final re-review

I performed a final science re-review against exact stable evidence commit
`354bbad8fb721871f82bb374ac9cfa62606067f4` and its newly regenerated
dossier. The two intervening implementation changes are reproducibility-only:
the exact checkpoint-contract comparison is factored into the directly
testable `retained_kinms_result` helper, and the fit-window cube that controls
visibility selection is added to the authenticated input hashes. Neither
change alters truth generation, likelihood evaluation, fitting, projected
profile calculation, aggregation, or a scientific gate.

The regenerated `metrics.json` SHA-256 is
`4317fd28e33a255a9ecfa8805a960ece1de9402620736ccac3af8ad8198f9f28`
and the top-level `MANIFEST.json` SHA-256 is
`e75ea038501cdd995ee99e19cf2883259b385e25a18418622e8294d68d3727db`,
exactly matching the submitted identities. The top manifest identifies
`354bbad`, contains 50 entries, and verifies completely; the nested synthetic
manifest identifies the same commit and all 49 entries verify. Every target
input hash matches its source, including the newly recorded fit-window cubes,
and all six KinMS checkpoint contracts name `354bbad` with the correct seed,
cube, mask, truth, and worker identities.

The science results are exactly unchanged from both earlier reviews. Synthetic
kinUV/KinMS projected-speed RMSE ratios remain `0.01180266879454277` for
KGAS066 and `0.04569431293329279` for KGAS007. Real held-out gains remain
`0.04265073718804183` and `0.00456082894209484` chi-square per real
component, with lower 95% bounds `0.03178592352114772` and
`0.0029030375555849555`. The bootstrap contract remains 20,000 draws with
seed `4404`. All previously recorded scientific scope limits remain unchanged.

**Stable-evidence verdict: Accept.** The final provenance fixes do not change
the physical comparison or its result. The `354bbad` dossier satisfies the S4
science gate within the exact-family and standard-use limits stated above.
