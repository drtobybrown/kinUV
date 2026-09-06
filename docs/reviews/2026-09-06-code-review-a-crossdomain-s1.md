---
role: reviewer
seat: science-numerics
phase: implementation
date: 2026-09-06
reviewer: /root/s0_review_a
canon_generation: 11
campaign_id: crossdomain-recovery-s1
proposal: docs/decisions/DEC-KINUV-CROSSDOMAIN-RECOVERY.md
reviewed_commit: 1b629a1fab9c64a25cb870d8f5ad43102b55cb0b
verdict: accept-with-required-changes
---
# Independent S1 science and numerics review

I reviewed exact commit
`1b629a1fab9c64a25cb870d8f5ad43102b55cb0b` and the durable S1 dossier at
`results/validation/crossdomain-recovery-s1-20260906/` without reading or
contacting Reviewer B. I did not modify the implementation.

## Attempted falsification

I traced the KinMS worker from its generated phase-space quadrature through
KinMS 3.0.13 normalization, the declared `(north, east, velocity)` cube, the
shared primary-beam and NUFFT operators, and the single Hann/bin operation. I
checked cube and flux units, velocity ordering, spectral registration,
quadrature resolution, analytic DFT closure, target rendering repeats, and
baseline replay. I also compared the spatially resolved first-moment signs of
the imported KinMS cubes with independently generated kinUV cubes at the same
target parameters. This last check falsified the physical position-angle
contract even though the scalar closure gates pass.

The focused S1/forward tests passed (`25 passed`) and the complete suite passed
(`230 passed, 8 skipped`). I recomputed the durable manifest's file hashes and
sizes and found no mismatch. The dossier identifies the exact reviewed commit,
the KinMS 3.0.13 environment, immutable worker and requirements hashes, the S0
input hash, and a clean source tree. No sampler was run and no target parameter,
prior, physics choice, or frozen numerical threshold changed in this commit.

## Findings

1. **Required — the KinMS position-angle convention is reflected relative to
   kinUV.** The worker passes the kinUV astronomical PA directly to KinMS as
   `posAng`. Under the cube's declared north/east axes, KinMS 3.0.13's rotation
   convention then puts the receding velocity gradient at `360 deg - PA`.
   Using the exact S1 target parameters and the worker's radial emissivity, I
   independently measured moment-1 gradient PAs of `199.721 deg` (kinUV) and
   `160.273 deg` (imported KinMS) for KGAS066, and `151.426 deg` versus
   `208.534 deg` for KGAS007. Native moment-1 correlations were only `0.8075`
   and `0.5682`; reflecting the KinMS cube's east axis raised them to `0.9984`
   and `0.9991`. The moment-0 centroids agree, so this is specifically a PA/sign
   conversion failure rather than a wholesale axis swap. Convert at the KinMS
   boundary, likely with `posAng = (360 - pa_deg) % 360`, and add a synthetic
   end-to-end test with a known receding side and an asymmetric spatial offset.
   Do not establish the convention by reflecting an already rendered cube.

2. **Required — the independent high-resolution repeat fails the frozen
   `0.1` chi-square convergence threshold for KGAS066, but that quantity is not
   gated.** The dossier reports KGAS066 high-common chi-square
   `223970.08732742647` and high-independent chi-square
   `223970.21781259932`, an absolute change of `0.13048517285`. The executable
   gate applies `<= 0.1` only to nominal-common versus high-common and applies
   only an RMS rendering-noise threshold to the independent high repeat. The
   decision explicitly requires high-cloud convergence within `0.1` "with
   fixed/common RNG and an independent high-cloud repeat." Gate the independent
   high-repeat chi-square comparison too, then raise deterministic resolution
   and regenerate the target dossier until both canonical targets pass without
   changing the frozen threshold.

3. **Required — the claimed doubled-sampling and high-cloud convergence do not
   test every numerical integration axis in the production comparator path.**
   Nominal and high target renders both use 256 radial samples, 512 azimuth
   samples, and ninth-order dispersion quadrature; high merely doubles the
   azimuth phase ensemble from 32 to 64. Radial and velocity-dispersion
   integration are therefore untested. Separately, the reported doubled
   spatial/spectral value `2.4017e-5` comes from normalized analytic Gaussian
   toy calculations outside the KinMS worker and target PB/NUFFT/Hann path.
   The spatial toy independently renormalizes two cube grids, and the spectral
   toy compares midpoint samples with two subchannel samples. This is useful
   analytic arithmetic, but it does not establish that doubling actual spatial
   and spectral rendering resolution changes the target likelihood by at most
   `0.1`. Add convergence runs through the real shared operator that separately
   double spatial sampling, radial quadrature, azimuthal quadrature, dispersion
   quadrature, and any native spectral sub-sampling. Record per-axis chi-square
   changes and enforce the frozen aggregate gate.

4. **Advisory — spectral centroid registration currently guarantees the
   centroid gate from the rendered answer.** After rendering, the worker
   measures the model centroid and linearly shifts the whole spectral cube to
   the requested systemic velocity, then renormalizes its flux. This is an
   additional spectral interpolation whose shift is about `0.0353` native
   channel for KGAS066 in this dossier. It is benign for the present symmetric
   radial models, but for later asymmetric emissivity or kinematics it can
   erase a physical flux-weighted centroid offset and makes the centroid gate
   partly circular. Derive registration from the two coordinate definitions or
   a zero-velocity calibration, then measure centroid error independently.
   Quantify the interpolation's effect before asserting that the native line
   spread function is followed by Hann exactly once.

5. **Advisory — post-render normalization hides edge loss.** KinMS clean-output
   normalization and the subsequent centroid-shift renormalization force the
   requested integrated flux, so the reported near-machine-precision flux
   errors do not reveal spatial or spectral clipping. Record pre-normalization
   deposited flux and the fraction falling outside the cube/channel support;
   retain the current final zero-baseline check as an operator-contract test.

## Gate assessment

The intrinsic/pre-beam flags are supported by the installed KinMS source:
`cleanOut=True` bypasses restoring-beam convolution, `spectral_resolution=0`
adds no KinMS response, and the clean cube is normalized as a spectral density
before the worker converts it to Jy per native channel. The worker exports the
documented x/y/velocity array as north/east/velocity. The consumer validates
the sidecar, checksums, velocity axis, and nonnegative finite cube. The kinUV
path and comparator call the same primary-beam and NUFFT functions, and Hann
plus channel binning occurs once after native visibility sampling.

The independent analytic DFT comparison is excellent: relative complex-
visibility L2 is `4.6965e-14`, noise-normalized component RMS is
`1.0379e-12`, zero-baseline flux error is `3.6973e-15`, and centroid error is
below machine precision. Target nominal/common-high rendering RMS is far below
`0.1` thermal SD, and baseline replay is exactly zero for both targets. These
checks validate substantial operator plumbing, but their symmetry and scalar
statistics do not detect the reflected kinematic PA.

Stage S1 cannot close at this exact commit. The comparator sends a physically
different velocity field through an otherwise consistent operator, KGAS066's
independent high repeat exceeds the frozen chi-square tolerance, and the
sampling-convergence evidence does not exercise the actual target path or all
quadrature axes.

## Residual risks

The present comparator is a deterministic operator-closure harness, not yet
the decision's complete matched/native fitting branches. It does not establish
equal nuisance optimization, identical fitting masks and covariance, held-out
prediction, or fair KinMS-versus-kinUV scientific performance. Axisymmetric
surface brightness made the PA defect invisible in total flux and centroid
checks; future contract tests need signed velocity structure and asymmetric
spatial structure. The unusually large deterministic cloud counts also make
full target validation expensive, so the corrected convergence study should
report peak memory and latency as required by the decision.

## Verdict rationale

**Accept with required changes.** The intrinsic adapter and shared operator
refactor are strong foundations, and their local numerical closure is precise.
The reflected PA changes the physical comparator model, however, and two
frozen convergence requirements are not currently demonstrated. Repair the PA
boundary, enforce both high-repeat chi-square comparisons, validate all
rendering axes through the target pipeline, regenerate the durable dossier,
and obtain a fresh exact-commit review before closing S1.
