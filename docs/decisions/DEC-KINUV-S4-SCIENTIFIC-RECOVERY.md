---
id: DEC-KINUV-S4-SCIENTIFIC-RECOVERY
status: accepted-for-implementation
date: 2026-09-07
authority: Astra-under-current-Project-PI-directive
scope: prospective-S4-recovery-and-S5-promotion
supersedes: prospective-S4-metric-domains-selection-and-campaign-sizing
historical_evidence: results/validation/crossdomain-recovery-s4-20260907-r3
---
# Scientific recovery of S4: prediction, identifiability, and intrinsic kinematics

## Decision and authority

The PI explicitly directed Astra to diagnose the S4 failure, redefine meaningful
scientific benchmarks, and unblock implementation. This decision licenses Sol
to repair the comparison, reassess the existing candidates, and implement the
bounded model changes below. It prospectively replaces the restored-cube
moment-1 and scalar voxel reduced-chi-square promotion contract. The required
10 percent projected-velocity recovery advantage and real-target improvement
remain; their statistical domains are made explicit below.

The r3 result at `5d08507` remains a failed historical experiment. The review
verdicts establish reproducibility of its arithmetic, not truth recovery or
proof that every diagnostic coordinate/operator is scientifically correct.
No existing failed result becomes a pass under this decision. S5 remains
gated on fresh S4 evidence. Sol owns implementation, numerical choices,
execution, and independent reviewer coordination. This is the complete
architectural handoff, not a request for further pre-implementation paperwork.

## Diagnosis: what the evidence does and does not show

### 1. The added S3 flexibility has not been isolated as the cause

The retained KGAS066 S3 ablation records show:

| Quantity | Baseline arctan | Selected emissivity plus two-zone dispersion |
|---|---:|---:|
| Projected amplitude u, km/s | 184.002 | 183.633 |
| Inclination, deg | 55.298 | 55.335 |
| Turnover / diagnostic BMAJ | 0.28620 | 0.28172 |
| Outer emissivity fraction | 0.33374 | 0.30298 |
| Inner dispersion, km/s | 9.062 | 7.473 |
| Outer dispersion, km/s | 9.062 | 9.878 |

The geometry and projected amplitude barely move. The outer component loses
relative flux; the selected model already includes a dispersion gradient.
S4 scored the selected candidate but did not score every retained S3 ancestor
through the same corrected diagnostic operator. Therefore, blaming the
regression on excessive added flexibility, an outer-flux runaway, or a missing
dispersion gradient is unsupported. Sol must first distinguish an inherited
baseline mismatch from a new ablation-induced mismatch.

The frozen KinMS fit uses inclination 48.409 deg, turnover 0.4652 arcsec,
dispersion 12.833 km/s, and LSRK optical systemic velocity 8300.968 km/s.
The selected kinUV values are inclination 55.335 deg, turnover about 0.2930
arcsec, and reported systemic velocity 8312.893 km/s. These are substantial
between-method differences even though the S3 within-method changes are small.
The systemic offset is about 11.9 km/s; it cannot be dismissed using the
0.037 km/s within-observation frame-drift result. Absolute frame accounting
and different fitted centroids are separate issues.

### 2. Diagnostic accounting must precede new astrophysics

Static inspection identifies concrete concerns in the existing diagnostic path:

- `major_axis_rotation_profile` in `src/kinuv/diagnostics/kinms_benchmark.py`
  forms a signed FITS offset using CDELT1, then negates that offset when
  CDELT1 is negative, while the FITS data remain unflipped. This is inconsistent
  with east-positive sky offsets for the registered unrotated celestial grid.
  Resolve this reflection against the celestial WCS and existing signed-axis
  convention before interpreting the slit profile. Shared use of an incorrect
  slit does not make its physical interpretation correct.
- The profile uses kinUV's fitted geometry and systemic velocity for the data
  and both models. It applies a separate five-percent-of-peak moment-0 cut to
  each cube, a fixed 0.5 arcsec deprojected half-slit, fixed radial bins, and
  then interpolates separately supported model curves. These are not one
  independently defined set of sky samples or independent measurements of
  intrinsic rotation. Its fixed 8 arcsec radial cap also leaves part of the
  KGAS066 template-based support unexamined. Three radial bins are not
  automatically three beams.
- `attenuate_intrinsic_cube` applies the PB at the median observed native
  frequency; `match_model_to_imaging` defaults to undoing it at RESTFRQ.
  These are different beams. Quantify and repair this inconsistency, including
  coordinate registration, before interpreting an outer-flux discrepancy.
- Diagnostic spectral matching currently overlap-averages channels, whereas
  the visibility path has an explicit Hann/bin response. Establish the actual
  response of the official cube from its extraction/imaging lineage and apply
  that response exactly once. Do not add Hann smoothing by assumption.
- The restored-cube score treats masked voxels as independent with one scalar
  RMS inferred from an integrated-error map. A shared scalar cancels in a raw
  score ratio; a non-diagonal covariance generally does not. CLEAN residuals,
  clipping, PB correction, and channel covariance prevent treating this number
  as a calibrated reduced chi-square or a sufficient statistic for the MS.

These are static findings and scientific accounting limitations. Their effect
on the measured S4 ratios has not been rerun or quantified in this architectural
turn. The voxel residual discrepancy remains an observed diagnostic concern;
its origin is not uniquely established.

### 3. The principal structural hypothesis is conditioned morphology

The brightness template comes from a Wiener-deconvolved 30 km/s moment-0 map
of the same observations. S3 clips it positive and splits it into three hard
radial bands at the S2 geometry. The band shapes stay fixed while the velocity
geometry changes. Two relative weights cannot move emission within a band,
remove a spurious clump independently, or propagate uncertainty in the
deconvolution. Unequal band weights also introduce artificial brightness steps.
This is useful conditional fitting but is not a fully joint intrinsic
brightness-and-geometry model.

There is also a conditional ablation confound: the template loader can retain
signed emission when its clipping check fails, whereas S3's basis construction
always takes the positive part. The retained record does not establish which
clipping state occurred. Record that state before attributing a gain entirely
to radial weighting. Preserve both the source Ico beam/Wiener resolution and
the different 10 km/s diagnostic beam; neither alone determines the native
visibility information available for an intrinsic velocity knot. Do not claim
sub-beam identifiability solely from a smaller diagnostic beam or a full-rank
conditional Hessian, and do not prohibit it solely from a beam-size rule.

The KGAS066 template-based r95 is 11.69 arcsec, while the retained KinMS record
lists an image-based extent of 7.85 arcsec. These use different definitions
and must not be compared as a measured excess halo. They motivate an audit of
faint template support, positive-clipped noise, and unmeasured short spacings.
The masked 97.75 percent flux ratio likewise is not an independent measurement
of total or zero-spacing flux.

Physically, a beam-smeared centroid depends jointly on brightness and velocity:

\[
\bar v_{\rm beam}(x,y) =
\frac{B * [\Sigma(x,y) v_{\rm los}(x,y)]}{B * \Sigma(x,y)}.
\]

This idealized expression assumes complete spectral integration; a clipped
channel mask also couples the centroid to line width. Correct total flux does
not fix the local brightness weighting. A rigid or noisy morphology can
therefore push the best-fitting inclination, turnover, systemic velocity, or
dispersion into compensating values without an optimizer failure.

KGAS066 is not globally an unresolved disk: its recorded emitting extent is
several beam widths. Its fitted turnover, however, is only about 0.28 BMAJ.
The inner rise is the vulnerable quantity. Four knots at approximately 0.52,
4.96, 7.45, and 11.69 arcsec do not provide a strong alternative description
of that sub-beam turnover. Rejection of that particular knot model does not
show that all non-arctan central profiles are unnecessary.

S3 selection uses penalized training fits and raw chi-square retention gains,
not a held-out predictive comparison. Those gains admit candidates; they do
not establish their generalization or calibrated effective complexity.

### 4. Lower-priority physical extensions

Retain u(r) = V_rot(r) sin(i) as the principal velocity chart. Projection still
couples inclination to deprojected radius, so this chart reduces but does not
eliminate the geometry degeneracy. An isotropic orientation prior is not an
external inclination measurement. Do not force either galaxy toward KinMS's
inclination or resurrect a catalogue value without a justified uncertainty.

After the accounting and morphology work, coherent approaching/receding
asymmetry, minor-axis residuals, or excess line wings can motivate one bounded
extension: smooth PA variation, a low-order noncircular component, or finite
thickness/LOS mixing. Do not introduce all three together. Current evidence
does not identify a bar, warp, thick disk, or turbulence gradient as the unique
cause. Gas streaming speed must not be relabeled gravitational circular speed
without downstream dynamical assumptions. DM decomposition remains excluded.

## Scientific benchmark contract for the next attempt

### What constitutes superiority

Visibility fitting is not guaranteed to beat a statistically sufficient image
representation using the same model, priors, and covariance. The achievable
advantage is more reliable inference under actual imaging, masking, resolution,
and noise limitations. KinMS already supports beam-convolved 3-D models,
variable dispersion, thickness, and warps; a comparison must not caricature it
as a two-dimensional moment fitter. See the [KinMS API](https://timothyadavis.github.io/KinMS_fitter/autoapi/kinms/index.html).

Use two complementary primary measurements, separately reported for each
canonical sampling regime:

1. **Known-truth projected kinematic recovery.** On paired simulations, use a
   fixed truth-defined radial support and weights shared by both methods:
   `E_m^2 = sum_j w_j [u_m(r_j)-u_truth(r_j)]^2 / sum_j w_j`.
   The aggregate ratio is `sqrt(sum E_kinUV^2 / sum E_KinMS^2)`, with equal
   prespecified regime weights, not a mean of noisy per-realization ratios.
   Require ratio <= 0.90 per target sampling and a paired 95 percent upper
   confidence bound below 1.0. Report absolute errors, signed bias, failures,
   and regime dependence; do not require dominance in every pixel or every
   stochastic realization. Low-SNR and asymmetric-brightness cases must be
   represented, without pooling a KGAS066 failure into a KGAS007 gain.
2. **Real held-out visibility prediction.** Fit both methods on the same
   training observations, image only those observations for stock KinMS,
   and evaluate their intrinsic predictions on the same held-out native
   visibilities with the same frozen covariance. Require improved aggregate
   held-out chi-square per retained real component for each target, and a
   group-resampled lower 95 percent bound above zero for
   `Delta = chi2_KinMS - chi2_kinUV`. This is the prospective real-data
   chi-square improvement gate. Held-out cells receive no parameter-count
   subtraction; quote a predictive chi-square per component, not fitted
   reduced chi-square. Groups, not the millions of correlated cells, determine
uncertainty. If covariance differs, compare predictive log likelihood
   including its normalization rather than raw chi-square.

KGAS066 must additionally have no more than 2 percent relative degradation
against its frozen kinUV reference in the upper 95 percent held-out bound,
using a reference trained under the same folds. This retains the earlier
aggregate non-regression guard. On paired mocks, the KGAS066 aggregate
new-to-frozen-kinUV recovery-error ratio must also have an upper 95 percent
bound <=1.05. These are target-level guards; the earlier intersection of
separate per-cell confidence limits is superseded. An uncertain comparison is inconclusive, not
permission to loosen the 10 percent scientific benefit criterion.

### Eligibility and supporting quantities

- Report u(r) and its error on emission-supported radii first. V_flat is a
  promoted quantity only when the data support a plateau and inclination is
  independently constrained or its uncertainty is propagated. Otherwise
  report an observed-radius velocity, not an extrapolated asymptote.
- R_turn is a model parameter, not a universally measurable radius. On
  matched arctan truths score its bias/RMSE directly. Across model families
  use a declared comparable rise scale, such as the radius reaching half a
  supported outer speed. If that scale is unresolved, report the degeneracy or
  limit; do not turn a bound or a narrow conditional Hessian into precision.
- Retain the existing mock flux gates: median absolute fractional error <=5
  percent and 90th percentile <=10 percent; signed radial velocity bias <=5
  percent of the measurable truth-u amplitude. Numerical flux accounting is
  separately required to close within 0.1 percent over the represented domain,
  with escaped flux recorded rather than renormalized away.
- Real total-flux adequacy uses aperture, PB, and calibration uncertainties;
  masked flux fractions remain descriptive. Calibration uncertainty does not
  excuse coherent spectral/phase errors or bias amplified by many samples.
- Retain common-sky moment maps, aperture spectra, both sides of the PVD,
  channel residuals, and the historical cube score. Geometry, sky support,
  and diagnostic masks must be common and fixed independently of the candidate
  under evaluation. Sparse profiles are reported as sparse; a fixed bin count
  cannot certify angular resolution or veto a valid visibility recovery test.
- Report runtime and peak memory on matched resources and comparable fitting
  budgets. Do not claim speed superiority from unmatched optimizer effort.
  Posterior coverage, R-hat, and ESS apply only if a separately authorized
  posterior product is attempted; no new sampler campaign is licensed here.

### Fair data and comparator boundary

The frozen stock KinMS software and historical full-data fits remain the
external baseline. A full-data fit cannot be reused as an un-leaked held-out
fit. Refit the frozen stock procedure on training-only cubes for predictive
comparisons. Both methods receive the same information, center/geometry prior
sources, spectral convention, and declared physically comparable parameter
support. Also retain a matched-family comparison to separate a domain benefit
from a richer model or prior benefit.

Any target-derived template, mask selection, covariance tuning, regularization
strength, and candidate selection must be learned from training data or from
an independently fixed external source. The official full-data Ico template
cannot silently cross a holdout boundary. Initial diagnostic replays may use
it, labeled conditional; predictive promotion may not. Keep complete correlated
MS groups intact through averaging and splitting. Nested selection or an
untouched evaluation split must protect the final score from model selection.

Use the existing S1-validated intrinsic continuum comparator and common
measurement operator to score fitted KinMS parameters in visibility space.
Do not Fourier-transform a restored KinMS cube as if it were intrinsic, refit
its amplitude on held-out data, or count a visibility-refitted KinMS result as
the stock image-fit baseline. Verify the adapter's empirical closure at newly
used parameters with the existing refinement machinery; no second Fourier
reference engine is requested.

Paired mock fitting must start from the same noisy visibilities; the image
fitter receives an image of that realization. Use existing validated renderers
and multiple known brightness/kinematic families so that an exact kinUV-family
injection does not become the entire superiority argument. The previous
16-cell/512-pair requirement is replaced by a prespecified pilot followed by
a fixed evaluation sample sized for the paired uncertainty. Sol owns the
resource budget and sample-size choice, fixed before final evaluation; do not
repeatedly stop sampling when a favorable confidence interval appears.

## Sol's sequential execution plan

| Step | Authorized work and physical purpose | Exit evidence |
|---|---|---|
| A: Repair the measurement comparison | Resolve sky handedness, reference pixels/phase centers, PB inversion, absolute spectral conventions, channel-edge coverage, and actual response lineage. Re-score the retained S3 baseline and each ablation without changing their parameters. | One common set of sky coordinates and response conventions; errors within existing empirical closure or sub-1 percent diagnostic fidelity; before/after scores for both targets. No new physical fitting before material comparison defects are resolved. |
| B: Release conditioned morphology if needed | If corrected replay still implicates morphology or fails predictive adequacy, replace hard template bands with a positive, smooth, low-dimensional emissivity representation fitted jointly. Retain the existing template as initialization or a conditional reference. Treat unsupported outskirts with smooth shrinkage, not a forced compact cutoff. Training-only input construction is required regardless. | Joint geometry and brightness share a consistent intrinsic projection, or an explicitly documented free sky-brightness model with no spurious morphology-derived inclination prior. Outer support, flux accounting, and u(r) sensitivity are reported. |
| C: Select identifiable flexibility | Keep the projected-speed chart and constant-dispersion/arctan baseline. Allow a smooth supported radial correction to u(r), with zero central velocity and regularized behavior outside information-bearing radii. Assess inner rise and the existing smooth dispersion gradient jointly. Select complexity and smoothing on training partitions; prefer the simpler model when predictive evidence is indistinguishable. | Stable top-likelihood cluster, no unresolved boundary pressure in promoted quantities, and held-out gain. A change in training chi-square or local knot-Hessian rank alone is insufficient. Recheck both targets after each retained change. |
| D: Verify S4, then seal S5 | Run the paired known-truth evaluation and untouched real grouped prediction for both methods; preserve all outcomes. Sol initiates independent science and reproducibility reviews of the implementation and evidence. | Both primary gates and supporting physical accounting pass. Then Sol may proceed to S5 verification and present the production record for Astra's scientific sign-off. Otherwise stop with the measured failure or identifiability limit. |

Sol chooses basis functions, quadrature, optimization, array layout, and
software boundaries. Freeze the existing intrinsic brightness transform during
Step A; if repaired scoring and predictive evaluation suffice, skip unnecessary
model expansion and proceed to verification. Do not prescribe an arbitrary narrow inclination prior,
force a flat tail into unsupported radii, tune parameters toward the known
KinMS answers, or add several noncircular/vertical components at once. If
steps A-C leave coherent residual morphology that demands one of the bounded
extensions above, Sol records the physical signature and tests that single
extension against the same predictive contract.

CASA-dependent imaging/extraction stays in the companion environment;
kinUV remains an observatory-agnostic visibility engine. Observation-specific
PB models and spectral response belong in the observation contract. Bind the
covariance input directly before reusing the S4 structured-residual report.
Use ASCII logs, scratch for transient I/O, and new durable run IDs; preserve
the r3 evidence. No auxiliary engine, extra ADR sequence, or float64-perfection
gate is licensed by this handoff.

## Evidence and scientific context

- Local fitted evidence: `results/validation/crossdomain-recovery-s3-20260907-r3/`,
  `results/validation/crossdomain-recovery-s4-20260907-r3/`, and each target's
  frozen KinMS fit record referenced by its configuration.
- [Di Teodoro and Fraternali (2015)](https://arxiv.org/abs/1505.07834) motivate
  distinguishing full-cube rotation/dispersion recovery from moment-field
  fitting under limited resolution; they do not establish this project's
  target-specific superiority.
- [Reid (2006)](https://academic.oup.com/mnras/article/367/4/1766/1747638)
  describes incomplete Fourier sampling and the dependence of reconstructed
  images on deconvolution assumptions. This supports treating a restored cube
  as a diagnostic reconstruction rather than independent ground truth.

No implementation, pipeline, or test execution was performed by Astra for
this directive. The diagnosis combines static inspection of code and retained
records with a delegated read-only scientific audit. Numerical attribution
and validation remain Sol's next task.
