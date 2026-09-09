---
id: DEC-KINUV-UNIFIED-FOUNDATION
status: architecture-licensed-for-bounded-prototyping
date: 2026-09-09
source: Project PI current directive
owner: senior-registrar
production_adoption: not-accepted
---
# Unified kinematic foundation

## Decision

The architecture is licensed for bounded prototyping.  Production adoption is
**NOT ACCEPTED**.  The Lead Architect owns the scientific architecture and Sol
implements it.  Prototypes must not overwrite production products, stop
existing sampler workers, or claim gates from historical scores.

All targets use one fixed-dimensional model, inference chart, and versioned
output schema.  Target metadata and data-dependent scales may enter through
one declared algorithm; target IDs and candidate names do not select different
mathematics.  One universal software chart does not guarantee that the thin,
axisymmetric circular model is adequate for every one of 452 targets.
Unresolved or noncircular cases retain the common schema with explicit
adequacy and missingness flags; they do not receive a hardcoded fallback or a
forced accepted measurement.

## Common kinematic model

The primary fitted quantity is a continuous projected rotation profile
`u(R)=V_rot(R) sin(i)` with `u(0)=0`, finite central `u/R`, and C2 continuity
(C1 is the minimum allowed at joins).  A cubic P-spline is the provisional
choice for compactness and local support pending comparison with a fixed-kernel
low-rank GP; no measured performance superiority is established.  The prototype may
use an arctan carrier with multiplicative smooth deviations in one universal
chart, but it must not choose arctan for one target and rings for another.
Manual knots and a hard 0.5-BMAJ resolution floor are excluded; the model must
retain central sub-beam support.

The current carrier-plus-shape P-spline chart is prohibited from production
sampling: its KGAS007 profile-only fit collapsed the carrier scale to
0.00069 arcsec while compensating with large shape latents.  Resolve that
redundant scale/shape freedom and remeasure posterior conditioning.  The
Implementer may orthogonalize the shape basis, remove the free carrier scale,
or use another explicit construction; an arbitrary tight turnover prior must
not hide the degeneracy.

Use a fixed, small-dimensional, continuous positive dispersion profile whose
deviations shrink to the exactly constant-dispersion limit.  Smoothness priors
must be proper, low rank, and regularize the nullspace.  Hyperparameters are
initially fixed by campaign or selected by one identical training-fold rule.
Do not freely sample a length scale until evidence shows that the resulting
funnel is tractable.

Use the empirical positive moment-0 morphology with floating total flux.  A
free two-dimensional brightness grid is excluded.  If held-out evidence later
requires morphology flexibility, one universal bounded first-harmonic modifier
with amplitude below unity may be added; it is not part of the initial core.
The existing three-component empirical weights may be frozen during a
prototype only if comparator parameter accounting is matched and the
conditioning is explicit.

## Likelihood, priors, and physical boundary

For the fixed C1 covariance, `log L = -chi2_vis/2`.  Prior and smoothness terms
remain separately visible.  “Visibility only” describes the data likelihood;
it does not mean an unregularized model.  Every sampler uses the same proper
prior, transformations, Jacobians, C1/channel response, primary beam, and
cached forward renderer.

The fitted profile is kinematic.  Smoothness removes derivative jumps but does
not prove gravitational circular speed or stability.  Report inclination
conditionality and negative kappa-squared shape diagnostics without clipping.
Mass, dark matter, epicyclic analysis, cosmology, distance conversion, and
asymmetric-drift or streaming corrections are downstream and cannot feed back
into the visibility likelihood.

The empirical moment-0 product from the full canonical cube is allowed by the
PI standard-use benchmark.  Real grouped hold-out scores therefore condition
on an all-data morphology product and are not described as fully end-to-end
leakage-free.  Mock workflows repeat template construction and regularization
when these are data-derived, or label the template as fixed independent input.

## Sampling contract

Use an affine chart `q = q_MAP + L z`, where `L L^T` is a regularized inverse
posterior Hessian in the correct nonlinear chart.  Retain nonlinear Jacobians,
do not apply the metric twice, and inspect negative curvature; taking absolute
Hessian eigenvalues does not establish a valid metric.  A local affine map can
reduce anisotropy but cannot remove every funnel.

NUTS remains the baseline until measured evidence shows that dynesty with
`rslice` gives adequate effective samples per wall hour and an independently
replicated posterior consistent with NUTS.  A compiled value-only likelihood
path is allowed without making gradients a production dependency.  Share the
forward renderer and caches; add no Fourier-transform engine or GPU dependency.
The initial sampler spike did not establish fairness because its KGAS007
callback retained the ring regularizer but misclassified it as likelihood;
the corrected fair KGAS007 comparison is deferred.  Corrected evidence is
required before any sampler decision.

## Output and turnover

Every fit emits the same versioned products: angular radius, `u(R)`,
`V_rot(R)`, `sigma(R)`, radial derivatives, geometry, q16/q50/q84 bands when
valid, sampler/stage/conditionality, covariance provenance, resolution and
measured-support masks, and explicit missingness flags.  Velocity frame and
convention are mandatory.  PVD overlays transform each native
`v_sys +/- u(R)` endpoint into the reporting frame.  A native radio velocity
difference must not be added directly to an optical systemic velocity.

Define turnover as `R50`, the first unique rising crossing of half a
demonstrably constrained outer or asymptotic projected-speed reference.  For
an arctan truth, `R50=r_t`.  Return unresolved or nonunique when no outer level
is measured, the crossing lies outside support, or multiple crossings are
ambiguous.  Do not force a flat tail or fit an arctan surrogate to rings.
Derive posterior R50 sample by sample and report the resolved-draw fraction.
A real fitted scale ratio is descriptive and is not proof of KinMS smearing.
The chart's amplitude coordinate `U_reference` at a declared measured radius
is distinct from the derived `U_outer` used by R50; only an independently
constrained asymptotic/outer reference enters the half-crossing definition.

## Diagnostic products

The automatic suite includes matched 3-by-5 moments, major and minor PVDs,
fixed-aperture spectra and residuals, and rotation plus dispersion profiles.
Posterior bands must come from the same accepted inference record.  Display
masks are fixed and shared.  Preserve unmasked cube/visibility residuals and
report channel coverage and excluded spectral flux so the display mask cannot
hide extra predicted wings.  Moment/PVD dipoles and spectral residuals are
supporting diagnostics, not an image objective or a guarantee that structure
will disappear.

## Verification before production adoption

After the model and sampler spikes, implement the common model and run uniform
fits on both targets and folds.  Use identical noisy realizations and operators
for kinUV and KinMS.  Include the current arctan truths, an independently
defined smooth nonmonotonic rotation truth, and constant and varying dispersion
truths.  Report flux recovery, inner-beam projected-speed RMSE, R50 error or
correct null, sigma recovery, profile bias/coverage, failure rate, and runtime.
Historical 40x/70x results do not transfer.

Production consideration requires all of the following:

- at least 10 percent lower projected-speed RMSE on both mock samplings, with
  no material KGAS066 regression;
- a positive lower 95-percent bound for real grouped-visibility prediction
  against KinMS;
- no more than 2 percent relative predictive degradation against the accepted
  kinUV baseline;
- the existing empirical numerical-refinement gate, with the PI-authorized
  sub-1-percent precision rule and no new asymptotic reference harness;
- for NUTS primary profile and geometry quantities, R-hat <= 1.05, ESS >= 400,
  and accepted divergence, BFMI, and tree-depth behavior;
- for weighted nested sampling, normalized weights, effective sampling per
  wall hour, an independent replicate and Monte Carlo error, and posterior
  agreement without applying R-hat to the weighted sequence.

Sequence: model spike, common implementation, fair sampler comparison, new
mock and real verification, independent reviews, then an explicit promotion
decision.  Queued, partial, blocked, and measured evidence remain distinct.

## Current bounded evidence

The representation and sampler measurements remain prototype evidence and do
not pass the gates above.  The diagnostics spike adapted both frozen targets
to one schema and rendered eight products in 11.238 seconds.  It exposed an
open production PVD-frame error of up to 10.280 km/s for KGAS066 and
9.306 km/s for KGAS007, restored the missing minor PVD in scratch, preserved
KGAS007 as a piecewise-ring adapter without a smooth surrogate, marked its R50
unresolved, and retained KGAS066 R50 only as conditional on its arctan model.
Details and the minimal adapter contract are in
[`UNIFIED_DIAGNOSTICS_SPIKE`](../technical/UNIFIED_DIAGNOSTICS_SPIKE.md).
The corrected P-spline pilots measured warmed value-plus-gradient times of
0.564/0.417 s for KGAS066/KGAS007 versus 0.539/0.441 s for the finite GP.  All
stopped at a three-iteration limit; their likelihood values are not
source-comparable and there is no throughput or scientific winner.  The
headless jobs sampled a dirty pre-final-audit source without digests, so exact
current-source scores remain unmeasured.  Their shared two-deviation dispersion basis missed the
retained KGAS066 two-zone fixture by 0.51 km/s RMS and 1.09 km/s maximum, so a
small target-neutral expansion may be required.  The automatic inner nodes
were about 0.574 arcsec; exact recovery of a 0.25-arcsec arctan checks the
shared carrier and does not validate an independent non-carrier sub-beam shape.
See [`UNIFIED_REPRESENTATION_SPIKE`](../technical/UNIFIED_REPRESENTATION_SPIKE.md).
The sampler cost and bounded-probe dispositions are recorded in
[`UNIFIED_SAMPLER_SPIKE`](../technical/UNIFIED_SAMPLER_SPIKE.md); no sampler
winner or posterior comparison is accepted.  The valid forced-bound `rslice`
probe took 102.807 s total and 12.572 s in the sampler for 28 actual likelihood
callbacks, three constrained replacements, and weighted ESS 1.  The paired
NUTS probe was stopped at its 10-minute bound after active 3.25-core/7.20-GB
use but produced no result or draw, so its sampler time and ESS are not measured.
