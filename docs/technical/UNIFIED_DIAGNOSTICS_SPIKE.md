# Unified diagnostics spike

Status: bounded adapter and rendering prototype; production adoption is not
accepted.  This work reads the frozen KGAS066 and KGAS007 products and writes
only to the unified-foundation validation dossier.  It performed no fit,
rescore, mask revision, or production-product replacement.

## Finding

The current inference-to-figure interface is not model neutral.  The closest
common artifact is `best_model/rotation_curve.npz`, which already carries
`radius_arcsec`, projected speed, and intrinsic speed.  Dispersion,
derivatives, geometry covariance, support, missingness, and velocity convention
remain distributed among candidate-specific parameter JSON, checkpoint JSON,
posterior files, and plotting logic.

The active delivery path has these branches and losses:

| Location | Current behavior | Consequence |
|---|---|---|
| `scripts/package_final_production.py:74` | admits posterior bands only when `target == "KGAS007"` and legacy knot fields exist | posterior availability is encoded as a target special case |
| `scripts/package_final_production.py:104` | branches on `candidate == "two_zone_dispersion"` and reconstructs sigma from legacy fields | renderer input is not a self-describing profile product |
| `scripts/package_final_production.py:121` | assigns no turnover to supported rings | correct caution, but no generic missingness record |
| `src/kinuv/diagnostics/delivery.py:159` | renders the major PVD plus a rotation panel | the minor PVD was lost from the final delivery path |
| `src/kinuv/diagnostics/delivery.py:245` | places panel `(f)` at `x=0.94`, top right | contradicts the requested common upper-left panel-label location |
| `src/kinuv/diagnostics/style.py:259` | returns `magma` for intensity | contradicts the requested navy/teal intensity treatment |
| `src/kinuv/diagnostics/delivery.py:133` | derives both external colorbar positions from the final residual panel | creates the observed label crowding and makes the common bar appear attached to the residual side |

There is also an open spectral-frame defect in the production PVD overlay.
The saved projected profile is a native TOPO-radio velocity difference, while
the PVD axis and systemic velocity are optical LSRK.  The renderer currently
adds the former directly to the latter.  Transforming each native
`v_sys +/- u(R)` endpoint through TOPO-to-LSRK radio and then radio-to-optical
changes the outer offset from 180.603 to +190.883/-190.647 km/s for KGAS066
and from 95.575 to +104.881/-104.811 km/s for KGAS007.  The naive overlay is
therefore low by as much as 10.280 and 9.306 km/s, respectively.  The scratch
renderer applies the endpoint transform, but this production correction
remains open and no accepted cube or inferred profile number was changed.

## Minimal output contract

The prototype emits `kinuv-unified-diagnostics-v0` as `contract.json` plus one
numeric `profiles.npz`.  A production version should retain the same separation:
JSON carries semantics and provenance; NPZ carries fixed-shape numeric arrays.
Every target and sampler exposes the same required fields.

| Group | Required content |
|---|---|
| Identity | schema version, target ID as metadata, artifact state, exact input hashes |
| Inference | point stage, sampler and posterior state, likelihood identity, posterior conditionality, common model/parameter-chart version |
| Geometry | PA, inclination, systemic velocity, east/north offsets, frame and conventions; MAP and q16/q50/q84 or an explicit missing reason |
| Profiles | one increasing angular-radius grid; MAP `u(R)`, `V_c(R)`, and `sigma(R)`; first radial derivatives; q16/q50/q84 arrays with NaN plus reason when unavailable |
| Support | measured-support, sub-beam, and derivative-supported masks; BMAJ and maximum constrained radius |
| Flags | posterior unavailable, inclination conditional, derivative discontinuity, R50 unresolved/nonunique/outside support, and negative kappa-squared shape diagnostic |
| Imaging | immutable data/model/comparator cubes, existing display mask, matched moments, WCS and spectral-response provenance |

`u` and `V_c` must state their velocity convention.  The current adapters
preserve native TOPO-radio differences and store the exact reporting transform.
A PVD renderer transforms `v_sys +/- u` endpoints; it does not apply one scale
factor or add native offsets to an optical systemic velocity.  `V_c=u/sin(i)`
is conditional on inclination and on the circular-motion model.  Gas streaming
and asymmetric-drift/dynamical corrections are outside this kinematic slot.
Distance conversion, physical-radius derivatives, mass, dark matter, and
epicyclic interpretation remain downstream and must propagate distance and
inclination uncertainty.

Profile bands must be evaluated sample by sample and must come from the same
model, parameter chart, and accepted inference record as the displayed point
profile.  Quantiles of derived profiles are then stored on the common grid.
No renderer may borrow an older posterior.  Nested-sampling records additionally
need normalized weights and effective sample size; chain diagnostics such as
R-hat do not apply to a weighted nested sequence.

The legacy adapter is allowed to understand the two old representations, but
it records that adaptation at the boundary.  Downstream renderers consume only
the common fields.  For KGAS066 it reproduces the exact arctan rotation and
two-zone tanh dispersion used by the forward model.  For KGAS007 it preserves
the accepted piecewise-linear supported-ring curve, labels its regularity as
`C0_piecewise_linear_derivative_jumps_at_knots`, and never fits or labels an
arctan/smooth surrogate.  The accepted KGAS007 NUTS product supplies
conditional bands; KGAS066 bands remain NaN with
`posterior_for_selected_checkpoint_not_accepted`.

Smoothness makes derivative diagnostics better behaved but does not establish
that the fitted velocity is a gravitational circular speed or guarantee
positive epicyclic frequency.  A negative kappa-squared proxy is a disclosure
flag.  It must never be clipped into a physical curve.

## Turnover definition

Use `R50`: the first unique rising crossing of half a demonstrably constrained
outer or asymptotic projected-speed reference,

`u(R50) = 0.5 u_ref`.

The reference and its support must be part of the inference product.  If the
outer level is imposed by a boundary rule, no plateau is measured, the crossing
falls outside measured support, or multiple rising crossings make the result
ambiguous, return null with a reason.  Derive R50 sample by sample before
reporting uncertainty and report the resolved-draw fraction.  Do not fit an
ad hoc arctan merely to obtain a scalar.

For the mock truth `u(R)=u_inf (2/pi) atan(R/r_t)`, `R50=r_t` exactly.  The
prototype self-check recovers `r_t=0.7` arcsec within `1e-5` arcsec and rejects
a multiple-crossing curve and an unconstrained reference.  The KGAS066 adapter
returns 0.28958 arcsec versus its analytic arctan `r_t=0.28926` arcsec; this is
resolved only within the retained conditional arctan model.  KGAS007 returns
unresolved because its apparent outer plateau is the imposed flat-after-last-
knot boundary, not independent evidence for an asymptote.

## Rendering proof

The scratch renderer consumes the common profile fields without target or
candidate branches.  For each target it automatically produced:

- a 3-by-5 Data, kinUV MAP, KinMS, and two-residual moment matrix;
- matched major- and minor-axis PVD rows with common/residual scales and the
  correctly transformed projected MAP overlay on the major row;
- the fixed-canonical-mask integrated spectrum and residuals;
- common rotation and dispersion panels, including accepted conditional bands
  only where available.

Intensity uses navy/teal, colorbars occupy independent gutters, every panel
label is upper left, and the minor PVD remains present.  The display mask is
the existing canonical data-emission mask.  No new threshold or mask hides a
residual.  A future product must also retain unmasked cube/visibility residual
summaries and report spectrum channel coverage and flux excluded by the display
mask, because a data-derived channel mask can hide predicted extra wings.

The empirical M0 morphology prior may use the full canonical cube under the PI
benchmark convention, but grouped held-out scores then condition on an
all-data morphology product.  They are not fully end-to-end leakage-free.
Mock workflow validation must repeat any data-derived template preparation and
regularization, or explicitly label the template as fixed independent input.

## Validation needed before adoption

Use the identical model, prior, operators, covariance, noisy realizations, and
selection procedure on both canonical samplings and on unseen mocks.  Include
the current arctan truths, an independently specified smooth nonmonotonic
rotation truth, and constant and radially varying dispersion truths.  Exercise
central sub-beam structure, low-S/N outer support, inclination uncertainty,
multiple/no R50 crossings, and misspecification.  KinMS must receive the same
realizations and declared comparison support.  Historical 40x/70x claims do
not transfer to this family.

Report parameter and profile bias, interval coverage, flux recovery, inner-
beam projected-speed RMSE, R50 error or correct null rate for each eligible
truth, sigma recovery, support/missingness calibration, bound pressure,
failure rate, and runtime.  Retain the existing prospective gates: at least
10 percent lower projected-speed RMSE on both mock samplings with no material
KGAS066 regression; positive lower 95-percent bound for real grouped
visibility prediction versus KinMS; no more than 2 percent relative predictive
degradation versus the accepted kinUV baseline.  Image dipole amplitude in
km/s, moments, PVDs, and fixed-aperture spectral residuals are supporting
diagnostics, not a second objective or a promised residual elimination.

For NUTS, primary profile and geometry parameters retain R-hat <= 1.05 and
ESS >= 400 plus divergences, BFMI, and tree-depth checks.  For a weighted
nested sampler, report normalized weights, effective sample size per wall
hour, an independent replicate with Monte Carlo error, and posterior agreement
with NUTS.  No sampler comparison is accepted yet: the initial KGAS007 callback
retained the ring regularizer but misclassified it as likelihood, so its
same-prior fairness claim was invalid; the corrected fair KGAS007 comparison
is deferred.

## Measured prototype evidence

Durable evidence is in
`results/validation/unified-foundation-spikes-20260909/diagnostics/`.  It holds
13 files (1.23 MB): two contracts, two common profile archives, eight PNGs, and
one summary.  On the local lightweight environment the final complete two-target
self-check, adaptation, cube loading, and rendering took 11.238 s wall time.
Recorded component times were 0.523 s adaptation plus 5.002 s rendering for
KGAS066 and 0.493 s plus 2.964 s for KGAS007.  Visual inspection confirmed all
eight products are populated, both PVD axes are present, colorbars do not
overlap panels or each other, and KGAS007 alone carries its accepted
conditional covariance band.

These figures are a schema/render smoke test, not evidence for the proposed
new model.  The scratch spectrum currently verifies the integrated canonical
mask aperture.  Production integration still requires the central one-BMAJ
spectrum, explicit KinMS projected overlays, a clearly keyed projected
posterior envelope, and final tick/colorbar layout refinement.

The prototype source is
`experiments/unified_foundation/diagnostics_prototype.py`.  Its executable
self-check covers the R50 arctan identity and missingness cases and the runtime
validator checks version, required fields, shared shapes, and increasing
radius.  These are interface and scientific-invariant checks; no test freezes
documentation wording.
