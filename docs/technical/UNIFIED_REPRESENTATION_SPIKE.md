# Unified smooth radial representation spike

**Date:** 2026-09-09
**State:** prototype evidence; no production model, prior, inference, or product change
**Code base:** `dev` at `71b9d9e0d55df19bdb6ac2b0fb475d4eeea30142`, plus the uncommitted files named below

## Decision

Keep the regularized cubic P-spline as the preferred physical representation,
but do not hand its present carrier-plus-shape chart to a production sampler.
On the nonmonotonic KGAS007 parent profile its unconstrained profile fit drove
the carrier scale to `0.00069"` while compensating with large shape latents;
that is direct evidence of a scale/shape degeneracy.  Orthogonalize the spline
shape modes against amplitude and carrier-scale derivatives, remove the free
carrier scale, or use another explicit construction that removes the redundant
freedom, then repeat the same-dimension comparison.  Orthogonalization is one
implementation option, not a frozen algorithm or proof of global
identifiability.  The fixed-kernel finite GP showed no comparable collapse in
this profile-only spike and is therefore the more stable runnable reference for
the next bounded test.  Posterior conditioning and Hessian rank remain
unmeasured for both families.  This is an architectural recommendation, not
approval for production inference.  Before either is
eligible, replace the prototype's index-space second-difference penalty with an
integrated or spacing-aware curvature penalty in dimensionless radius and
select its fixed strength using training folds only.  No candidate justifies a
second Fourier/likelihood engine.

Do not carry the present two-mode global dispersion spline into production
unchanged.  It recovers a constant exactly, but approximates the retained
KGAS066 sharp two-zone profile with RMS `0.51 km/s` and maximum error
`1.09 km/s`.  Because that earlier dispersion extension gained about 154
visibility chi-square, the loss could matter.  Use one common target-neutral
log-dispersion spline with automatic beam/emission-scale internal nodes and the
same active deviation count for both galaxies.  All zero deviations must still
give an exactly constant dispersion.  Choose its node rule and shrinkage on
training folds, never by target-specific branching or scored data.

## Common chart and profile contract

Both candidate families have the same 15 unconstrained coordinates, in the
same order, for both KGAS066 and KGAS007:

`log_flux, pa_rad, vsys_channel_offset, dx_over_bmaj, dy_over_bmaj,`
`cos_inclination_logit, log_u_reference_kms,`
`log_rotation_scale_over_bmaj, rotation_shape_1..4, log_sigma0_kms,`
`dispersion_deviation_1..2`.

The observable amplitude is projected speed at the automatically selected,
emission-supported `Rref=R70`; it is not an asymptotic velocity.  Inclination
is needed for spatial deprojection, and intrinsic speed is derived as
`Vc=u/sin(i)`.  The prototype exposes chart-to-physical and inverse transforms,
the chart Jacobian, separate log likelihood/prior/posterior functions, and a
JIT value-and-gradient callable.  The rotation-scale Jacobian includes its
third `BMAJ` factor in addition to the two positional factors.  Its physical
priors are provisional and conditional; they must be frozen and independently
reviewed before inference.  The unwrapped PA currently has a broad proper
normal prior around the parent mode, so this is explicitly a local PA chart;
production needs a normalized periodic prior or explicitly bounded periodic
chart.

For both families,

\[
u(R)=u_{\rm ref}
 {\arctan(R/R_s)\over\arctan(R_{\rm ref}/R_s)}
 \exp\{\boldsymbol\phi(R)^T\mathbf z\}.
\]

Every one of the four shape coordinates remains active.  The arctangent is an
everywhere-smooth carrier and coordinate choice rather than a fallback branch.
It makes `u(0)=0`, gives a finite central `u/R`, and guarantees that a sub-beam
turnover is representable without a `0.5 BMAJ` floor.  The exponential shape
factor keeps the rotation direction positive.  At `R=0`, the forward model
sets the azimuthal projection to zero, so no `0/0` enters the line-of-sight
velocity.  The implementation leaves the analytic carrier unmasked at zero so
autodiff retains this finite central slope rather than differentiating through
an exact-zero `where` branch.

The P-spline uses six raw clamped cubic B-spline coefficients projected onto a
four-dimensional subspace.  One constraint removes the constant mode already
absorbed by `u_ref`; the other makes the shape correction's outer derivative
zero.  It is C2 at internal knots and C1 at the constant-extension endpoint;
the smooth carrier continues outside the reported support, so velocity is not
forced flat at the last data node.  The prototype precision is
`D2.T D2 + 0.05 I`.  The ridge makes the second-difference nullspace proper,
but raw index differences across unequal physical knots are not a production
curvature prior.

The finite GP is a rank-four squared-exponential Nyström basis with inducing
locations and fixed correlation length derived from the model cell, beam, and
emission quantiles.  Subtracting its value at `Rref` preserves the exact
amplitude interpretation.  It is smooth at every finite radius and uses four
standard-normal whitened latents.

Dispersion is

\[
\log \sigma(R)=\log\sigma_0+\boldsymbol\psi(R)^T\mathbf d.
\]

The cubic basis is C2 internally and C1 at its outer extension.  `d=(0,0)` is
exactly the constant-dispersion model, with no discrete switch and no positive
amplitude funnel.

The neutral profile artifact schema contains `radius_arcsec`,
`u_projected_kms`, `vc_intrinsic_kms`, `sigma_kms`, their three radial
derivatives with units and method, geometry and conditionality, beam/emission
support, provenance, and explicit flags.  Posterior bands remain absent in
this spike.  `R50` is deliberately `UNASSESSED_PLATEAU_IDENTIFIABILITY`: it may
be reported only as the first rising half crossing of a constrained plateau or
outer reference.  It is exactly `rt` for an arctangent; unresolved, nonunique,
or outside-support crossings require flags rather than a surrogate arctangent
fit.

## Automatic radial design

The beam is used as a scale and never as a resolution cutoff.  The inner node
is `max(2*cell, min(0.25*BMAJ, 0.5*R20))`; the outer reporting radius is
`max(R95+0.5*BMAJ, 2*BMAJ)`.  The actual grids were:

| Target | fit shape | cell | BMAJ | Rref=R70 | R95 | inner node | outer |
|---|---:|---:|---:|---:|---:|---:|---:|
| KGAS066 | 881 x 95 | 0.2870" | 1.0400" | 7.1748" | 11.5091" | 0.5740" | 12.0291" |
| KGAS007 | 912 x 66 | 0.2877" | 1.2542" | 3.4171" | 5.3615" | 0.5754" | 5.9886" |

The `2*cell` term, rather than the beam, controls these particular inner nodes.
The smooth carrier supplies the smaller free turnover scale; it recovered an
analytic `rt=0.25"` profile exactly in the profile-only check.  A future node
rule should test a finer inner basis if non-carrier core departures are the
scientific target.

## Measured evidence

The basis implementation was compared against SciPy's cubic `BSpline` on and
beyond its endpoint.  Maximum absolute disagreement was `1.11e-16`; partition
of unity was one at all test radii.  The constant-mode and outer-derivative
constraint residuals were `4.72e-16` and `1.11e-16`.  These are local analytic
checks, not real-data likelihood measurements.

All real-data evaluations used the unchanged production visibility operator,
the same cells and fixed accepted emissivity per target, and the frozen C1
AR(1) covariance.  KGAS066 used propagated `(scale,rho)=(1.9964079,0.0959301)`;
KGAS007 used `(2.0351473,0.0959238)`.  There was no image likelihood, mass
model, physical-radius grid, or secondary Fourier transform.

The completed corrected pilots measured:

| Target/family | compile + first value/grad | warmed value/grad median | 3-step chi2 start -> end | gradient / complex | state |
|---|---:|---:|---:|---:|---|
| KGAS066 P-spline | 5.45 s | 0.564 s | 177665.16 -> 175081.76 | 0.0492 | iteration limit |
| KGAS066 finite GP | 4.52 s | 0.539 s | 174329.20 -> 171155.14 | 0.0717 | iteration limit |
| KGAS007 P-spline | 5.16 s | 0.417 s | 109853.24 -> 107232.81 | 0.0224 | iteration limit |
| KGAS007 finite GP | 4.14 s | 0.441 s | 106510.68 -> 106214.49 | 0.00850 | iteration limit |

These are three-iteration profile pilots.  They are neither converged MAPs nor
fair likelihood rankings against the accepted parent or each other.  The
finite-GP first-derivative jump diagnostic was `4.34e-6` and `4.02e-5
km/s/arcsec` at the reporting edge.  Corrected P-spline maximum numerical
first-derivative jumps were `0.0141` and `0.0305 km/s/arcsec`, at the innermost
knot where slopes are hundreds of `km/s/arcsec`; endpoint jumps were only
`1.04e-4` and `4.41e-4 km/s/arcsec`.  All profiles had exactly zero projected
speed at the origin and finite `u/R`.  On the audited current source, JAX's
origin derivative agreed with the small-radius `u/R` limit to relative
`4.36e-8` (P-spline) and `8.79e-9` (finite GP).

Profile-only fits used 256 radii and the same four shape latents.  Both families
fit the analytic sub-beam arctangent (`rt=0.25"`) to floating-point error because
it is exactly the shared carrier; this checks scale access but is not an
independent flexibility test.  The GP fit to KGAS007's retained nonmonotonic
four-knot curve had full/inner-beam RMSE `1.82/2.58 km/s` and maximum error
`5.84 km/s`.  The P-spline fit had full/inner-beam RMSE `2.18/2.57 km/s` and
maximum error `5.83 km/s`, but exposed the scale collapse described above.
KGAS066's retained parent is arctangent and was reproduced to floating-point
error.  A non-carrier sub-beam injection is still required to measure
basis-induced core bias.

The first attempt evaluated the outer
endpoint indicator before completing the Cox-de Boor recursion, causing a
false endpoint derivative jump and then a GP bookkeeping failure.  Those
sessions are retained as failed evidence.  The corrected source passed the
SciPy comparison above before retry dispatch; both corrected P-spline retries
succeeded.

## Execution record

Flexible headless sessions used `skaha/astroml:latest`, no CPU or memory flags,
CPU JAX with x64, one BLAS/XLA thread, node-local `/scratch` caches/logs, and
atomic bounded results under
`/arc/projects/KILOGAS/analysis/toby_sandbox/results/incoming/`.

Initial failed sessions (preserved): `y5tz7vy6` (KGAS066) and `scq5r2v0`
(KGAS007), both failing because `USER` was unset in the headless environment.
The next sessions `cspzow89` and `u1i009p9` preserved the invalid first
P-spline case and failed before GP on the endpoint bookkeeping defect.  GP-only
retries `c3imwg8o` and `a08nvjwa` succeeded.  Corrected P-spline retries
`eh2awk3f` and `rszm087s` also succeeded.

Submission command pattern:

```bash
/usr/bin/python3 /arc/home/thbrown/.local/bin/canfar create headless \
  skaha/astroml:latest --name NAME -- \
  /bin/bash /arc/projects/KILOGAS/analysis/toby_sandbox/kinUV/experiments/unified_foundation/representation_headless.sh \
  TARGET /arc/projects/KILOGAS/analysis/toby_sandbox/results/incoming/unified-representation-spike-20260909 \
  3 FAMILY
```

The prototype entry point is
`experiments/unified_foundation/representation_model.py`; the bounded worker
and headless wrapper are the sibling `representation_worker.py` and
`representation_headless.sh`.  The sampler spike can reuse
`build_target_context`, `PARAMETER_NAMES`, `initial_z`, `z_to_physical`,
`physical_to_z`, `log_abs_det_jacobian`, `log_likelihood`, `log_prior_chart`,
`log_posterior`, and `value_and_grad`.  Until these uncommitted files receive a
commit, every consumer must bind both the base commit above and the exact dirty
diff; no result may claim an implementation SHA for the prototype.

The headless wrapper read the shared uncommitted repository rather than a
frozen source snapshot, and the worker did not record source digests.  The four
successful sessions imported the revision immediately before the final audit
fixes to the proper local PA prior, the missing constant `log(BMAJ)` in the
separately exposed physical Jacobian, and the origin autodiff branch.  The
Jacobian was not used by the short optimizer, and the reported chi-square is
likelihood-only.  The omitted PA term still makes the three-step trajectories
local timing/feasibility evidence rather than valid posterior objectives.  The
current source contains all three fixes and has the analytic derivative check
reported above; exact-current-source C1 scores remain unmeasured.  This
provenance limitation is why the table cannot be promoted or treated as an
exact benchmark of a future commit.

## Remaining gates

The following are unmeasured: exact-current-source and converged same-start MAP chi-square for both
families and targets; held-out selection of common rotation/dispersion
shrinkage; a non-carrier sub-beam shape injection; Hessian rank and the
`log_u_ref`/`log_Rs` scale degeneracy; posterior geometry correlations; plateau
identifiability and R50; derivative uncertainty; and sampler convergence.
Production inference, promotion, and scientific superiority claims remain
ineligible.
