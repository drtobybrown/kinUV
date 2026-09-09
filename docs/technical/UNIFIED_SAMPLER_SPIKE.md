# Unified bounded-sampler spike

**Date:** 2026-09-09
**Code baseline:** `71b9d9e0d55df19bdb6ac2b0fb475d4eeea30142` plus the isolated
`experiments/unified_foundation/sampler*` spike files
**State:** benchmark incomplete; no production launch, promotion, or posterior
claim
**Durable evidence:**
`/arc/projects/KILOGAS/analysis/toby_sandbox/results/incoming/unified-sampler-spike-20260909/`

## Decision

Keep preconditioned NUTS as the provisional sampler for a future smooth unified
representation.  Do not replace it with dynamic nested sampling on present
evidence.  The current experiment is intentionally too short to establish
posterior fidelity, R-hat, ESS, evidence accuracy, or an ESS/hour winner.

Dynamic nested sampling remains useful as a bounded independent mode/evidence
audit once the unified representation exposes a normalized proper prior
transport.  Its full broad-prior initialization and constrained-draw calls must
be charged to runtime.  A MAP-conditioned or shrunken prior is not an eligible
speed comparison.

This recommendation is provisional.  KGAS066 is already a smooth arctan
rotation curve with a smooth tanh two-zone dispersion profile, so its observed
tree-depth saturation cannot be blamed on piecewise ring kinks.  The finite
MAP curvature is positive definite in the tested chart, but that also does not
identify a unique cause.  The unified spline representation must be timed
directly rather than assumed to cure the existing geometry.

## Posterior identity used by the spike

The baseline experiment reuses the production visibility forward model and
does not implement another Fourier engine.  Both sampler paths target the same
KGAS066 probability measure:

\[
  \log p(q\mid d) = \log L_{\rm vis,C1}(d\mid q) + \log p(q) + C.
\]

`log L` is solely the correlated complex-visibility likelihood with the frozen
C1 covariance.  The prior is uniform in the registered bounded physical
coordinates except for independent, properly normalized, bound-truncated
`N(0, 0.5 arcsec)` centre offsets.  Frozen emissivity weights are not sampled.
KGAS066's selected arctan chart has no ring smoothness penalty.

For NUTS, the existing nonlinear `CampaignTransform` maps bounded coordinates
to unconstrained `y` and contributes its nonconstant Jacobian.  The spike then
uses one affine whitening only,

\[
  y = y_{\rm MAP} + Lz, \qquad LL^T = C_y \simeq H_y^{-1}.
\]

The constant affine `log|L|` is recorded separately from the nonlinear chart
Jacobian.  It can be omitted from HMC dynamics but must not be confused with
the nonlinear change-of-measure term.

For dynesty, the prior transform maps the unit cube directly to the same
bounded physical prior and the callback returns only `log L_vis,C1`.  Posterior
sample calculations use normalized nested weights
`exp(logwt - logsumexp(logwt))`; raw nested samples are not equally weighted.

KGAS007 is deliberately not compared with dynesty on this historical chart.
Its selected `supported_rings` model has a coupled, bounded ring smoothness
prior.  Moving that penalty into the likelihood would violate the common-prior
contract, while a MAP-local prior would make the comparison artificially easy.
The first two KGAS007 spike sessions were stopped as soon as this was detected,
before any result was accepted.  `same-prior dynesty KGAS007 = NOT_MEASURED`.
The unified representation's planned proper Gaussian coefficient priors should
make an exact prior transform straightforward; compare both targets there.

The representation worker subsequently exposed the 15-dimensional prototype
at `experiments/unified_foundation/representation_model.py`, with separate
likelihood, prior, posterior, Jacobian, and value-and-gradient interfaces.
That callable arrived after this controlled sampler pair began, so every timing
in this report remains a historical-chart baseline and is not evidence about
the unified chart.

## Dense metric convention

NumPyro's source implements kinetic energy as
`K(p) = 0.5 * p.T @ inverse_mass_matrix @ p`.  Therefore its
`inverse_mass_matrix` argument is mathematically `M^-1`.  Near a quadratic
posterior with curvature `H` and covariance `C = H^-1`, choosing `M = H`
requires passing `M^-1 = C`.  The existing runner's inverse-Hessian object is
thus oriented correctly despite its confusing variable name.  In the explicitly
whitened `z` chart, the correct initial inverse mass is the identity; passing
`C` again would whiten twice.

The production helper currently regularizes curvature with
`abs(eigenvalues)`.  That could silently turn a materially negative-curvature
direction into a positive one.  The spike instead records negative eigenvalues
and floors eigenvalues below `max(abs(lambda))*1e-6` without absolute-value
folding.  No material negative eigenvalue occurred for the KGAS066 local probe,
so this is an audit risk rather than the observed saturation cause.

References: the official [NumPyro MCMC API](https://num.pyro.ai/en/stable/mcmc.html)
documents the dense inverse-mass input; the installed NumPyro 0.21 source gives
the kinetic-energy multiplication used above.  The official
[dynesty API](https://dynesty.readthedocs.io/en/stable/api.html) states that
initial live points must come from the prior and that `rslice` performs repeated
slice updates; the [dynesty quick start](https://dynesty.readthedocs.io/en/stable/quickstart.html)
defines `logwt`, `logz`, and importance-weighted posterior use.

## Live KGAS066 audit

No existing worker was stopped or modified.  At the 20:24 UTC audit:

- Four original attempt-4 workers remained in warm-up at step 100 with no
  durable warm-up checkpoint or draw.
- Four attempt-5 dense-MAP-metric workers had completed 200 warm-up iterations
  and 100 draws each.  Their draw archives contained mean leapfrog counts
  `249.56`, `251.16`, `250.52`, and `255.00`; `98`, `98`, `99`, and `100` of
  100 draws reached 255 steps, the sampling cap for depth 8.  All 400 retained
  draws reported zero divergences.
- The status field `current_num_steps=127` is the last warm-up value and is not
  the retained-draw maximum.  It reflects the separate warm-up depth-7 cap.
- Attempt-5 reported positive curvature eigenvalues from `593.862` to
  `3,608,362.342`, condition number `6076.09`, in all four chains.

Those 400 draws are still an unfinished campaign.  Their zero divergences do
not override the near-universal depth saturation, and no R-hat or ESS claim is
made from them here.

## Measured local objective cost

These local probes are real fixed-C1 likelihoods on the historical charts.
They are useful for call budgeting but are not CANFAR throughput.

| Quantity | KGAS066 | KGAS007 |
|---|---:|---:|
| Dimension | 10 | 11 |
| Load/build | 23.277 s | 16.508 s |
| First jitted value call | 2.054 s | 1.538 s |
| First separately jitted value-and-gradient call | 3.084 s | 2.968 s |
| Warmed value, median of 2 | 0.4121 s | 0.3041 s |
| Warmed value-and-gradient, median of 2 | 0.8973 s | 0.6986 s |
| Central-difference curvature | 17.212 s (20 gradient calls) | 14.592 s (22 gradient calls) |
| Peak process RSS | 2458.1 MiB | 2443.1 MiB |
| Curvature eigenvalue range | 593.862 to 3,608,362.342 | 94.408 to 89,473.375 |
| Curvature condition number | 6076.09 | 947.73 |

Compilation is reported separately from setup and warmed evaluation.  The two
first-call entries compile separate jitted functions, so they should not be
added as a projection for one executable that compiles only its selected path.

## Controlled headless probes

All submitted sessions are flexible CANFAR headless sessions with no CPU or RAM
flags.  The valid KGAS066 pair uses 20 broad-prior live points and at most two
initial dynamic-nested iterations with `sample='rslice'`, two slice updates,
and an 80-call cap; the NUTS peer uses 8 warm-up plus 8 retained iterations and
depth 6.  These are timing probes, not posterior runs.

| Target | Sampler | Session | Disposition |
|---|---|---|---|
| KGAS066 | DynamicNestedSampler `rslice` | `fobpqe5x` | completed exit 0; startup plus three forced-bound constrained replacements |
| KGAS066 | affine-whitened NUTS | `xzhdx43q` | stopped at 10-minute bound; no result record |
| KGAS007 | DynamicNestedSampler `rslice` | `liip4cd7` | stopped; historical coupled prior was not yet transported |
| KGAS007 | affine-whitened NUTS | `jn5vzhrl` | stopped with its invalid peer; no head-to-head result |
| KGAS066 | first draft dynesty/NUTS | `oioo21m8`, `mszhfro4` | stopped and superseded so dynesty uses the jitted value callback |
| KGAS066 | second dynesty draft | `drkwsmxa` | calls finished, then checkpoint pickle failed; no accepted metrics |
| KGAS066 | third dynesty draft | `k77cot6e` | completed, then superseded because the tiny run did not force the first bound update |

The valid dynesty run separated 64.210 s setup/load, 2.087 s first jitted value,
3.067 s first separately jitted value-and-gradient, 13.952 s curvature, and
12.572 s sampler execution, for 102.807 s total in-session elapsed.  Warmed
value and value-and-gradient medians were 0.2969 and 0.6678 s; peak RSS was
2433.8 MiB.  Queue delay was not instrumented and is `NOT_MEASURED`.

The dynesty run used 20 broad-prior live points, forced the first bounding
update (`min_ncall=0`, `min_eff=101`), and made 8 additional measured likelihood
callback calls while replacing three dead points.  Thus it exercised bounded
`rslice`, but only for three constrained replacements.  It produced 23 stored
points with weighted ESS exactly 1.0 and did not reach its stopping criterion.
The callback counter recorded 28 actual Python likelihood invocations while
dynesty's nested accounting reported 44 calls; both raw values are preserved.
`maxbatch=0` means no dynamic allocation batch was attempted.
The reported log-evidence and error are invalid as scientific estimates.  This
measures startup and a few constrained callbacks; it does not measure dynamic
allocation efficiency or posterior exploration.

The valid NUTS session showed active compute at about 3.25 cores and 7.20 GB
RAM near minute nine, but its zero-byte log did not reveal whether it was in
compilation, curvature construction, initialization, or adaptation.  It was
stopped at the declared 10-minute wall bound with no atomic result, completed
draw, or `num_steps` record.  Its sampler runtime and contemporaneous ESS/hour
are therefore `NOT_MEASURED`.

There is no head-to-head winner.  The historical attempt-5 NUTS draws establish
that the current preconditioned chart still saturates depth, while the tiny
dynesty run establishes only broad-prior startup and three constrained-replacement
costs.  Neither establishes a faithful posterior, stable primary profiles, or
effective samples per run hour.

## Practical next experiment

Run the same contract on each runnable unified representation with four short,
independent NUTS chains and one broad-prior dynesty audit per target.  Allocate
enough iterations to estimate rank-normalized split R-hat and weighted profile
stability, but retain the existing production gates: primary R-hat at most
1.05 and bulk/tail ESS at least 400.  Report all failures and modes.  Choose a
sampler only from effective primary-profile samples per compute/run wall hour.
Report scheduler queue time separately, and also report total time-to-result
because it matters operationally.  Compute/run time includes load, compile,
curvature, warm-up, prior live-point initialization, and constrained nested
calls.

The output adapter should retain target, representation family, parameter
names, sample values, nested log/normalized weights (or unit MCMC weights),
sampler state, conditionality, prior and covariance provenance, queue/setup/
compile/run times, call counts, memory, divergences and `num_steps`, and explicit
missing reasons for unavailable profile bands or derived radii.  Pilot samples
must remain labeled `posterior_fidelity: unresolved`.
