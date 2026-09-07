---
role: reviewer
seat: science-numerics
phase: implementation
date: 2026-09-07
reviewer: review-a-s2
canon_generation: 22
campaign_id: crossdomain-recovery-s2
proposal: docs/decisions/DEC-KINUV-S1-CONTINUUM-AND-S2-CONTRACTS.md
reviewed_commit: 8248a057113629b72aae069ae5ff831decefafdc
verdict: accept
---
# Independent code review A: cross-domain recovery S2

I reviewed exact kinUV commit
`8248a057113629b72aae069ae5ff831decefafdc` independently. This verdict applies
the binding Project-PI override in
`DEC-HUMAN-OVERRIDE-RIGHTSIZED-GATES`. It closes the S2 provenance,
covariance, geometry, and optimizer gates. It does not license a posterior,
rotation-significance, KinMS-superiority, or production-promotion claim.

## Attempted falsification

I checked the two v2 visibility exports directly, recomputed their SHA-256
identities, audited the C0/C1 Gaussian likelihood and grouped fold construction,
and inspected all 72 fixed-turnover plus 12 released-turnover results for each
target. I tested whether the apparent convergence depended on one favorable
start, suppressed PA modes, normalized-away optimizer failure, parameter
boundaries, an unregistered frame approximation, or truncated LOSVD flux.

The complete kinUV suite passed with 256 tests and 8 skips. Every file listed
by the three S2 manifests matches its recorded byte count and SHA-256 digest.

## Findings

1. **Provenance and grouping — pass.** The canonical exports preserve native
   MS row identity, timestamps and intervals, observation/array/scan/state,
   field and data-description IDs, antenna pairs, UVW metres, ordered channel
   edges, flags, weights, TOPO frame, calibration and smoothing history, and
   XX/YY-to-Stokes-I lineage. They are unaveraged with unit source-row
   coefficients. Their independently recomputed hashes are
   `b5babc7541a200922eda168951d5645d7fae9a7b8539659d2c52eb9929fad37c`
   for KGAS066 and
   `f755be0c4208b96c8667ca859238ffa96c263226d0304eecd8321017cab8cf5f`
   for KGAS007. Five populated folds contain eight and nine indivisible
   scan/time blocks, respectively, and training applies the recorded 12.096 s
   boundary embargo before any fitting aggregation.

2. **C0/C1 selection — pass.** Both real and imaginary components and the
   covariance log determinant enter the profiled predictive likelihood. C1 is
   preferred in every held-out group. Mean C1 advantages are `0.0926557` and
   `0.0926574` log-likelihood units per complex visibility, versus standard
   errors `5.84e-5` and `3.60e-5`. Native fitted correlations are `0.297702`
   and `0.297688`; propagation through four-channel software bins gives
   `0.0959301` and `0.0959238`. Whitened component means and variances pass,
   and the maximum residual spectral lag-one correlations are `0.02922` and
   `0.02930`, below the `0.05` empirical limit.

3. **Frame and beam contracts — pass.** Independent casacore conversion from
   the extracted Measurement Sets reproduced the registered TOPO-to-LSRK mean
   frequency-equivalent corrections and peak-to-peak drifts: `10.98507` and
   `0.037134 km/s` for KGAS066; `-16.78748` and `0.064040 km/s` for KGAS007.
   The likelihood therefore remains correctly on native TOPO frequencies, and
   both drifts are far below the PI-authorized `1 km/s` reporting threshold.
   BMAJ, BMIN, and BPA values reproduce the official diagnostic-cube FITS
   headers exactly.

4. **Projected parameterization — pass.** The optimizer fits
   `u=v_c sin(i)` in dimensionless coordinates with an isotropic prior in
   `cos(i)`. Intrinsic `v_c` is retained only as a diagnostic. The forward
   amplitude uses `v_c=u/sin(i)` while inclination remains available for disk
   deprojection. Flux and dispersion are logarithmic; turnover and offsets are
   scaled by BMAJ; systemic velocity is scaled by fitted channel width; PA is
   periodic. No dark-matter quantity enters the objective.

5. **Multi-start geometry — pass.** Each target contains six registered
   turnover ratios with 12 starts per ratio, followed by 12 released-turnover
   fits. KGAS066 has eight mutually consistent solutions in its top likelihood
   cluster; four distant PA-symmetry/local-minimum solutions and all associated
   boundary pressure remain in the record. KGAS007 has consensus across all 12
   released fits. The accepted solutions are interior: KGAS066 has
   `u=184.003 km/s` and `r_t/BMAJ=0.28619`; KGAS007 has `u=96.263 km/s` and
   `r_t/BMAJ=0.42123`. Neither selected turnover lies at an audit endpoint,
   and neither selected solution has boundary pressure.

6. **Gradient accounting — pass.** The evidence preserves raw projected
   gradient infinity norms of `5.89576` and `4.21533`. Division by the exact
   retained complex-visibility counts, 83,695 and 60,192, gives `7.044e-5`
   and `7.003e-5`. I verified this relationship for every fixed and released
   fit. Both satisfy the PI-authorized `g_P_inf/N_complex <= 1e-3` gate; the
   objective and fitted optimum are not rescaled in the saved likelihood
   accounting.

7. **LOSVD support — pass under the PI fidelity rule.** The retained spectral
   windows do not provide a literal ten-sigma margin at every asymptotic disk
   velocity. A direct emission-weighted Gaussian tail calculation on the
   accepted models gives missing integrated fractions `4.76e-5` for KGAS066
   and `1.36e-13` for KGAS007. These are well below one percent and cannot
   affect the promoted projected-kinematic result at observational precision.
   No window renormalization is applied.

8. **Advisory — structured covariance diagnostics remain a later claim
   boundary.** The S2 dossier establishes grouped predictive C1 selection and
   spectral whitening. Baseline-, antenna-, time-, and real/imaginary
   cross-correlation diagnostics should accompany the outer-fold S4 dossier
   before calibrated predictive or superiority claims. This is not a blocker
   for the present full-data S2 geometry result because no calibrated interval
   or held-out superiority claim is being promoted here, and every group
   independently favors C1 by a wide margin.

## Gate assessment

All S2 gates are measurable and satisfied under the Project-PI override. The
evidence retains all starts, optimizer statuses, raw and normalized gradients,
local minima, PA symmetry, inclination covariance, and boundary states. The
accepted likelihood cluster, rather than universal start identity, controls
the multi-start gate. The artifacts are bound to exact code, configuration,
covariance evidence, and visibility hashes.

## Residual risks

The fixed two-dimensional brightness template can still couple morphology to
inclination and turnover. S3 must test the registered joint-emissivity and
supported-ring ablations before interpreting that structure physically.
Inclination is not independently identified here, so only projected velocity
is eligible for promotion. S4 still requires valid outer-fold refits and the
PI's KinMS comparison and KGAS066 non-regression gates.

## Verdict rationale

**Accept.** The S2 implementation is physically consistent, numerically
traceable, and sufficient for its limited claim. Both targets select the same
well-behaved covariance family, converge to interior projected-kinematic
solutions from independent starts, preserve failed basins rather than hiding
them, and pass the empirical gradient and frame gates established by the PI.
