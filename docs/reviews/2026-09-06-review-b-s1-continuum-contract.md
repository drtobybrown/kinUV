---
reviewer: reviewer-b-software-reproducibility
proposal_commit: e7a1e71eba9ffe68a21754bb41dc9bb03d6c2acc
verdict: accept-with-required-changes
date: 2026-09-06
independent_of_reviewer_a: true
---
# Reviewer B: S1 continuum and S2 contracts

The adapter is honestly labeled, isolated from kinUV runtime, and responsive
to the sealed `r2` failure. All 76 dossier payload hashes and manifest SHA-256
`97b4d4f526680fbfaad9fbb47147ab47aef8d9803cf923a4123b5c7cc674c45d`
were independently verified.

Implementation is blocked until the proposal defines:

1. A new machine-readable adapter schema with Jy/pixel values, channel
   edges/order/frame, kernel and quadrature settings, support bounds, source
   hashes, operator flags, and an analytic-versus-cube flux-ledger identity.
2. The disposition of every old S1 gate plus norms, selection, covariance,
   residuals, frozen parameters, refinement levels/phases, aggregation, and a
   corrected-baseline source for actual-data sensitivity.
3. Checksum-bound campaign configuration containing target priors, bounds,
   turnover grid, starts, gates, and all beam values and input hashes.
4. A real-S2 block for both targets: neither retained visibility export meets
   the new grouping/provenance contract.
5. Executable covariance/fold/whiteness definitions, finite inclination bounds,
   periodic-PA projection, prior/Jacobian convention, exact starts, optimizer
   budgets, and fail-closed behavior.

The resolved decision must also close the superseded escalation record.
