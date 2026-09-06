---
reviewer: reviewer-a-science-numerics
proposal_commit: e7a1e71eba9ffe68a21754bb41dc9bb03d6c2acc
verdict: accept-with-required-changes
date: 2026-09-06
independent_of_reviewer_b: true
---
# Reviewer A: S1 continuum and S2 contracts

The continuum-adapter strategy, analytic LOSVD, direct continuous reference,
isotropic-orientation fallback with projected-velocity-only promotion, and
unchanged S4 boundary are scientifically appropriate. The failed `r2` evidence
is valid; all 76 manifest payload hashes and sizes were independently checked.

Implementation is blocked until the proposal defines:

1. S1 residual signs, the stacked real/imaginary covariance inner product,
   direct chi-square identity, named S0 covariance, and the exact rule requiring
   every target/axis and two successive factor-two refinements to pass.
2. The B-spline pixel-unit conversion `h^2 K_h=B3 B3`, the pre-normalization
   flux identity, and spatial/spectral/joint exclusion accounting.
3. The heteroscedastic `C0`/`C1` formulas, real/imaginary convention, flags and
   gaps, response propagation, positive-definiteness floor, estimators, and
   one-standard-error calculation.
4. Endpoint-safe `cos(i)` computation, a line-of-sight implementation that
   never divides by `sin(i)`, boundary reporting, and transform Jacobian.
5. A preregistered twelve-start design and exact mapping from fixed-turnover to
   released-turnover starts.

The verified BMAJ values are `1.254195896 arcsec` for KGAS007 and
`1.040022196 arcsec` for KGAS066.
