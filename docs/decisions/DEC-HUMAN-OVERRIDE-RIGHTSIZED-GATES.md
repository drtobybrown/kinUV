---
id: DEC-HUMAN-OVERRIDE-RIGHTSIZED-GATES
status: accepted
date: 2026-09-07
authority: Project-PI
scope: S2-S5-crossdomain-recovery
---
# Human authority override: empirically right-sized gates

The Project PI directs that S2 through S5 acceptance measure scientific
sufficiency on interferometric data. A mathematically stricter threshold does
not control when it is disconnected from native resolution, calibration
uncertainty, thermal noise, or the promoted physical claim. This decision
supersedes conflicting thresholds in earlier cross-domain recovery records.

## S2 geometry and frame gates

Multi-start fitting passes when the top-ranked, lowest-chi-square solution
cluster reaches a common projected-kinematic result. Every initialization need
not converge to the same point. The evidence must retain all starts and expose
discrete PA symmetry, projected-speed/inclination covariance, local minima,
optimizer failures, and boundary pressure. A favorable start may not be hidden
or selected after changing the data.

Native observed frequencies remain the likelihood coordinates. A registered
TOPO-to-LSRK reporting conversion passes when its unmodeled time variation is
less than 1.0 km/s. Visibility resampling is neither required nor preferred at
that level. The full frame, conversion method, mean correction, and variation
must remain in configuration and run provenance.

## S3 posterior diagnostics

For posterior campaigns, R-hat <= 1.05 and ESS >= 400 apply to the primary
kinematic and geometric parameters: projected velocity amplitude or profile,
turnover scale, inclination, PA, systemic velocity, and velocity dispersion.
Peripheral nuisance parameters are reported and diagnosed, but their ESS does
not veto promotion unless their poor mixing changes a primary parameter or a
promoted prediction. This rule does not itself convert an unlicensed sampler
or uncalibrated interval into an accepted science product.

## S4 comparison gate

Cross-domain superiority requires both:

1. at least 10 percent lower projected-velocity RMSE against registered ground
   truth or rotation curves than the frozen stock KinMS comparator; and
2. improved overall reduced chi-square on the real KGAS066 and KGAS007
   benchmarks under the fair, common-data comparison contract.

Per-region residual dominance is diagnostic and is not an intersection gate.
Reviewers must examine flux conservation and may not reward stock KinMS
post-crop flux rescaling or low-SNR image-plane optimization as truth recovery.
KGAS066 non-regression remains required at the aggregate promoted-quantity
level.

## Review and implementation authority

The Implementer may right-size intermediate numerical tolerances when the
change is documented, preserves the physical claim, and demonstrates sub-1
percent empirical fidelity or a stated observation-limited tolerance. Reviewer
A and Reviewer B must reject gates that demand asymptotic float64 agreement,
universal multi-start identity, or local residual dominance without a physical
need. They review the implementation and evidence against the empirical claim.

Astra must incorporate this PI decision in later specifications and may not
reinstate a superseded gate without a new explicit PI directive.
