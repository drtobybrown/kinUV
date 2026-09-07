---
id: DEC-PI-S4-STANDARD-USE-BENCHMARK
status: accepted
date: 2026-09-07
authority: Project-PI
scope: S4-standard-use-comparison-and-synthetic-truth
supersedes: S4-training-fold-CASA-reimaging-requirement
---
# DEC-PI-S4-STANDARD-USE-BENCHMARK: Standard-use S4 evaluation

**Status:** Accepted by direct Project PI authority, 2026-09-07  
**Supersedes:** the training-fold CASA reimaging requirement in
`DEC-KINUV-S4-SCIENTIFIC-RECOVERY`

## Decision

S4 evaluates each method in its normal scientific domain:

* Stock KinMS is frozen as the external image-plane baseline and is evaluated
  from the canonical, pipeline-delivered science cube made from the complete
  calibrated observation.
* kinUV is fitted to calibrated visibility tables and evaluated by prediction
  on disjoint, provenance-complete visibility groups under the selected C1
  covariance model.
* Controlled projected-velocity recovery uses Python-native simulations for
  which the analytic three-dimensional sky cube and corresponding Fourier
  visibilities share one declared truth. CASA is not used in simulation.

No S4 gate requires `tclean`, a CASA runtime, or imaging of training-only
Measurement Set folds. The abandoned CASA experiment produced no science
products and is not part of the evidence chain. CASA remains outside kinUV;
Measurement Set ETL remains the responsibility of the standalone `ms2kinuv`
companion.

## Real-data predictive gate

The retained grouped audit is deliberately conservative. kinUV is refitted on
each training fold and scored only on its held-out native rows. The frozen
stock KinMS fit saw the complete canonical cube, including information derived
from those held-out rows, before its intrinsic continuum rendering was sampled
and scored on the same rows. A positive lower 95 percent bound for
`chi2_KinMS - chi2_kinUV` per real visibility component therefore establishes
the required standard-use predictive advantage despite favoring KinMS.

The immutable audit at
`results/validation/crossdomain-recovery-s4-remediation-20260907-r2/grouped/`
passes on every fold. Aggregate lower 95 percent bounds are +0.03178592 for
KGAS066 and +0.00290304 for KGAS007 chi-square per real component. Reimaging
the training folds cannot be reintroduced as a promotion prerequisite.

## Synthetic recovery gate

The synthetic evaluation must report projected speed
`u(r) = v_c(r) sin(i)` on a common supported radial grid. For each canonical
target sampling, kinUV must have at least 10 percent lower paired RMSE than
stock KinMS. The injection, analytic cube, and exact model visibilities must be
generated in Python. Random seeds, parameter support, noise, masks, inputs,
code commit, and file checksums are retained in the final dossier.

## Consequences

Restored-cube residuals, moment maps, spectra, channel maps, and PVDs remain
valuable supporting diagnostics on real targets. They do not replace known
truth or held-out visibility prediction and cannot veto S4 by themselves.
The earlier r3 failure and the repaired replay remain preserved as diagnostic
history.
