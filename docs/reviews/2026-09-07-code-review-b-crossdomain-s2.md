---
role: reviewer
seat: software-reproducibility
phase: implementation
date: 2026-09-07
reviewer: review-b-s2
canon_generation: 22
campaign_id: crossdomain-recovery-s2
proposal: docs/decisions/DEC-HUMAN-OVERRIDE-RIGHTSIZED-GATES.md
reviewed_commit: 8248a057113629b72aae069ae5ff831decefafdc
verdict: accept
---
# Independent code review B: cross-domain recovery S2

I reviewed exact commit `8248a057113629b72aae069ae5ff831decefafdc`
without reading or contacting Reviewer A. This verdict applies the binding PI
override in `DEC-HUMAN-OVERRIDE-RIGHTSIZED-GATES` and covers the generic S2
covariance and geometry implementation plus the three named validation trees.

## Attempted falsification

I checked that the fit runner refuses a dirty tree or a branch other than
`dev`, that target identity and scientific values enter through configuration,
and that the new `kinuv.infer.s2` interface contains no KGAS007/KGAS066 branch
or package default. I recomputed every manifest byte count and SHA-256 digest,
checked exact file-set coverage, verified all JSON products are ASCII-only, and
rehashed each geometry configuration, visibility input, and frozen covariance
record. The focused S2 geometry suite passed with `7 passed`; the complete CPU
suite passed with `256 passed, 8 skipped`.

I independently enumerated the fit records. Each target contains all 72 unique
fixed-turnover combinations (six registered turnover ratios times twelve start
IDs) and all twelve unique released-turnover fits. KGAS066 retains five failed
fixed fits and KGAS007 retains seven; every one records its iteration-limit
message, evaluation counts, raw and normalized projected gradients, parameters,
and boundary state. No failed fit was deleted. All twelve released-turnover
fits succeed for each target.

## Findings

1. **Verified:** Both geometry summaries identify clean `dev` commit
   `8248a057113629b72aae069ae5ff831decefafdc`. Their manifests name the same
   commit. The frozen covariance evidence identifies clean commit `40c04c8` and
   is bound into both geometry summaries by SHA-256.
2. **Verified:** The three manifests exactly cover their payload files. All
   recorded sizes and hashes match. The configuration and visibility hashes in
   both complete geometry checkpoints also match the current inputs.
3. **Verified:** Raw and normalized projected gradients are retained for every
   fit, and each normalized value equals the raw norm divided by the selected
   complex-visibility count. The best values are `7.0443e-5` for KGAS066 and
   `7.0031e-5` for KGAS007, below the PI gate of `1e-3`.
4. **Verified:** Checkpoint content is complete and restart-addressable by the
   unique fixed `(turnover ratio, start ID)` pair and joint start ID. Optimizer
   failures, boundary pressure, PA alternatives, and local minima remain
   visible in the final records. Both accepted solutions have no promoted
   boundary pressure.
5. **Verified:** `src/kinuv/infer/s2.py` exposes target-neutral covariance,
   parameter-chart, start-grid, objective, and fit interfaces. Target paths,
   beam values, frame corrections, turnover grids, and seeds remain in the two
   versioned target configurations.
6. **Advisory:** Checkpoint replacement uses a direct text write and the
   validation directories remain writable. The completed artifacts are
   internally intact and checksum-bound, so this does not invalidate S2.
   Before publication promotion, use atomic checkpoint replacement and seal the
   accepted evidence tree. Also bind the FITS brightness template and fit-window
   cube hashes directly into the geometry record rather than relying on their
   paths inside the hashed configuration.

## Gate assessment

The covariance record contains five populated, embargoed folds for each target,
the selected C1 parameters, whitening diagnostics, input hashes, and clean code
identity. Both geometry records pass all seven registered gates under the PI
override. The top likelihood cluster has eight mutually consistent starts for
KGAS066 and twelve for KGAS007. Best projected gradients pass after the required
per-selection normalization, turnover solutions are interior, promoted
boundary pressure is absent, and registered frame drift is below 1 km/s.

The evidence exposes all starts rather than requiring universal optimizer
identity. This is the acceptance rule mandated by the PI and preserves the
diagnostic failures needed for later S3/S4 assessment.

## Residual risks

The S2 solutions remain MAP geometry products. Their intrinsic circular speeds
are inclination-dependent diagnostics; projected velocity is the supported
quantity. The writable development artifacts and non-atomic checkpoint helper
are operational risks for later production promotion, as noted above, but no
manifest discrepancy or missing S2 fit is present in the reviewed dossiers.

## Verdict rationale

**Accept.** The generic implementation, complete fit matrices, clean exact-code
identity, failure visibility, ASCII records, reproducible input bindings, and
independently verified manifests satisfy the right-sized S2 software and
reproducibility contract. The full regression suite passes, and the retained
failure evidence shows that acceptance did not depend on hiding unfavorable
starts.
