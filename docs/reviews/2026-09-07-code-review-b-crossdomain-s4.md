---
role: reviewer
seat: software-reproducibility
phase: implementation
date: 2026-09-07
reviewer: review-b-s4
canon_generation: 24
campaign_id: crossdomain-recovery-s4
proposal: docs/decisions/DEC-HUMAN-OVERRIDE-RIGHTSIZED-GATES.md
reviewed_commit: 5d08507d2b9fa048a7760d57eedd7cbff722238c
verdict: accept-failed-gate-evidence
---
# Independent code review B: failed cross-domain recovery S4 gate

I reviewed exact clean implementation commit
`5d08507d2b9fa048a7760d57eedd7cbff722238c` and the r3 evidence without
reading or contacting Reviewer A. This verdict accepts the integrity of the
failed result; it does not promote S4 or authorize S5.

## Attempted falsification

I independently rehashed the complete artifact and all seven declared source
inputs for each target. I recomputed the common-cube chi-square, degrees of
freedom, reduced chi-square, projected-velocity RMSE, eligibility, gate
booleans, and top-level acceptance from the retained FITS and NPZ products.
I also checked the clean-commit guard, manifest publication order, explicit
nonzero failure exit, and the absence of a subsequent S5 artifact or commit.

The focused S4/comparator/architecture suite passed with `18 passed`. The
complete CPU suite passed with `263 passed, 8 skipped`.

## Findings

1. **Verified:** `summary.json` and `MANIFEST.json` identify clean `dev` commit
   `5d08507d2b9fa048a7760d57eedd7cbff722238c`. The manifest exactly covers all
   23 payload files; every byte count and SHA-256 digest matches.
2. **Verified:** Both target records bind and match the target configuration,
   accepted S3 ablation record, official data cube, mask, integrated-error
   image, frozen KinMS cube, and KinMS fit result. The copied KinMS comparison
   cubes are also manifest-bound.
3. **Verified:** KGAS066 fails both PI gates. Its projected-velocity RMSE is
   `16.3203 km/s` versus KinMS `12.8486 km/s`, ratio `1.27020`. Its common
   reduced chi-square is `7.43475` versus KinMS `6.50510`.
4. **Verified:** KGAS007 improves common reduced chi-square, `2.04657` versus
   `2.27507`. Its diagnostic velocity RMSE is `2.50486 km/s` versus KinMS
   `4.08720 km/s`, ratio `0.612855`, but only two common radial bins survive.
   The record therefore marks the velocity gate ineligible and failed under
   the predeclared minimum of three bins.
5. **Verified:** Independent recomputation reproduces all RMSE values and every
   gate boolean exactly. Recomputing kinUV chi-square from the retained
   float32 FITS cubes differs from the pre-serialization float64 value by only
   `1.19e-9` relative for KGAS066 and `3.81e-10` for KGAS007; neither verdict
   can change. KinMS chi-square reproduces exactly.
6. **Verified:** The runner writes both target records, the failed summary, and
   the complete manifest before raising `SystemExit` with the S4 escalation
   message. No S5 validation directory or post-result S5 commit exists.
7. **Advisory:** The covariance record used for structured line-free
   diagnostics is not directly path-and-hash bound in the S4 summary. Those
   diagnostics do not enter either failed PI gate, so this omission does not
   weaken the stop decision. Bind that auxiliary input before reusing the
   structured diagnostics in a later promotion record.

## Gate assessment

The PI requires both at least ten-percent better projected-velocity RMSE and
better overall reduced chi-square on each target. KGAS066 demonstrates an
aggregate regression on both quantities. KGAS007 passes only reduced
chi-square; its favorable but sparse two-bin velocity estimate is diagnostic,
not gate-eligible. Thus each target is rejected and the overall S4 result is
correctly `accepted: false`.

## Verdict rationale

**Accept failed-gate evidence.** Exact code and input identity, complete
checksums, independently reproduced arithmetic, explicit sparse-profile
handling, passing tests, and fail-closed runner behavior establish a valid S4
failure. The authorized cascade must stop and escalate; S5 remains prohibited.
