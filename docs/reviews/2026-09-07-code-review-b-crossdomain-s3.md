---
role: reviewer
seat: software-reproducibility
phase: implementation
date: 2026-09-07
reviewer: review-b-s3
canon_generation: 23
campaign_id: crossdomain-recovery-s3
proposal: docs/decisions/DEC-HUMAN-OVERRIDE-RIGHTSIZED-GATES.md
reviewed_commit: b2ac6bc44991766ab8c790de1f351b812df0db45
verdict: accept
---
# Independent code review B: cross-domain recovery S3

I reviewed exact clean commit
`b2ac6bc44991766ab8c790de1f351b812df0db45` without reading or contacting
Reviewer A. This review applies the Project PI's right-sized gate override to
the generic S3 visibility-model ablations and the r3 evidence tree.

## Attempted falsification

I independently rebuilt every recorded gate and model-selection decision from
the two target ablation records. I verified the candidate ordering, checked
that rejected optional candidates remain visible, recomputed gradient
normalization, and compared target records with the top-level summary. I
rehashed the visibility, brightness-template, propagated-error, fit-window,
target-configuration, S2-summary, and frozen-covariance inputs.

I also recomputed every manifest byte count and SHA-256 digest, checked exact
file-set coverage, searched for abandoned atomic-write temporaries, verified
that every JSON byte is ASCII, and scanned the reviewed S3 source for legacy
prototype or dark-matter dependencies. Fourteen focused S3, hot-path,
standalone, and forbidden-import tests passed. The complete CPU suite passed
with `259 passed, 8 skipped`.

## Findings

1. **Verified:** `summary.json` and `MANIFEST.json` both identify clean `dev`
   commit `b2ac6bc44991766ab8c790de1f351b812df0db45`. The manifest exactly covers
   its three payload files; every recorded size and hash matches.
2. **Verified:** All external inputs match their target-level byte counts and
   SHA-256 digests. Both target configurations and S2 summaries match their
   recorded hashes, and the summary's frozen covariance hash matches the S2
   covariance record.
3. **Verified:** Both targets execute the deterministic sequence
   `baseline_arctan`, `joint_emissivity`, `supported_rings`, then
   `two_zone_dispersion`. Each candidate is present exactly once. The runner
   atomically checkpoints the initial record, every candidate result, the
   final selection, summary, and manifest using file flush, `fsync`, and
   `os.replace`; no temporary file remains.
4. **Verified:** Optional models remain visible even when rejected. KGAS066
   retains the unsupported ring result with delta chi-square `4.82079`, then
   correctly evaluates two-zone dispersion from the retained joint-emissivity
   parent and selects it at delta chi-square `152.50021`. KGAS007 retains the
   rejected two-zone result at delta chi-square `-0.0003841` after selecting
   supported rings at delta chi-square `32.87504`. The implementation appends
   optimizer status and messages before selection, so a failed candidate would
   remain visible; all eight reviewed fits succeeded.
5. **Verified:** S2 likelihood replay errors are `6.11e-10` for KGAS066 and
   `5.82e-11` for KGAS007. Maximum normalized candidate gradients are
   `2.3666e-4` and `8.3170e-5`, respectively, below the PI-authorized `1e-3`
   gate. Every objective is finite and both preferred solutions have no
   boundary pressure.
6. **Verified:** `kinuv.infer.s3` exposes target-neutral emissivity-basis,
   supported-knot, chart, objective, and candidate-fit interfaces. It contains
   no target-ID branches, site paths, legacy imports, halo profiles, or mass
   decomposition. Target values remain in versioned configuration and the
   scientific objective remains visibility chi-square with the frozen C1
   covariance.
7. **Advisory:** The atomic helper syncs file contents before replacement but
   does not `fsync` the parent directory, and this validation tree remains
   writable. Its current manifest is exact and no interrupted temporary exists,
   so this does not invalidate the development-stage result. Seal promoted
   evidence and add directory synchronization when these checkpoints become a
   publication record.

## Gate assessment

All six recorded gates are independently reproducible and pass for both
targets. The retained evidence distinguishes empirical model selection from
posterior calibration: the Hessians are local identifiability diagnostics,
while the PI's S3 R-hat and ESS gates remain reserved for a later licensed
posterior campaign.

The selection logic respects the sequential parent relationship. The ring
candidate must meet its delta-chi-square, rank, boundary, and gradient checks
before two-zone dispersion inherits it; otherwise dispersion returns to the
joint-emissivity parent. The two reviewed targets exercise both branches.

## Residual risks

S3 selects model structure on the fit sample. It does not establish the S4
held-out KinMS superiority claim or calibrated posterior intervals. The
unsealed validation permissions and parent-directory durability point above
must be addressed before publication promotion, but the checksum-bound r3
evidence is complete and internally consistent.

## Verdict rationale

**Accept.** Exact clean-code provenance, atomic checkpointing, complete input
hashes, deterministic and fully retained candidate evidence, independently
reproduced selection decisions, passing regression suites, and the absence of
legacy or dark-matter dependencies satisfy the S3 software and reproducibility
contract under the binding PI gates.
