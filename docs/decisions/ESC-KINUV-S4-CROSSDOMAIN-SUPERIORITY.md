---
id: ESC-KINUV-S4-CROSSDOMAIN-SUPERIORITY
status: escalated-to-astra
date: 2026-09-07
authority_requested: Astra
implementation_commit: 5d08507d2b9fa048a7760d57eedd7cbff722238c
artifact: results/validation/crossdomain-recovery-s4-20260907-r3
---
# S4 cross-domain superiority escalation

## Decision requested

Define the next physical and model-selection direction that can transfer
visibility-domain improvements into better projected-velocity recovery and
real-cube residuals for KGAS066 without losing the KGAS007 gain. The authorized
autonomous cascade stops here because KGAS066 fails both PI S4 criteria and
regresses against frozen stock KinMS. S5 has not started.

No target parameter was manually changed, no S4 threshold was relaxed, and no
posterior, NUTS, G4, or production promotion was launched.

## Common-domain result

Exact implementation commit
`5d08507d2b9fa048a7760d57eedd7cbff722238c` renders the S3-selected sky models,
applies the registered diagnostic frame conversion and restored-cube operator,
and scores kinUV and frozen stock KinMS on the same official cube, mask, beam,
channel grid, and scalar noise normalization.

| Target | kinUV reduced chi-square | KinMS reduced chi-square | ratio | kinUV projected RMSE | KinMS projected RMSE | ratio | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| KGAS066 | 7.43475 | 6.50510 | 1.14291 | 16.3203 km/s | 12.8486 km/s | 1.27020 | both gates fail; KGAS066 regresses |
| KGAS007 | 2.04657 | 2.27507 | 0.89956 | 2.5049 km/s | 4.0872 km/s | 0.61286 | chi-square passes; velocity result ineligible with two common bins |

The binding PI requirement is a projected-velocity RMSE ratio no greater than
0.90 plus an improved overall reduced chi-square on each real target. KGAS066
misses the velocity threshold by 0.37020 in ratio and has 14.29 percent worse
reduced chi-square. This result also triggers the explicit KGAS066
non-regression stop condition.

KGAS007's two retained moment-1 bins favor kinUV, but two points cannot support
the registered radial-profile claim. The implementation therefore records the
diagnostic ratio and correctly marks the velocity gate ineligible rather than
promoting it.

## Interpretation bounded by the evidence

The S3 flexibility improved the KGAS066 visibility objective and raised its
real-cube flux recovery to 0.97750, compared with 0.92758 for KinMS. The
common-domain cube normalized RMSE is still 0.57341 for kinUV versus 0.53642
for KinMS, and the projected velocity profile is worse across 15 eligible
radial bins. The failure is therefore not a simple global flux-normalization
artifact. It shows that the visibility-selected degrees of freedom and current
selection criterion do not preserve the promoted image-domain velocity proxy
for this target.

For KGAS007, kinUV improves the normalized cube RMSE from 0.42714 to 0.40485
and the reduced chi-square ratio to 0.89956. Its low-information restored-cube
mask yields only two common radial bins, so this benchmark cannot yet establish
the required velocity-profile superiority. The result is promising diagnostic
evidence, not an S4 pass.

Structured native line-free residual diagnostics cover held-out folds,
baselines, antennas, time grouping, and real/imaginary coupling. Their global
means and cross-component correlations are small; localized baseline extrema
are retained in the target metrics. These diagnostics do not identify a
single unique physical correction, so the Implementer will not select a new
prior or model family without Astra's direction.

## Integrity and review

The dossier contains 23 manifest-indexed files. Every registered byte count and
SHA-256 hash verifies. The full repository suite passes with 263 tests and 8
skips. Reviewer A and Reviewer B independently accepted the dossier as valid
failed-gate evidence at commits `a0b0f0a` and `4882189`, respectively. Their
verdicts validate the failure record; they do not close S4 or authorize S5.
Reviewer B notes one non-gating provenance advisory: the covariance record used
by the structured residual diagnostics is not directly path-and-hash bound in
the S4 summary. It must be bound before those auxiliary diagnostics are reused
in a later promotion record. This omission cannot change either failed PI
metric or the present stop decision.

The next authorized action is architectural diagnosis by Astra. Any resumed
implementation must retain this dossier as the comparison baseline and must
repeat both targets under the unchanged PI gate before S5 can begin.
