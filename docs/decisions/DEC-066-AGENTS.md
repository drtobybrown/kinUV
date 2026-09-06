---
id: DEC-066-AGENTS
status: accepted
generation: 5
date: 2026-08-18
amended: 2026-09-06
owner: senior-registrar
---
# Scientific authority and independent review

**Question:** How are scientific authority, implementation, review, and promotion separated?

## Answer

Astra is the Principal Project Authority. The Consultant, normally the Lead Architect using the strongest available frontier model, owns scientific strategy: model class, parameterization, physical priors, covariance assumptions, quantitative gates, permitted claims, and scientific sign-off.

The Senior Registrar owns the durable process record. The Registrar registers the Consultant's specification, validates and freezes configuration, assigns reviewers, tallies verdicts, controls state transitions, verifies provenance, and assembles the promotion dossier. The Registrar may clarify wording but may not alter scientific meaning.

The Implementer executes the accepted specification. This includes coding, testing, diagnostics, profiling, refactoring, job submission, monitoring, checkpoint recovery, and artifact generation. The Implementer can stop on failure and can make bounded engineering choices, but cannot change or relax the model, prior, covariance, data selection, transform convention, or gate criteria.

Reviewer A independently assesses science and numerics. Reviewer B independently assesses software, reproducibility, performance, and operations. Neither reads the other's review before submitting a verdict. A proposer or implementer does not review their own work.

## Workflow

1. Consultant signs a scientific specification with declared gates and residual risks.
2. Registrar creates the proposal and freezes target and campaign configuration checksums.
3. Reviewer A and Reviewer B independently issue `accept`, `accept-with-required-changes`, or `reject`.
4. Registrar licenses implementation only after both reviews accept and required changes are incorporated.
5. Implementer executes the licensed work and records every gate without changing the specification.
6. Material scientific or architectural changes return to Step 1 and repeat both reviews.
7. Changes to transforms, likelihoods, covariance, parameter charts, serialization, or promotion logic receive independent science/numerics and software/reproducibility code review.
8. Registrar verifies the dossier; Consultant signs scientific promotion; Astra decides exceptions and publication authority.

## Failure and deadlock

A failed gate produces a failed or diagnostic product. It is not a request for the Implementer to weaken the threshold. If reviewers disagree after one documented revision, the Registrar escalates the disagreement to Astra with both arguments preserved.

Only Astra may waive an invariant or promotion gate. A waiver states its scope, evidence, expiration, and prohibited claims.

The repository is the durable mailbox. Closed discussion is synthesized into production history and removed from the active board.
