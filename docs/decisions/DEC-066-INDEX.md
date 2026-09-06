---
id: DEC-066-INDEX
status: accepted
generation: 5
date: 2026-08-18
amended: 2026-09-06
owner: senior-registrar
---
# Source of truth and decision scope

**Question:** Which record controls when documents disagree, and how far does a campaign decision apply?

## Authority order

1. Astra's current written directive.
2. Accepted `docs/decisions/DEC-*.md` records within their declared scope.
3. [`field-guide/index.md`](../../field-guide/index.md), the data-agnostic production operating standard.
4. The active accepted proposal and frozen target/campaign configuration.
5. `docs/architecture/STATUS.md` and `docs/reviews/BOARD.md`.
6. `PLAN.md`, diagnostic notes, historical reviews, editor plans, and chat transcripts.

Higher authority does not erase evidence. When a directive supersedes a decision, the Registrar records the change and affected products.

## Scope rule

The `DEC-066-*` series originated in one campaign. Numeric priors, coordinates, paths, seeds, fixed geometry, thresholds, and product names in those records apply only to products that explicitly cite them. They are not software defaults and do not transfer to another target.

Cross-target rules must be stated as architectural invariants or generalized decisions and tested independently of campaign data. Current target state belongs in configuration, run manifests, `STATUS.md`, methodology, and production history.

## Change rule

- The Consultant owns proposed changes to scientific models, priors, covariance, parameterization, and gates.
- The Registrar owns canon consistency, configuration freeze, review tally, and promotion records.
- Two independent reviewers assess science/numerics and software/reproducibility.
- The Implementer executes the accepted specification and may not relax it.
- Astra resolves disputes and is the only authority who may waive an invariant or promotion gate.

The detailed workflow is normative in the field guide and `DEC-066-AGENTS`. A worker encountering an unresolved scientific choice records it and escalates to the Consultant; it does not create a hidden default.
