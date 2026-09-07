---
id: DEC-066-INDEX
status: accepted
generation: 5
date: 2026-08-18
amended: 2026-09-07
owner: senior-registrar
---
# Source of truth and decision scope

**Question:** Which record controls when documents disagree, and how far does a campaign decision apply?

## Authority order

1. The Project PI's current written directive.
2. Astra's current written directive within the PI-defined boundary.
3. Accepted `docs/decisions/DEC-*.md` records within their declared scope.
4. [`field-guide/index.md`](../../field-guide/index.md), the data-agnostic production operating standard.
5. The active accepted proposal and frozen target/campaign configuration.
6. `docs/architecture/STATUS.md` and `docs/reviews/BOARD.md`.
7. `PLAN.md`, diagnostic notes, historical reviews, editor plans, and chat transcripts.

Higher authority does not erase evidence. When a directive supersedes a decision, the Registrar records the change and affected products.

## Scope rule

The `DEC-066-*` series originated in one campaign. Numeric priors, coordinates, paths, seeds, fixed geometry, thresholds, and product names in those records apply only to products that explicitly cite them. They are not software defaults and do not transfer to another target.

Cross-target rules must be stated as architectural invariants or generalized decisions and tested independently of campaign data. Current target state belongs in configuration, run manifests, `STATUS.md`, methodology, and production history.

The current generalized inference boundary is [`DEC-KINUV-VISLIK`](DEC-KINUV-VISLIK.md): visibility chi-square is the scientific likelihood, while cosmology and mass decomposition are downstream-only.

## Change rule

- The Consultant owns proposed changes to scientific models, priors, covariance, parameterization, and gates.
- The Registrar owns canon consistency, configuration freeze, review tally, and promotion records.
- Two independent reviewers assess science/numerics and software/reproducibility.
- The Implementer executes the accepted specification and may right-size an
  intermediate gate when explicit PI authority permits it and the change is
  documented against the promoted physical claim.
- Astra resolves technical disputes within the PI-defined boundary. The
  Project PI may waive or supersede an invariant or promotion gate. The active
  S2-S5 override is
  [`DEC-HUMAN-OVERRIDE-RIGHTSIZED-GATES`](DEC-HUMAN-OVERRIDE-RIGHTSIZED-GATES.md).

The detailed workflow is normative in the field guide and `DEC-066-AGENTS`. A worker encountering an unresolved scientific choice records it and escalates to the Consultant; it does not create a hidden default.
