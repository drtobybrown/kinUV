# Review board

The board tracks one active proposal and its independent reviews. Scientific values live in frozen configuration and proposals; durable results live in run manifests and `docs/PRODUCTION_RECORD.md`.

## Active card

`crossdomain-recovery-s0`: Astra formally approved
[`DEC-KINUV-CROSSDOMAIN-RECOVERY`](../decisions/DEC-KINUV-CROSSDOMAIN-RECOVERY.md)
and licensed S0/S1 on 2026-09-06. S0 implementation is complete locally;
independent Reviewer A science/numerics and Reviewer B
software/reproducibility code verdicts are pending. S0 remains open until both
verdicts accept the same commit.

## Roles and files

| Role | Responsibility | File |
|---|---|---|
| Consultant | Sign scientific strategy and acceptance criteria | `YYYY-MM-DD-propose-<slug>.md` |
| Senior Registrar | Register, freeze, assign, tally, verify | STATUS front matter and proposal record |
| Reviewer A | Science and numerical review | `YYYY-MM-DD-review-a-<slug>.md` |
| Reviewer B | Software and reproducibility review | `YYYY-MM-DD-review-b-<slug>.md` |
| Implementer | Execute accepted specification | Code, tests, run manifest, gate artifacts |
| Reviewer A/B after critical code changes | Verify implementation against the frozen specification | `YYYY-MM-DD-code-review-{a,b}-<slug>.md` |

Templates: [`_template.md`](_template.md) and [`_review_template.md`](_review_template.md).

## Independence

Reviewers receive the frozen proposal, configuration checksums, relevant decisions, and acceptance criteria. They do not read each other's review before committing a verdict. Neither reviewer implements the proposal under review. A generic approval without an attempted falsification, missing-gate check, or residual-risk assessment is invalid.

## Tally

- Two `accept` verdicts license implementation.
- `accept-with-required-changes` licenses implementation only after the Registrar verifies incorporation into the frozen proposal.
- Any `reject` returns the scope to the Consultant and requires two fresh reviews after revision.
- Persistent disagreement is escalated to Astra; the last writer does not win.

The Implementer may resolve routine defects inside the accepted specification. A change to physics, priors, covariance, data selection, transform conventions, quantitative gates, or promoted claims reopens the proposal.

## Promotion

After execution, Reviewer A checks scientific and numerical code changes and Reviewer B checks software and reproducibility changes where the field guide requires dual code review. These are implementation reviews, distinct from the proposal verdicts, and both must name the reviewed commit. The Registrar verifies manifests, checksums, gate states, and artifact completeness. The Consultant then signs or rejects scientific promotion.

Closeout folds durable findings into `docs/PRODUCTION_RECORD.md`, clears active STATUS pointers, and removes closed discussion from the board.
