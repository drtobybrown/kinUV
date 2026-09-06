# Review board

The board tracks one active proposal and its independent reviews. Scientific values live in frozen configuration and proposals; durable results live in run manifests and `docs/PRODUCTION_RECORD.md`.

## Active card

`crossdomain-recovery-s1`: Astra formally approved
[`DEC-KINUV-CROSSDOMAIN-RECOVERY`](../decisions/DEC-KINUV-CROSSDOMAIN-RECOVERY.md)
and licensed S0/S1 on 2026-09-06. S0 closed after independent Reviewer A and
Reviewer B accepted exact commit
`fb4a14543d579168c9224c8ebca6a7591147f4db`. The S1 intrinsic KinMS adapter
and operator-parity implementation reached its first frozen review. Both
independent reviewers returned `accept-with-required-changes` on exact commit
`1b629a1fab9c64a25cb870d8f5ad43102b55cb0b`. Required repairs cover the KinMS
PA boundary, target-path convergence axes and independent-repeat chi-square,
fail-closed metadata and cube-derived flux, atomic dossier publication,
executable environment/input locks, and complete runtime provenance. S1 stays
open until both reviewers accept one revised exact commit; S2 has not begun.
Commit `0c95240b2f7e238ba6c0666f61871c65558494b0` resolves the bounded review
findings, but the newly complete target-path matrix fails the frozen resolution
gate. The active card is escalated through
[`ESC-KINUV-S1-RENDERER-CONVERGENCE`](../decisions/ESC-KINUV-S1-RENDERER-CONVERGENCE.md).
No fresh code verdict is requested until Astra freezes the eligible intrinsic
KinMS deposition/LOSVD transform.

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
