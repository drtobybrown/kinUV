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

Astra froze that transform and the missing S2 contracts in
[`DEC-KINUV-S1-CONTINUUM-AND-S2-CONTRACTS`](../decisions/DEC-KINUV-S1-CONTINUUM-AND-S2-CONTRACTS.md).
Reviewer A and Reviewer B independently returned `accept-with-required-changes`
on exact proposal commit `e7a1e71eba9ffe68a21754bb41dc9bb03d6c2acc`; see
[`review-a`](2026-09-06-review-a-s1-continuum-contract.md) and
[`review-b`](2026-09-06-review-b-s1-continuum-contract.md). Both require exact
metric/refinement/unit definitions, checksum-bound campaign state, and an
executable S2 covariance/start/optimizer contract. Both retained target exports
are insufficient for real S2 covariance selection and held-out scoring.
Those verdicts remain historical evidence. The later pragmatic S1 execution
directive classifies the renderer correction as a localized mathematical bug
fix and authorizes the Field Guide's proportional-verification fast path. S1
therefore closes on the unchanged two-target refinement gates, without a new
proposal tally. The r4 attempt at `c55c985` passed every tested dimension except
radial quadrature. The composite radial implementation at exact commit
`3a734693bdd023d01490f77c8181c5d1551072bb` then passed every frozen S1 gate in
the sealed r5 two-target matrix. S1 is closed and the active card has advanced
to S2 geometry and covariance. Real grouped covariance selection remains
blocked pending compliant `ms2kinuv` exports; independent S2 implementation
and geometry/optimization audits are licensed. Full dual review remains
required for scientific-contract changes and production promotion.

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

## Governance and engineering velocity mandate

Reviewer A and Reviewer B enforce the Field Guide's Governance & Engineering
Velocity Invariant. They must return `changes-requested` on any architectural
proposal that mandates redundant test harnesses, unnecessary secondary
implementations, or excessive paperwork gates for localized mathematical,
numerical, unit, or interpolation repairs.

For those repairs, reviewers evaluate the implementation against the declared
empirical refinement gate, mathematical correctness, and focused test passage.
They must not demand an auxiliary verification engine when the primary method
achieves empirical closure. A secondary reference engine becomes eligible only
after the primary numerical implementation demonstrably fails empirical
refinement closure on target data.

Reviewer A and Reviewer B must actively flag and push back on architectural
proposals from Astra that prescribe implementation mechanics, low-level data
structures, non-essential secondary test harnesses, or paperwork delays beyond
the risk of the change. Such proposals receive `changes-requested` until the
division of responsibility is restored: Astra defines what must be achieved and
the physical reason; the Senior Implementer, including Sol when assigned,
decides how to implement it.

This mandate governs the upcoming S2 and S3 proposal cycles and all later
stages. Reviewers enforce scientific acceptance gates while preserving the
Implementer's authority over software design, numerical methods, efficiency,
and code organization.

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
