# kinUV agent bootstrap

At the start of work, read [`field-guide/index.md`](field-guide/index.md), [`docs/decisions/DEC-066-INDEX.md`](docs/decisions/DEC-066-INDEX.md), [`docs/architecture/STATUS.md`](docs/architecture/STATUS.md), and [`docs/reviews/BOARD.md`](docs/reviews/BOARD.md).

The field guide is the data-agnostic production operating standard. Target values, data paths, priors, and campaign thresholds belong in versioned configuration and immutable run manifests. Current results belong in STATUS, methodology, and `docs/PRODUCTION_RECORD.md`.

Authority flows from Astra to the Consultant (Lead Architect / frontier model), then through the Senior Registrar to the Implementer. The Consultant owns scientific strategy and sign-off. The Registrar freezes the specification, tallies reviews, and verifies promotion evidence. The Implementer executes without relaxing the model, prior, covariance, or gates. Reviewer A covers science/numerics; Reviewer B covers software/reproducibility; both are independent.

Authorized fixes and recovery plans proceed through implementation and relevant
test verification in one continuous turn; do not pause to re-propose accepted
work. Test physical invariants, numerical convergence, calculations, and data
interfaces. Do not gate documentation wording, log prose, review phrasing, or
procedural formatting.

Use high-tier subagents only for complex derivations, independent formal
reviews, or bounded audits that require independent judgment. Keep routine
tests, file inspection, checksums, and ordinary edits lightweight. At every
stage boundary, update `docs/architecture/STATUS.md` with exact code and
evidence commit hashes, scalar metrics with units and thresholds, gate state,
durable evidence path, and the next responsible role before handoff.

kinUV must remain importable without legacy packages or CASA tooling. Measurement Set extraction belongs exclusively to the separate `ms2kinuv` repository. Use node-local `/scratch` for high-frequency temporary I/O and the configured `/arc` run root for durable, bounded artifacts.

If `code_freeze: true`, do not change production code unless Astra explicitly overrides the freeze. Never commit secrets or overwrite a promoted product.
