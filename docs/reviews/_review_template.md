---
role: reviewer
seat: science-numerics | software-reproducibility
phase: specification | implementation
date: YYYY-MM-DD
reviewer: independent-agent-id
canon_generation: 5
campaign_id: campaign-id
proposal: docs/reviews/YYYY-MM-DD-propose-<slug>.md
reviewed_commit: null
verdict: accept | accept-with-required-changes | reject
---
# Independent review

Do not read the other review before recording this verdict. Do not implement the proposal under review. For an implementation review, replace `reviewed_commit: null` with the exact commit and verify code plus artifacts against the frozen proposal.

## Attempted falsification

Identify the strongest failure mode tested: physical inconsistency, unit/sign error, non-identifiability, covariance mismatch, missing recovery case, package-boundary violation, irreproducibility, or operational data loss.

## Findings

Number each finding and label it `required` or `advisory`. Cite the specification, code, test, or missing evidence.

## Gate assessment

State whether every declared gate is measurable, precommitted, and sufficient for the proposed claim.

## Residual risks

List risks remaining even if required findings are fixed.

## Verdict rationale

Explain why the verdict follows. A generic approval is invalid.
