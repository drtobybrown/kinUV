---
role: reviewer
seat: a | b
date: YYYY-MM-DD
agent: subagent-id
canon_generation: 4
ids: []
verdict: accept | reject
severity: major | minor
propose: docs/reviews/YYYY-MM-DD-propose-<slug>.md
---

# Review <a|b>: title

Do not read the other seat's review file. Do not implement.

## Attacks / bounds

At least one of: ADR contradiction; tighter quantitative bound; missing test/gate; residual risk the propose omitted. "Looks good" is invalid.

## Comments

Numbered. Tag each `major` or `minor`.

## Residual risks

Numbered. Carry forward anything still open.

## STATUS updates required

- `verdict` and `severity` as in the header
- `last_review_a` or `last_review_b`: this file
- Do not set `board: accepted` (parent tallies)
