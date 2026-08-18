---
role: proposer | reviewer
date: YYYY-MM-DD
agent: chat-A | chat-B | name
canon_generation: 2
ids: [DEC-066-…]
verdict: propose | challenge | ack
---

# Title

## Scope

Which DEC ids. Do not open ids that are not in `STATUS.pending` unless the user added them.

## What changed / what was checked

Bullets. Quote generation numbers.

## Rejected alternatives (proposer) or attacks (reviewer)

Reviewer must include **at least one** of: contradiction that changes an ADR; tighter quantitative bound; missing unit test/gate; ACK that names a residual risk the proposer did not list.

“Looks good” with no residual risk is invalid. Do not update `status: accepted` in that case.

## Residual risks

Numbered. Carry forward anything still open.

## STATUS updates required

- `next_role:`
- `pending:`
- `last_propose` / `last_review:` this file
