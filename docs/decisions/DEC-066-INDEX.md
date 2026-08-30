---
id: DEC-066-INDEX
status: accepted
generation: 2
date: 2026-08-18
owner: planner
---
# Source of truth

**Question:** If several documents disagree, which wins?

## Answer

Rank, highest first (`DEC-066-AGENTS`):

1. `docs/decisions/DEC-*.md` — closed (or proposed) physics.
2. `field-guide/index.md` — operating system; ids only, no essays.
3. `docs/architecture/STATUS.md` — mailbox (`next_role`, `board`, `pending`, `code_freeze`).
4. `docs/reviews/` — handshake log (propose + review-a + review-b); not physics.
5. `PLAN.md` — evidence base. If it disagrees with an ADR, the ADR wins; fix PLAN.md in the same generation.
6. Cursor plan files (`~/.cursor/plans/*.plan.md`) and chat transcripts — **not canon**. A Cursor plan without `canon_generation` matching STATUS is stale; read the repo.

A worker that needs a new decision **stops**. It does not create a `DEC-*` id. The user adds stubs. Edits follow `DEC-066-AGENTS`: no `accepted` without dual-board accept (or a user tie); no rubber-stamp. After dual accept the parent implements the named stages without a third review.
