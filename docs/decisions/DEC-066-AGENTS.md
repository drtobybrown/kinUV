---
id: DEC-066-AGENTS
status: accepted
generation: 2
date: 2026-08-18
amended: 2026-08-30
owner: planner
---
# Adversarial handshake (dual-review board)

**Question:** How do agents share one architecture without forking it?

## Answer

They do not share memory. The repo is the mailbox. The **parent chat is the proposer**. Two **independent** reviewer sub-agents must accept or reject on the board (`docs/reviews/`). The user is the only tie-breaker and the only one who may create a new `DEC-*` id.

User 2026-08-30: methodology is licensed. After a dual accept, the parent **implements and executes** the accepted scope without a second review cycle. A user **build** command means run that loop through all licensed stages (propose → board → execute), not stop at the propose.

## Canon ranking

1. `docs/decisions/DEC-*.md` — physics. `accepted` only after dual-board accept (or a user tie).
2. `field-guide/index.md` — 80-line OS.
3. `docs/architecture/STATUS.md` — live mailbox (`next_role`, `board`, `pending`, `code_freeze`).
4. `docs/reviews/` — propose + review-a + review-b (how agents talk).
5. `PLAN.md` — evidence, not policy.
6. Cursor `~/.cursor/plans/*.plan.md` — **not canon** unless `canon_generation` matches STATUS.

Human-facing science: `docs/methodology.md`. Board ops: `docs/reviews/BOARD.md`.

## Start of every turn

Read `AGENTS.md` → `field-guide/index.md` → `STATUS.md` → latest `docs/reviews/`. If `code_freeze: true`, do not write production code unless the user unfreezes.

## Build command

When the user says **build** (or build and execute):

1. Parent writes `docs/reviews/YYYY-MM-DD-propose-<slug>.md` (scope, stages, residual risks).
2. Parent sets STATUS `next_role: board`, `board: open`, `last_propose` that file.
3. Parent launches **two** reviewer sub-agents **in parallel**. They must not read each other's review files. Each writes `…-review-a-<slug>.md` or `…-review-b-<slug>.md`.
4. Tally (parent, not a reviewer):
   - Both `accept` → `board: accepted`, `next_role: implementer`. Execute the accepted scope through all named stages. No further review.
   - Either `reject` → `board: rejected`, `next_role: proposer`. Revise and re-board both reviewers.
5. Commit and push after the propose, after the tally, and after each stage deliverable.

The proposer must not write `review-a` / `review-b`. Rubber-stamp ("looks good") is invalid.

## Reviewer verdict

Each review file:

- `verdict: accept | reject`
- `severity: major | minor` (required on accept if there are comments; omit only if there are truly no comments **and** a residual risk is named)
- At least one of: contradiction that changes an ADR; tighter quantitative bound; missing unit test/gate; residual risk the proposer did not list

**Accept + major:** implementer must fix those comments during execute (note in STATUS).  
**Accept + minor:** optional; mention in CHANGELOG if taken.  
**Reject:** do not implement.

## Deadlock

If the two reviewers still disagree after one revise cycle, `next_role: user` and append `deadlocks`. Neither agent wins by writing last.

## New DEC ids

Neither agent. Recommend in a review file; the user adds the stub.

## Physics stops (unchanged)

Hann on binned channels, restored Ico as intrinsic SB, vis phase ramp after PB, `(dx,dy)` frozen at 0, MAP that cannot beat V=0, new `DEC-*` without a user stub — stop. Dual accept does not override these.
