---
id: DEC-066-AGENTS
status: accepted
generation: 1
date: 2026-08-18
owner: planner
---
# Two-agent adversarial handshake

**Question:** How do two Cursor chats share one architecture without forking it?

## Answer

They do not share memory. The repo is the mailbox. Handshake is **adversarial**: one proposes, the other must challenge or ACK. The user is the only tie-breaker and the only one who may create a new `DEC-*` id.

## Canon ranking

1. `docs/decisions/DEC-*.md` — physics. `accepted` only after a review ACK (generation 2 is provisional until chat B reviews).
2. `field-guide/index.md` — 80-line OS.
3. `docs/architecture/STATUS.md` — live mailbox (`next_role`, `pending`, `code_freeze`).
4. `docs/reviews/` — append-only propose/review notes (how agents talk).
5. `PLAN.md` — evidence, not policy. If it disagrees with an ADR, fix PLAN.md in the same generation.
6. Cursor `~/.cursor/plans/*.plan.md` — **not canon**. Ignore unless the file headers `canon_generation` matching STATUS. Chat transcripts are not canon. Untracked `src/` is not canon.

## Start of every turn

Read `AGENTS.md` → `field-guide/index.md` → `STATUS.md` → latest `docs/reviews/*.md`. If `code_freeze: true`, do not write Python.

## Proposer (`STATUS.next_role: proposer`)

- May set existing DEC files to `status: proposed` (do not set `accepted`).
- Write `docs/reviews/YYYY-MM-DD-propose.md` from `_template.md`.
- Update STATUS: `next_role: reviewer`, list `pending` ids, `last_propose`.

## Reviewer (`STATUS.next_role: reviewer`)

- Must not ACK in the same turn as proposing.
- Write `docs/reviews/YYYY-MM-DD-review.md` with **at least one** of: contradiction that changes an ADR; tighter quantitative bound; missing unit test/gate; ACK naming a residual risk the proposer did not list.
- “Looks good” is invalid; leave `next_role: reviewer`.
- Challenge → `next_role: proposer`. ACK → `accepted`, Field Guide DEC row if needed, STATUS `pending` cleared, `next_role: proposer` for the next topic.

## Deadlock

If both still disagree after a challenge cycle, set STATUS `next_role: user` and append `deadlocks`. Neither agent wins by writing last.

## New DEC ids

Neither agent. Recommend in a review file; the user adds the stub.
