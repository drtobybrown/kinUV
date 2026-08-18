---
generation: 2
phase: architecture
code_freeze: true
next_role: reviewer
pending:
  - DEC-066-SPECRESP
  - DEC-066-SB
  - DEC-066-PB
  - DEC-066-OSCMETRIC
  - DEC-OPS-AUTH
last_propose: docs/reviews/2026-08-18-hello.md
last_review: null
open_questions: []
deadlocks: []
canon_generation: 2
---

# Architecture mailbox

**Hello from chat A (2026-08-18).** This repo is the only shared memory. If you are the other agent, you are the **reviewer**. Do not propose new ADR patches until you have filed a real review of generation 2 (challenge, tighter bound, missing gate, or ACK with a residual risk the hello note did not list). Rubber-stamp is a process failure (`DEC-066-AGENTS`).

Generation 2 ADRs were written in chat A **without** an adversarial ACK. Treat `status: accepted` on those five ids as **provisional** until you ACK or challenge. Do not write Python (`code_freeze: true`). Do not create a new `DEC-*` id.

## Rank (DEC-066-INDEX)

1. `docs/decisions/DEC-*.md`
2. `field-guide/index.md`
3. `PLAN.md` (evidence only)
4. Cursor `~/.cursor/plans/*.plan.md` — stale unless the file headers `canon_generation: 2`

## What to attack first

SPECRESP (native Hann vs weights), SB (Wiener K, no B̃²), PB (pbcor + FWHM), OSCMETRIC (20×5 mocks, not a SE-GP), OPS-AUTH (session names, no fake cron).

When done, write `docs/reviews/YYYY-MM-DD-review.md` from the template, then update this file: `last_review`, `next_role` (`proposer` if challenge, or keep `reviewer` until ACK is real). Only after ACK may `pending` clear and `accepted` be treated as closed.
