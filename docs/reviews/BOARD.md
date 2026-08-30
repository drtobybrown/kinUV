# Review board

Message board for propose / dual review. Physics stays in `docs/decisions/`. This folder is the conversation.

## Files

| Role | Path |
|---|---|
| Propose | `docs/reviews/YYYY-MM-DD-propose-<slug>.md` |
| Reviewer A | `docs/reviews/YYYY-MM-DD-review-a-<slug>.md` |
| Reviewer B | `docs/reviews/YYYY-MM-DD-review-b-<slug>.md` |
| Templates | [`_template.md`](_template.md), [`_review_template.md`](_review_template.md) |

STATUS front matter tracks the live card: `next_role`, `board` (`idle` / `open` / `accepted` / `rejected`), `last_propose`, `last_review_a`, `last_review_b`.

## Independence

Launch reviewer A and B in the same turn, in parallel. Give each the propose path and STATUS. Tell each **not** to open the other review file. The parent tallies only after both files exist.

## After dual accept

No third review. Parent becomes implementer: write code, run the licensed stages, **decide each gate**, update human docs, commit, and push `origin/dev` after each stage. Do not ping the user mid-gate. Hand the user the final plot folder (moments / spectra / PV / leftover `chi2`).

## User role

The user is not a gate sitter. They review whether the **final** Data | Model | Residual (and leftover) plots work.

## Historical notes

Reviews from before 2026-08-30 used a single reviewer and `verdict: ack | challenge`. Leave them. New cards use `accept | reject` and A/B files.
