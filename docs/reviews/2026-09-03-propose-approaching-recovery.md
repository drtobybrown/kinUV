---
role: proposer
date: 2026-09-03
agent: parent
canon_generation: 4
ids:
  - DEC-066-INFER
  - DEC-066-PA
  - DEC-066-SHIFT
  - DEC-066-TARGET
  - DEC-066-VC
  - DEC-066-ZEROMODEL
  - DEC-067-RUNNER
verdict: propose
---

# Approaching-PA recovery: understand the failure, MAP first, NUTS only if competitive

## Scope

Existing ids only. No new `DEC-*`. Official MAP `kinuv-KGAS066-uvsign-map` stays read-only. Receding G3 (`sd3ckpf2`, `docs/reviews/artifacts/2026-08-30-g3-nuts/`) stays the 066 NUTS product. `KGAS066-latest` stays receding. No G4. No G5. No GPU. No KGAS007. Do not quote S2 16/50/84 or inner `dV/dr`. Do not stack receding and approaching draws. Do not overwrite `docs/reviews/artifacts/2026-08-30-g3-nuts/` or the four-chain `pa25/` merge already on disk.

Human review surface: `docs/reviews/artifacts/2026-09-02-kgas066-leftover-and-modes/` (plus `pa25/failure.md` and `pa25/approaching-map/`).

## What failed (evidence, not chat)

Parallel approaching NUTS (`KGAS066-20260902T170918Z-nuts-pa25-c{1..4}`) finished. Merge wrote `pa25/` at 2026-09-03T13:10Z as `COMPLETED_UNMIXED` (`mixing_pass: false`). Sampler label leaked `laplace_mh` because `product_record` still defaults that name — do not call this merge NUTS.

| Chain | seed | wall | mean steps | PA median | vsys | V_0 | gas_sigma | r_t |
|---|---|---|---|---|---|---|---|---|
| c1 | 11 | 8.8 h | 90.9 | 14.7° | 8515 | 391 | 37.7 | 6.8e-4″ |
| c2 | 12 | 2.7 h | 22.6 | 64.4° | 8390 | 275 | 36.0 | 7.1e-4″ |
| c3 | 13 | 7.7 h | 76.0 | 14.9° | 8514 | 389 | 38.1 | 6.1e-4″ |
| c4 | 14 | 19.8 h | 195 | exploded (~414°, flux 1e262) | — | — | — | — |

Merged R_hat: PA 22.3, vsys 8.6, V_0 20.6. ESS tens–low hundreds. Receding comparison (`sd3ckpf2`): R_hat ≤ 1.004, ESS ≥ 889, mean steps **~10**, PA locked at 200.05°.

Serial `xgepg7qy` (same seeds on c1–c3) is leftover duplicate work. Race watcher merged but did not delete it.

**Root cause is not “c4 was unlucky.”** Three independent facts:

1. **Init was not an approaching MAP.** Worker copied official receding MAP θ and overwrote only `pa_deg=25.2` (`scripts/run_kgas066_nuts_headless.py`). Receding NUTS started from a converged MAP (chi2 168675.6). Approaching NUTS started ~31k χ² away from the vis minimum.
2. **Official two-start L-BFGS already scored the 180° start and discarded it.** `stage_a_map.json` message: `PA=205.2 Δχ²=35552.7, PA=25.2 Δχ²=4260.2`. Approaching start is a local basin ~8× weaker on the V=0 gate. DEC-066-PA is “fit; seed 205.2 receding” — the loser start was never a peer product.
3. **c1–c3 do not form one mode.** Even without c4: PA 14.7° vs 64.4°, vsys spread ~125 km/s, V_0 spread ~116 km/s, `r_t` collapsed to ~0 (below the L-BFGS box 0.5″). Dropping c4 cannot pass R_hat ≤ 1.01 / ESS > 400.

SB leftover remains structured (vel span > uv span). Frozen Wiener I_CO is not 180°-symmetric, so PA+180° is not a free kinematic flip.

## Architect verdict (selected path)

**Do not relaunch four 20 h NUTS jobs from PA=25.2 + receding MAP.** That experiment is done. It failed for geometry, not wall-clock.

**Track 0 — stop the leak.** Delete serial `xgepg7qy` if still Running. Do not wait for its chain 4.

**Track 1 — diagnostic merge c1–c3 only.** Local, no CANFAR. Write `pa25/c1c3-diagnostic/` (do not overwrite `pa25/kgas066_nuts.json`). Expect `COMPLETED_UNMIXED`. Record per-chain physical medians and mixing. This answers “is c4 the whole problem?” with a file, not a chat.

**Track 2 — approaching-only Stage A L-BFGS to a new tree.** Single start PA=25.2 (and optional finite starts at c1 / c2 medians, `r_t` boxed at 0.5″ so NUTS-collapsed `r_t` is not a MAP seed). Write `docs/reviews/artifacts/2026-09-02-kgas066-leftover-and-modes/pa25/approaching-map/`. Do **not** write `kinuv-KGAS066-uvsign-map`. Report chi2, Δχ² vs V=0, Δχ² vs official receding MAP, landed PA. Also eval vis chi2 at finite c1/c2/c3 medians (clamp `r_t` to 0.5″ for a second column so the 6e-4″ draw is not the quoted number).

**Track 3 — gate before any new NUTS.** Implementer decides on the numbers:

- If approaching MAP PA walks to within 20° of 200°, or Δχ² vs V=0 stays ~4e3 while receding is 3.6e4: **approaching is not a competitive 066 mode.** Write that in `pa25/failure.md`. Do not launch NUTS. Receding remains the product.
- If approaching MAP stays approaching (PA in [0, 90] or [270, 360]) **and** Δχ² vs official MAP is better than −5000 (i.e. approaching chi2 within 5000 of 168676): launch **4×1-chain CPU** `nuts-pa25` from that **full MAP θ** (not PA-only overwrite), same 200/600, flexible, no GPU, artifact still `pa25/` but a dated subdir `pa25/nuts-from-map/`. Merge must reject non-finite chains; require ≥3 finite chains; mixing gate unchanged (R_hat < 1.01, ESS/tail > 400). Fail → `COMPLETED_UNMIXED`, label `sampler: nuts` only if autodiff + mixing_pass.

The −5000 bound is a gate, not an ADR. Official loser Δχ² gap is ~31300; this bound exists so a *new* approaching MAP that actually competes can still earn NUTS.

**Track 4 — merge hygiene (small code).** `merge_nuts_chains.py` accepts 3 or 4 shards; drops non-finite draws/chains; `product_record` must not write `sampler: laplace_mh` on a NUTS merge. Unit test: exploded c4-like shard is dropped; 3 finite disagreeing chains stay `COMPLETED_UNMIXED`.

## Rejected alternatives

- “Rerun four NUTS from 25.2 because c4 exploded” — c1–c3 already disagree; official L-BFGS already discarded that start.
- “dynesty / nested sampling this card” — evidence Z with leftover structured is not G4/G5; live-point wall is not cheaper than a MAP. Literature notes may recommend it later.
- “Average c1 and c3 because they look similar” — two chains at 15° with `r_t~0` is not a mixing pass.
- “New official MAP” — leftover-vs-velocity still True; DEC-066-INFER product stays `kinuv-KGAS066-uvsign-map`.
- GPU, G4, G5, KGAS007, unfreeze `i`, logit `[0.5, 15]`, stack modes, steal `KGAS066-latest`.

## Residual risks

1. Approaching L-BFGS may again land at Δχ²~4260. That is a science result (Ico + vis prefer receding), not a sampler bug. Do not “fix” it by stacking.
2. c1/c2 median chi2 with `r_t~0` is unphysical; the clamped-0.5″ column is the one that may be compared to MAP.
3. Serial delete is `canfar delete --force`. If the session is already gone, log and continue.
4. `product_record` sampler-name leak can confuse STATUS if Track 4 is skipped.
5. A competitive approaching MAP is unlikely given the 31k Δχ² gap; the expected execute is Tracks 0–2 + failure note, not another 20 h NUTS.
6. Real-066 16/50/84 stay uncalibrated. Leftover remains SB-dominated.

## Execute if accepted

1. Kill serial `xgepg7qy` if Running. STATUS one-liner.
2. Diagnostic c1–c3 merge → `pa25/c1c3-diagnostic/`. Commit the JSON + note, not the 589 kB four-chain draw file again unless needed.
3. Approaching-only L-BFGS + chi2 at chain medians → `pa25/approaching-map/`. New tree only.
4. Apply Track 3 gate. If fail: write `pa25/failure.md`, leftover PNG at approaching MAP if cheap, stop. If pass: launch 4×1-chain `nuts-pa25` from full MAP θ; watcher optional.
5. Merge hygiene + tests in the same execute (needed for c1–c3 diagnostic and any later NUTS).
6. Literature notes (astro + CS) land under `docs/reviews/` as non-ADR. Field Guide mailbox + CHANGELOG + STATUS. Do not start G4. Official MAP unchanged.
7. Commit and push `origin/dev` after propose, after tally, after diagnostic/MAP artifacts, and after any dispatch.

## STATUS updates required

- `next_role: board`
- `board: open`
- `last_propose:` this file
