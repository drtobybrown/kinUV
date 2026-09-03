---
role: reviewer
seat: b
date: 2026-09-03
agent: review-b
canon_generation: 4
ids:
  - DEC-066-INFER
  - DEC-066-PA
  - DEC-066-SHIFT
  - DEC-066-TARGET
  - DEC-066-VC
  - DEC-066-ZEROMODEL
  - DEC-067-RUNNER
verdict: accept
severity: major
propose: docs/reviews/2026-09-03-propose-approaching-recovery.md
---

# Review b: approaching-PA recovery

Do not read the other seat's review file. Do not implement.

Scope check: do not relaunch four 20 h jobs from PA=25.2 plus receding MAP; kill leftover serial `xgepg7qy`; write a diagnostic c1–c3 merge; run approaching-only Stage A L-BFGS to a **new** tree; official `kinuv-KGAS066-uvsign-map` stays read-only; receding G3 (`sd3ckpf2`) stays the 066 NUTS product; `KGAS066-latest` stays receding; no G4; no G5; no GPU; no 007; do not stack modes; do not quote S2 16/50/84 or inner `dV/dr`. That recovery is accept-eligible. Execute as typed still auto-launches a second approaching NUTS product if a new L-BFGS lands within 5000 χ² of 168676, will quote unphysical `r_t~0` medians as a table column, and will let a 3-shard merge mint `sampler: nuts` after dropping c4.

Live numbers used below (official MAP tree + landed JSONs; not a chat summary):

| Product | source | value |
|---|---|---|
| Official two-start | `kinuv-KGAS066-uvsign-map/stage_a_map.json` `message` | PA=205.2 Δχ²=35552.7, PA=25.2 Δχ²=4260.2 |
| Receding MAP | same JSON `chi2_map` / `delta_chi2` / `chi2_zero` | 168675.59555208942 / 35552.65225039818 / 204228.2478024876 |
| Approaching basin χ² | `chi2_zero − 4260.2` | 199968.0 (Δ vs official MAP = **+31292.5**) |
| Receding NUTS | `docs/reviews/artifacts/2026-08-30-g3-nuts/` | PA 200.054°, `mean_num_steps` 9.99, R_hat ≤ 1.004, ESS ≥ 889, χ² at mean 167486.8 (Δ vs MAP = −1189) |
| pa25 merge | `…/leftover-and-modes/pa25/summary.json` + `wall.json` | `COMPLETED_UNMIXED` 2026-09-03T13:10:25Z; `sampler: laplace_mh`; R_hat PA 22.30 / vsys 8.60 / V_0 20.56; `leftover_chi2_structured: false` |

Worker init (on disk, `scripts/run_kgas066_nuts_headless.py`): `start = dict(params)` then `start["pa_deg"] = float(pa_init)`. Approaching NUTS copied receding MAP θ and overwrote only PA. Receding NUTS started from a converged MAP. That experiment is done.

## Attacks / bounds

1. **DEC-066-PA already chose receding. A new approaching MAP is diagnostic-only. Track 3 auto-NUTS is not licensed.** PA: “Fit receding-side PA (E of N). Seed 205.2° from the CO YAML.” Official two-start already scored the 180° start and discarded it (Δχ² 4260.2 vs 35552.7). INFER: MAP first; “Do not sample a likelihood that cannot beat the noise pedestal.” The leftover-and-approaching card already licensed one diagnostic approaching NUTS from a PA-only overwrite. It failed for geometry (`mixing_pass: false`; c1/c3 PA ~15°, c2 ~64°, c4 exploded; receding mixed at PA ~200.05 with ~10 steps). Track 2 to a new tree is the right autopsy. Track 3 then says: if that autopsy MAP stays approaching **and** χ² is within 5000 of 168676, launch 4×1-chain `nuts-pa25` and treat it as a product path (`pa25/nuts-from-map/`, `sampler: nuts` after mix). That is a second 066 NUTS product on the approaching side. DEC-066-PA does not authorize a peer product. A surprising MAP that closed ~26k of a 31k gap would be a science event (leftover still SB-dominated; official MAP read-only). It is not an implementer gate.

   **Bound:** Track 2 approaching L-BFGS is diagnostic-only. Do **not** write `kinuv-KGAS066-uvsign-map`. Do **not** retarget `KGAS066-latest`. Do **not** overwrite `docs/reviews/artifacts/2026-08-30-g3-nuts/` or `pa25/kgas066_nuts.json`. Track 3 does **not** dispatch NUTS this card. Write `pa25/failure.md` (and the approaching-map README) with landed PA, χ², Δχ² vs V=0, Δχ² vs official 35552.65. If a new approaching MAP is ever within 200 of official χ², STATUS one-liner and a **new propose** — do not auto-launch. Do not start G4.

2. **The −5000 Track 3 bound is a loophole relative to the 31292.5 historical gap and the 1189 NUTS-vs-MAP move.** Known approaching basin χ² ≈ 199968 is 31k worse than 168676; it cannot pass “within 5000.” The bound exists so a *different* L-BFGS (Track 2’s optional c1/c2-median starts) can invent a basin the official two-start never ran and still earn ~80 h of CPU NUTS while remaining 4999 χ² worse than the receding product. Receding NUTS vs MAP was Δχ² = −1189. −5000 is **4.2×** that same-basin move and **16%** of the official discard (35552.7 − 4260.2 = 31292.5). ZEROMODEL reports Δχ² vs V=0, not “within 3% of the winner’s raw χ².” A mode with Δχ² vs V=0 = 30553 (the −5000 edge) is still ~5k worse than receding 35553 and was never the DEC-066-PA seed.

   **Bound:** no approaching NUTS this card (attack 1). If a later card needs a numerical “competes” gate, require **both** Δχ² vs official MAP ≥ **−200** (same-basin noise, ~1/6 of the receding NUTS-vs-MAP 1189) **and** approaching Δχ² vs V=0 ≥ **35000** (must close to within ~553 of official 35552.7, not 5000 of raw χ²). −5000 is rejected as the gate.

3. **Do not quote vis χ² at NUTS medians with `r_t ~ 0`.** Track 2 writes a second column at clamped 0.5″ “so the 6e-4″ draw is not the quoted number,” then still evaluates the unclamped median. Residual 2 admits that column is unphysical. `RT_BOUNDS_ARCSEC=(0.5, 15)` is the L-BFGS box (G2: do not logit it). Receding NUTS already showed leaving 0.5″ for 0.224″ bought Δχ² = −1189. Quoting c1/c3 at `r_t ~ 6e-4″` (V_0 ~390, vsys ~8515 vs official vsys 8098.8) as a number next to 168676 will be copied into STATUS / CHANGELOG / `failure.md`. That is not a MAP and not inside the box.

   **Bound:** quoted approaching-map / `failure.md` / STATUS / CHANGELOG χ² columns are **clamped `r_t=0.5″` only**. Unclamped `r_t < 0.5″` evals, if computed at all, live only under `pa25/approaching-map/raw/` with `unphysical_rt_below_lbfgs_box: true` and must not appear in the README table. Do not compare unclamped median χ² to official 168675.6. Do not seed Track 2 L-BFGS from full NUTS-median θ (vsys ~8515, V_0 ~391, `gas_sigma` ~38). Seeds are official MAP other-θ with PA ∈ {25.2, optional 14.7, optional 64.4} and `r_t` boxed at 0.5″. Using unmixed draws as MAP starts is circular and can invent a third basin neither official two-start scored.

4. **3-chain merge is diagnostic. DEC mixing for `sampler: nuts` is four finite chains.** Live `scripts/merge_nuts_chains.py` is `nargs=4`. Licensed 066 product (`sd3ckpf2`) is 4×600, R_hat < 1.01, ESS/tail > 400 on six names. Track 1 c1–c3 diagnostic is the right file answer to “is c4 the whole problem?” (no: PA 14.7 vs 64.4). Track 4 then “accepts 3 or 4 shards; require ≥3 finite chains; mixing gate unchanged.” That lets execute drop exploded c4, keep three chains, and — if a later from-MAP run agrees — write `sampler: nuts` on n_chain=3. Gelman–Rubin on three shards is not the receding product bar. Propose already says dropping c4 cannot pass R_hat on *this* merge; the hygiene change outlives this card.

   **Bound:** `sampler: nuts` requires **n_chain == 4** finite shards **and** `mixing_ok(..., rhat_max=1.01, ess_min=400.0, ess_tail_min=400.0)` **and** autodiff. 3-shard merge writes only `pa25/c1c3-diagnostic/` (or a dated diagnostic dir); it may compute R_hat but cannot set `sampler: nuts` or retarget latest. Unit test: exploded c4-like shard dropped; 3 finite **disagreeing** stay `COMPLETED_UNMIXED`; **3 finite agreeing** still `sampler != "nuts"`. Four finite + mix → `nuts`. `product_record` must not write `laplace_mh` on a NUTS merge that failed mix (`COMPLETED_UNMIXED`, sampler unset or a non-`nuts` label that is not the MH path). Do not call the unmixed pa25 merge NUTS.

5. **pa25 merge already lies about leftover. Track 4 as typed only names the sampler leak.** `merge_nuts_chains.py` passes `leftover_chi2_structured=False` into `product_record` without leftover arrays. Landed `pa25/summary.json` has `leftover_chi2_structured: false` and `r_t_at_floor: false` while official leftover-vs-velocity is True and c1–c3 `r_t` collapsed below the box. G3 hardcoded `r_t_at_floor: true`; leftover-and-approaching already forbade hardcoded leftover True. Writing False when leftover was not measured is the same class of product lie.

   **Bound:** if leftover arrays were not computed, omit `leftover_chi2_structured` or set it only from `leftover_velocity_structured` on arrays. Do not write `false` as “not measured.” Do not overwrite the four-chain `pa25/kgas066_nuts.json` to “fix” the leak in place; note the lie in `failure.md`. New diagnostic / from-map JSON follows the omit-or-measure rule. Unit test: merge without leftover npz does not persist `leftover_chi2_structured: false`.

## Comments

1. **major.** DEC-066-PA: approaching MAP is diagnostic-only. Track 3 does not launch NUTS this card. Official MAP unchanged. `KGAS066-latest` stays receding. Do not start G4. If a new MAP ever competes, new propose — not an implementer auto-dispatch. Attack 1.

2. **major.** Reject −5000 as a NUTS gate. Historical discard is Δχ² 31292.5; receding NUTS-vs-MAP is 1189. A later “competes” bound, if any, is Δ vs official MAP ≥ −200 **and** Δχ² vs V=0 ≥ 35000. Attack 2.

3. **major.** Do not quote χ² at `r_t ~ 0` NUTS medians. README / `failure.md` / STATUS / CHANGELOG use the `r_t=0.5″` clamp only. Track 2 seeds = official MAP other-θ + PA variant; not full c1/c2 median θ. Attack 3.

4. **major.** 3-chain merge is diagnostic only. `sampler: nuts` needs four finite chains + mix + autodiff. Tests: drop exploded shard; 3 disagree → `COMPLETED_UNMIXED`; 3 agree → still not `nuts`. Attack 4.

5. **major.** Do not persist `leftover_chi2_structured: false` when leftover was not measured. Do not in-place rewrite `pa25/kgas066_nuts.json`. Attack 5.

6. **minor.** Track 0: `canfar delete --force` on `xgepg7qy` if Running; if already gone, log and continue. Do not wait for serial chain 4. Do not relaunch PA-only overwrite.

7. **minor.** Receding `sd3ckpf2` remains the only 066 `sampler: nuts` product. Do not stack receding and approaching draws. Do not quote inner `dV/dr` or S2 16/50/84. Stage A remains arctan. Frozen `(dx, dy)` at MAP for any NUTS that a later card might run.

## Residual risks

1. Approaching L-BFGS may again land at Δχ² ~4260. That is Ico + vis preferring receding, not a sampler bug. Do not “fix” it by stacking or by loosening the NUTS gate. (propose residual 1, kept)

2. Serial delete is `--force`. Session may already be gone. (propose residual 3)

3. Real-066 16/50/84 stay uncalibrated (S2 Laplace SBC failed 68/95). Leftover remains SB-dominated. Do not start G4. (propose residual 6)

4. **New.** Optional c1/c2-median L-BFGS starts (vsys ~8515, V_0 ~391) can invent a third basin the official two-start never scored. Comment 3 forbids those seeds; if execute ignores it, Track 3’s old −5000 gate could have fired on nonsense.

5. **New.** 3-chain `sampler: nuts` hygiene will be copied by the next parallel race. Comment 4 is the lock; a later card that “three finite is enough” would mint a weaker product than `sd3ckpf2`.

6. **New.** Landed `pa25/summary.json` already ships `leftover_chi2_structured: false` and `sampler: laplace_mh` on a NUTS merge. Readers will quote it. `failure.md` must say both are merge bugs, not science.

7. Competitive approaching MAP is unlikely given the 31k gap. Expected execute is Tracks 0–2 + Track 4 hygiene + failure note. Not another 20 h NUTS. Official MAP read-only.

## STATUS updates required

- `verdict: accept`
- `severity: major`
- `last_review_b:` this file
- Do not set `board: accepted` (parent tallies)
