---
role: reviewer
seat: a
date: 2026-09-03
agent: review-a
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

# Review a: approaching-PA recovery (MAP first; no NUTS this card)

Do not read the other seat's review file. Do not implement.

Scope check: official MAP `kinuv-KGAS066-uvsign-map` stays read-only. Receding G3 (`sd3ckpf2`, `docs/reviews/artifacts/2026-08-30-g3-nuts/`) stays the 066 NUTS product. `KGAS066-latest` stays receding. No G4. No G5. No GPU. No KGAS007. Do not quote S2 16/50/84 or inner `dV/dr`. Do not stack modes. Do not overwrite the four-chain `pa25/` merge already on disk. Existing ids only. That selected path (Tracks 0–2 + failure note + merge hygiene) is accept-eligible.

Disk (not chat). Official `stage_a_map.json` message: `PA=205.2 Δχ²=35552.7, PA=25.2 Δχ²=4260.2`. Canon numbers: `chi2_map=168675.59555208942`, `chi2_zero=204228.2478024876`, `delta_chi2=35552.65225039818`, landed PA `199.7298`. Approaching two-start chi2 is `204228.2478 − 4260.2 = 199968.05` (gap vs official MAP **31292.5**). `pa25/summary.json`: `COMPLETED_UNMIXED`, `mixing_pass: false`, `sampler: "laplace_mh"`, `leftover_chi2_structured: false`, `pa_deg.rhat=22.299`. Receding `2026-08-30-g3-nuts/summary.json`: `sampler: "nuts"`, `mixing_pass: true`, `leftover_chi2_structured: true`, R_hat ≤ 1.004, ESS ≥ 889, `chi2_nuts_mean=167486.7639`. Three-way leftover (`comparison.json`): MAP / NUTS-mean / Stage B all `leftover_chi2_structured: true`; NUTS−MAP = **−1188.83**; Stage B χ² = **167302.19** (Δ vs A = **1373.41**). Execute as typed can still launch four 20 h `nuts-pa25` jobs on a 31 k loser because −5000 is not a competitive-mode bar, and Track 4 as typed does not catch the leftover false-negative or a finite c4 explosion.

## Attacks / bounds

1. **ADR: DEC-066-INFER + DEC-066-PA already closed approaching NUTS; the −5000 gate is too loose vs the 31 k two-start gap and vs the 1.2–1.4 k gains that actually moved 066.** DEC-066-PA is “fit receding-side PA; seed 205.2°.” Official two-start already scored the 180° start and discarded it (`Δχ²=4260.2` vs `35552.7`, factor **8.35**). DEC-066-INFER: MAP first; NUTS only if MAP Δχ² vs V=0 is real **and** vsys/PA/flux mocks recover. Approaching Δχ² vs V=0 is real (4260) but is the loser start. S1 mock recovery is the receding inject; no approaching mock exists. Track 3 still licenses 4×1-chain CPU NUTS if a new L-BFGS stays in `[0, 90]∪[270, 360]` and lands within 5000 of 168676 (chi2 ≤ **173675.6**). That bar is **26292** better than the official approaching start (199968 → 173676) and still **5000** worse than receding MAP — **4.21×** the receding NUTS-mean gain (1188.83) and **3.64×** the Stage B N=7 λ=0 gain (1373.41). `exp(−Δχ²/2)=exp(−2500)` is not a peer posterior. Propose residual 5 already expects Tracks 0–2 + `failure.md`, not another 20 h NUTS. The −5000 number is a second lottery on a start vis already rejected, not a competitive-mode test.

   **Bound:** this card does **not** launch `nuts-pa25`. Track 3 is write `pa25/failure.md` (and a cheap leftover PNG at the approaching MAP if the chi2 eval is already in hand). Do not treat “implementer decides on the numbers” as a license to dispatch. Numeric abort stays: PA walks to within 20° of 200°, **or** Δχ² vs V=0 stays ~4e3 while receding is 3.6e4, **or** Δ vs official MAP is worse than **−1373** (Stage B scale; prefer **−1189**, the NUTS-mean scale). The −5000 window is void. If Track 2 somehow lands approaching **and** `chi2_approaching − 168675.5956 < 1373.41` (i.e. beats or matches Stage B on Stage A arctan), STATUS one-liner and a **new** propose — INFER still owes approaching mock recovery of PA/vsys/flux, leftover is still SB-dominated at MAP / NUTS-mean / Stage B, and leftover structured is the live science blocker. Do not stack. Do not retarget `KGAS066-latest`. Official MAP unchanged.

2. **`pa25/summary.json` wrote `leftover_chi2_structured: false` without evaluating leftover; Track 4 as typed does not fix it.** `scripts/merge_nuts_chains.py` calls `product_record(..., leftover_chi2_structured=False)` and `write_nuts_product_plots(..., leftover=False)`, then swallows plot exceptions. Receding product and the three-way comparison all have leftover True (vel span > uv span; gate SB-dominated). STATUS blocker is leftover True. The unmixed merge’s False is a default, not a measurement — the same class of bug as G3’s hardcoded `r_t_at_floor: true` and the leftover card’s hardcoded True (now inverted). `test_product_record_does_not_force_leftover_true` only locks the opposite polarity. A reader of `pa25/summary.json` can conclude leftover cleared on the approaching run.

   **Bound:** merge / c1–c3 diagnostic must **not** write `leftover_chi2_structured: false` unless `leftover_velocity_structured` ran on leftover arrays at a finite θ. If leftover is skipped (unmixed, exploded, `leftover=False`), omit the key or write `null` / `"unevaluated"` — do not serialize a science False. Track 2 approaching MAP **must** measure leftover on 881×95 at `s=0.5136098555284736` (`hann_then_bin`, `NPZ_UV_SIGN=-1`) and record the bit from arrays. Unit test: merge of four shards with `leftover=False` does not emit `leftover_chi2_structured is False` as a measured flag (omit/null). Do not copy True from G3 either.

3. **Track 4 “drop non-finite” does not drop the recorded c4; `product_record` fallback *is* `laplace_mh`.** Propose table: c4 PA ~414°, flux `1e262`. Those are finite floats. `merge_nuts_chains.py` `nargs=4` stacks all four; `np.stack` keeps c4. `product_record` does `label = NUTS_SAMPLER if (autodiff_ok and mixing_pass) else SAMPLER_NAME` with `SAMPLER_NAME="laplace_mh"` (`kinuv.infer.posterior`). Unmixed autodiff merge therefore **must** write `laplace_mh` today — the leak is the function, not a merge typo. Track 4 “accepts 3 or 4 shards; drops non-finite; must not write `laplace_mh`” leaves both holes if the test uses NaN/Inf instead of a c4-like finite explosion, and if it only patches the script while `product_record` still falls back to S2’s name.

   **Bound:** (a) Drop a shard if any sampled-name draw is non-finite **or** `flux` not in `(0, 1e4]` **or** `gas_sigma_kms` not in `(0, 200]` **or** `v0_kms` not in `[0, 400]` (Stage B box) **or** `|r_t_arcsec|` not in `(0, 15]`. Flux `1e262` / PA 414 must be dropped by that test, not only by `np.isfinite`. (b) Require ≥3 finite chains after the drop; else `COMPLETED_UNMIXED` and do not call it a 4-chain product. (c) `product_record` on a NUTS-kind merge: `sampler: "nuts"` only if autodiff **and** `mixing_ok(..., rhat_max=1.01, ess_min=400.0, ess_tail_min=400.0)` on the six sampled names; else `sampler` is **not** `"nuts"` and **not** `"laplace_mh"` (new string e.g. `"nuts_unmixed"`, or omit). S2 records keep `laplace_mh`. Corner plotter already refuses `!= "nuts"`. Unit test: fixture with three finite disagreeing chains (PA 14.7 / 64.4 / 14.9) plus a c4-like `flux=1e262` shard → c4 dropped, `COMPLETED_UNMIXED`, `mixing_pass is False`, `sampler != "laplace_mh"`, leftover key omitted or unevaluated. Do not overwrite `pa25/kgas066_nuts.json`; write `pa25/c1c3-diagnostic/`.

4. **Killing serial `xgepg7qy` is in scope; leftover-structured False is not a license to quote the merge as a mode.** Parallel `20260902T170918Z` all SUCCEEDED; `pa25/wall.json` `COMPLETED_UNMIXED` at `2026-09-03T13:10:25Z`. Serial is the same seeds on a leftover 4-loop. Track 0 `canfar delete --force` is ops hygiene, not a science gate — do it if `canfar` still lists Running, else log and continue (propose residual 3). It does not reopen approaching NUTS. Do not wait on chain 4. Do not treat the unmixed merge as a second 066 product.

## Comments

1. `major` -- Do not launch approaching NUTS this card. Track 3 = `pa25/failure.md`. Void the −5000 gate. Competitive bar if a future propose exists: Δ vs official MAP better than **−1373.41** (Stage B), not −5000. Official two-start + DEC-066-INFER mock-recovery (receding-only S1) + DEC-066-PA seed 205.2 already discarded 25.2. Attack 1.

2. `major` -- Merge / diagnostic must not write `leftover_chi2_structured: false` unless leftover arrays were scored. Measure leftover at the Track 2 approaching MAP on 881×95. Unit test for omit/null when leftover is skipped. Attack 2.

3. `major` -- Drop exploded-but-finite shards (c4 flux `1e262` / PA 414), not only NaN/Inf. Fix `product_record` fallback so an unmixed NUTS-kind merge is never `sampler: laplace_mh`. Unit test as in Attack 3. Do not overwrite the on-disk four-chain `pa25/` JSON.

4. `minor` -- Track 0 kill serial if Running. Same seeds; parallel merge already exists. Log if the session is gone. Not a science blocker.

5. `minor` -- Track 2 chi2 at c1/c2/c3 medians: the quoted column is `r_t` clamped to 0.5″. Do not seed L-BFGS at `r_t ~ 6e-4`. Optional extra starts at those medians stay optional; the 25.2 single start plus the three evals are required. DEC-066-SHIFT `(dx, dy)` are MAP parameters on that new tree (freeze after MAP if you follow the G3 leave; do not copy receding `(dx, dy)` into an approaching MAP start).

6. `minor` -- Reject-this-wave list stays: no GPU, no G4/G5, no 007, no logit of `[0.5, 15]`, no in-place official MAP write, no stack, no steal of `KGAS066-latest`, no S2 16/50/84, no inner `dV/dr`. Receding G3 folder untouched. CHANGELOG + Field Guide mailbox. Official MAP unchanged.

## Residual risks

1. Approaching L-BFGS lands again at Δχ² ~ 4260. That is the two-start science result (frozen Wiener I_CO is not 180°-symmetric). Do not “fix” it by stacking or by another NUTS. Carry-forward from the propose; comment 1 is the lock.

2. c1/c2 median chi2 with `r_t ~ 0` is unphysical. Clamped-0.5″ column only. Carry-forward.

3. Serial `canfar delete --force` on a session that is already gone. Log and continue. Carry-forward.

4. **(new)** `leftover_chi2_structured: false` on the unmixed `pa25/` merge is already on disk. Do not quote it. Comment 2 is the lock for new files; leave the existing four-chain JSON as the failed experiment (propose: do not overwrite).

5. **(new)** `product_record` else-branch writes `laplace_mh` for every unmixed autodiff merge until comment 3 lands. STATUS/mailbox can re-leak if Track 4 is “script only.”

6. Real-066 16/50/84 stay uncalibrated. Leftover remains SB-dominated at the official points. Do not start G4. Carry-forward.

7. A competitive approaching MAP is unlikely given the 31292.5 Δχ² gap. Expected execute is Tracks 0–2 + `failure.md` + hygiene, not 80 h of CANFAR. Carry-forward; comment 1 forbids the surprise dispatch.

## STATUS updates required

- `verdict` and `severity` as in the header (`accept`, `major`)
- `last_review_a`: this file
- Do not set `board: accepted` (parent tallies)
