---
role: reviewer
seat: b
date: 2026-09-02
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
propose: docs/reviews/2026-09-02-propose-leftover-and-approaching.md
---

# Review b: leftover decomposition + approaching-mode NUTS

Do not read the other seat's review file. Do not implement.

Scope check: Track A (approaching PA 25.2°, no pool with receding) plus Track B leftover/dirty-residual diagnostics on 066 is the right card. Existing ids only. Official MAP `kinuv-KGAS066-uvsign-map` stays read-only. No G4, no GPU, no 007, no KinMS import, no logit of `RT_BOUNDS_ARCSEC=(0.5, 15)`, no S2 16/50/84, no velocity fudge. Identity PA, frozen `(dx, dy)`, mixing on six sampled names. That is accept-eligible. Execute as typed still ships a three-way leftover that copies Stage B `chi2=167302` from JSON, hardcodes `leftover_chi2_structured=True` the same way G3 hardcoded `r_t_at_floor`, and can sell a dirty-map PNG as a KinMS posterior.

Live numbers used below (read-only MAP tree + landed G3 JSON; not a chat summary):

| Product | source | chi2 |
|---|---|---|
| Stage A MAP | `stage_a_map.json` `chi2_map` | 168675.59555208942 |
| Receding NUTS mean | `leftover_chi2.json` `chi2_sum` | 167486.7639374534 |
| Stage B N=7 λ=0 | `stage_b_map.json` `chi2_map` | 167302.18673431588 |

Gap NUTS-mean vs Stage B ≈ 184.6. Stage B JSON has **no** `n_row`, `n_chan`, or `s`. Stage A does (`881`, `95`, `s=0.5136098555284736`). `v0_recovered=270.188`, `r_t_recovered=0.5000000000000019` — rings did not move recovered-arctan off the L-BFGS wall.

## Attacks / bounds

1. **Track B three-way vis χ² isolation of the ~185 gap is unearned unless Stage B leftover is recomputed on the same 881×95 Hann+bin operator.** Propose: compare official Stage A (168675.6), receding NUTS mean (167487), official Stage B rings (167302); "Reuse `kinuv.diagnostics.figures` and existing leftover/imaging plotters; do not invent a second plotter." Live leftover path cannot evaluate rings. `scripts/plot_leftover_chi2.py` loads `stage_a_map.json`, calls `kinuv.infer.map.predict_binned` with the eight Stage A names, and **exits** unless leftover sum matches that Stage A JSON within 1. `write_leftover_at_params` is the same arctan dict. `plot_fit_diagnostics.py` leftover is Stage A; `--imaging` then swaps in the Stage B *cube*. `leftover_chi2(data, model)` in `s1.py` will score any vis model, and `kinuv.infer.stage_b.predict_binned` already takes `r_knots_arcsec` / `v_knots_kms` through `hann_then_bin`. Execute item 5 never names that call. Copying `chi2_map: 167302.186` from a JSON that does not even record `(881, 95)` or `s` is the same class of product lie as G3's hardcoded `r_t_at_floor: true`. Using `v0_recovered` / `r_t_recovered` through Stage A `predict_binned` is also wrong: that recovered arctan is still on the 0.5″ wall and is not the ring vis model.

   **Bound:** Track B leftover at Stage B is `stage_b.predict_binned(data, nuisance_from_params(stage_a), tmpl, grid, r_knots_arcsec=…, v_knots_kms=…)` on `load_kgas066` with `assert vis.shape == (881, 95)` and frozen `s = 0.5136098555284736`, `NPZ_UV_SIGN=-1`. Gate: `|chi2_sum − 167302.186| < 1` (same identity Stage A leftover already enforces vs 168675.6). Store `leftover_chi2.npz` + measured `leftover_uv_span` / `leftover_vel_span` at **all three** vis points. Scalar Δχ² of ~185 without those spans does not isolate frozen Wiener Ico vs `s_1`/`c_3`. Do not quote the MAP JSON χ² as the leftover sum until that identity holds.

2. **`leftover_chi2_structured=True` in the headless `product_record` is the G3 floor bug, now on leftover.** `scripts/run_kgas066_nuts_headless.py` currently measures `r_t_at_floor` from draws (`median(rt) <= 0.5 + 1e-6`) then still passes `leftover_chi2_structured=True` into `product_record` *before* leftover arrays exist. `write_leftover_at_params` later overwrites the flag from `leftover_velocity_structured` — only if plots succeed. Plots are in `try/except Exception: log.exception` (fail-open). Approaching product JSON can ship the MAP G0 leftover bit without leftover arrays, exactly as receding shipped `r_t_at_floor: true` until draws were inspected. `flags.py` already refuses to set the leftover bit from Stage A JSON alone (`leftover_npz is None` → False).

   **Bound:** approaching (and any Track B NUTS-mean record) must call `leftover_velocity_structured` on leftover arrays at that theta. Do not pass a literal `True`. If leftover plots fail, omit the key or set it from arrays in the worker before `product_record`; do not leave the MAP G0 copy. Unit test: `product_record(..., leftover_chi2_structured=False)` plus a mock leftover that is white-in-velocity / structured-in-uv must not become `True` in the written JSON.

3. **`quote_inner_slope` will flip True at the receding NUTS mean; leftover-vs-velocity is still True. Do not quote inner `dV/dr`.** `map_quality_flags` sets `quote_inner_slope: bool(not rt_floor)` with `RT_FLOOR_ARCSEC=0.5`. Receding posterior `r_t` mean is 0.224 (`leftover_chi2.json`); `r_t_at_floor` is already false on the product. Calling flags on that mean (or teaching Track B to "the floor is gone, quote inner slope") yields `quote_inner_slope=True` while leftover vs velocity remains structured (G3 leftover JSON `leftover_chi2_structured: true`; official MAP leftover vel span > uv span in `test_map_quality_flags.py`). Roadmap: "Do not quote inner `dV/dr` when `r_t` is on the floor." That sentence is now weaker than the leftover flag. Propose residual list forbids quoting inner `dV/dr` but does not lock the flag function. Stage B recovered arctan is still `r_t=0.5`; rings are not a license to quote NUTS inner slope either.

   **Bound:** while `leftover_chi2_structured` is True, `quote_inner_slope` stays False on every 066 product this card writes (NUTS mean, approaching, Stage B leftover README). Do not print `dV_c/dr` at 0.25 BMAJ, at `r_t`, or from the inner ring. Extend `map_quality_flags` (or do not call it on NUTS-mean θ as if leftover cleared). Test: flags on `{r_t_arcsec: 0.224, …}` plus the official leftover npz → `quote_inner_slope is False`. DEC-066-VC quoted rotation curve remains Stage A arctan.

4. **DEC-066-VC: Track B "does the ~185 need `s_1`/`c_3`?" is not a warp stub and not a license to unfreeze `i` / add `h_z`.** VC: Stage A arctan is the quoted `V_c`; Stage B is 6–8 rings initialised to that arctan; keep Stage A if B does not beat A on AIC (official B did beat A; quoted product is still A). Gold-standard: "Do not treat Stage B rings as a warp." `flags.py` already ships `rings_are_not_a_warp: True`, `i_held_fixed: True`, `h_z_in_model: False`, `axisym_assumed: True`. Propose reject-list names "Unfreeze `i`. Add `h_z`." and "Do not add Fourier non-circular terms this card" but Track B's isolation question still frames a ~185 vis residual as a possible `s_1`/`c_3` discovery. If leftover-vs-velocity stays True at Stage B on the recomputed arrays, the remainder is frozen Wiener Ico, not a Fourier term, and not a later-propose stub invented in a README.

   **Bound:** Track B README states `rings_are_not_a_warp: true`. Quoted `V_c` is Stage A arctan (NUTS-mean `V_0`, `r_t` are Stage A names, not ring `V_k`). Do not unfreeze `i`. Do not add `h_z`. Do not add `s_1`/`c_3`. If Stage B leftover-vs-velocity is still True, the post-leftover gate is "still SB-dominated" and STATUS says so. A quiet image-plane M1 residual does not clear leftover-vs-velocity (propose residual 4 is correct; keep it as a gate on the vis leftover spans, not on dirty M1).

5. **KinMS "benchmark" folder name will be sold as a KinMS posterior comparison. S1 used restoring-beam M1/M2, not KinMS.** Propose stores under `docs/reviews/artifacts/2026-09-02-kinuv-vs-kinms-benchmark/` and "restates that table and adds F^{-1}{ΔV} dirty residuals." S1 (`docs/diagnostics/s1-mock.md`): vis recovered `r_t=0.25″`; cube estimator is `sky_cube` → restoring beam of the 10 km/s cube → major-axis M1/M2 (`imaging.restoring_beam_kernel`); "3DBarolo was **not on PATH**"; "the exact 94.7 number is not a 3DBarolo posterior." Tests: `test_forward.py` / `test_map.py` forbid `uvkin` / `from kinms`. Dirty `F^{-1}{ΔV}` of three 066 models vs CLEAN residual of the same sky is an image-plane diagnostic (`imaging.py` docstring: "not a second fit"). It is not KinMS, not uvkin, not a posterior comparison.

   **Bound:** do not create a folder whose name contains `kinms` unless the README first sentence is `Not KinMS. Not uvkin. S1 was restoring-beam M1/M2; 3DBarolo was not on PATH.` Prefer a subsection of `2026-09-02-kgas066-leftover-and-modes/` named `dirty-residuals/` (or `s1-restating/`). Source grep this card: no `uvkin`, no `from kinms`, no `KinMS` import in `src/` or the new scripts. Dirty-map PNG captions must say restoring-beam / dirty vis residual, not "KinMS benchmark."

6. **Field Guide mailbox is still "G3 NumPyro is a separate propose." Stale OS is a process failure if execute leaves it.** `field-guide/index.md` Mailbox: "G2 chart landed; G3 NumPyro is a separate propose." Gold-standard roadmap G3 row still says "066 CPU NUTS skipped: projected 8.3 h > 2 h cap." Receding 066 NUTS mixed (`sd3ckpf2`, 4.84 h, `sampler: nuts`). Propose execute item 6 names Field Guide; it does not name the roadmap G3 row. An implementer who patches STATUS and CHANGELOG and leaves the 80-line OS on "G3 is next" will dispatch G3 again or skip leftover.

   **Bound:** execute refreshes Field Guide Mailbox to: G3 receding NUTS landed; this card is leftover + PA 25.2; no G4. Roadmap G3 row must record 066 CPU NUTS mixed (receding; approaching this card). Do not start G4.

7. **Approaching worker/watcher will rewrite Agent Run Status as receding G3 and point plots at `2026-08-30-g3-nuts/`.** `write_job_status_md` hardcodes Phase `G3 066 NUTS {state}`, Next Step `Copy posteriors into docs/reviews/artifacts/2026-08-30-g3-nuts/`, default note `Official MAP unchanged. Do not start G4`. Worker and watcher both call it. `patch_agent_run_status` correctly stops at `# Architecture mailbox` (DEC-067), but the approaching job still overwrites the live Agent Run Status block with receding-G3 copy instructions. DEC-067 itself still says copy PNGs into `2026-08-30-g3-nuts/` — approaching must not follow that sentence. Propose residual 2 names `write_nuts_product_plots` clobber; it does not name this STATUS patcher clobber.

   **Bound:** `nuts-pa25` must not call `write_job_status_md` with the G3 Phase string or the `2026-08-30-g3-nuts` Next Step. Kind-specific artifact dir only. Worker/watcher still must **not** rewrite Architecture mailbox history (existing `patch_agent_run_status` stop is the contract; do not "fix" it by appending mailbox bullets). Parent writes the leftover-card mailbox line after Track B, not the job. Unit test already in `tests/test_canfar_runner.py` for mailbox preservation: add a case that a `nuts-pa25` note does not contain `2026-08-30-g3-nuts`.

## Comments

1. **major.** Recompute Stage B vis leftover on `stage_b.predict_binned` + official knots; `|chi2_sum − 167302.186| < 1` on shape `(881, 95)` at `s=0.5136098555284736`. Do not copy `stage_b_map.json` `chi2_map`. Do not evaluate recovered arctan (`r_t_recovered=0.5`) as Stage B. Record leftover uv/vel spans at MAP / NUTS-mean / Stage B. Attack 1.

2. **major.** Measure `leftover_chi2_structured` from leftover arrays at approaching NUTS mean (and Track B points). Delete the hardcoded `True` in `run_kgas066_nuts_headless.py` `product_record`. Fail-closed if leftover plots throw. Attack 2.

3. **major.** Do not quote inner `dV/dr` even though receding `r_t_at_floor` is false. `quote_inner_slope` stays False while leftover-vs-velocity is True. Test on `r_t=0.224` + official leftover npz. Attack 3.

4. **major.** DEC-066-VC: quoted `V_c` stays Stage A arctan. Track B does not treat rings as a warp, does not unfreeze `i`, does not add `h_z` / `s_1` / `c_3`. If Stage B leftover-vs-velocity remains True, STATUS post-leftover gate is SB-dominated. Attack 4.

5. **major.** KinMS section is S1 restating + dirty vis residuals. No folder name that reads as a KinMS posterior unless the README first sentence forbids that reading. No `uvkin` / `from kinms`. Captions: restoring-beam / dirty residual, not KinMS. Attack 5.

6. **major.** Refresh Field Guide Mailbox and roadmap G3 row (G3 receding landed; this card leftover + PA 25.2; no G4). Stale "G3 is a separate propose" after execute is a process failure. Attack 6.

7. **major.** Approaching STATUS patcher: do not rewrite Architecture mailbox history (DEC-067). Do not set Phase/Next Step to receding G3 / `2026-08-30-g3-nuts`. Kind `nuts-pa25` artifact dir only; do not retarget `KGAS066-latest`. Attack 7.

8. **major.** Reusing `plot_stage_b_vs_imaging.py` as-is labels every model "Stage B" (`label="Stage B"`, suptitle `Stage B vs 10 km/s cube`, WCS report `stage_b_model`). G3 NUTS-mean cubes already go through this path. Track B three-way moments/spectra must take an explicit model label (`Stage A MAP` / `NUTS-mean Stage A` / `Stage B rings`). Default for the official cube path may stay Stage B. Do not invent a second plotter; add a label argument.

9. **minor.** Approaching mixing is `mixing_ok(..., rhat_max=1.01, ess_min=400.0, ess_tail_min=400.0)` as the receding worker already passes — not the function default `ess_min=200`. Fail-to-mix toward PA~200° is a 180° result (`DEC-066-PA` receding-side product). Do not stack. Do not set `sampler: nuts` on an unmixed approaching JSON.

10. **minor.** `CORNER_TITLE` in `runner/plots.py` still says `066 NUTS PA 199.73`. Approaching corner must not inherit that title.

## Residual risks

1. Approaching init at receding MAP kinematics may not mix. 180° mode result. Do not stack runs to "make ESS." (propose residual 1)
2. `write_nuts_product_plots` defaults `artifact_dir=ARTIFACT_G3`. Approaching must override. (propose residual 2)
3. `point_latest` will steal `KGAS066-latest` unless `--kind nuts-pa25` skips it. (propose residual 3)
4. Track B dirty residuals still use frozen Wiener Ico; quiet M1 does not clear leftover-vs-velocity. Gate is vis leftover spans, not dirty M1. (propose residual 4)
5. Real-066 16/50/84 remain uncalibrated (S2 Laplace SBC failed 68/95; leftover structured). Product README must say so. (propose residual 5)
6. Headless wall ~5 h. Interactive agents must not block (DEC-067-RUNNER). Track B does not wait on Track A. (propose residual 6)
7. First JIT of the 066 potential can be minutes. (propose residual 7)
8. **New.** Stage B `chi2_map` was written 2026-08-28 without `n_row`/`n_chan`/`s`. If recompute misses `|Δchi2|<1`, stop and record the operator mismatch on STATUS; do not "isolate" 185 from a JSON you cannot regenerate.
9. **New.** `quote_inner_slope = not r_t_at_floor` will lie on this posterior until flags AND leftover. Execute comment 3 is the lock; a later card that "the floor is gone" can still quote inner slope.
10. **New.** DEC-067 text still says copy PNGs into `2026-08-30-g3-nuts/`. Approaching must not follow that sentence. Do not amend DEC-067 this card (no new/changed `DEC-*`).

## STATUS updates required

- `verdict: accept`
- `severity: major`
- `last_review_b:` this file
- Do not set `board: accepted` (parent tallies)
eftover_chi2_structured=True` plus a plots exception republishes the G0 flag. Track B and the approaching worker must measure or omit.

5. Real-066 16/50/84 stay uncalibrated (S2 Laplace SBC failed 68/95; leftover structured at the NUTS mean). Product README must say so. G4 is still the calibration wave.

6. Stage B image-plane M1 can look quiet while vis leftover-vs-velocity stays True: frozen Wiener Ico. A pretty cube does not clear the 185 vis gap or the leftover flag.

7. First JIT of the 066 potential can be minutes. Wall ~5 h on CPU. No GPU because the recovery venv is CPU jax-finufft (`DEC-067-RUNNER`).

8. Official MAP `r_t=0.5` is still the L-BFGS floor. NUTS leaving that floor does not license a MAP rewrite or inner \(dV/dr\).

## STATUS updates required

- `verdict` and `severity` as in the header (`accept`, `major`)
- `last_review_b`: this file
- Do not set `board: accepted` (parent tallies)
