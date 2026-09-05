---
role: reviewer
seat: b
date: 2026-09-05
agent: review-b-revise
canon_generation: 4
ids:
  - DEC-066-TARGET
  - DEC-066-INFER
  - DEC-066-INC
  - DEC-066-PA
  - DEC-066-VC
  - DEC-066-ZEROMODEL
  - DEC-067-RUNNER
verdict: accept
severity: major
propose: docs/reviews/2026-09-05-propose-kgas007-nuts-and-live-s3.md
---

# Review b: KGAS007 NUTS revise (steal_latest / G3 / i_rad / loader / quote / tests)

Do not read the other seat's review file. Do not implement.

Scope check (disk, not chat): approaching closed. Receding `sd3ckpf2` is the 066 NUTS product. Official MAP `kinuv-KGAS066-uvsign-map` stays read-only. `code_freeze: false`. No new `DEC-*` id. Do not start G4. Live runner is still a two-way switch; this card is a propose, not a landed patch.

007 leftover on disk (`docs/reviews/artifacts/2026-09-05-kgas007-stage-a-map/leftover_chi2.json`): `leftover_chi2_structured: false`, uv span 0.737 > vel 0.214, `quote_inner_slope: false` only because `r_t=0.5`. 007 MAP (`results/KILOGAS007/kinuv-KGAS007-stage-a-map/stage_a_map.json`): `i_rad_frozen: 0.5044`, PA 151.602°, `dx=-0.0186`, `dy=-0.0910`, χ²=122070.76, Δ vs V=0 = +6211.63, `uv_stored: wavelengths`, `n_row=956`, `n_chan=66`, `s=0.570735`. That is not the 066 SB-dominated gate.

Revise now names the six first-round holes. Live code still steals if execute skips any of them. Rubber-stamp is invalid. The six are tight enough to accept if the majors below are fixed during execute and stay red before any `canfar create`.

## Attacks / bounds

1. **Live `kind.py` is still `pa25` vs else.** `steal_latest` is `not is_approaching_kind`. `artifact_dir_for_kind` returns `2026-08-30-g3-nuts` unless `"pa25" in kind`. `pa_init_deg` returns 199.73 unless approaching or override. `tests/test_canfar_runner.py` still only locks `steal_latest("nuts-pa25") is False` and `steal_latest("nuts") is True`. Revise step 2 names the third kind. Bound: those three functions plus `--kind nuts` / default-galaxy exit 2 plus `point_latest` not called must be unit-tested on `nuts-kgas007` before dispatch. `steal_latest("nuts")` stays True.

2. **`merge_nuts_chains.py` still hardcodes the official 066 MAP and `kgas066_nuts.json`.** Default `--artifact-dir` is `ARTIFACT_G3`. `map_path` is `…/KILOGAS066/kinuv-KGAS066-uvsign-map/stage_a_map.json` with no `--map-path`. It writes `kgas066_nuts.json` and `kind: "nuts"`. `--artifact-dir` alone does not stop a 066 `(dx, dy)` / PA stitch. Revise step 6 names refuse-G3 + 007 MAP + `kgas007_nuts.json`. Bound: a unit test must lock all four, not only dest refuse. Do not run live defaults on 007 shards.

3. **`write_nuts_product_plots` still defaults `artifact_dir=ARTIFACT_G3`, leftover-loads `load_kgas066`, and copies `kgas066_nuts.json`.** `write_leftover_at_params` is called without `i_rad` from that path (066 43.9°). Revise only says pass `artifact_dir=`. Bound: 007 leftover/plots must pass 007 `data`/`tmpl`/`grid`/`i_rad=0.5044`; never take the 066 fallback; never write `kgas066_nuts.json`; G3 sentinel untouched. 007 worker must refuse dest containing `2026-08-30-g3-nuts` (same as merge), not only the merge script.

4. **`make_potential` has no `i_rad`.** `potential_unconstrained` neither. `predict_binned` / `sky_cube` default `inclination_rad()` = DEC-066-INC 43.9°. 007 MAP froze `i_rad=0.5044`. A host `predict_binned(..., i_rad=0.5044)` identity can pass while NUTS `U` still closes 43.9°. Bound: MAP-θ identity `|chi2-122070.76|<1` on 956×66 must go through `make_potential(..., i_rad=0.5044)` at MAP `z6`, not a side-channel `predict_binned`. A second assert: `make_potential` with `i_rad=None` still matches official 066 `|chi2-168675.6|<1` when 066 vis exist. Do not edit `geometry.py` 066 ba/PA.

5. **`quote_inner_slope` in `flags.py` flips true when `r_t` leaves 0.5″ and leftover is uv-dominated.** 007 leftover is already unstructured (uv 0.737 > vel 0.214). `write_leftover_at_params` copies `qflags["quote_inner_slope"]`. Hard-set on NUTS JSON only leaves `leftover_chi2.json` true. Bound: hard-write `quote_inner_slope: false` on every 007 NUTS product including leftover JSON, even if `r_t` leaves the floor. Do not copy `leftover_gate: SB-dominated`. Do not quote `dV/dr` or form `V_0/r_t`.

6. **007 vis is wavelengths; `load_kgas066` requires `u_m`.** Landed `_load_007` converts `u * c / f_ref`, refuses `u_m`, does not invent `time`/`baseline`. Revise step 4 names this. Bound: 007 worker source must not call `load_kgas066` or any 066 vis/MAP/Ico path (KGAS66 / KILOGAS066). Identity at MAP θ binds the scale.

7. **DEC-066-INFER first sentence still requires mock recovery.** User waiver is this card only. Do not amend INFER. Product JSON records `infer_mock_recovery: waived-this-card` / `infer_007_s1_waived: true` and `intervals_calibrated: false`. Not a 452/γ/G4 license.

8. **DEC-067-RUNNER items 3–4 still say copy into G3 and `kinuv-KGAS066-{sha6}-nuts`.** Live `session_name()` ignores `--galaxy`. `write_job_status_md` non-`pa25` branch writes Phase `G3 066 NUTS` and Next Step copy-into-G3. User licensed a TARGET amend, not a DEC-067 amend. Bound: session `kinuv-KGAS007-{sha6}-nuts-kgas007-c{N}` (63-char; length on disk is 36). STATUS one-liner that items 3–4 are left. `write_job_status_md` must not name G3 for this kind. Entrypoint execs `run_kgas007_nuts_headless.py` only when `KINUV_KIND=nuts-kgas007`; default/`nuts` stays the 066 worker. Unit-test that branch (live `test_entrypoint_uses_scratch_and_venv` only asserts the 066 worker string).

9. **`DEC-066-TARGET` on disk already contains the 2026-09-05 007 license; Field Guide TARGET line is already `KGAS066 + KGAS007`.** Execute step 1 is verify-in-place, not a new DEC id. Do not rewrite `kinuv-KGAS007-stage-a-map`. Official 066 MAP unchanged.

## Comments

1. `major` — Kind `nuts-kgas007` only. `steal_latest("nuts-kgas007") is False` including `--dry-run`. `steal_latest("nuts")` stays True. `artifact_dir_for_kind` → `docs/reviews/artifacts/2026-09-05-kgas007-nuts/` (not G3, not leftover `pa25`). `pa_init_deg` → 151.6. `--kind nuts` on a 007 launch is exit 2. Default galaxy + 007 kind is exit 2. `point_latest` is not called. Tests green before any `canfar create`. Attack 1.

2. `major` — 007 merge must not run live `merge_nuts_chains.py` defaults. Unit test locks: refuse dest containing `2026-08-30-g3-nuts`; read 007 MAP (`dx=-0.0186`, `dy=-0.0910`, PA 151.6) not official 066 MAP; write `kgas007_nuts.json` never `kgas066_nuts.json`; `kind: nuts-kgas007`. `--artifact-dir` alone is not that lock. Attack 2.

3. `major` — `write_nuts_product_plots` / leftover on 007 must pass `artifact_dir=` and 007 `data`/`tmpl`/`grid`/`i_rad=0.5044`. Never the `load_kgas066` fallback. Never `kgas066_nuts.json`. 007 worker refuses a G3 dest. G3 sentinel untouched. Attack 3.

4. `major` — Close `i_rad=0.5044` into `make_potential` (optional, default None = 066 `inclination_rad()`). Identity `|chi2-122070.76|<1` on 956×66 through that `U`, not a side-channel `predict_binned`. 066 path with `i_rad=None` unchanged. Do not edit `geometry.py`. Attack 4.

5. `major` — Hard-write `quote_inner_slope: false` on 007 NUTS JSON **and** leftover JSON even if `r_t` leaves 0.5″. Measure leftover spans; record `leftover_chi2_structured` from `leftover_velocity_structured`. Do not copy `leftover_gate: SB-dominated`. Do not quote inner `dV/dr` or `V_0/r_t`. Attack 5.

6. `major` — Wavelength→metre loader as landed `_load_007` (`uv_stored: wavelengths`; refuse `u_m`; refuse `load_kgas066` and 066 vis/MAP/Ico). Attack 6.

7. `major` — Do not amend `DEC-066-INFER`. Waiver this card only. Product records `infer_mock_recovery: waived-this-card` and `intervals_calibrated: false`. No 452/γ/G4. Attack 7.

8. `major` — `session_name` uses `--galaxy` (`kinuv-KGAS007-{sha6}-nuts-kgas007-c{N}`). STATUS one-liner: DEC-067 items 3–4 left. `write_job_status_md` must not name G3 or copy-into-G3 for this kind. Entrypoint 007 worker only on `KINUV_KIND=nuts-kgas007`. Test that branch. Official 066 MAP unchanged. Attack 8.

9. `major` — Amend existing TARGET only if the on-disk text still needs the Field Guide line; no new DEC id. New tree `results/KILOGAS007/kinuv-KGAS007-nuts/` only. `sampler: nuts` only after autodiff + four finite chains + `R_hat≤1.01` and ESS>400; else `nuts_unmixed`. Never `laplace_mh` on this kind; autodiff fail → STATUS stop. Dispatch 4×1 flexible CPU only after comments 1–8 tests are green. No GPU. No G4. No G5. Attack 9.

10. `minor` — Track A isolated env is `toby_sandbox/external_fitters/` only. No `from kinms` under `src/` or `scripts/`. PATH miss → keep S1. README first body sentence: vis χ² is the fit. No pip into recovery or `$HOME`.

11. `minor` — Track B literature note is not an ADR. Ground in S1 (vis r_t 0.254″ vs CLEAN M1 94.7 vs 236.7) and 066 leftover SB-dominated. Do not claim 0.2239″ is a published inner scale.

12. `minor` — No new writes under `/arc/home/thbrown/`. HTTPS fail → `toby_sandbox/patches/`. TARGET on disk already has the 2026-09-05 license (attack 9).

## Residual risks

1. conda-forge `bbarolo` / `kinms` missing. S3 stays S1. (propose 1)
2. 007 NUTS without S1 (user waiver, this card only). Do not amend INFER. (propose 2; comment 7)
3. HTTPS push may fail; patches → `toby_sandbox/patches/`. Job `git pull` is safe only if the `/arc` repo the entrypoint cds into already has the kind/worker/merge patches. (propose 3)
4. Real-066 16/50/84 stay uncalibrated. Do not start G4. (propose 4)
5. `write_leftover_at_params` still copies live `quote_inner_slope`. Comment 5 is the override. (new, tightened)
6. Merge `--artifact-dir` without a 007 MAP path still stitches 066 `(dx, dy)`. Comment 2. (new, tightened)
7. Identity via `predict_binned` can hide a missing `i_rad` on `U`. Comment 4. (new, tightened)

## STATUS updates required

- `verdict: accept`, `severity: major`
- `last_review_b:` this file
- Do not set `board: accepted` (parent tallies)
