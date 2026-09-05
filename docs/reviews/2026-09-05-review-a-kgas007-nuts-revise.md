---
role: reviewer
seat: a
date: 2026-09-05
agent: review-a-revise
canon_generation: 4
ids:
  - DEC-066-TARGET
  - DEC-066-INFER
  - DEC-066-INC
  - DEC-066-PA
  - DEC-066-VC
  - DEC-066-ZEROMODEL
  - DEC-067-RUNNER
  - DEC-OPS-AUTH
verdict: accept
severity: major
propose: docs/reviews/2026-09-05-propose-kgas007-nuts-and-live-s3.md
---

# Review a: KGAS007 NUTS + live S3 (revise)

Do not read the other seat's review file. Do not implement. First-round `review-a` was accept+major on the selected path; this file judges the **revise** only.

Execute-as-typed of this revise does **not** still steal `KGAS066-latest`, write G3, omit `i_rad` on `make_potential`, run live merge defaults, or pip recovery. Those five first-round holes are now named locks (Track C 2–7, rejected alternatives, execute 3 then 4). Live disk is still the two-way `pa25` switch; that is why tests must go green **before** `canfar create`, not a remaining steal in the typed execute order.

Scope check (disk, not chat): approaching closed. Receding `sd3ckpf2` stays the 066 NUTS product (V_0 **255** km/s, r_t **mean** 0.2239″ not science, chi2 **167486.8**). Official MAP `kinuv-KGAS066-uvsign-map` read-only (PA=199.73°, chi2=168675.6). 007 MAP `/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS007/kinuv-KGAS007-stage-a-map/stage_a_map.json`: PA=151.602°, V_0=195.98 km/s, r_t=0.5″ floor, χ²=122070.76, Δχ² vs V=0 = +6211.63, `i_rad_frozen=0.5044001538263612` (28.9°), `sampler: map`, `diagnostic_only: true`, `dec_066_target_amended: false`. Leftover `leftover_chi2_structured: false` (uv 0.737 > vel 0.214). `quote_inner_slope: false`. No G4. No GPU. No new `DEC-*`. User 2026-09-05: amend existing TARGET after accept; waive 007 S1 **this card only**; kind `nuts-kgas007` only.

Live runner (still true until execute lands the named patches): `steal_latest("nuts-kgas007") is True`; `artifact_dir_for_kind` → `2026-08-30-g3-nuts`; `pa_init_deg` → 199.73; `make_potential(data, template, grid, dx_map, dy_map)` has no `i_rad` and calls `predict_binned(..., xla=True)` so U uses `inclination_rad()` = 43.9°; `merge_nuts_chains.py --artifact-dir` defaults `ARTIFACT_G3`, hardcodes official 066 MAP, writes `kgas066_nuts.json`; `canfar_entrypoint.sh` execs `run_kgas066_nuts_headless.py` unless `map-pa25`; `session_name` is `kinuv-KGAS066-{sha6}-{tag}`; `point_latest` runs when `steal_latest` is True **before** `--dry-run` returns.

## Attacks / bounds

1. **Revise closes the five first-round execute-as-typed holes. Confirm they stay closed on the landed tree, then dispatch.** Track C.2: `steal_latest("nuts-kgas007") is False` including dry-run; `steal_latest("nuts")` stays True; dest `docs/reviews/artifacts/2026-09-05-kgas007-nuts/` (not G3, not leftover `pa25`); `pa_init_deg` **151.6**; `point_latest` not called; `--kind nuts` on 007 is exit 2; default galaxy + 007 kind is exit 2. Track C.4–5: new 007 worker; entrypoint execs it **only** when `KINUV_KIND=nuts-kgas007`; close `i_rad` into `make_potential` / `predict_binned` / leftover (optional, default None = 066 `inclination_rad()`); init `i_rad` **0.5044**. Track C.6: do **not** run live merge defaults; refuse dest containing `2026-08-30-g3-nuts`; 007 MAP for `(dx, dy)` / PA; `kgas007_nuts.json`; `kind: nuts-kgas007`. Track A / rejected alternatives: no pip into recovery or `$HOME`. Execute 3 then 8: tests green before any `canfar create`.

   **Bound (execute must still nail):** `make_potential` is not closed by adding an unused kwarg. Live U calls `predict_binned` without `i_rad`. The landed signature must pass `i_use` into that call (and leftover). Before dispatch: 007 MAP-θ `|chi2-122070.76|<1` on 956×66 at `i_rad=0.5044`; 066 official `|chi2-168675.6|<1` at default None. Tests: `steal_latest("nuts-kgas007") is False`; G3 sentinel untouched; `point_latest` not called; `session_name` is not `kinuv-KGAS066-…` when galaxy is 007; merge with default dest or dest containing `2026-08-30-g3-nuts` exits nonzero and does not mkdir G3; `--kind nuts` + KGAS007 exits 2.

2. **CANFAR `git pull` can still run the unpatched 066 worker after a clean local steal_latest.** `scripts/canfar_entrypoint.sh` pulls `origin/dev` unless `KINUV_SKIP_PULL=1`. `steal_latest` / `point_latest` run on the **launch host**; G3 write and the 066 worker run on the **job**. Execute item 4 (dispatch) is listed before item 5 (push or format-patch). If HTTPS push fails (propose residual 3) and the job still pulls unpatched `origin/dev`, the live last line is `run_kgas066_nuts_headless.py` → 066 vis/MAP/Ico and G3 dest even though local kind.py is patched.

   **Bound:** Do not `canfar create` until the job will exec the patched entrypoint: `origin/dev` contains the kind/worker/entrypoint/`i_rad`/merge patches, **or** launch `--skip-pull` against a repo that already has them. Format-patch under `toby_sandbox/patches/` is not a substitute for a job-visible tree. Official MAP and `KGAS066-latest` stay untouched.

3. **Live merge still hardcodes the official 066 MAP even if dest is the 007 artifact dir.** `scripts/merge_nuts_chains.py` reads `/…/KILOGAS066/kinuv-KGAS066-uvsign-map/stage_a_map.json` for `pa_init` / `dx` / `dy` (066 `0.091 / 0.019 / 199.73`, not 007 `dx=-0.0186″`, `dy=-0.0910″`, PA=151.6°). Watcher `write_job_status_md` non-`pa25` branch is still `G3 066 NUTS` / “Copy posteriors into `2026-08-30-g3-nuts/`”. Propose C.6 names the refuse and the 007 MAP read; execute must land both, not only `--artifact-dir`.

   **Bound:** 007 merge reads the landed **007** MAP JSON only. Product is `kgas007_nuts.json` (never `kgas066_nuts.json`). `rec["kind"]` is `nuts-kgas007`. `write_nuts_product_plots` / leftover pass `artifact_dir=`. `write_job_status_md` for this kind does not name G3 or “copy into G3”. Product tree: `/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS007/kinuv-KGAS007-nuts/` (new; do not overwrite `kinuv-KGAS007-stage-a-map`). Leave DEC-067 items 3–4 / DEC-OPS-AUTH as 066-only (STATUS one-liner; do not amend).

4. **TARGET on disk is already KGAS066+KGAS007; the diagnostic MAP is not.** `DEC-066-TARGET` and Field Guide `TARGET | KGAS066 + KGAS007` already match the licensed amend. 007 MAP JSON still says `diagnostic_only: true`, `dec_066_target_amended: false`, note “No 007 NUTS”. Flipping those keys in place makes the diagnostic MAP look always-official. INC/PA still answer **066 only** (`i = arccos(0.721) = 43.9°`, seed 205.2°). `inclination_rad()` has no galaxy argument.

   **Bound:** After accept: TARGET + Field Guide line only if still needed; no new `DEC-*`; do not amend INC / PA / VC / INFER / OPS-AUTH / RUNNER. Do not edit `geometry.py` `_CATALOGUE_BA` / `_PA_SEED_DEG`. 007 NUTS is a **new** tree. NUTS JSON: `galaxy: KGAS007`, `dec_066_target_amended: true`, `infer_007_s1_waived: true` / `infer_mock_recovery: waived-this-card`, `quote_inner_slope: false`, `intervals_calibrated: false`. Do not rewrite `kinuv-KGAS007-stage-a-map` in place.

5. **INFER waiver is this-card only; autodiff-fail still labels `laplace_mh`.** DEC-066-INFER is unamended (correct). `sampler_label(autodiff_ok=False)` still returns `laplace_mh`. Isolated S3 still missing on PATH; `external_fitters/` still absent. 007 leftover is **not** SB-dominated.

   **Bound:** Do not amend INFER. `sampler: nuts` only after autodiff + four finite chains + mix (`R_hat≤1.01`, ESS>400); else `nuts_unmixed`. Never `laplace_mh` on this kind; autodiff fail → STATUS stop. Never call Laplace-MH “NUTS”. No pip / conda / `--user` into recovery or `$HOME`. PATH miss → keep S1. Do not copy `leftover_gate: SB-dominated` onto 007. Do not quote 0.2239″ as a science inner scale; do not form `V_0/r_t`. No G4. No GPU.

## Comments

1. `major` -- Land C.2–C.7 **and** tests green before any `canfar create`. `steal_latest("nuts-kgas007") is False`; dest not G3; `point_latest` not called; `--kind nuts` on 007 is exit 2. Attack 1.

2. `major` -- `make_potential` must pass `i_rad` into `predict_binned` (and leftover). 007 identity `|chi2-122070.76|<1` at `i_rad=0.5044` on 956×66; 066 `|chi2-168675.6|<1` at default None. Unused kwarg is still 43.9°. Attack 1.

3. `major` -- Do not dispatch until the job execs the patched entrypoint (`origin/dev` has the patches, or `--skip-pull` on that tree). Format-patch alone + live `git pull` is the remaining 066-worker / G3 vector. Attack 2.

4. `major` -- 007 merge reads the 007 MAP, refuses G3 dest, writes `kgas007_nuts.json`, `kind: nuts-kgas007`. `write_job_status_md` must not name G3. Leave DEC-067 / OPS-AUTH (STATUS one-liner). Attack 3.

5. `major` -- Do not rewrite `kinuv-KGAS007-stage-a-map` in place. Do not edit `geometry.py` 066 ba/PA. Do not amend INC/PA/INFER. Attack 4.

6. `major` -- Never `laplace_mh` on this kind. No pip into recovery. No `/arc/home/thbrown/` writes. 066 leftover stays SB-dominated; do not copy that gate onto 007. `quote_inner_slope: false`. Attack 5.

7. `minor` -- Reject-this-wave stays: no approaching NUTS, no GPU, no logit of `[0.5, 15]`, no in-place official 066 MAP write, no S2 16/50/84, no inner `dV/dr`, V_0 mean is 255 km/s not 353. Official MAP unchanged. Do not start G4.

## Residual risks

1. User waiver of 007 S1 is chat + this card, not an INFER amendment. Later 007 science still owes mocks. Comment 6.

2. HTTPS push may fail. Dispatch after format-patch-only while the entrypoint still `git pull`s unpatched `origin/dev` writes G3 via the 066 worker. Comment 3. Tightened from propose residual 3.

3. conda-forge `bbarolo` / `kinms` still missing; `external_fitters/` does not exist. S3 may stay S1. Comment 6 forbids pip-to-fix or `$HOME`.

4. 007 leftover is not SB-dominated. Copying the 066 leftover_gate string onto 007 NUTS is a science error. Comment 6.

5. Real-066 16/50/84 stay uncalibrated (S2 Laplace SBC failed 68/95). Leftover 066 remains SB-dominated. Do not start G4.

6. DEC-OPS-AUTH / DEC-067 session template is still `kinuv-KGAS066-…`. Leave with a STATUS one-liner (OPS-AUTH already warns `KILOGAS066[:8]` collides with 007).

7. `sampler_label` returns `laplace_mh` when autodiff fails. Comment 6 is the lock.

## STATUS updates required

- `verdict` and `severity` as in the header (`accept`, `major`)
- `last_review_a`: this file
- Do not set `board: accepted` (parent tallies)
