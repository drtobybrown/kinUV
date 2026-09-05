---
role: reviewer
seat: a
date: 2026-09-05
agent: review-a
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

# Review a: live 066 S3, literature note, user-licensed KGAS007 NUTS

Do not read the other seat's review file. Do not implement.

Scope check: approaching stays closed (`pa25/failure.md`). Receding `sd3ckpf2` stays the 066 NUTS product (V_0 **255** km/s not 353; r_t **mean** 0.2239″ not science; chi2 **167486.8**). Official MAP `kinuv-KGAS066-uvsign-map` stays read-only (disk PA=199.73°, chi2=168675.6). `quote_inner_slope: false`. Leftover on 066 stays **SB-dominated**. No G4. No GPU. No new `DEC-*` id. User 2026-09-05 licensed: amend existing `DEC-066-TARGET` after accept; waive 007 S1 mock **for this card only**; dispatch **`nuts-kgas007` only**. That selected path is accept-eligible. The live runner is not.

Execute as typed can still: `--kind nuts` (or an unpatched `nuts-kgas007`) steal `KGAS066-latest` and write G3; `session_name` / entrypoint / merge run the 066 worker and freeze 066 MAP `(dx,dy)`; `pip` into recovery or write `/arc/home/thbrown/`; quote 0.224″ as a science inner scale; edit `geometry.py` 066 seeds; label Laplace `laplace_mh` as NUTS; start G4.

Canon (disk, not chat): `comparison.json` `nuts_mean.params.r_t_arcsec = 0.22392216472996415`, `v0_kms = 254.9834109292598`; leftover_gate `SB-dominated`. 007 MAP `/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS007/kinuv-KGAS007-stage-a-map/stage_a_map.json`: PA=151.6°, V_0=196 km/s, r_t=0.5″ floor, χ²=122071, Δχ² vs V=0 = +6212, `i_deg_frozen=28.9`, `sampler: map`, leftover `leftover_chi2_structured: false` (uv 0.737 > vel 0.214). Live `steal_latest("nuts-kgas007") is True`; `artifact_dir_for_kind` → `2026-08-30-g3-nuts`; `pa_init_deg` → 199.73. `KGAS066-latest` → `kinuv_runs/KGAS066-20260831T194009Z-nuts`.

## Attacks / bounds

1. **Live kind `nuts-kgas007` is receding today: steal True, dest G3, PA 199.73, session KGAS066, entrypoint is the 066 worker.** `src/kinuv/runner/kind.py`: `steal_latest` is `not is_approaching_kind` (`"pa25" in kind`); `artifact_dir_for_kind` returns `ARTIFACT_G3_REL` (`docs/reviews/artifacts/2026-08-30-g3-nuts`) unless pa25; `pa_init_deg` returns official 066 MAP 199.73 unless pa25. Executed this turn: `steal_latest("nuts-kgas007") is True`, artifact name `2026-08-30-g3-nuts`, PA 199.73. `tests/test_canfar_runner.py` only locks `steal_latest("nuts") is True` and `steal_latest("nuts-pa25") is False`. `scripts/launch_headless.py` `session_name` is `kinuv-KGAS066-{sha6}-{tag}` and ignores `--galaxy`. Default `--galaxy KGAS066`, `--kind nuts`. `point_latest` runs when `steal_latest` is True — default galaxy retargets `KGAS066-latest` (live symlink to the receding run). `scripts/canfar_entrypoint.sh` last line is always `run_kgas066_nuts_headless.py` except `KINUV_KIND=map-pa25`. That worker hardcodes official 066 MAP / `KILOGAS066.npz` / 066 Ico. `--kind nuts` on 007 is the propose's own rejected alternative and is still the launcher default.

   DEC-067-RUNNER item 3 still says copy PNGs/JSON into `2026-08-30-g3-nuts/`. Item 4 and DEC-OPS-AUTH still say session `kinuv-KGAS066-{sha6}-{map|nuts}`. Propose Track C names the patches but execute item 4 is "kind + worker + entrypoint + tests + **dispatch**". Dispatch before the tests go green is the steal.

   **Bound:** Do not dispatch until unit tests are green. `steal_latest("nuts-kgas007") is False` including `--dry-run`; `steal_latest("nuts")` stays True (066 receding). `artifact_dir_for_kind("nuts-kgas007")` is `docs/reviews/artifacts/2026-09-05-kgas007-nuts/` (not G3, not leftover `pa25`). `session_name` uses `--galaxy` (`kinuv-KGAS007-{sha6}-nuts-kgas007`, 63-char cap). Entrypoint execs a **007** worker only when `KINUV_KIND=nuts-kgas007`; `nuts` / default still the 066 worker. `--kind nuts` on 007 is forbidden (exit 2). `--galaxy KGAS007` is required; default galaxy + 007 kind is exit 2. `point_latest` is not called. Official MAP and `KGAS066-latest` untouched. Leave DEC-067 items 3–4 and DEC-OPS-AUTH as the **066 receding** path; STATUS one-liner; do not amend those ADRs. Tests: `nuts-kgas007` does not write a G3 sentinel; does not call `point_latest`; `session_name` is not `kinuv-KGAS066-…` when galaxy is 007.

2. **`merge_nuts_chains.py` "merge later" as typed writes G3, names `kgas066_nuts.json`, and freezes 066 MAP `(dx,dy)` / PA.** Live default `--artifact-dir` is `ARTIFACT_G3`. It reads `/…/KILOGAS066/kinuv-KGAS066-uvsign-map/stage_a_map.json` for `pa_init`, `dx_arcsec`, `dy_arcsec`. It writes `artifact_dir / "kgas066_nuts.json"` and sets `rec["kind"] = "nuts"`. `write_job_status_md` non-`pa25` branch is `G3 066 NUTS` and Next Step "Copy posteriors into `2026-08-30-g3-nuts/`". 007 MAP shifts are `dx=-0.0186″`, `dy=-0.0910″`, PA=151.6° — not 066 `0.091 / 0.019 / 199.73`. Execute item 4's merge with the live script is a G3 write and a silent 066 geometry soak.

   **Bound:** 007 merge must pass `--artifact-dir docs/reviews/artifacts/2026-09-05-kgas007-nuts` (or refuse if dest contains `ARTIFACT_G3_REL`). Product filename is not `kgas066_nuts.json`. Init / freeze `(dx, dy)` and PA from the landed **007** MAP JSON only. `rec["kind"]` is `nuts-kgas007`. `write_job_status_md` for this kind does not name G3 and does not tell anyone to copy into G3. Worker `write_nuts_product_plots` must pass `artifact_dir=` (function default is still G3). Product tree: `/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS007/kinuv-KGAS007-nuts/` (new tree; do not overwrite `kinuv-KGAS007-stage-a-map` or the official 066 MAP).

3. **007 worker that reuses `load_kgas066` / `inclination_rad()` / `pa_start_degs()` / `vsys_seed_radio_kms()` is an INC/PA/TARGET contradiction, not a loader bug.** 007 npz on disk is keys `u, v, vis, weights, freqs` (wavelengths, 2.05 GB). `load_kgas066` requires `u_m, v_m, time, baseline` and will KeyError — or, if pointed at `KILOGAS066.npz`, it samples 066. Landed 007 MAP already has `_load_007` (wavelength → metre via `C_LIGHT_M_S / f_ref`; refuses `u_m`). `geometry.py` `_CATALOGUE_BA = 0.721` → i ≈ 43.9°, `_PA_SEED_DEG = 205.2`, both `@requires("DEC-066-INC"|"DEC-066-PA")`. `inclination_rad()` has no galaxy argument. `seeds.pa_start_degs()` is 205.2 / 25.2; `vsys_seed_radio_kms()` is 066 optical 8299.563. 007 MAP froze i=28.9°, PA init **151.6°** (both catalogue starts landed there), vsys radio 13584. Propose says "do not edit `geometry.py` 066 ba/PA" and "catalogue/MAP overrides" — a copied 066 worker ignores that.

   DEC-066-INC / DEC-066-PA still answer **066 only**. Amending TARGET does not rewrite those answers to 28.9° / 151.6°. Editing `_CATALOGUE_BA` or `_PA_SEED_DEG` for 007 corrupts 066.

   **Bound:** Do not edit `_CATALOGUE_BA`, `_PA_SEED_DEG`, `inclination_rad()`, `pa_seed_deg()`, or 066 `stage_a_seeds`. 007 NUTS init is the landed 007 MAP JSON: `i_rad` frozen 28.9°, PA **151.6°**, vsys / `(dx, dy)` from that file. Wavelength→metre loader is the 007 path (refuse `u_m`; refuse any 066 vis/MAP/Ico path). Do not call `load_kgas066` on 007. Do not amend DEC-066-INC or DEC-066-PA.

4. **TARGET amend is user-licensed; the live ADR is still 066-only. A sloppy amend licenses 452 / GPU / a silent INC rewrite.** Disk `DEC-066-TARGET`: "KGAS066 / KILOGAS066 only. No 007, 452, γ, GPU…". Field Guide table: `TARGET | KGAS066 only`. User 2026-09-05 licensed **amend existing** TARGET after accept (no new DEC id) so 007 is a second official galaxy. That unlocks this card's NUTS tree. It does not unlock G4, GPU, 452, or rewriting INC/PA. The landed 007 MAP JSON still says `diagnostic_only: true`, `dec_066_target_amended: false`. Flip that in place and a reader thinks the diagnostic MAP was always official.

   **Bound:** After accept only: amend `DEC-066-TARGET` + Field Guide TARGET line to `KGAS066 + KGAS007`. No new `DEC-*`. Do not drop the 066 MAP/zero-model gates. Do not add 452, γ, GPU, uvfit, stellar M/L. Do not amend INC / PA / VC / INFER / OPS-AUTH / RUNNER. 007 NUTS is a **new** tree; do not rewrite `kinuv-KGAS007-stage-a-map` in place. NUTS product JSON: `galaxy: KGAS007`, `dec_066_target_amended: true`, `infer_007_s1_waived: true`, `quote_inner_slope: false`. Official 066 MAP unchanged.

5. **INFER mock waiver is this-card only; autodiff-fail still writes `laplace_mh`.** DEC-066-INFER: NUTS only if MAP Δχ² vs V=0 is real **and** vsys/PA/flux mocks recover. 007 MAP Δχ² = +6212 (ZEROMODEL gate holds). 007 S1 mock does not exist; user waived it **for this card only**. Propose lists INFER in `ids` but does not amend the ADR — correct. `sampler_label(autodiff_ok=False)` returns `SAMPLER_NAME` = `laplace_mh` (`src/kinuv/infer/nuts.py`, `posterior.py`). A 007 NUTS-kind job that fails autodiff is Laplace on a NUTS card. Relabeling that `nuts` is the forbidden "call Laplace NUTS". Propose `sampler: nuts` only after autodiff + mix (`R_hat≤1.01`, ESS>400); else `nuts_unmixed`. Keep that. Do not start G4 (DEC-066-VC Stage B rings).

   **Bound:** Do not amend `DEC-066-INFER`. Waiver is this card only; later 007 science still owes mocks. 007 product `sampler` is `nuts` only after autodiff + four finite chains + mix; else `nuts_unmixed`. Never write `sampler: laplace_mh` on a `nuts-kgas007` product; if autodiff fails, STATUS stop. Never label Laplace-MH "NUTS". No 007 G4. No 066 G4. No GPU.

6. **Isolated S3 / KinMS can still pip into recovery or land under `$HOME`.** `toby_sandbox/external_fitters/` does **not** exist. `/arc/home/thbrown/kinuv_runs` **does**. Entrypoint / `headless_job_env` default `KINUV_VENV=/arc/home/thbrown/kinuv-venv-recovery` (read is licensed; pip is not). `test_scripts_have_no_pip_install` scans `scripts/*.py` only — not `external/`, not a shell `python -m pip` in `canfar_entrypoint.sh`, not conda into `$HOME`. Prior card already bans `from kinms` / `importlib` / `pip install kinms` under `src/kinuv` and `scripts`. Propose Track A is `external/run_image_benchmarks.py` + isolated env. A "make S3 work" install into recovery, `--user`, or `/arc/home/thbrown/external_fitters/` is the jax hole and the home-write hole.

   S3 on disk is still S1: `barolo.status=missing_on_path`, `kinms.status=missing`. 066 leftover_gate in that table is `SB-dominated`. 007 leftover is **not** SB-dominated. Copying that string onto 007 NUTS is a leftover lie. Literature note that treats 0.2239″ as a published 066 inner scale is inner `dV/dr` (`quote_inner_slope` must stay false; do not form `V_0/r_t` from 255 / 0.224).

   **Bound:** Create `external_fitters/` only under `/arc/projects/KILOGAS/analysis/toby_sandbox/external_fitters/`. Patches only under `toby_sandbox/patches/`. No new writes under `/arc/home/thbrown/` (including `kinuv_runs`, `external_fitters`, `patches`). Recovery venv is **read** as `KINUV_VENV` for jax; no `pip` / `conda` / `--user` of kinms/bbarolo into it. No `from kinms` under `src/` or `scripts/`. If tools still miss: STATUS one-liner and keep S1. S3 leftover_gate stays `SB-dominated`. 007 NUTS product must **not** copy `leftover_gate: SB-dominated`; use the 007 leftover eval (today `leftover_chi2_structured: false`) or omit the key if unevaluated. Literature note is not an ADR. Do not claim 0.224″ / 0.2239″ is a science inner scale. `quote_inner_slope: false` on every 066 row that mentions that number.

## Comments

1. `major` -- `nuts-kgas007` must have `steal_latest` False, artifact `2026-09-05-kgas007-nuts` (not G3), `session_name` from `--galaxy KGAS007`, entrypoint 007 worker. Tests green **before** dispatch. `--kind nuts` on 007 is exit 2. Do not retarget `KGAS066-latest`. Leave DEC-067 / OPS-AUTH 066 session+G3 copy as 066-only (STATUS one-liner). Attack 1.

2. `major` -- 007 merge must not default to G3, must not write `kgas066_nuts.json`, must freeze `(dx, dy)` / PA from the 007 MAP JSON, must set `kind: nuts-kgas007`. `write_job_status_md` must not name G3. Attack 2.

3. `major` -- Do not edit `geometry.py` 066 ba/PA. 007 i=28.9°, PA init 151.6°, vsys and shifts from the 007 MAP. Wavelength→metre 007 loader only; refuse `load_kgas066` and 066 paths. Do not amend INC/PA. Attack 3.

4. `major` -- Amend existing TARGET + Field Guide TARGET line after accept only. No new DEC id. No 452/GPU/γ. New 007 NUTS tree; do not rewrite the diagnostic MAP in place. Attack 4.

5. `major` -- Do not amend INFER. S1 waiver is this card only. `sampler: nuts` only after autodiff + mix; else `nuts_unmixed`. Never `laplace_mh` on this kind; never call Laplace NUTS. No G4. Attack 5.

6. `major` -- Isolated fitters / patches only under `toby_sandbox/`. No `/arc/home/thbrown/` writes. No pip into recovery. PATH miss → keep S1. 066 leftover stays SB-dominated; do not copy that gate onto 007. Do not quote 0.2239″ as science; `quote_inner_slope: false`. Attack 6.

7. `minor` -- Reject-this-wave stays: no approaching NUTS, no GPU, no logit of `[0.5, 15]`, no in-place official 066 MAP write, no S2 16/50/84, no inner `dV/dr`, V_0 mean is 255 km/s not 353. Official MAP unchanged. Do not start G4.

## Residual risks

1. User waiver of 007 S1 is chat + this card, not an INFER amendment. A later reader will treat 007 NUTS 16/50/84 as calibrated. Comment 5 is the lock. Carry-forward from propose residual 2, tightened.

2. Live `steal_latest` is True for every non-`pa25` kind, including the name `nuts-kgas007` today. Dispatch-before-test retargets `KGAS066-latest` and writes G3. Comment 1. Carry-forward from propose residual 3, verified on disk this turn.

3. `merge_nuts_chains.py` default dest is G3 and hardcodes 066 MAP + `kgas066_nuts.json`. "Merge later" without comment 2 writes G3. **(new)**

4. conda-forge `bbarolo` / `kinms` still missing; `external_fitters/` does not exist. S3 may stay S1. Comment 6 forbids pip-to-fix or `$HOME`. Carry-forward from propose residual 1.

5. 007 leftover is not SB-dominated. Copying the 066 leftover_gate string onto 007 NUTS is a science error. Comment 6. Carry-forward from propose residual 4.

6. HTTPS push may fail; format-patch only to `toby_sandbox/patches/`, never `$HOME`. Carry-forward from propose residual 5.

7. Real-066 16/50/84 stay uncalibrated (S2 Laplace SBC failed 68/95). Leftover 066 remains SB-dominated. Do not start G4. Carry-forward.

8. **(new)** DEC-OPS-AUTH / DEC-067 session template is still `kinuv-KGAS066-…`. 007 must leave that with a STATUS one-liner, not silently reuse the 066 name (OPS-AUTH already warns `KILOGAS066[:8]` collides with 007).

9. **(new)** `sampler_label` returns `laplace_mh` when autodiff fails. Comment 5 is the lock.

## STATUS updates required

- `verdict` and `severity` as in the header (`accept`, `major`)
- `last_review_a`: this file
- Do not set `board: accepted` (parent tallies)
