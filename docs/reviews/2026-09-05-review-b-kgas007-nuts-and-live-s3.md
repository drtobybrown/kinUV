---
role: reviewer
seat: b
date: 2026-09-05
agent: review-b
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

# Review b: live 066 S3, literature note, user-licensed KGAS007 NUTS

Do not read the other seat's review file. Do not implement.

Scope check (disk, not chat): approaching closed (`docs/reviews/artifacts/2026-09-02-kgas066-leftover-and-modes/pa25/failure.md`). Receding `sd3ckpf2` is the 066 NUTS product. Official MAP `kinuv-KGAS066-uvsign-map` stays read-only. `quote_inner_slope: false`. `intervals_calibrated: false`. Do not start G4. No new `DEC-*` id. User 2026-09-05 licensed: amend existing `DEC-066-TARGET` after accept; waive 007 mock-recovery **for this card only**; kind must be `nuts-kgas007` not `nuts`. Path isolation: no new writes under `/arc/home/thbrown/`; fitters only under `toby_sandbox/external_fitters/`; patches only under `toby_sandbox/patches/`. Recovery venv may be **read** as `KINUV_VENV`; do not pip into it.

007 MAP on disk (`results/KILOGAS007/kinuv-KGAS007-stage-a-map/stage_a_map.json`): `sampler: map`, `diagnostic_only: true`, `dec_066_target_amended: false`, i frozen 28.9°, both PA starts → 151.602°, V_0=195.98 km/s, r_t=0.5″ floor, χ²=122070.76, Δχ² vs V=0 = +6211.63, `uv_stored: wavelengths`. 007 leftover (`docs/reviews/artifacts/2026-09-05-kgas007-stage-a-map/leftover_chi2.json`): `leftover_chi2_structured: false`, uv span 0.737 > vel 0.214, `quote_inner_slope: false`. That is not the 066 SB-dominated gate (066 MAP leftover: structured true, uv 0.115 < vel 0.355).

User TARGET stub + INFER waiver make Track C accept-eligible. Live runner/merge/STATUS paths still treat every non-`pa25` kind as receding 066. Those holes are bindable. Rubber-stamp is invalid.

## Attacks / bounds

1. **ADR / live kind: `steal_latest` / `artifact_dir_for_kind` / `pa_init_deg` are a two-way switch (`pa25` vs else).** `src/kinuv/runner/kind.py`: `steal_latest` is `not is_approaching_kind`; `artifact_dir_for_kind` returns `docs/reviews/artifacts/2026-08-30-g3-nuts` unless `"pa25" in kind`; `pa_init_deg` returns official 066 MAP PA 199.73 unless approaching or an override. `tests/test_canfar_runner.py` only locks `steal_latest("nuts-pa25") is False` and `steal_latest("nuts") is True`. Adding the string `nuts-kgas007` without patching those three functions keeps `steal_latest True`, writes G3, and starts at 199.73°. Propose residual 3 names session/entrypoint; it does not name the `pa_init` default or the dry-run `point_latest` call. `scripts/launch_headless.py` calls `point_latest` **before** the `--dry-run` return. `--kind nuts` is forbidden even with `--galaxy KGAS007`: default galaxy still retargets `KGAS066-latest`; `--galaxy KGAS007 --kind nuts` still writes G3 and still execs the 066 worker (samples 066 vis). `point_latest` itself is galaxy-tagged (`{KGASID}-latest`); the G3 write is the 066-product clobber.

2. **Entrypoint is a 066 worker.** `scripts/canfar_entrypoint.sh` execs `run_kgas066_nuts_headless.py` unless `KINUV_KIND == map-pa25`. That worker hardcodes official 066 MAP / `KILOGAS066.npz` / KGAS66 Ico+cube, `_load_066()`, writes `kgas066_nuts.json`, and G3-guards only `kind == nuts-pa25`. `KINUV_GALAXY` is already in `headless_job_env` and is unused. A `nuts-kgas007` env without a 007 worker samples 066 and can overwrite G3.

3. **`make_potential` has no `i_rad`; live default is DEC-066-INC 43.9°.** `src/kinuv/infer/nuts.py` `make_potential` → `predict_binned(..., xla=True)` with no `i_rad`. `src/kinuv/forward/model.py` / `infer/map.py`: `i_use = inclination_rad() if i_rad is None`. `geometry.inclination_rad()` is `arccos(0.721)` (066). 007 MAP froze `i_rad` from catalogue 28.9° (`i_rad_frozen: 0.5044`) and passed it into `_lbfgs_one_start`. Propose says “Frozen `i_rad`” but does not name this default. 007 NUTS on the live potential is a different likelihood than the landed MAP. DEC-066-INC is 066-only; do not edit `geometry.py` 066 ba/PA (propose is correct). The 007 worker must close `i_rad` from the 007 MAP JSON into U, leftover, and plots.

4. **`merge_nuts_chains.py` is a G3 066 overwrite.** Default `--artifact-dir` is `ARTIFACT_G3`. It reads official `KILOGAS066/.../stage_a_map.json` for `dx`/`dy`/`pa_deg`, sets `rec["kind"] = "nuts"`, writes `kgas066_nuts.json` into the artifact dir, leftover unevaluated. Propose “merge later” without this lock still clobbers the receding 066 product JSON. 007 merge must read the 007 MAP (`dx=-0.0186`, `dy=-0.0910`, PA 151.6°), write `docs/reviews/artifacts/2026-09-05-kgas007-nuts/` (and/or `results/KILOGAS007/kinuv-KGAS007-nuts/`), refuse any dest containing `2026-08-30-g3-nuts`, and never write `kgas066_nuts.json`.

5. **007 leftover is not SB-dominated; live `quote_inner_slope` will flip true if NUTS leaves the 0.5″ floor.** Disk: 007 `leftover_chi2_structured: false` because uv span 0.737 > vel 0.214 (`leftover_velocity_structured` is `vel > uv`). 066 MAP/NUTS-mean leftover is the opposite (vel > uv) and the S3 gate string is `SB-dominated`. `src/kinuv/diagnostics/flags.py` `quote_inner_slope` is `(not r_t_at_floor) and leftover_measured and (not leftover_chi2_structured)`. 007 MAP stays false only because `r_t=0.5`. 066 NUTS already left that wall (`r_t` mean 0.2239″). If 007 NUTS does the same and leftover stays uv-dominated, `write_leftover_at_params` will write `quote_inner_slope: true`. That is not a clean leftover (uv span 0.737) and this card has no 007 S1. Card rule `quote_inner_slope: false` must be a hard write, not the live flag default. Do not copy `leftover_gate: SB-dominated` onto 007 products.

6. **DEC-066-INFER first sentence still requires mock recovery; user waiver is this card only.** Live INFER: “NUTS only if MAP Δχ² vs the zero model is real and vsys/PA/flux mocks recover.” 007 Δχ² +6212 is real (ZEROMODEL). User waived the mock clause for this card. Do **not** amend `DEC-066-INFER`. Do **not** treat the waiver as a standing 007/452/γ license. Product JSON must record the waiver. Reviewers were invited to reject on INFER; this seat accepts the named user waiver and binds the text freeze.

7. **DEC-067-RUNNER session name and G3 copy contradict a 007 dispatch unless left with a STATUS line.** Item 4: session name stays `kinuv-KGAS066-{sha6}-nuts`. Item 3: copy PNGs/JSON into `2026-08-30-g3-nuts/`. Live `session_name()` ignores `--galaxy`. `write_job_status_md` non-`pa25` branch writes Phase `G3 066 NUTS` and Next Step copy into G3, then clears `pending`. User licensed a TARGET amend, not a DEC-067 amend. AGENTS: leave a DEC with one STATUS line. 007 session name must include `KGAS007` and `nuts-kgas007` (63-char cap, include `-c{N}` for 4×1). STATUS one-liner that DEC-067 items 3–4 are left for 007 isolation. 007 jobs must not call the live `write_job_status_md` G3 branch.

8. **007 vis is wavelengths; `load_kgas066` is metres + `u_m`.** Disk `KILOGAS007.npz` keys `u,v,vis,weights,freqs` (no `u_m`; 44550 rows native). `load_kgas066` requires `u_m` and would KeyError — fail-closed, not a silent wrong scale. The landed 007 MAP used a local `_load_007` (`u * c / f_ref`). NUTS must reuse that conversion. Identity at MAP θ: `n_row=956`, `n_chan=66`, `s≈0.570735`, `|chi2-122070.76|<1`. Do not call `load_kgas066` on 007. Do not invent `time`/`baseline`.

## Comments

1. `major` — Kind `nuts-kgas007` only. `steal_latest("nuts-kgas007") is False` including `--dry-run`. `steal_latest("nuts")` stays True (066 receding). `artifact_dir_for_kind("nuts-kgas007")` is `docs/reviews/artifacts/2026-09-05-kgas007-nuts/` (not G3, not leftover `pa25`). `pa_init_deg("nuts-kgas007")` is 151.6 from the 007 MAP JSON, not 199.73. `--kind nuts` is exit-2 / refuse for any 007 launch. `point_latest` is not called. Unit tests must lock all of the above plus a G3 sentinel untouched. Attack 1.

2. `major` — Entrypoint execs a 007 worker when `KINUV_KIND` is `nuts-kgas007` (or `KINUV_GALAXY` is KGAS007 with that kind). The 066 worker must not run. 007 worker loads the landed 007 MAP JSON, wavelength→metre vis, KGAS7 Ico/cube, freezes `i_rad` from that JSON. Product tree `/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS007/kinuv-KGAS007-nuts/` is a **new** tree. Do not write `kinuv-KGAS066-uvsign-map` or `docs/reviews/artifacts/2026-08-30-g3-nuts/`. Durable run dir stays `toby_sandbox/kinuv_runs/{KGAS007}-{ts}-nuts-kgas007-c{N}/`. Attack 2.

3. `major` — Close `i_rad=0.5044` (28.9°) into `make_potential` / `predict_binned` / leftover. Live default `inclination_rad()` is 066 43.9°. Do not edit `geometry.py` 066 ba/PA/seeds. Attack 3.

4. `major` — Do not run live `scripts/merge_nuts_chains.py` on 007 shards. Merge must not default to G3, must not read the official 066 MAP, must not write `kgas066_nuts.json`, must set `kind: nuts-kgas007`. `sampler: nuts` only after autodiff + four finite chains + `R_hat≤1.01` and ESS>400; else `nuts_unmixed`. Attack 4.

5. `major` — 007 leftover_gate is **not** `SB-dominated`. Measure 007 leftover spans; record `leftover_chi2_structured` from `leftover_velocity_structured`. Hard-set `quote_inner_slope: false` on every 007 NUTS product even if `r_t` leaves 0.5″ and leftover-vs-velocity is false. Do not quote inner `dV/dr` or form `V_0/r_t`. 066 leftover_gate stays SB-dominated. Attack 5.

6. `major` — Do not amend `DEC-066-INFER`. Waiver is this card only. 007 product JSON records `infer_mock_recovery: waived-this-card` and `intervals_calibrated: false`. 007 NUTS is not a license for 452/γ or G4. Attack 6.

7. `major` — `session_name` must include `KGAS007` and `nuts-kgas007` (and `-c{N}`). STATUS one-liner: DEC-067-RUNNER items 3–4 left for 007 isolation (user did not stub a DEC-067 amend). `write_job_status_md` must not write Phase `G3 066 NUTS` or Next Step copy-into-G3 for this kind. Official 066 MAP unchanged. Attack 7.

8. `major` — Wavelength→metre loader as in the landed 007 MAP (`uv_stored: wavelengths`, no `u_m`). Do not call `load_kgas066`. MAP-θ identity `|chi2-122070.76|<1` on 956×66 before dispatch. Attack 8.

9. `major` — Amend existing `DEC-066-TARGET` only (Field Guide TARGET line `KGAS066 + KGAS007`). No new DEC id. 007 still uses catalogue/MAP overrides (`i=28.9°`, PA init 151.6°, vsys radio from the 007 MAP). Do not start G4. No GPU. No G5.

10. `minor` — Track A isolated env is `toby_sandbox/external_fitters/` only. Runner `kinUV/external/run_image_benchmarks.py` (does not exist yet; that is execute). No `from kinms` under `src/` or `scripts/`. If conda-forge `bbarolo`/`kinms` miss: STATUS one-liner, keep S1. 066 S3 leftover_gate stays SB-dominated. README first body sentence: vis χ² is the fit. No pip into recovery or `$HOME`.

11. `minor` — Track B literature note is not an ADR. Ground in S1 (vis r_t 0.254″ vs CLEAN M1 94.7 vs 236.7) and 066 leftover SB-dominated. Do not claim 0.2239″ is a published 066 inner scale.

12. `minor` — Propose residual 1 (conda miss) and 5 (HTTPS push → `toby_sandbox/patches/`) stand. No writes under `/arc/home/thbrown/`.

## Residual risks

1. conda-forge `bbarolo` / `kinms` missing on the image. S3 then stays S1. (propose 1)
2. 007 NUTS without S1 inject (user waiver, this card only). Do not amend INFER. (propose 2; comment 6)
3. `session_name` / entrypoint / merge / STATUS still hardcode KGAS066 unless patched. Comments 1–4 and 7 are the lock. (propose 3, tightened)
4. 007 leftover is not SB-dominated; live `quote_inner_slope` will flip if NUTS leaves the r_t floor. Comment 5. (propose 4, tightened)
5. HTTPS push may fail; patches go to `toby_sandbox/patches/`. (propose 5)
6. Real-066 16/50/84 stay uncalibrated. Do not start G4. (propose 6)
7. New. `make_potential` default i is 066 43.9°. Comment 3.
8. New. Live merge defaults G3 and reads the official 066 MAP. Comment 4.

## STATUS updates required

- `verdict: accept`, `severity: major`
- `last_review_b:` this file
- Do not set `board: accepted` (parent tallies)
