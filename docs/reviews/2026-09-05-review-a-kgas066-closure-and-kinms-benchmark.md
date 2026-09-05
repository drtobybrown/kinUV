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
verdict: accept
severity: major
propose: docs/reviews/2026-09-05-propose-kgas066-closure-and-kinms-benchmark.md
---

# Review a: KGAS066 closure, external cube S3, KGAS007 Stage A MAP

Do not read the other seat's review file. Do not implement.

Scope check: approaching search stays closed (`pa25/failure.md`). Receding NUTS `sd3ckpf2` stays the 066 sampling product (PA 200.05°, V_0 **255** km/s not 353, r_t mean **0.224″**, chi2 **167486.8**). Official MAP `kinuv-KGAS066-uvsign-map` stays read-only. Existing DEC ids only. Agents do not write a `DEC-*`. No approaching NUTS. No 007 NUTS. No G4. No G5. No GPU. Leftover gate stays **SB-dominated** (not `s_1` / `c_3`). `quote_inner_slope: false`. `intervals_calibrated: false`. KinMS stays outside `src/kinuv/` and `scripts/*.py`. User 2026-09-05 licensed 007 **MAP only**. That selected path (Track A external S3 + Track B diagnostic 007 MAP) is accept-eligible.

Execute as typed can still amend TARGET by running 007 through 066-hardcoded `inclination_rad()` / `pa_start_degs()` / `vsys_seed_radio_kms()`, put KinMS on the recovery `PYTHONPATH` while leaving `test_no_uvkin_or_kinms_import` green, label NUTS **mean** r_t 0.224″ as a science (or median) inner scale, `pip install kinms` into `kinuv-venv-recovery`, and sneak 007 NUTS through `steal_latest=True` / a `nuts` job kind.

Canon (disk, not chat): `docs/reviews/artifacts/2026-09-02-kgas066-leftover-and-modes/comparison.json` `nuts_mean.params.r_t_arcsec = 0.22392216472996415`, `v0_kms = 254.9834109292598`; leftover_gate `SB-dominated`. S1 (`docs/diagnostics/s1-mock.md`): inject r_t 0.25″, vis 0.254″, CLEAN M1 94.7 vs 236.7 km/s/arcsec. Official two-start Δχ² vs V=0: 35552.7 (205.2°) vs 4260.2 (25.2°).

## Attacks / bounds

1. **ADR: DEC-066-TARGET is still 066-only; Track B as typed will apply DEC-066-INC / DEC-066-PA / 066 vsys seeds to KGAS007 and look like a silent TARGET amendment.** TARGET answer: KGAS066 only; no 007 until 066 MAP beats V=0 **and** injected vsys/PA/flux recover on 066 uv. Those 066 gates have landed (Δχ² +35553; S1 recovered). That unlocks 066 inference, not a second official galaxy. Field Guide table still reads `TARGET | KGAS066 only`. Gold-standard: hard targets get flags until the **user** writes a stub. DEC-066-AGENTS / INDEX: agents do not create or amend `DEC-*`. User 2026-09-05 licensed 007 MAP + leftover only and asked for a TARGET stub — the stub is not in the tree.

   Live code is 066-hardcoded. `src/kinuv/geometry.py`: `_CATALOGUE_BA = 0.721` → i ≈ 43.9°, `_PA_SEED_DEG = 205.2`, both `@requires("DEC-066-INC"|"DEC-066-PA")`. `stage_a_seeds` / `stage_a_bounds` / `pa_start_degs()` use that PA, 066 radio vsys (`optical_to_radio_kms(VSYS_SEED_KM_S)`), and the 066 ±100 km/s vsys box. `inclination_rad()` has no galaxy argument. Propose Track B says “freeze i from 007 catalogue, two-start PA” without saying those values are **overrides**, not edits to `geometry.py`. A default `stage_a_map(007_vis)` freezes 066 i, seeds 066 PA 205.2/25.2, and boxes vsys around the 066 line — that is an INC/PA/TARGET contradiction, not a diagnostic.

   **Bound:** Do not write or amend `DEC-066-TARGET` (or any new `DEC-*`). Do not edit `_CATALOGUE_BA`, `_PA_SEED_DEG`, `inclination_rad()`, or the Field Guide `TARGET | KGAS066 only` line. Track B is a **new-tree diagnostic** only: `/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS007/kinuv-KGAS007-stage-a-map/`. Product JSON must carry `galaxy: KGAS007`, `diagnostic_only: true`, `dec_066_target_amended: false`, `sampler` absent or `"map"` (never `nuts` / `nuts_unmixed` / `laplace_mh`). 007 i, PA seeds, and vsys box come from the **007 catalogue on disk**, passed as explicit `i_rad=` / `pa_deg=` / bounds overrides into the existing MAP entry points. If 007 vis, Ico, or catalogue i/PA/vsys is missing, STATUS one-liner and **stop Track B** — do not invent, do not fall back to 43.9° / 205.2°. Official 066 MAP and `KGAS066-latest` untouched.

2. **`test_no_uvkin_or_kinms_import` does not lock the path the propose uses; KinMS can still enter the recovery interpreter.** The test (`tests/test_forward.py`) rglob-bans only the substrings `from kinms`, `import kinms`, `from uvkin`, `import uvkin` under `src/kinuv/**` and `scripts/*.py`. It does **not** scan `external/`, `docs/`, or `tests/`. It does **not** ban `importlib.import_module("kinms")`, `__import__("kinms")`, `from kinms_kgas66`, `sys.path.insert(..., "external")`, `pip install kinms`, or `jupyter nbconvert` of `/arc/projects/KILOGAS/analysis/toby_sandbox/uvkin/kinMS_kgas66_example.ipynb` (cell source already `from kinms import KinMS`). Propose place `external/kinms_kgas66.py` specifically so that test stays green. That is a scan loophole, not a likelihood firewall. `scripts/run_s3_barolo.py` is inside the scan for `import kinms` but can still subprocess the uvkin notebook or `python -c "import kinms"` with `KINUV_VENV` on `PATH`.

   **Bound:** KinMS lives only at `external/kinms_kgas66.py` (or a sibling under `external/`, never `src/kinuv/` or `scripts/`). `scripts/run_s3_barolo.py` is Barolo CLI via `subprocess` only — no `import kinms`, no `importlib` kinms, no `sys.path` insert of `external/`, no spawn of the uvkin notebook. Extend the existing test (or a sibling in `test_forward.py`) so `src/kinuv/**/*.py` and `scripts/*.py` also fail on `importlib.import_module("kinms")`, `__import__("kinms")`, `kinms_kgas66`, `pip install kinms`, and `sys.path` + `external`. Do not execute the uvkin notebook from kinUV. Artifact README **first body sentence** (after H1): vis χ² `s * sum w |ΔV|^2` is the fit; Barolo/KinMS are comparators, not a kinUV likelihood. Same class of gate as `test_dirty_residuals_readme_opens_not_kinms`.

3. **S3 table item 1 names NUTS “median” r_t=0.224″; that number is the uncalibrated posterior *mean* and is not a science inner scale.** Propose canon table correctly lists Receding NUTS **mean** r_t **0.224″**, V_0 **255** km/s, chi2 **167486.8**. Track A row 1 and residual 4 then say “NUTS **median** r_t=0.224″” and tell the README to say median. Disk: `comparison.json` `nuts_mean.params.r_t_arcsec = 0.22392216472996415` (mean). Approaching `failure.md` already warns that a “NUTS-median” χ² at r_t=0.5″ is a clamp, not a science radius. DEC-066-VC quoted V_c stays Stage A arctan; leftover card set `quote_inner_slope: false` while leftover-vs-velocity. V_0/r_t or arctan′ at 0.25 BMAJ from 255 / 0.224 is inner `dV/dr`.

   **Bound:** S3 JSON key is `nuts_mean_r_t_arcsec` (value from `comparison.json`, ~0.2239″) plus `quote_inner_slope: false` and `intervals_calibrated: false` on **every** row that mentions 0.224. Label **mean**, not median, unless the implementer actually computes a median from `sd3ckpf2` draws and writes both. README first sentence: vis χ² is the fit; 0.224″ is an uncalibrated NUTS mean that left the L-BFGS 0.5″ wall, not a quoted 066 inner scale. Do not compute or quote `V_0/r_t`, arctan′(r), or km/s/arcsec from that number. S1 comparator row restates inject 0.25″ / vis 0.254″ / M1 94.7 vs 236.7 only. Cube-fit r_t or inner slope is a beam-smearing comparator, not a replacement V_c.

4. **Leftover `s_1` / `c_3` can land in the new S3 folder even though the propose forbids it in prose.** DEC-066-VC has no harmonic m=1. Methodology note: leftover-vs-velocity at Stage B is frozen Wiener Ico (uv span 0.093, vel span 0.335), not `s_1`/`c_3`; adding those terms needs a user DEC. Track A item 3 says “Do not call leftover `s_1`” but the S3 artifact path is new and untested. A Barolo warp / floating-i row titled “s1 residual” or a JSON key `s_1` is a VC contradiction.

   **Bound:** S3 JSON `leftover_gate` is `SB-dominated` (same string as `comparison.json`). No `s_1` / `c_3` keys, plot titles, or column headers except in a single negative sentence. Do not convert Δi / residual PV into a harmonic amplitude. kinUV still freezes 066 i (DEC-066-INC). Geometry soak-up is Δi / residual PV of the **cube** fitter only.

5. **`pip install` into `kinuv-venv-recovery` (or any interpreter `KINUV_VENV` can see) is a one-way jax hole; PATH-miss is not a license to install.** Propose already says do not pip into recovery and to ship S3 from S1 if Barolo/KinMS are missing. Recovery is CPU jax 0.11.1 + jax-finufft (GPU card: pip is a one-way hole). `scripts/canfar_entrypoint.sh` defaults `KINUV_VENV=/arc/home/thbrown/kinuv-venv-recovery`. A “make S3 work” `python -m pip install kinms` (or barolo wheels, or uvkin deps) from `scripts/` or `external/` using that interpreter mutates the 066 NUTS/MAP venv. `--user` and image site-packages are the same class.

   **Bound:** If `BBarolo` / `3dbarolo` / KinMS is missing on PATH, STATUS one-liner and ship S3 from S1 plus whichever tool actually ran. No `pip install` / `python -m pip` of kinms, KinMS extras, 3DBarolo, or uvkin into `kinuv-venv-recovery`, `--user`, or any env that `KINUV_VENV` resolves. An isolated env for `external/kinms_kgas66.py` is allowed only if it is **not** `KINUV_VENV` and is **not** on `src/kinuv`’s import path. Unit test: `scripts/*.py` contain no `pip install`.

6. **007 NUTS sneak-in: DEC-066-INFER plus live runner defaults.** INFER: MAP first; NUTS only if MAP Δχ² vs V=0 is real **and** vsys/PA/flux mocks recover. 007 has no S1 mock. Propose: “No 007 NUTS until MAP beats V=0 and a later propose” — beating V=0 on this card is not a NUTS license (INFER still owes 007 inject recovery). Live `steal_latest(kind)` is True for every kind that is not `pa25`. Live kinds are `nuts` / `nuts-pa25`; there is no map kind. “Headless flexible CPU if the MAP is long (DEC-067-RUNNER)” as typed can reuse the NUTS worker, steal `KGAS066-latest`, and write `sampler: nuts` under a 007 run id.

   **Bound:** This card never dispatches a 007 job whose `kind` contains `nuts`. 007 product JSON `sampler` is not `nuts`, not `nuts_unmixed`, not `laplace_mh`. No numpyro import on the 007 path. `steal_latest` must be False for any 007 job; do not write `KGAS066-latest` or `kinuv-KGAS066-uvsign-map`. Headless MAP, if needed, is a **map-only** kind (or an interactive/flexible MAP process) that does not call the NUTS worker. Even if 007 MAP Δχ² vs V=0 is large, stop; a later propose still owes 007 mock recovery before any 007 sample.

## Comments

1. `major` -- TARGET stays 066. Do not write or amend any `DEC-*`. Track B is a diagnostic new tree only. Do not edit `geometry.py` 066 ba/PA. 007 i/PA/vsys are catalogue overrides or Track B stops. Product JSON: `diagnostic_only: true`, `dec_066_target_amended: false`. Attack 1.

2. `major` -- KinMS only under `external/`. Extend `test_no_uvkin_or_kinms_import` (or sibling) for `importlib` / `__import__` / `kinms_kgas66` / `pip install kinms` / `sys.path`+`external` under `src/kinuv` and `scripts`. Do not run the uvkin notebook. S3 README first body sentence: vis χ² is the fit. Attack 2.

3. `major` -- S3 quotes NUTS **mean** r_t (~0.2239″ from `comparison.json`), not median, not a science inner scale. `quote_inner_slope: false` on every 0.224 row. No V_0/r_t or arctan′. Attack 3.

4. `major` -- S3 leftover gate is `SB-dominated`. No `s_1`/`c_3` keys or titles except in the negative. Attack 4.

5. `major` -- PATH miss → S1-only S3. No pip into `kinuv-venv-recovery`, `--user`, or `KINUV_VENV`. Isolated KinMS env must not be the recovery interpreter. Test: `scripts/*.py` have no `pip install`. Attack 5.

6. `major` -- No 007 NUTS this card even if MAP beats V=0. No `nuts` kind, no numpyro, no steal of `KGAS066-latest`. INFER still owes 007 mocks before any later sample propose. Attack 6.

7. `minor` -- Reject-this-wave stays: no approaching NUTS, no G4/G5, no GPU, no logit of `[0.5, 15]`, no in-place official MAP write, no S2 16/50/84, no inner `dV/dr` as a 066 science number, V_0 mean is 255 km/s not 353. Point methodology `user_review` at the S3 folder only after the README first-sentence bound. Official MAP unchanged. Do not start G4.

## Residual risks

1. User has not written the TARGET stub. Track B remains a diagnostic tree. A later reader will treat `kinuv-KGAS007-stage-a-map` as a licensed second galaxy. Comment 1 is the lock; Field Guide TARGET line stays 066-only. Carry-forward from propose residual 3, tightened.

2. `external/` is outside today’s import scan. A “cleanup” move into `scripts/` either fails the test or (if someone weakens the test) imports KinMS into kinUV. Comment 2 is the lock. **(new)**

3. Cube-fit r_t vs 0.224″ will still be copied as a science inner scale if the S3 table header says `r_t` without `nuts_mean` / `quote_inner_slope: false`. Comment 3. Carry-forward from propose residual 4, corrected mean vs median.

4. 3DBarolo / KinMS absent on PATH. S3 then restates S1 only. Carry-forward from propose residual 1; comment 5 forbids pip-to-fix.

5. 007 vis / Ico / catalogue geometry not staged on `/arc`. Track B stops; 066 S3 still ships. Carry-forward from propose residual 2; comment 1 forbids 066-seed fallback.

6. Isolated KinMS env can still be mistaken for a kinUV likelihood. README first sentence (comment 2) + propose residual 5. Carry-forward.

7. Real-066 16/50/84 stay uncalibrated (S2 Laplace SBC failed 68/95). Leftover remains SB-dominated. Do not start G4. Carry-forward.

8. **(new)** Live `steal_latest` defaults True off the `pa25` path. A 007 headless job that reuses `kind="nuts"` retargets `KGAS066-latest`. Comment 6 is the lock.

9. **(new)** 007 MAP that beats V=0 will tempt same-card NUTS. INFER still requires 007 vsys/PA/flux mock recovery. Not this card.

## STATUS updates required

- `verdict` and `severity` as in the header (`accept`, `major`)
- `last_review_a`: this file
- Do not set `board: accepted` (parent tallies)
