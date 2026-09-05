---
role: reviewer
seat: b
date: 2026-09-05
agent: review-b
canon_generation: 4
ids:
  - DEC-066-INFER
  - DEC-066-PA
  - DEC-066-TARGET
  - DEC-066-VC
  - DEC-066-INC
  - DEC-066-ZEROMODEL
  - DEC-066-SHIFT
  - DEC-067-RUNNER
verdict: accept
severity: major
propose: docs/reviews/2026-09-05-propose-kgas066-closure-and-kinms-benchmark.md
---

# Review b: KGAS066 closure, external cube S3, KGAS007 Stage A MAP

Do not read the other seat's review file. Do not implement.

Scope check: approaching search is closed (`pa25/failure.md`). Receding NUTS `sd3ckpf2` is the 066 sampling product (PA 200.05°, V_0 **255** km/s, r_t **0.224″**, χ² 167486.8). Official MAP `kinuv-KGAS066-uvsign-map` stays read-only (Stage A PA=199.73°, V_0=267.7, r_t=0.5″ L-BFGS box, χ² 168675.6; Δχ² vs V=0 = +35553). Stage B N=7 λ=0 χ² 167302.2. Leftover gate stays **SB-dominated**; do not label leftover `s_1` / `c_3` (not in DEC-066-VC). `i` frozen at catalogue **43.86°** (DEC-066-INC). `quote_inner_slope: false`. `intervals_calibrated: false`. Existing DEC ids only; agents do not write a new `DEC-*`. No approaching NUTS. No 007 NUTS. No G4. No G5. No GPU. Official MAP unchanged. That selected path (Track A image-plane S3 comparators + Track B 007 Stage A MAP only if vis exist on `/arc`) is accept-eligible.

Execute as typed still lets two cube fitters become a second likelihood, treats “find vis + Ico on `/arc`” as an unbounded inventory, keeps `test_no_uvkin_or_kinms_import` blind to a `tools/` KinMS wrapper, reads 066 closure as a G4 unlock, and can retarget `KGAS066-latest` via the default `--galaxy KGAS066` headless path.

Canon used below (not chat): S1 (`docs/diagnostics/s1-mock.md`) vis recovered inject r_t=0.254″ vs truth 0.25″; CLEAN-beam M1 inner slope **94.7 vs truth 236.7** km/s/arcsec; M2 56 vs 8; **3DBarolo was not on PATH**. Leftover-vs-velocity True at Stage B (uv span 0.093, vel span 0.335). S2 Laplace SBC failed 68/95.

## Attacks / bounds

1. **Two cube fitters on the real 10 km/s cube become a second likelihood unless they stay S1-style comparators.** Leftover and 2026-09-02 methodology dual-accept already rejected a cube likelihood and forbade `from kinms` in `src/kinuv/**` or `scripts/*.py`. This card re-opens **running** 3DBarolo and KinMS. That is accept-eligible only as comparators. Propose Track A item 2 then compares “Barolo/KinMS formal errors vs NUTS ESS widths.” That is a posterior-vs-posterior table. Item 3 lets the cube fitter float `i` or warp and report Δ`i`. DEC-066-INC freezes kinUV `i` at 43.86°. DEC-066-ZEROMODEL / methodology: the fit is `chi2 = s * sum w |ΔV|^2` on 881×95. S1 already scored the vis-vs-cube claim (CLEAN M1 94.7 vs truth 236.7); that cube estimator was restoring-beam M1/M2, not 3DBarolo. Shipping KinMS + Barolo `r_t` next to NUTS median 0.224″ will be copied as “KinMS agrees,” then as an inner scale (`quote_inner_slope` is false).

   **Bound:** vis χ² stays the only likelihood. Cube outputs are comparators, not calibrated posteriors, not a joint Barolo+KinMS likelihood, and not a replacement leftover model. S1 row in the S3 README **restates** CLEAN M1 94.7 vs truth 236.7 and “3DBarolo was not on PATH”; do not rewrite S1 as if Barolo ran in August. Do not write cube-fitter χ², formal errors, or `i` into `kinuv-KGAS066-uvsign-map` or `sd3ckpf2` JSON. Do not unfreeze kinUV `i`. Do not call leftover `s_1` / `c_3`. Artifact README **first sentence** after the H1: vis χ² is the fit; KinMS/Barolo are image-plane comparators. KinMS home is exactly `external/kinms_kgas66.py`. `scripts/run_s3_barolo.py` may only `subprocess` `BBarolo` / `3dbarolo` — **no** `import kinms`. Do not `pip install` into `kinuv-venv-recovery`. Do not execute the uvkin KinMS notebook. If a tool is missing on PATH, STATUS one-liner and ship S3 from S1 plus whichever tool ran. `quote_inner_slope: false` on every cube-fit `r_t` vs 0.224″ row.

2. **Track B inventory is “find vis + Ico on `/arc`.” That is not a gate.** Propose: find continuum-subtracted vis + Ico; if missing, stop; do not invent an npz. No named candidates. `DEC-066-TARGET` answer is still **KGAS066 only** until the user adds a stub (agents do not write `DEC-*`). 066 MAP already beats V=0 (Δχ² +35553) and S1 recovered inject; that satisfies TARGET’s *gate language*, not a license to treat a chat date as a TARGET amendment. A glob that hits `KILOGAS066.npz`, a laptop path, or a copied 066 array under a 007 name would “find vis.” STAGE A on the wrong galaxy still writes a tree that looks like a 007 product.

   **Bound:** Track B is a **new-tree diagnostic** only. Inventory must record the exact paths searched and the exact files used (or “none”). Accept only continuum-subtracted vis whose galaxy id is 007 (not 066 renamed) plus 007 Ico on `/arc`. If either is absent after that named search, STATUS one-liner and **stop Track B**. Do not invent an npz. Do not copy 066 vis. Do not pull a laptop file onto `/arc` to unblock the MAP. Do not write a TARGET DEC. 007 MAP only if those vis exist; Hann+bin, empirical `s`, two-start PA, freeze `i` from the 007 catalogue, Δχ² vs V=0. No 007 NUTS this card.

3. **`test_no_uvkin_or_kinms_import` is blind to `tools/` (and any new top-level package).** `tests/test_forward.py` rglob-bans `from kinms` / `import kinms` / `from uvkin` / `import uvkin` only under `src/kinuv` and `scripts`. `tools/` does not exist today. Propose execute item 4 only re-asserts that existing test. An implementer who drops `tools/run_s3_kinms.py` (or `bin/`, or a second file next to `external/`) keeps the test green and puts KinMS on the kinUV PATH. Propose already says KinMS lives in `external/kinms_kgas66.py` so the import ban stays green — that is the hole, not the lock.

   **Bound:** KinMS import is allowed in **one** file: `external/kinms_kgas66.py`. Extend `test_no_uvkin_or_kinms_import` (or a sibling) to fail on `from kinms` / `import kinms` / `from uvkin` / `import uvkin` under `tools/`, `bin/`, and any new top-level `*.py` / package other than that single external file. `scripts/run_s3_barolo.py` stays subprocess-only. Do not add a `kinuv.tools` package that imports KinMS.

4. **066 closure is not a G4 unlock.** Gold-standard G4 is Talts SBC on the exact kernel after G3 CPU smoke, with a leftover caveat. Receding G3 landed (`sd3ckpf2`). Leftover is still structured (SB-dominated) at MAP / NUTS-mean / Stage B. S2 Laplace SBC already failed 68/95. `intervals_calibrated: false`. Propose reject-list names G4, but Track A item 2 (formal errors vs NUTS ESS) plus “approaching search is closed” is the sentence that lets execute treat widths as calibrated and start Talts “now that 066 is done.” Leftover structured is a G4 **physics** stop, not a STATUS one-liner that quietly unlocks SBC.

   **Bound:** do not start G4 or G5. Do not quote S2 or NUTS 16/50/84 as calibrated intervals. Cube-fitter formal errors are not a calibration of NUTS ESS. Leftover gate stays **SB-dominated**. `quote_inner_slope: false`. Do not add `s_1` / `c_3` to DEC-066-VC. Closing approaching does not change those flags.

5. **Track B / any new runner can steal `KGAS066-latest`.** `scripts/launch_headless.py` defaults `--galaxy` to `KGAS066`. `steal_latest(kind)` is **True** for every kind that is not `*pa25*` (`tests/test_canfar_runner.py`: `steal_latest("nuts") is True`). `point_latest(galaxy, run_id)` writes `{galaxy_tag}-latest`. A 007 MAP that reuses the headless launcher without `--galaxy KGAS007` **and** a kind that returns `steal_latest False` retargets the receding 066 pointer. Propose says “Do not steal `KGAS066-latest`” and names the 007 tree, but does not lock the runner. Official MAP overwrite is a separate forbidden write: never `kinuv-KGAS066-uvsign-map`.

   **Bound:** Track B does **not** call `launch_headless.py` / `point_latest` / `steal_latest` on KGAS066. Do not retarget `KGAS066-latest` (stays receding). Do not write `kinuv-KGAS066-uvsign-map`. 007 product path is only `/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS007/kinuv-KGAS007-stage-a-map/`. If a headless job is used at all, `--galaxy KGAS007` is required and `point_latest` must not run for that job (kind that `steal_latest` is False, or skip the call). Prefer a local Stage A L-BFGS to that tree — MAP is not NUTS (DEC-066-INFER). No 007 NUTS.

## Comments

1. `major` — Vis χ² is the only likelihood. Barolo and KinMS are S1-style comparators, not a second posterior. Do not rewrite S1 (CLEAN M1 94.7 vs 236.7; 3DBarolo was not on PATH). Do not unfreeze `i`. README first sentence: vis χ² is the fit. Attack 1.

2. `major` — Track B inventory is a named 007 vis+Ico search on `/arc`, then stop if missing. No invent npz. No TARGET DEC from agents. New-tree diagnostic only. Attack 2.

3. `major` — KinMS home is exactly `external/kinms_kgas66.py`. Scripts may only subprocess Barolo. Extend the import ban to `tools/` / `bin/` / new top-level py. Attack 3.

4. `major` — Do not start G4 or G5. Leftover stays SB-dominated. No calibrated 16/50/84. No `s_1` / `c_3`. Attack 4.

5. `major` — Do not steal `KGAS066-latest`. Do not write the official 066 MAP tree. 007 MAP, if vis exist, is the named 007 tree only. Attack 5.

6. `minor` — Do not quote V_0 = 353 km/s as the NUTS mean (mean is 255 km/s). Stage A remains arctan. Approaching catalogue L-BFGS (PA 25.2 stuck, V_0=0 box, χ² 199968) stays a closed search, not a second product.

7. `minor` — If `BBarolo` / KinMS is missing, do not `pip install` into `kinuv-venv-recovery`. STATUS one-liner; S3 still ships from S1 plus whichever tool ran.

## Residual risks

1. 3DBarolo / KinMS absent on CANFAR/astroml PATH. S3 then restates S1 only. (propose residual 1)

2. 007 vis not staged on `/arc`. Track B stops; 066 S3 still ships. (propose residual 2)

3. TARGET without a user stub: 007 MAP is a **new-tree diagnostic** only. Official 066 product unchanged. (propose residual 3)

4. Cube-fit `r_t` vs 0.224″ will be copied as a science inner scale. README must say NUTS median, uncalibrated, `quote_inner_slope: false`. (propose residual 4)

5. External KinMS script can still be mistaken for a kinUV likelihood. Artifact README first sentence: vis χ² is the fit. (propose residual 5)

6. **New.** Dual cube-fitter formal errors vs NUTS ESS will be pasted as “S3 calibrated the widths.” Comment 1 + 4 forbid that reading; leftover structured still fails G4 for physics.

7. **New.** A `tools/` wrapper would bypass today’s import test. Comment 3 is the lock; without the extended test the green suite is not evidence.

8. **New.** Default `--galaxy KGAS066` plus `steal_latest("nuts") is True` will retarget latest if Track B reuses the NUTS launcher. Comment 5 is the lock.

## STATUS updates required

- `verdict: accept`
- `severity: major`
- `last_review_b:` this file
- Do not set `board: accepted` (parent tallies)
