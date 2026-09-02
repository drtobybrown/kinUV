---
role: reviewer
seat: a
date: 2026-09-02
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
propose: docs/reviews/2026-09-02-propose-leftover-and-approaching.md
---

# Review a: leftover decomposition + approaching-mode NUTS (066 kernel)

Do not read the other seat's review file. Do not implement.

Scope check: finish the G3 approaching PA 25.2° run and document leftover at three vis points. Existing ids only. No new `DEC-*`. Official MAP `kinuv-KGAS066-uvsign-map` stays read-only. `DEC-066-TARGET` still 066. No G4. No G5. No KGAS007. No GPU. Do not logit `RT_BOUNDS_ARCSEC=(0.5, 15)`. Do not quote S2 16/50/84 or inner `dV/dr`. Do not fudge velocity. Do not import KinMS. Canon chi2 is the landed JSON, not a chat summary: Stage A MAP `168675.59555208942` (`docs/reviews/artifacts/2026-08-30-final-fit/leftover_chi2.json`), receding NUTS-mean `167486.7639374534` (`docs/reviews/artifacts/2026-08-30-g3-nuts/leftover_chi2.json` and `summary.json` `chi2_nuts_mean`), Stage B N=7 λ=0 `167302.18673431588` (`kinuv-KGAS066-uvsign-map/stage_b_map.json`; Δ vs A = `1373.4088177735393`). Propose table (168675.6 / 167487 / 167302) matches those files. It does **not** quote the false 157178 set. Receding mixing on disk: R_hat ≤ 1.004, ESS ≥ 889 on six sampled names; `(dx, dy)` frozen; `leftover_chi2_structured: true`. vsys_shift.json: optical MAP−catalog +24.07 km/s; aperture Δv_M-D approaching +12.71 vs receding +36.31 km/s. Reject-this-wave list (GPU, `$HOME` run dirs, KGAS007, G4, logit `[0.5, 15]`, MAP overwrite, stack PA runs, KinMS as a fitter) stays rejected. That selected path is accept-eligible. Execute as typed can still launch a second receding chain, quote Stage B 167302 next to image-plane moments without recomputing ring vis leftover, and label a CLEAN-matched cube `F^{-1}{ΔV}`.

## Attacks / bounds

1. **`KINUV_PA_INIT` is not a delivery path today; execute item 1 can land as dead flags and still init at 199.73°.** Live chain:

   - `scripts/run_kgas066_nuts_headless.py` argparse is `--run-id` only. `OFFICIAL_PA = 199.72980072503037` is assigned to `start["pa_deg"]`, `product_record(pa_init_deg=…)`, and the product note (`"066 headless NUTS PA 199.73; …"`). The worker never reads a manifest.
   - `scripts/launch_headless.py` writes `pa_init_deg: 199.72980072503037` into the **manifest** and always calls `point_latest` **before** the dry-run return. The `env` dict passed to `submit_headless` is `KINUV_RUN_ID`, `KINUV_GALAXY`, `KINUV_PROJECT`, `KINUV_RUNS`, `JAX_*`, `PYTHONUNBUFFERED`. There is no `KINUV_PA_INIT`.
   - `scripts/canfar_entrypoint.sh` takes `RUN_ID` as `$1` only (`command: ["/bin/bash", entry, run_id]`). Last line: `python …/run_kgas066_nuts_headless.py --run-id "${RUN_ID}"`. It does not forward a PA.
   - `KINUV_RUN_ID` reaches the worker because it is both `--env` **and** argv. Manifest JSON is documentation. Adding `--pa-init` to the launcher argparse and writing `pa_init_deg: 25.2` into the same manifest (already a field) does not change `start["pa_deg"]`.

   Execute item 1 says “pass it from `launch_headless.py` / entrypoint env” and does not name the `--env` dict. Execute item 4 then dispatches; item 7 commits after the runner patch **and** after dispatch. The entrypoint `git pull --ff-only origin dev` unless `KINUV_SKIP_PULL=1`. Dispatch before that commit is on `origin/dev` pulls a worker that still hardcodes 199.73°. Five hours of “approaching” is receding again.

   **Bound:** (a) `launch_headless.py --kind nuts-pa25` puts `KINUV_PA_INIT=25.2` in the `env` dict passed to `submit_headless` (`--env`, same path as `KINUV_RUN_ID`). Manifest `pa_init_deg: 25.2` is not a delivery path. (b) Worker `--pa-init` defaults to `os.environ.get("KINUV_PA_INIT")`; `start["pa_deg"]`, `product_record(pa_init_deg=…)`, and the product note use that float. Receding default without the env remains `OFFICIAL_PA`. Entrypoint may keep argv = `RUN_ID` only if (a)+(b) hold; optional `--pa-init "${KINUV_PA_INIT}"` is belt-and-suspenders, not a substitute for `--env`. (c) Unit tests: launcher env for `nuts-pa25` contains `KINUV_PA_INIT=25.2` and `JAX_PLATFORMS=cpu` and does not pass `--gpu`; worker init at 25.2 when that env is set. (d) Dispatch only after the runner patch is on `origin/dev` (or `KINUV_SKIP_PULL=1` on a tree that already has the patch). Do not hardcode 199.73 in the approaching product note or `CORNER_TITLE` (`src/kinuv/runner/plots.py`).

2. **Track B can quote vis chi2 167302 and only plot image-plane Stage B moments.** Official Stage B leftover on vis is `stage_b_map.json` `chi2_map=167302.18673431588` from `kinuv.infer.stage_b.predict_binned` (rings, Hann+bin, same 881×95 operator). Live leftover writer `write_leftover_at_params` calls `kinuv.infer.map.predict_binned` (Stage A arctan 8-vector). Existing Stage B figures (`docs/reviews/artifacts/2026-08-28-stage-b-imaging/`) are CLEAN-matched D/M/R of `stage_b_model_cube.fits`, not vis leftover. Execute item 5 says leftover at MAP / receding NUTS-mean / Stage B and “reuse existing plotters.” Feeding ring knots through Stage A `predict_binned`, or copying 167302 next to those imaging PNGs, does not measure the ~185 vis gap (167486.76 − 167302.19). Residual 4 already says a quiet M1 does not clear leftover-vs-velocity; it does not require recomputing ring vis leftover.

   **Bound:** Track B leftover of Stage B must call `kinuv.infer.stage_b.predict_binned` with official read-only `v_knots_kms` / `r_knots_arcsec` from `stage_b_map.json`, then `leftover_chi2(data, model)` on the 881×95 array (`s=0.5136098555284736`, `hann_then_bin`, `NPZ_UV_SIGN=-1`). Require `|chi2_sum − 167302.18673431588| < 1`, `n_row=881`, `n_chan=95`. Same identity for Stage A `|chi2_sum − 168675.59555208942| < 1` and receding NUTS-mean `|chi2_sum − 167486.7639374534| < 1`. Recompute `leftover_velocity_structured` per model (`src/kinuv/diagnostics/flags.py`); do not copy `leftover_chi2_structured: true` from the G3 JSON. If the Stage B vis leftover bound fails, STATUS one line and do not hand the user a leftover of a different operator. Image-plane moments are a second product, not a substitute for that vis sum.

3. **`F^{-1}{ΔV}` is not defined on the live type-2 path, and it is not a 2-D FFT of the 881×95 vis array.** `src/kinuv/transforms/nufft.py` is type-2 degrid only (`nufft2` / `nufft2d2`; “Type-3 is not implemented”). There is no type-1 adjoint. `dirty_cube_from_truth` (`diagnostics/s1.py`) is `sky_cube` → restoring-beam match onto the 10 km/s WCS, not an inverse FT of residual vis. ΔV is `data.vis − model` with shape `(881, 95)` on irregular `(u,v)` samples. `numpy.fft` of that array is not an image. The CLEAN residual of the same sky is a different operator (restoring beam + CLEAN components). Propose KinMS section: “adds `F^{-1}{ΔV}` dirty residuals of the three 066 models vs CLEAN residual of the same sky” and “reuse existing plotters.” Those plotters will emit CLEAN-matched cubes and call them dirty. `tests/test_forward.py` `test_no_uvkin_or_kinms_import` only glob `src/kinuv/forward/*.py`; `tests/test_map.py` only glob `src/kinuv/infer/*.py`. A new leftover script under `scripts/` can `import kinms` and both tests stay green.

   **Bound:** either (a) type-1 NUFFT of `(data−model)` per channel on the 881×95 array (natural weights; no PB divide; 95 dirty images → ΔM0/M1/M2) and label it adjoint residual, **or** (b) drop the `F^{-1}{ΔV}` claim and plot existing CLEAN-matched cubes with a caption that they are not vis inversions. Do not `numpy.fft` the vis array. Do not call `dirty_cube_from_truth` / `plot_stage_b_vs_imaging` output `F^{-1}`. Do not install or import KinMS / emcee / 3DBarolo. Extend the KinMS import ban this card to `src/kinuv/**/*.py` and `scripts/*.py` (rglob), not only `forward/` and `infer/`. Restate the S1 table as vis vs CLEAN-beam (`r_t` 0.254 vs truth 0.25; M1 slope 94.7 vs 236.7), not NUFFT vs KinMS clouds.

4. **Approaching mixing must lock to DEC-067 (`ESS > 400`), not G3 tiny-mock `ESS > 200`; `point_latest` and `ARTIFACT_G3` will clobber receding unless tests fail closed.** Propose Track A: `R_hat ≤ 1.01`, ESS bulk/tail > 400. That matches DEC-067-RUNNER and the live worker `mixing_ok(..., ess_min=400.0, ess_tail_min=400.0)`. Live `mixing_ok` default is still `ess_min=200.0` (`src/kinuv/infer/nuts.py`); G3 tiny-mock used 200. `write_nuts_product_plots(..., artifact_dir=ARTIFACT_G3)` defaults to `docs/reviews/artifacts/2026-08-30-g3-nuts/`. The worker calls it **without** `artifact_dir`. DEC-067-RUNNER literally says copy PNGs + posterior JSON into that G3 folder. Execute as typed (do not write G3) **leaves** that copy dest; Field Guide allows a STATUS one-liner. Residual 2 names the clobber; execute item 2 names a unit test. Live `test_write_nuts_product_plots_corner_only` already passes a tmp `artifact_dir` and never asserts `ARTIFACT_G3` is untouched. `point_latest` always runs, including `--dry-run`. DEC-067 also says `{KGASID}-latest` is the newest run; skipping it for `nuts-pa25` is a second leave-the-DEC.

   **Bound:** approaching `sampler: nuts` only if `mixing_ok(mix, rhat_max=1.01, ess_min=400.0, ess_tail_min=400.0)` on the six sampled names. Tiny-mock `ESS > 200` stays G3 gate 4 only. Fail-to-mix (`COMPLETED_UNMIXED`, walk toward PA ~200°) is a 180° result; do not stack with receding; do not flip PA in the product; identity chart does not wrap. Worker approaching call must pass an explicit `artifact_dir` that is **not** `ARTIFACT_G3`; receding default stays `2026-08-30-g3-nuts`. Unit test: approaching kind writes `docs/reviews/artifacts/2026-09-02-kgas066-leftover-and-modes/` (or `pa25/` subdir) and does not create or modify `docs/reviews/artifacts/2026-08-30-g3-nuts/` (monkeypatch `ARTIFACT_G3` to a sentinel). `launch_headless.py --kind nuts-pa25` does not call `point_latest` (including dry-run). STATUS records both DEC-067 leaves (artifact dest; latest symlink). Do not retarget `KGAS066-latest`. Durable path remains `/arc/projects/KILOGAS/analysis/toby_sandbox/kinuv_runs/{KGASID}-{YYYYMMDDTHHMMSSZ}-nuts-pa25/`, never `$HOME`.

## Comments

1. `major` -- `KINUV_PA_INIT` must ride `submit_headless --env` (and worker argparse defaulting to that env). Manifest `pa_init_deg` is not the worker init. Dispatch only after the runner patch is on `origin/dev`. Do not hardcode 199.73 in the approaching product note / corner title.

2. `major` -- Track B Stage B leftover is `stage_b.predict_binned` on official knots, then vis leftover on 881×95. Gate `|chi2_sum − 167302.18673431588| < 1` (and the MAP / NUTS-mean identities above). Recompute `leftover_velocity_structured` per model. Image-plane moments are not that vis sum.

3. `major` -- Bound `F^{-1}{ΔV}`: type-1 adjoint of `(881, 95)` residual vis, or drop the claim and caption CLEAN-matched cubes as not vis inversions. No `numpy.fft` of the vis array. No KinMS import. Extend the import ban to `src/kinuv/**` and `scripts/`.

4. `major` -- Approaching mixing is DEC-067 `ESS > 400` / `R_hat < 1.01` on six names, not tiny-mock 200. Explicit `artifact_dir` so approaching does not write `2026-08-30-g3-nuts/`. Unit test must fail closed. `nuts-pa25` skips `point_latest`. Do not stack PA runs. STATUS one-liner for the two DEC-067 leaves.

5. `minor` -- Reject-this-wave list stays: no GPU / CUDA image, no `$HOME` or `/arc/home/thbrown/kinuv_runs/`, no KGAS007, no G4/G5, no logit of `[0.5, 15]`, no in-place MAP write, no vsys nudge, no `i` / `h_z`. Operator `hann_then_bin`; `s=0.5136098555284736`; `NPZ_UV_SIGN=-1`. `intervals_calibrated: false`. Do not quote inner `dV/dr`. DEC-066-PA quoted product remains receding-side; approaching is a 180° mode test. CHANGELOG + Field Guide mailbox. Official MAP unchanged.

## Residual risks

1. Approaching init at receding MAP kinematics may not mix. That is a 180° mode result. Do not stack runs to “make ESS.” Carry-forward from the propose.

2. `write_nuts_product_plots` default `ARTIFACT_G3` will clobber receding unless `artifact_dir` is overridden **and** the unit test fails closed. Carry-forward; comment 4 is the lock.

3. `point_latest` will steal `KGAS066-latest` unless the launcher skips it for `nuts-pa25` (including dry-run). Carry-forward.

4. Track B dirty / CLEAN-matched residuals of Stage B still use frozen Wiener I_CO. A quiet M1 does not clear leftover-vs-velocity. Carry-forward. Comment 2 is the vis-sum lock this propose omitted.

5. Real-066 16/50/84 remain uncalibrated (S2 Laplace SBC failed 68/95; leftover structured). Product README must say so. Carry-forward.

6. Headless wall ~5 h. Interactive agents must not block (DEC-067-RUNNER). Track B does not wait on Track A. Carry-forward.

7. First JIT of the 066 potential can be minutes. Speed notes are post-warmup. Carry-forward.

8. **(new)** Entrypoint `git pull origin dev` plus “commit after dispatch” can run the unpatched worker. Comment 1(d) is the lock; if skip-pull is used, STATUS must name the tree SHA the session actually ran.

9. **(new)** DEC-067-RUNNER still names `docs/reviews/artifacts/2026-08-30-g3-nuts/` as the git copy dest. Approaching uses a new folder only as a documented leave. Do not silently treat that as an ADR rewrite. No new `DEC-*`.

## STATUS updates required

- `verdict` and `severity` as in the header (`accept`, `major`)
- `last_review_a`: this file
- Do not set `board: accepted` (parent tallies)
