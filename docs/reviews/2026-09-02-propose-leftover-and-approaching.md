---
role: proposer
date: 2026-09-02
agent: parent
canon_generation: 4
ids:
  - DEC-066-INFER
  - DEC-066-PA
  - DEC-066-SHIFT
  - DEC-066-TARGET
  - DEC-066-VC
  - DEC-066-ZEROMODEL
  - DEC-067-RUNNER
verdict: propose
---

# Leftover decomposition + approaching-mode NUTS (066 kernel)

## Scope

G3 receding NUTS is landed (`sd3ckpf2`; mixing pass; leftover still structured). G3 dual-accept already named two PA runs; approaching 25.2° was not launched. This card finishes that run and documents leftover structure at three vis points. Existing ids only. No new `DEC-*`. Official MAP `kinuv-KGAS066-uvsign-map` stays read-only. `DEC-066-TARGET` still 066. No G4. No G5. No KGAS007. No GPU. Do not logit `RT_BOUNDS_ARCSEC=(0.5, 15)`. Do not quote S2 16/50/84 or inner `dV/dr`. Do not fudge velocity.

Canon numbers are the landed JSON, not a chat summary:

| Product | PA (deg) | r_t (arcsec) | V_0 (km/s) | chi2 |
|---|---|---|---|---|
| Stage A MAP | 199.73 | 0.5 (L-BFGS floor) | 267.7 | 168675.6 |
| Receding NUTS mean | 200.05 | 0.224 | 255 | 167487 (Delta=-1189 vs MAP) |
| Stage B N=7 lambda=0 | — | rings | — | 167302 (Delta=+1373 vs A) |

Gap NUTS-mean vs Stage B is ~185, not 103. Receding mixing: R_hat <= 1.004, ESS >= 889 on six sampled names; (dx, dy) frozen. Image-plane centroids already in `docs/reviews/artifacts/2026-08-30-final-fit/vsys_shift.json`: optical MAP-catalog +24.07 km/s; aperture Delta v_M-D approaching +12.71 vs receding +36.31 km/s. Root cause on file: vis-weighted vsys vs CLEAN weighting, not WCS/Hann. Do not apply a velocity nudge.

## Architect verdict (selected path)

**Track A — approaching PA 25.2° (complete G3; do not pool with receding).**

CPU NumPyro NUTS, 4 chains x 200 warmup + 600 draws, init PA=25.2°, other theta at official MAP, (dx, dy) frozen at MAP host floats. Mixing on six sampled names only: R_hat <= 1.01, ESS bulk/tail > 400. Report chi2 at that run's mean vs receding 167487. Identity PA does not wrap. Do not average modes. Fail-to-mix toward PA~200° is a 180° result, not a license to flip PA in the product.

Headless worker today hardcodes `OFFICIAL_PA=199.73` and `write_nuts_product_plots` copies into `docs/reviews/artifacts/2026-08-30-g3-nuts/`. Execute must add `--pa-init` / `KINUV_PA_INIT`, kind `nuts-pa25`, a new artifact dir, and must not retarget `KGAS066-latest` or overwrite the receding G3 folder. Durable run: `/arc/projects/KILOGAS/analysis/toby_sandbox/kinuv_runs/{KGASID}-{YYYYMMDDTHHMMSSZ}-nuts-pa25/`. Scratch/JAX: `/scratch/kinuv-$USER/<session>`. Image `skaha/astroml:latest`, flexible CPU/RAM, **no `--gpu`** (recovery venv is CPU jax-finufft; DEC-067-RUNNER). There is no `config/nuts_pa25.yaml`; do not invent one. Launch via `scripts/launch_headless.py`.

**Track B — leftover decomposition (receding three-way; may run while A samples).**

Same vis operator (`hann_then_bin`, s=0.5136098555284736, `NPZ_UV_SIGN=-1`). Reuse `kinuv.diagnostics.figures` and existing leftover/imaging plotters; do not invent a second plotter. Artifacts: `docs/reviews/artifacts/2026-09-02-kgas066-leftover-and-modes/`.

Compare:

1. Official Stage A MAP (chi2=168675.6).
2. Receding NUTS mean (chi2=167487).
3. Official Stage B rings (chi2=167302).

Deliver leftover chi2 vs uv and vs velocity; per-channel chi2; dirty residual moments Delta M0, M1, M2; 4-panel spectral overlays with the existing aperture Delta v_M-D. Isolate whether the remaining ~185 vis chi2 is frozen Wiener I_CO (M0/spiral leftover; leftover-vs-velocity already True) vs something that would need s_1 / c_3. Do not add Fourier non-circular terms this card.

**KinMS benchmark (documentation, not a fitter).**

Tests forbid `from kinms` in kinUV. S1 already is the vis vs image-plane claim (`docs/diagnostics/s1-mock.md`: vis recovered r_t=0.25 arcsec; CLEAN M1 slope 95 vs truth 237 km/s/arcsec; 3DBarolo was not on PATH). This card restates that table and adds F^{-1}{Delta V} dirty residuals of the three 066 models vs CLEAN residual of the same sky. Store under `docs/reviews/artifacts/2026-09-02-kinuv-vs-kinms-benchmark/` (or a subsection of the leftover folder). Do not install KinMS. Do not run a cube fitter.

**Post-leftover gate (STATUS only).**

Decide whether leftover is still SB-dominated. Do not land a new official MAP tree on this card even if r_t is off the floor. A floor-free MAP rewrite is a later propose.

**Reject this wave:**

- GPU / `--gpu 1` / CUDA image (DEC-067: recovery venv is CPU jax-finufft).
- Durable products under `$HOME` or `/arc/home/thbrown/kinuv_runs/`.
- Overwrite `docs/reviews/artifacts/2026-08-30-g3-nuts/` or `kinuv-KGAS066-uvsign-map`.
- Retarget `KGAS066-latest` at the approaching run.
- Stack/average receding and approaching chains.
- Logit of `[0.5, 15]` arcsec. Quote S2 16/50/84. Quote inner `dV/dr`.
- Velocity fudge / ad-hoc vsys nudge.
- Import KinMS / emcee / 3DBarolo as a kinUV likelihood.
- G4 Talts SBC. G5 PSIS-LOO. KGAS007. Unfreeze `i`. Add `h_z`. New `DEC-*`.

Human review surface: `docs/reviews/artifacts/2026-09-02-kgas066-leftover-and-modes/` (plus approaching PNGs when the job finishes). Not a mid-gate chat.

## What changed / what was checked

- Receding NUTS: `docs/reviews/artifacts/2026-08-30-g3-nuts/` (`sampler: nuts`, chi2_nuts_mean=167486.76, r_t mean 0.224, leftover_chi2_structured true, r_t_at_floor false on draws).
- Official Stage A: chi2=168675.59555208942, s=0.5136098555284736, PA=199.7298, r_t=0.5, V_0=267.67, dx=0.091, dy=0.019.
- Stage B: chi2=167302, Delta vs A = +1373. Quoted rotation curve stays Stage A arctan.
- vsys_shift.json centroids as above. Do not fudge.
- Headless worker `scripts/run_kgas066_nuts_headless.py` hardcodes `OFFICIAL_PA`. Entrypoint takes `RUN_ID` only. `launch_headless.py` writes `pa_init_deg: 199.73` into the manifest and always `point_latest`.
- `write_nuts_product_plots(..., artifact_dir=ARTIFACT_G3)` defaults to the receding G3 folder.
- S1 vis vs CLEAN-beam: `docs/diagnostics/s1-mock.md`. `tests/test_forward.py` forbids KinMS imports.
- DEC-067-RUNNER: flexible CPU/RAM, no GPU unless live JAX sees CUDA. Durable path is `toby_sandbox/kinuv_runs`, never `$HOME`.
- DEC-066-TARGET still 066. Gold-standard dual-accept: MAP vs V=0 does not unlock a survey runner.

## Rejected alternatives

- "Silent-execute approaching because G3 already accepted it" — leftover analysis and artifact isolation are new scope; board them.
- "GPU because 066 NUTS is slow" — receding mixed in 4.84 h on CPU; DEC-067 forbids a GPU the venv cannot use.
- "Install KinMS / 3DBarolo on CLEAN cubes this card" — S1 already scored the image-plane claim; kinUV tests forbid the import; methodology 10x is vis vs CLEAN, not NUFFT vs KinMS clouds.
- "New official MAP without the r_t floor" — leftover_vs_velocity still True at the NUTS mean; a MAP rewrite is a later propose.
- "Start KGAS007 / G4 / G5" — leftover kernel is misspecified; SBC would fail for physics, not sampler bugs. TARGET still 066.

## Residual risks

1. Approaching init at receding MAP kinematics may not mix. That is a 180° mode result. Do not stack runs to "make ESS."
2. `write_nuts_product_plots` will clobber G3 receding artifacts unless `artifact_dir` is overridden. Gate: approaching must not write `2026-08-30-g3-nuts/`.
3. `point_latest` will steal `KGAS066-latest` unless the launcher skips it for `nuts-pa25`.
4. Track B dirty residuals of Stage B still use the frozen Wiener I_CO; a quiet M1 residual does not clear leftover-vs-velocity.
5. Real-066 16/50/84 remain uncalibrated (S2 Laplace SBC failed 68/95; leftover structured). Product README must say so.
6. Headless wall ~5 h. Interactive agents must not block on the sampling loop (DEC-067-RUNNER). Track B does not wait on Track A.
7. First JIT of the 066 potential can be minutes. Speed notes are post-warmup.

## Execute if accepted

Boundary: approaching-PA CPU NUTS + leftover/dirty-residual diagnostics on 066. No G4. No GPU. No MAP write. No KinMS import.

1. Add `--pa-init` / `KINUV_PA_INIT` to the headless worker and pass it from `launch_headless.py` / entrypoint env. Kind `nuts-pa25`. Do not hardcode 199.73 in the approaching product note.
2. `write_nuts_product_plots` takes an explicit artifact dir for this kind. Default for receding stays `2026-08-30-g3-nuts`. Approaching writes `docs/reviews/artifacts/2026-09-02-kgas066-leftover-and-modes/` (or a `pa25/` subdir). Unit test: approaching path does not write the G3 receding folder.
3. `launch_headless.py --kind nuts-pa25` does **not** retarget `KGAS066-latest`. Manifest records `pa_init_deg: 25.2`. CPU flexible, no `--gpu`.
4. Dispatch `python scripts/launch_headless.py --galaxy KGAS066 --kind nuts-pa25`. Watcher patches Agent Run Status. Receding run dir stays untouched.
5. Track B (no sampling): leftover chi2, per-channel chi2, dirty residual moments, spectra at MAP / receding NUTS-mean / Stage B into `docs/reviews/artifacts/2026-09-02-kgas066-leftover-and-modes/`. Reuse existing plotters. README with canon chi2 table and vsys_shift numbers. KinMS section restates S1 + dirty vis residuals (no KinMS binary).
6. Patch STATUS Agent Run Status + mailbox. Refresh Field Guide mailbox (G3 landed; this card is leftover + PA 25.2). CHANGELOG. Do not start G4. Do not write `kinuv-KGAS066-uvsign-map`.
7. Commit and push `origin/dev` after the runner patch, after dispatch (docs), and after Track B artifacts. Conventional subject. Do not skip hooks.

## STATUS updates required

- `next_role: board`
- `board: open`
- `last_propose:` this file
