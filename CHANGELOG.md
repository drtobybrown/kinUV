# Changelog

## 2026-09-02 — GPU rejected; CPU-parallel canonical

CUDA 0.55 vs CPU 3.01 eval/s on 881×95; GPU sessions killed; CUDA venv and GPU code purged. Production NUTS is flexible CPU headless; parallel path is 4×1-chain + merge. Note: `docs/architecture/notes/2026-09-02-gpu-rejection-cpu-parallel.md`. `xgepg7qy` still Running. Official MAP unchanged.

## 2026-09-02 — GPU NUTS smoke runner

Kind `nuts-gpu` skips `KGAS066-latest` and G3. `--gpu` requires pinned CPU+RAM and `KINUV_CHAIN_ID`. CUDA venv builder source-builds jax-finufft. Official MAP unchanged.

## 2026-09-02 — GPU NUTS smoke dual accept (major)

nuts-gpu skips KGAS066-latest and G3. GPU requires pinned CPU+RAM and KINUV_CHAIN_ID. jax stays 0.11.1. 10× wall is max(chain)+merge vs 17440 s. Official MAP unchanged.

## 2026-09-02 — GPU NUTS smoke propose

Board open. CUDA venv + 4 parallel GPU chains; 10× wall vs serial CPU 17440 s. Do not mutate recovery. Propose: `docs/reviews/2026-09-02-propose-gpu-nuts-smoke.md`. Official MAP unchanged.

## 2026-09-02 — CANFAR GPU headless ops doc

Document fixed-resource GPU submit, probe scripts, and 2026-09-02 scheduling results. Production NUTS still CPU. Official MAP unchanged.

## 2026-09-02 — uv-vs-image methodology notes landed

Dual accept (major). Notes are not an ADR. Not KinMS first sentence. No cube fitter, no 007, no G4. `xgepg7qy` still Running. Official MAP unchanged.

## 2026-09-02 — uv-vs-image methodology propose

Board open. Docs-only S1/leftover restating. No KinMS fitter, no KGAS007, no G4. Propose: `docs/reviews/2026-09-02-propose-uv-vs-image-methodology.md`. Official MAP unchanged.

## 2026-09-02 — approaching-PA NUTS dispatched (`xgepg7qy`)

Flexible headless session Running, PA init 25.2, run `KGAS066-20260902T085027Z-nuts-pa25`. KGAS066-latest still receding. Official MAP unchanged. Do not start G4.

## 2026-09-02 — leftover three-way identity (SB-dominated)

Recomputed vis leftover on 881x95: MAP 168675.6, receding NUTS-mean 167486.8, Stage B 167302.2 (gap +184.6). leftover-vs-velocity True at Stage B. Plots: `docs/reviews/artifacts/2026-09-02-kgas066-leftover-and-modes/`. Official MAP unchanged. Do not start G4.

## 2026-09-02 — leftover + approaching-PA runner

KINUV_PA_INIT rides `--env`. `nuts-pa25` skips KGAS066-latest and writes leftover-and-modes/pa25, not G3. Leftover flag is measured. quote_inner_slope stays false while leftover-vs-velocity. Official MAP unchanged. Do not start G4.

## 2026-09-02 — leftover + approaching-PA dual accept (major)

KINUV_PA_INIT must ride `--env`; Stage B leftover is `stage_b.predict_binned` identity vs 167302.19; approaching must not write `2026-08-30-g3-nuts/` or steal `KGAS066-latest`. `quote_inner_slope` false while leftover-vs-velocity. No KinMS import or folder. Official MAP unchanged. Do not start G4.

## 2026-09-02 — leftover + approaching-PA propose

Board open. Receding NUTS chi2=167487 vs MAP 168676 vs Stage B 167302. Approaching PA 25.2 not run. No GPU, no G4, no KGAS007, official MAP unchanged. Propose: `docs/reviews/2026-09-02-propose-leftover-and-approaching.md`.

## 2026-09-01 — 066 NUTS corner and imaging plots

Headless worker writes corner, leftover chi2, and Data|Model|Residual PNGs at the NUTS mean (FITS stay in the run dir). 066 product in `docs/reviews/artifacts/2026-08-30-g3-nuts/`. Official MAP unchanged.

## 2026-09-01 — Job writes Agent Run Status

Headless worker and watcher patch `docs/architecture/STATUS.md` Agent Run Status and YAML `pending` on finish. They do not rewrite Architecture mailbox history. Official MAP unchanged.

## 2026-08-31 — Do not balloon /arc

Verbose NUTS stdout stays on scratch and is overwrite-copied to `worker.log` every 60 s. Watcher snapshots platform logs without appending full dumps. Chain checkpoints remain kB parameter `npz` only — no JAX cache, vis, or cubes on `/arc`. Official MAP unchanged.

## 2026-08-31 — Scratch-then-/arc chain checkpoints

`np.savez('*.tmp')` appended `.npz` and dropped chain-1 draws after 67 min. Checkpoints now write via a file handle on `/scratch`, then copy+fsync to `/arc`; crash/SIGTERM flushes scratch `*.npz`. Official MAP unchanged.

## 2026-08-31 — Flexible RAM; dated KGAS run dirs

Headless jobs stay flexible (≤8 CPU / ≤32 GB). Run dir is `{KGASID}-{YYYYMMDDTHHMMSSZ}-{kind}` with `{KGASID}-latest` pointing at the newest. Official MAP unchanged.

## 2026-08-31 — Project-volume runs, scratch compute

Durable logs, checkpoints, and posteriors go to `/arc/projects/KILOGAS/analysis/toby_sandbox/kinuv_runs/`, not `$HOME`. Jobs run on ephemeral `/scratch`; tee + fsync keep logs on `/arc` after fail/OOM/success. Official MAP unchanged.

## 2026-08-31 — Headless job-owned logs

Platform `canfar logs` expire in ~1 hour and vanish on 404. Each job tees stdout/stderr to `/arc/home/thbrown/kinuv_runs/<run_id>/worker.log`, writes `logs/run.log` + fsync heartbeats, and a submit-host watcher copies `canfar logs`/`info`/`events`. Official MAP unchanged.

## 2026-08-30 — CANFAR headless runner (DEC-067-RUNNER)

User: jobs > 15 min run as `canfar create headless` (flexible CPU/RAM). Interactive 7200 s cap is not a batch ceiling. GPU only if the live JAX build sees CUDA. KGAS066 NUTS 4×600 dispatched as session `h2dlc07f`. Official MAP unchanged.

## 2026-08-30 — G3 autodiff potential + CPU NumPyro NUTS

Dual accept (major). JAX `U(z)=0.5(chi2+shift_prior_const)-log|J|` on the G2 chart; sampled-name `float()` removed on the XLA vis path. `(dx, dy)` frozen at MAP host floats. Tiny-mock 4-chain NUTS mixed (`R_hat<1.01`, ESS>200) → `sampler: nuts`. 066 wall projection 8.3 h > 2 h cap; no 066 `sampler: nuts`; no GPU. numpyro 0.21.0 `--no-deps` keeps jax 0.11.1. Official MAP unchanged. Do not quote S2 16/50/84 or inner `dV/dr`.

## 2026-08-30 — Stage B spectral redshift is vis-weighted vsys, not WCS/Hann

Image-plane +30–50 km/s look is MAP optical vsys (~8323.6) vs catalogue/CLEAN (~8299.6), with asymmetric aperture centroids. Locked: radio↔optical, CO(2–1) `RESTFRQ`, `CRPIX3` centres, centred Hann `[0.25, 0.5, 0.25]`. Model cubes stamp `SPECSYS=LSRK` + `RESTFRQ`. Spectra annotate \(\Delta v_{\rm M-D}\); matched cube stays in the artifact dir. No vis-axis fudge; official MAP unchanged.

## 2026-08-30 — G3 NumPyro NUTS propose

Board open. Autodiff `chi2(θ(z))` + two CPU Stage A NUTS runs (PA 199.73 and 25.2, 4 chains). Freeze `(dx, dy)` at MAP. `sampler: nuts` only after autodiff. G2 already on `origin/dev` (`ee459af`, 17/17). No GPU, no logit of `[0.5, 15]`, official MAP unchanged.

## 2026-08-30 — G2 unconstrained Stage A chart + Jacobian

Dual accept (major). `kinuv.infer.chart`: log on flux / gas_sigma / r_t, stable `logaddexp` softplus on `V_0` (no Python `if`; `V_0=0` is `-inf`), identity on PA / vsys / (dx, dy). JIT surface is a length-8 vector; `log_prob_unconstrained` is host `log_prob(θ)+log|J|`, not autodiff. Official `r_t=0.5` maps to finite `z`. No NumPyro, no NUTS label, official MAP unchanged.

## 2026-08-30 — /scratch I/O, NUTS-only corners, senior handoff

Dual accept (major) on ops-scratch. High-frequency writes go to `/scratch/kinuv-$USER` (else `/tmp`), never NFS `/arc`. `plot_posterior_corner` requires `sampler == "nuts"` and an 8-column draw array; S2 `laplace_mh` JSON raises. Handoff: `docs/reviews/2026-08-30-handoff-senior.md`. No NUTS run, no G2, official MAP unchanged.

## 2026-08-30 — G1 CPU JAX `predict_binned`

Dual accept (major). `predict_binned(..., xla=True)` keeps JAX arrays from sky through NUFFT, Hann+bin, and `chi2`. NumPy path stays the identity reference. Official Stage A `chi2` matches 168675.6 at frozen `s=0.5136`. Post-warmup 3.01 eval/s vs S2 FD 0.329. Tiny-grid `jax.grad` vs FD. Tests set CPU/x64/`/tmp` cache. No NUTS, no GPU, no G2 logit of the `r_t` floor. Official MAP unchanged.

## 2026-08-30 — G0 MAP quality flags (066 kernel)

Dual accept (major) on the gold-standard sequence. Flags live in `kinuv.diagnostics.flags` (not `infer/`). Official 066 fires `r_t_at_floor` and leftover-vs-velocity vs leftover-vs-uv; PA=199.73 does not fire the 21.9 alias; `Delta_chi2` = +35553 records `beats_zero`. Roadmap rewritten as the 066 kernel sequence. No JAX, no NUTS, no GPU, no 400-galaxy runner. Official MAP unchanged.

## 2026-08-30 — Final-fit plot handoff

Dual accept (major). Regenerated Stage A leftover + Stage B D/M/R into `docs/reviews/artifacts/2026-08-30-final-fit/`. Leftover `chi2` = 168675.6. Official MAP not written. Matched cube stays in the artifact dir (not committed).

## 2026-08-30 — User reviews final plots; implementer owns gates

Physics stops are no longer user-blocking. The implementer decides each gate, notes it on STATUS, and continues. The human review surface is the final Data | Model | Residual + leftover `chi2` folder. Still forbidden: new `DEC-*`, in-place overwrite of `kinuv-KGAS066-uvsign-map`, calling Laplace-MH NUTS. Amended `DEC-066-AGENTS`.

## 2026-08-30 — Dual-review board; human methodology

User-directed handshake: the parent proposes; two independent reviewer sub-agents accept or reject (major/minor) on `docs/reviews/`. Dual accept → implement and execute the named stages with no third review. A user **build** command runs that loop end-to-end. Docs: `docs/methodology.md`, `docs/reviews/BOARD.md`, amended `DEC-066-AGENTS`. `build_licensed` stays false until that command. Official MAP unchanged. No new `DEC-*`.

## 2026-08-29 — S1/S2 land; kernel hygiene; diagnostic suite

### Production pointer

Official Stage A/B MAP remains `kinuv-KGAS066-uvsign-map`. S1 benchmark: `docs/reviews/artifacts/2026-08-29-s1-mock/`. S2 hybrid coverage: `docs/reviews/artifacts/2026-08-29-s2/` (`sampler: laplace_mh`, not NUTS). Image-plane check: `docs/reviews/artifacts/2026-08-28-stage-b-imaging/` (the 2026-08-27 folder is the pre-sign inverted-PA set).

### Deprecated kernels

- `native_diagonal` is gone from the mock and MAP paths. `kinuv.response.spectral.native_diagonal` now raises and points at `hann_then_bin`.
- `kinuv.likelihood` does not export `hann_then_bin` (that miss used to skip SPECRESP).
- Gate 2 (`tests/test_mock_recovery.py`) and S1 assert `kinuv.response.spectral.hann_then_bin` before generating vis.

### kinUV vs uvkin

Canonical Hann+bin, `NPZ_UV_SIGN=-1`, `fits_image_east_north`, and vis `chi2` live only in kinUV. uvkin is the KinMS/emcee science-matrix repo; see `docs/diagnostics/repos.md`. Tests already forbid uvkin imports in SPECRESP/vis/chi2 sources.

### Diagnostic plotters

- `kinuv.diagnostics.figures`: leftover `chi2` vs uv/velocity and PA/`gas_sigma`/`r_t` slices. ASCII labels. Style via `kinuv.diagnostics.style` (not viridis).
- `scripts/plot_fit_diagnostics.py` runs leftover (and optional imaging D/M/R). Preview dir `docs/reviews/artifacts/fit-diagnostics/` is gitignored.
- Guide: `docs/diagnostics/plotting.md`.

### What was not deleted

Stage A/B runners, `infer/campaign.py` (lambda-reg), historical reviews that mention `native_diagonal`, and the superseded 2026-08-27 imaging PNGs stay. No new `DEC-*`. No uvkin merge.
