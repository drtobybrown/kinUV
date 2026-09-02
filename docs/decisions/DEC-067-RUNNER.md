---
id: DEC-067-RUNNER
status: accepted
generation: 4
date: 2026-08-30
owner: user
---
# CANFAR headless execution ceiling

**Question:** When may an agent run a long NUTS (or other) job, and where does it run?

## Answer

User 2026-08-30 mission directive (user creates `DEC-*` ids). The 7200 s interactive/subagent cap is **not** a batch ceiling.

1. Jobs with expected wall-clock **> 15 minutes** run as asynchronous CANFAR **headless** sessions (`canfar create headless`). Interactive agents must not block on the sampling loop.
2. Flexible resources by default: omit `--cpu` and `--memory` (platform grows to ≤8 cores / ≤32 GB and is easier to schedule than a pinned 64 GB request). Pass `--gpu` only when the live JAX build actually sees CUDA (this repo's recovery venv is CPU jax 0.11.1 / jax-finufft; do not request a GPU that that venv cannot use). **GPU jobs:** pin `--cpu` and `--memory`; do not use flexible mode with GPU. Agent ops: [`docs/diagnostics/canfar-gpu.md`](../diagnostics/canfar-gpu.md).
3. Compute on `/scratch/kinuv-$USER/<session>` (TMP, JAX cache, chain `npz`, verbose stdout). Durable products — manifests, **job-owned logs** (`worker.log` overwrite-copied from scratch every 60 s, `logs/run.log`), chain-draw `npz` (kB parameter arrays, not vis), posterior JSON, **and PNGs** (6D corner, leftover chi2, moments/spectra/PV at the NUTS mean) — go to `/arc/projects/KILOGAS/analysis/toby_sandbox/kinuv_runs/{KGASID}-{YYYYMMDDTHHMMSSZ}-{kind}/`, never `$HOME`. Copy those PNGs plus posterior JSON into `docs/reviews/artifacts/2026-08-30-g3-nuts/`. `{KGASID}-latest` is a symlink to the newest run. After each chain, write draws on scratch then copy+fsync that `npz` to `/arc` (file-handle `savez`, not `savez(path.tmp)`). Crash/SIGTERM copies checkpoint `*.npz` onto `/arc`. Do not rsync the JAX cache, vis, cubes, or scratch tmp onto `/arc`. Do not tee tqdm onto NFS. FITS cubes for imaging stay in the run dir (`plots/`), not the official MAP tree and not git (`docs/reviews/artifacts/**/*.fits` is ignored). Platform `canfar logs` expire in ~1 hour and vanish on 404; a submit-host watcher snapshots `canfar logs`/`info`/`events` (overwrite, no append of full dumps into `platform.log`). On success or fail the worker (and the watcher, if still alive) patches `docs/architecture/STATUS.md` **Agent Run Status** and YAML `pending`; it does not rewrite Architecture mailbox history.
4. Session name stays `kinuv-KGAS066-{git_sha[:6]}-nuts` (`DEC-OPS-AUTH`). `cadc-get-cert` immediately before `canfar create` when the cert is missing or near expiry.
5. Image preference: `skaha/astroml:latest`. If the platform refuses that image for `headless`, fall back to a headless-tagged image (`skaha/base-notebook:latest`).
6. Official MAP `kinuv-KGAS066-uvsign-map` stays read-only. `sampler: nuts` on a 066 product still requires autodiff + mixing (`R_hat < 1.01`, bulk/tail ESS > 400 on sampled names).

Interactive debug and unit tests may keep a short wall cap. Headless workers ignore it.
