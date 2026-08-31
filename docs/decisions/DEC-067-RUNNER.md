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
2. Flexible resources by default: omit `--cpu` and `--memory` (platform grows to ≤8 cores / ≤32 GB and is easier to schedule than a pinned 64 GB request). Pass `--gpu` only when the live JAX build actually sees CUDA (this repo's recovery venv is CPU jax 0.11.1 / jax-finufft; do not request a GPU that that venv cannot use).
3. Compute on `/scratch/kinuv-$USER/<session>` (TMP, JAX cache, chain `npz`). Durable products — manifests, **job-owned logs** (`worker.log`, `logs/run.log`), chain checkpoints, posterior JSON — go to `/arc/projects/KILOGAS/analysis/toby_sandbox/kinuv_runs/{KGASID}-{YYYYMMDDTHHMMSSZ}-{kind}/`, never `$HOME`. `{KGASID}-latest` is a symlink to the newest run. After each chain, write draws on scratch then copy+fsync to `/arc` (file-handle `npz`, not `savez(path.tmp)`). Crash/SIGTERM copies scratch `*.npz` onto `/arc`. `/scratch` is ephemeral: tee worker stdout to both scratch and `/arc` so logs remain on `/arc` after fail, OOM, or success. Platform `canfar logs` expire in ~1 hour and vanish on 404; a submit-host watcher also copies `canfar logs`/`info`/`events` onto that run dir until the session is gone.
4. Session name stays `kinuv-KGAS066-{git_sha[:6]}-nuts` (`DEC-OPS-AUTH`). `cadc-get-cert` immediately before `canfar create` when the cert is missing or near expiry.
5. Image preference: `skaha/astroml:latest`. If the platform refuses that image for `headless`, fall back to a headless-tagged image (`skaha/base-notebook:latest`).
6. Official MAP `kinuv-KGAS066-uvsign-map` stays read-only. `sampler: nuts` on a 066 product still requires autodiff + mixing (`R_hat < 1.01`, bulk/tail ESS > 400 on sampled names).

Interactive debug and unit tests may keep a short wall cap. Headless workers ignore it.
