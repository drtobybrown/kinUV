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
2. Flexible resources by default: omit `--cpu` and `--memory`. Pass `--gpu` only when the live JAX build actually sees CUDA (this repo's recovery venv is CPU jax 0.11.1 / jax-finufft; do not request a GPU that that venv cannot use). If a session vanishes without a product, pin `--cpu`/`--memory` on relaunch (flexible cap is ≤8 cores / ≤32 GB; platform max is 16 cores / 192 GB).
3. Fast compile / TMP: `/scratch/kinuv-$USER`. Manifests, **job-owned logs**, and posterior JSON: `/arc/home/thbrown/kinuv_runs/<run_id>/`. Platform `canfar logs` expire in ~1 hour and vanish on 404; the worker tees stdout/stderr to `worker.log`, writes structured `logs/run.log` + 30 s `status.json` (fsync), and a submit-host watcher copies `canfar logs`/`info`/`events` until the session is gone.
4. Session name stays `kinuv-KGAS066-{git_sha[:6]}-nuts` (`DEC-OPS-AUTH`). `cadc-get-cert` immediately before `canfar create` when the cert is missing or near expiry.
5. Image preference: `skaha/astroml:latest`. If the platform refuses that image for `headless`, fall back to a headless-tagged image (`skaha/base-notebook:latest`).
6. Official MAP `kinuv-KGAS066-uvsign-map` stays read-only. `sampler: nuts` on a 066 product still requires autodiff + mixing (`R_hat < 1.01`, bulk/tail ESS > 400 on sampled names).

Interactive debug and unit tests may keep a short wall cap. Headless workers ignore it.
