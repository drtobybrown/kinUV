# CANFAR scratch and checkpoints (human)

High-frequency I/O (JIT cache, TMP) goes on **node-local** `/scratch`. `/scratch` is ephemeral: a preempted or OOM-killed node loses it. Durable products (logs, MCMC checkpoints, posterior JSON, manifests) go to **`/arc/projects/KILOGAS/analysis/toby_sandbox/kinuv_runs/<run_id>/`**, never `$HOME`.

Root: `/scratch/kinuv-$USER/$SKAHA_SESSION_ID` (mode 0700), else `/tmp/kinuv-$USER/...`. Helper: `kinuv.scratch.kinuv_scratch_root`. Tests set `TMPDIR` and `JAX_COMPILATION_CACHE_DIR` there **before** jax import (`tests/conftest.py`).

## Do

- `export TMPDIR=$root/tmp` (and `TEMP`, `TMP`) in long workers.
- JAX: `JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`, compile cache under that root. Cache is disposable; a preempted node recompiles. Do not rsync the JAX cache onto NFS.
- Worker stdout: tee to `$root/worker.log` **and** the project run dir on `/arc` (line-buffered). Structured `logs/run.log` and `status.json` fsync to `/arc` so fail/OOM/success all leave logs.
- Timer sync (30 s heartbeat) and per-chain checkpoints: small JSON + `checkpoints/chain_N.npz` (draws, not vis) onto the project run dir. Do not `git add` on a timer.
- Checkpoints are kB-MB: parameter vectors, RNG, chain metadata. Native 066 vis (43240 x 1920 complex128 ~ 1.3 GB) is **not** a per-eval checkpoint, including on `/scratch` and `/dev/shm`.

## Do not

- Loop unbuffered vis or 881x95 cubes over `/arc/projects` or `/arc/home`.
- Rsync the JAX cache onto NFS `/arc`.
- Set `TMPDIR=/arc/home/...` or write run products under `$HOME`.
- Treat `scripts/plot_fit_diagnostics.py`'s preview dir `docs/reviews/artifacts/fit-diagnostics/` as filesystem `/scratch`.
- Use `/dev/shm` for native vis (RAM OOM).
- Rely on `canfar logs` (expire ~1 hour; vanish on 404).

Composer 2.5 (`composer-2.5-fast`) may edit only `## Agent Run Status` in `docs/architecture/STATUS.md` (up to `# Architecture mailbox`). It must not touch YAML `board` / `next_role` / reviews, skip hooks, amend, or force-push. Parent writes physics mailbox lines.
