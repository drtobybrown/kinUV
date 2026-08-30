# CANFAR scratch and checkpoints (human)

High-frequency I/O (JIT cache, TMP, run logs, MCMC parameter snapshots) goes on **node-local** disk. This host's fastest local volume is `/scratch` (not Ceph `/arc`). If `/scratch` is missing (laptop/CI), use `/tmp`.

Root: `/scratch/kinuv-$USER/$SKAHA_SESSION_ID` (mode 0700), else `/tmp/kinuv-$USER/...`. Helper: `kinuv.scratch.kinuv_scratch_root`. Tests set `TMPDIR` and `JAX_COMPILATION_CACHE_DIR` there **before** jax import (`tests/conftest.py`).

## Do

- `export TMPDIR=$root/tmp` (and `TEMP`, `TMP`) in long workers.
- JAX: `JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`, compile cache under that root. Cache is disposable; a preempted node recompiles.
- Long runs: `python script.py > $root/run.log 2>&1 &`. Inspect `tail` or a small JSON, not a vis dump.
- Timer sync (60 s or every N evals): **small JSON only** (last theta, `chi2`, `nfev`, eval/s). Destination: git artifact dir or `/arc` project results. Do not `git add` on a timer.
- Checkpoints are kB-MB: parameter vectors, RNG, chain metadata. Native 066 vis (43240 x 1920 complex128 ~ 1.3 GB) is **not** a per-eval checkpoint, including on `/scratch` and `/dev/shm`.

## Do not

- Loop unbuffered vis or 881x95 cubes over `/arc/projects` or `/arc/home`.
- Rsync the JAX cache onto NFS `/arc`.
- Set `TMPDIR=/arc/home/...`.
- Treat `scripts/plot_fit_diagnostics.py`'s preview dir `docs/reviews/artifacts/fit-diagnostics/` as filesystem `/scratch`.
- Use `/dev/shm` for native vis (RAM OOM).

There is no NUTS chain to resume yet. This policy is for G3 later; this card does not start a sampler.

Composer 2.5 (`composer-2.5-fast`) may edit only `## Agent Run Status` in `docs/architecture/STATUS.md` (up to `# Architecture mailbox`). It must not touch YAML `board` / `next_role` / reviews, skip hooks, amend, or force-push. Parent writes physics mailbox lines.
