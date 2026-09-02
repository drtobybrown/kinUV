# CANFAR CPU-parallel headless NUTS (production)

Canonical 066 NUTS path under `DEC-067-RUNNER`. GPU acceleration was benchmarked and rejected (see `docs/architecture/notes/2026-09-02-gpu-rejection-cpu-parallel.md`).

## Production engine

| Mode | When | Command pattern |
|---|---|---|
| **Serial 4-chain** | Default receding product | `launch_headless.py --kind nuts` (flexible CPU/RAM) |
| **Parallel 4×1-chain** | Wall-clock speedup | Four jobs with `--chain-id {1..4}`, then `merge_nuts_chains.py` |
| **Approaching PA 25.2** | Conjugate-mode science | `--kind nuts-pa25 --pa-init 25.2` (does not steal `KGAS066-latest`) |

Image: `skaha/astroml:latest`. Venv: `/arc/home/thbrown/kinuv-venv-recovery` (CPU jax 0.11.1 + jax-finufft). `JAX_PLATFORMS=cpu`.

Durable run root: `/arc/projects/KILOGAS/analysis/toby_sandbox/kinuv_runs/{run_id}/`. Never `$HOME`. JAX cache on `/scratch/kinuv-$USER` (see `scratch.md`).

## Submit: serial receding (landed `sd3ckpf2`)

```bash
cd /arc/projects/KILOGAS/analysis/toby_sandbox/kinUV
python scripts/launch_headless.py --galaxy KGAS066 --kind nuts
```

Flexible CPU/RAM (`--cpu 0 --memory 0`). Worker loops four chains sequentially in one session.

## Submit: parallel 4×1-chain CPU

```bash
for c in 1 2 3 4; do
  python scripts/launch_headless.py --kind nuts --chain-id "$c" --no-watch
done
```

Each session runs one chain (200 warmup + 600 draws), writes `checkpoints/chain_{id}.npz`, exits with `sampler: pending_merge`. Host merge:

```bash
python scripts/merge_nuts_chains.py \
  KGAS066-{ts}-nuts-c1 KGAS066-{ts}-nuts-c2 \
  KGAS066-{ts}-nuts-c3 KGAS066-{ts}-nuts-c4
```

Mixing gate: `R_hat ≤ 1.01`, `ESS > 400` on six sampled names. Fail → `COMPLETED_UNMIXED`, not `sampler: nuts`.

## Monitor

```bash
canfar ps
canfar info SESSION_ID
cat kinuv_runs/{run_id}/status.json
tail kinuv_runs/{run_id}/worker.log
```

## Do not

- Request GPU for production NUTS (empirically slower on 881×95 at MIG 1g.12gb).
- Interrupt approaching runs (`xgepg7qy` / `nuts-pa25`) for speed experiments.
- Write approaching products into `docs/reviews/artifacts/2026-08-30-g3-nuts/`.
- Block interactive agents on the sampling loop (DEC-067).

## Related

- Entrypoint: `scripts/canfar_entrypoint.sh`
- Watcher: `scripts/watch_headless.py`
- Runner: `src/kinuv/runner/canfar.py`
- Tests: `tests/test_canfar_runner.py`
