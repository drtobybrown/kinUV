# CANFAR GPU headless jobs (agent ops)

How to **submit and schedule** GPU headless sessions on CANFAR for kinUV. Policy: [`DEC-067-RUNNER`](../decisions/DEC-067-RUNNER.md). Scratch/checkpoints: [`scratch.md`](scratch.md).

**Status (2026-09-02):** fixed-resource GPU jobs **schedule successfully** on H100 MIG slices. Production NUTS is **not** GPU-ready yet: `kinuv-venv-recovery` is CPU jax 0.11.1; image Python on `astroml-cuda` hit cuBLAS/MIG errors. Use this doc to **probe and submit**; use a separate GPU-smoke propose before production NUTS on CUDA.

## Rules

| Rule | Detail |
|---|---|
| GPU count | **`--gpu` is integer only.** `--gpu 0.1` is rejected by the CLI. |
| No flexible on GPU | Always pin **`--cpu`** and **`--memory`** when requesting a GPU. Omitting both defaults to flexible CPU/RAM (DEC-067 CPU path). |
| JAX env | `launch_headless.py` / `headless_job_env()` set `JAX_PLATFORMS=cuda` when `--gpu > 0`, else `cpu`. |
| Venv | Default entrypoint uses **`/arc/home/thbrown/kinuv-venv-recovery`** (CPU jax-finufft). GPU NUTS needs a **different venv** once built. |
| Production gate | Do not label `sampler: nuts` on GPU until autodiff chi2 identity holds: `\|chi2 − 168675.6\| < 1` at the official MAP (G1 gate). |
| Durable I/O | Run dir: `/arc/projects/KILOGAS/analysis/toby_sandbox/kinuv_runs/{run_id}/`. Never `$HOME`. |

## Prerequisites

- `canfar` CLI on PATH (`~/.local/bin/canfar`, v1.4.x tested 2026-09-02).
- Valid CADC cert: `~/.ssl/cadcproxy.pem` (`cadc-get-cert` if near expiry).
- Repo on CANFAR: `/arc/projects/KILOGAS/analysis/toby_sandbox/kinUV`.

## Submit: raw `canfar create` (probe or custom script)

Minimal pattern — **fixed 1 GPU, 2 CPU, 8 GB RAM**:

```bash
RUN_ID="KGAS066-$(date -u +%Y%m%dT%H%M%SZ)-gpu-probe-test"
RUNS="/arc/projects/KILOGAS/analysis/toby_sandbox/kinuv_runs"
PROBE="/arc/projects/KILOGAS/analysis/toby_sandbox/kinUV/scripts/gpu_probe_image_python.sh"

mkdir -p "${RUNS}/${RUN_ID}/logs"

canfar create headless skaha/astroml-cuda:latest \
  --name kinuv-gpu-probe-test \
  --gpu 1 \
  --cpu 2 \
  --memory 8 \
  --env "KINUV_RUN_ID=${RUN_ID}" \
  --env "KINUV_RUNS=${RUNS}" \
  --env "KINUV_GPU_PROBE_IMAGE=skaha/astroml-cuda:latest" \
  --env "JAX_PLATFORMS=cuda" \
  --env "JAX_ENABLE_X64=1" \
  --env "PYTHONUNBUFFERED=1" \
  -- /bin/bash "${PROBE}"
```

Dry-run (parse only, no session):

```bash
canfar create headless skaha/astroml-cuda:latest \
  --name kinuv-gpu-dry \
  --gpu 1 --cpu 2 --memory 8 \
  --dry-run
```

Monitor:

```bash
canfar info SESSION_ID
canfar logs SESSION_ID
cat "${RUNS}/${RUN_ID}/gpu_probe.json"
```

Delete when done:

```bash
canfar delete -f SESSION_ID
```

## Submit: `launch_headless.py` (kinUV runner)

When a GPU-capable venv exists, the repo launcher passes `--gpu`, `--cpu`, and `--memory` into `submit_headless()` and sets `JAX_PLATFORMS=cuda`:

```bash
cd /arc/projects/KILOGAS/analysis/toby_sandbox/kinUV

python scripts/launch_headless.py \
  --galaxy KGAS066 \
  --kind nuts \
  --gpu 1 \
  --cpu 4 \
  --memory 16 \
  --image skaha/astroml-cuda:latest
```

Notes:

- **`--cpu 0` and `--memory 0` mean flexible** — do not use with `--gpu`.
- **`--gpu 0` omits `--gpu`** (CPU job); manifest records `"gpu": null`, `JAX_PLATFORMS=cpu`.
- Implementation: `src/kinuv/runner/canfar.py` (`submit_headless`, `headless_job_env`).

## Images (registry)

| Image | Headless | Notes |
|---|---|---|
| `skaha/astroml:latest` | yes | Production CPU NUTS today; recovery venv. |
| `skaha/astroml-cuda:latest` | yes | CUDA stack in image; JAX 0.9.x in conda; no `jax_finufft` in image Python (2026-09-02). |
| `skaha/astroflow-cuda:latest` | yes | Alternative CUDA notebook/headless; not probed for kinUV yet. |

List: `canfar image ls | rg -i 'astroml|cuda'`

## Probe scripts

| Script | Purpose |
|---|---|
| [`scripts/gpu_probe_image_python.sh`](../../scripts/gpu_probe_image_python.sh) | Container default Python; writes `{run_dir}/gpu_probe.json`. **Use for GPU scheduling smoke.** |
| [`scripts/gpu_probe_canfar.sh`](../../scripts/gpu_probe_canfar.sh) | Activates `kinuv-venv-recovery`; checks whether that venv sees CUDA (today: no). |

Success criteria for a **scheduling** probe:

- `canfar create` returns `(ID: …)`.
- `canfar info` shows `Status Running` then `Completed`.
- `gpu_probe.json` has `nvidia_smi_L.rc == 0`.

Success criteria for a **production** GPU venv (not yet met):

- `has_cuda_device: true` in probe JSON.
- `jax_finufft: true`.
- Optional: tiny `predict_binned` + `chi2` identity vs 168675.6.

## Verified 2026-09-02 (scheduling)

All used **`--gpu 1 --cpu 2 --memory 8`** (fixed, not flexible).

| Session | Image | Schedule | GPU (nvidia-smi) | JAX CUDA |
|---|---|---|---|---|
| `okzj0cod` | `skaha/astroml:latest` | yes | H100 NVL MIG 1g.12gb | no (recovery venv CPU jax 0.11.1) |
| `lyddomx5` | `skaha/astroml-cuda:latest` | yes | H100 NVL MIG 1g.12gb | no (recovery venv CPU jax 0.11.1) |
| `dnm4sey1` | `skaha/astroml-cuda:latest` | yes | H100 NVL MIG 1g.12gb | no (image jax 0.9; cuBLAS load error on MIG) |

Example probe artifact: `kinuv_runs/KGAS066-20260902T130208Z-gpu-probe-cuda-imagepy/gpu_probe.json`.

Platform load: `canfar stats` (2026-09-02: ~1292/3000 CPU, ~11.7/15 TB RAM in use).

## Blockers before GPU NUTS

1. **Recovery venv** — CPU-only JAX; `JAX_PLATFORMS=cuda` raises `Backend 'cuda' is not in the list of known backends`.
2. **Image JAX** — `astroml-cuda` conda Python is jax **0.9.0.1**, not kinUV's pinned **0.11.1**; missing `jax_finufft`.
3. **MIG / cuBLAS** — image Python on MIG slice: `Unable to load cuBLAS. Is it installed?` May need full GPU quota or platform fix.
4. **Board** — production GPU NUTS requires a propose + dual review (chi2 identity, mixing gates unchanged).

## Do not

- Request fractional GPU (`--gpu 0.1`) — unsupported.
- Use flexible CPU/RAM on GPU jobs (always pin `--cpu` and `--memory`).
- Point production NUTS at GPU until chi2 identity passes on CUDA.
- Use `--gpu` when the worker venv cannot see CUDA (DEC-067).
- Block interactive agents on the sampling loop; GPU NUTS is still headless + watcher.

## Related

- CPU headless: `scripts/launch_headless.py`, `scripts/canfar_entrypoint.sh`, `scripts/watch_headless.py`
- Tests: `tests/test_canfar_runner.py` (`JAX_PLATFORMS=cpu` for `nuts-pa25`)
