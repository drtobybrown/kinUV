---
role: proposer
date: 2026-09-02
agent: parent
canon_generation: 4
ids:
  - DEC-067-RUNNER
  - DEC-066-INFER
  - DEC-066-TARGET
  - DEC-066-ZEROMODEL
verdict: propose
---

# GPU NUTS smoke: CUDA venv + 4 parallel chains (10× wall)

## Scope

User 2026-09-02: production stack GPU-ready on CANFAR; **10× wall-clock** of a full 066 NUTS product vs serial CPU (~4.84 h → ~29 min). Metric is `max(chain wall) + merge` vs receding serial **17440 s**, not eval/s alone. Path: GPU kernel × four 1-chain GPU sessions.

Existing ids only. No new `DEC-*`. Official MAP `kinuv-KGAS066-uvsign-map` read-only. `DEC-066-TARGET` still 066. Do not interrupt `xgepg7qy`. Do not steal `KGAS066-latest`. Do not write `docs/reviews/artifacts/2026-08-30-g3-nuts/`. Do not mutate `/arc/home/thbrown/kinuv-venv-recovery`. No G4/G5. No KGAS007. No logit `[0.5, 15]`. Do not quote inner `dV/dr` or S2 16/50/84.

CPU production NUTS stays the default until CUDA identity **and** mixing pass. Short of 10× is a STATUS number, not a license to skip identity.

## Architect verdict (selected path)

**Stage 1 — dedicated CUDA venv.** Path: `/arc/projects/KILOGAS/analysis/toby_sandbox/venvs/kinuv-cuda`. Build on a pinned GPU headless session. Image try order: `skaha/astroml-cuda:latest`, then `skaha/astroflow-cuda:latest`. Pins: jax/jaxlib **0.11.1** CUDA12, jax-finufft **1.3.1**, numpyro **0.21.0 `--no-deps`**. If no CUDA jax-finufft wheel, build jax-finufft from source in that session; do not silently upgrade jax. Gate: `jax.devices()` has CUDA; `BACKEND == "jax-finufft"`; nvidia-smi OK.

**Stage 2 — identity + timing, fail-closed.** Same 881×95, `s=0.5136098555284736`, `NPZ_UV_SIGN=-1`. Official MAP `|chi2 − 168675.6| < 1` on CUDA. Finite `jax.grad(U)` on six sampled names. Artifacts: `docs/reviews/artifacts/2026-09-02-gpu-nuts-smoke/timing.json`. Do not claim 10× from eval/s.

**Stage 3 — four parallel GPU chains.** Kind `nuts-gpu`. Each session: 1 chain, 200 warmup + 600 draws, `--gpu 1 --cpu 4 --memory 16` (tune after OOM), `JAX_PLATFORMS=cuda`, CUDA venv via `KINUV_VENV` on `--env`. `KINUV_CHAIN_ID` 1–4; distinct `rng_seed`. Run dirs `{KGASID}-{ts}-nuts-gpu-c{N}`. Skip `point_latest`. Artifact dest `docs/reviews/artifacts/2026-09-02-gpu-nuts-smoke/` (not G3). Merge `(4, 600, 8)` then `mixing_ok(R_hat≤1.01, ESS>400)` on six names. Fail-to-mix → `COMPLETED_UNMIXED`, not `sampler: nuts`. Report run wall and submit-to-done separately (queue is not kernel time).

**Runner locks.** `launch_headless.py --gpu` without `--cpu` and `--memory` refuses. `--gpu` without CUDA devices in the live venv refuses (DEC-067). `headless_job_env` sets `KINUV_VENV` when gpu. `steal_latest("nuts-gpu")` is False. `artifact_dir_for_kind("nuts-gpu")` is not `ARTIFACT_G3`. Worker `--n-chain` / `KINUV_CHAIN_ID` runs one chain and exits; merge is a separate host script.

**Reject this wave:** mutate recovery venv; flexible GPU; `--gpu 0.1`; poll `xgepg7qy`; G4; 007; MAP rewrite; retarget latest; write G3 receding folder.

## What changed / what was checked

- Receding CPU serial: chain elapsed 4269 / 4446 / 4624 / 4102 s; sum ≈ 17440 s (`kinuv_runs/KGAS066-20260831T194009Z-nuts/logs/chain_*.json`). Mixing pass.
- G1 CPU: 3.01 eval/s, chi2 168675.6 (`docs/reviews/artifacts/2026-08-30-g1-jax/timing.json`).
- GPU schedule OK, compute not: `docs/diagnostics/canfar-gpu.md`. Probes `okzj0cod` / `lyddomx5` / `dnm4sey1`. Recovery jax backends `cpu,tpu`. Image jax 0.9 cuBLAS/MIG fail.
- Entrypoint `KINUV_VENV` defaults to recovery. `headless_job_env` does not pass `KINUV_VENV`. `submit_headless` already takes `--gpu` as int.
- Worker always loops `N_CHAINS = 4` sequential. No chain-id.
- `steal_latest` is False only for `pa25` in kind name. A kind `nuts-gpu` would **steal latest** unless execute changes `kind.py`.

## Rejected alternatives

- “Just `--gpu 1` on the recovery venv” — JAX has no CUDA backend.
- “Use image `/opt/conda` jax 0.9” — wrong jax pin; no jax-finufft; cuBLAS fail.
- “10× from eval/s on one chain” — user metric is full-product wall; serial 4 chains cap speedup at ~4× even with infinite GPU.
- “Wait for `xgepg7qy`” — DEC-067; this card does not block on approaching mixing.
- “Upgrade jax past 0.11.1 to get a CUDA wheel” — identity `|chi2−168675.6|<1` and G3 autodiff pins 0.11.1.

## Residual risks

1. No CUDA jax-finufft wheel for jax 0.11.1 → source build or no GPU NUTS. STATUS one-liner; keep CPU production.
2. MIG 1g.12gb (~12 GB) + x64 NUFFT OOM or weak speedup. Retry `--cpu 4 --memory 16` then STATUS; do not invent full-GPU quota.
3. Image cuBLAS vs pip jax CUDA mismatch (`dnm4sey1`).
4. Four GPU slots queue. Report **run wall** (`max` chain elapsed) and **submit-to-done** separately. 10× gate uses run wall vs 17440 s.
5. Interactive agents must not block on sampling (DEC-067).
6. `xgepg7qy` still Running; do not rewrite Agent Run Status to hide it.
7. GPU NUTS mean leftover still structured; 16/50/84 still uncalibrated. Speed smoke is not a new science product.

## Execute if accepted

1. Kind `nuts-gpu` in `src/kinuv/runner/kind.py`: `steal_latest` False; artifact dir `docs/reviews/artifacts/2026-09-02-gpu-nuts-smoke` (not G3, not leftover `pa25`). Unit test: approaching and gpu kinds do not write G3; gpu does not call `point_latest`.
2. `submit_headless` / `launch_headless.py`: `--gpu` requires `--cpu` and `--memory` else exit 2. `headless_job_env` sets `KINUV_VENV` (default CUDA path when gpu else recovery) and `KINUV_CHAIN_ID` if given. Entrypoint sources `"${KINUV_VENV}"`. Tests: gpu env has `JAX_PLATFORMS=cuda` and `KINUV_VENV` containing `kinuv-cuda`; cpu env stays recovery + `cpu`.
3. Worker: `--chain-id` / `KINUV_CHAIN_ID` runs **one** chain (`num_chains=1`), seed `11 + chain_id`, writes `checkpoints/chain_{id}.npz`, does **not** merge. Without chain-id, existing 4-loop CPU behaviour unchanged. Merge script `scripts/merge_nuts_chains.py` stacks four npz → mixing + product JSON into gpu-nuts-smoke; mixing fail → `COMPLETED_UNMIXED`.
4. Dispatch a **venv-build** pinned GPU job (`--gpu 1 --cpu 4 --memory 16`, image astroml-cuda first) that creates `/arc/projects/KILOGAS/analysis/toby_sandbox/venvs/kinuv-cuda`. Do not pip into recovery. If jax-finufft CUDA missing, source-build in that session. Probe JSON: CUDA device + `BACKEND`. If fail: astroflow-cuda then STATUS stop; no 066 GPU NUTS.
5. Same session or follow-up: G1 identity `|chi2−168675.6|<1`; six-axis `jax.grad(U)` finite; `timing.json` eval/s and `t_grad_s` vs 3.01 / 0.43. Identity fail → no Stage 3.
6. After identity: four `launch_headless.py --kind nuts-gpu --gpu 1 --cpu 4 --memory 16 --chain-id {1..4}` (or env). Do not wait interactively. When all four SUCCEEDED, merge. Wall vs 17440 s. Do not retarget latest. Do not write G3.
7. Patch `docs/diagnostics/canfar-gpu.md` with venv path, image, identity, eval/s, 4-chain command, wall table. Field Guide one line. CHANGELOG. Mailbox: GPU smoke landed or blocked; CPU NUTS default until identity+mixing. Official MAP unchanged. Commit/push after runner patch, after venv+identity, after merge.

## STATUS updates required

- `next_role: board`
- `board: open`
- `last_propose:` this file
- Do not clear Agent Run Status for `xgepg7qy`
- Do not start G4
