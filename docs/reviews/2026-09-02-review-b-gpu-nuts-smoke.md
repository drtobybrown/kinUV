---
role: reviewer
seat: b
date: 2026-09-02
agent: review-b
canon_generation: 4
ids:
  - DEC-067-RUNNER
  - DEC-066-INFER
  - DEC-066-TARGET
  - DEC-066-ZEROMODEL
verdict: accept
severity: major
propose: docs/reviews/2026-09-02-propose-gpu-nuts-smoke.md
---

# Review b: GPU NUTS smoke (CUDA venv + 4 parallel chains)

Do not read the other seat's review file. Do not implement.

Scope check: dedicated CUDA venv on the project volume, G1 identity on CUDA, then four 1-chain GPU sessions merged to a 066 product, 10× vs serial receding CPU, official MAP read-only, TARGET 066, no G4, do not interrupt `xgepg7qy`, do not mutate recovery. Existing ids only. That path is accept-eligible. Execute as typed still has holes that void the 10× claim, clobber the receding G3 folder, hide the approaching job, and can pip a newer jax. Rubber-stamp is invalid.

On disk (HEAD `8e23b60`): `steal_latest` is False only for `pa25`; `artifact_dir_for_kind("nuts-gpu")` returns `ARTIFACT_G3`; `write_nuts_product_plots` defaults `artifact_dir=ARTIFACT_G3`; `write_job_status_md` non-`pa25` Next Step is `2026-08-30-g3-nuts/`; worker loops `N_CHAINS = 4` and does not read `KINUV_CHAIN_ID`; `headless_job_env` does not set `KINUV_VENV`; entrypoint defaults `/arc/home/thbrown/kinuv-venv-recovery`; `launch_headless.py --gpu` without `--cpu`/`--memory` is flexible; `session_name` is `kinuv-KGAS066-{sha6}-{kind}` with no chain id; `scripts/merge_nuts_chains.py` does not exist; `toby_sandbox/venvs/` does not exist. Serial chain walls `kinuv_runs/KGAS066-20260831T194009Z-nuts/logs/chain_{1..4}.json`: 4268.698 + 4445.501 + 4624.001 + 4101.832 = **17440.032 s**. G1 `timing.json`: `eval_per_s=3.0119`, `chi2=168675.596`, `BACKEND=jax-finufft`, `JAX_PLATFORMS=cpu`. `pyproject.toml` extras are `jax>=0.4.30`, `jax-finufft>=1.3` (no 0.11.1 ceiling). `product_record` sets `sampler: nuts` only if `autodiff_ok and mixing_pass`, else `laplace_mh`.

## Attacks / bounds

1. **10× gate as typed can count queue, or four GPU jobs that still loop `N_CHAINS=4`.** User metric is full-product **run** wall vs 17440 s, not eval/s, not Pending/Queued. Propose residual 4 already splits run wall vs submit-to-done; execute item 6 only says “Wall vs 17440 s.” Live worker always `for c in range(N_CHAINS)` with `N_CHAINS=4`; argparse has no `--chain-id`; entrypoint does not forward `KINUV_CHAIN_ID`; runner-lock line says `--n-chain` (count) while execute item 3 says `--chain-id` (identity). Four sessions each running four sequential chains make wall ≈ `4 × t_gpu_chain`. Parallelism then buys nothing: 10× vs 17440 requires kernel ≥10×. Correct 4×1-chain wall is `max(chain elapsed)+merge`; 10× then needs kernel ≳ 4624/1744 ≈ **2.65×**. MIG 1g.12gb can sit in that gap. Merging `chain_1.npz` from four jobs that each wrote `chain_1..4` with `rng_seed=11+c` stacks four copies of the same chain.

   **Bound:** 10× uses `T_run = max_i(elapsed_s of the four GPU `chain_{id}.json`) + merge_script wall`. Denominator **17440.032 s**. Pass iff `T_run ≤ 1744.003 s`. Submit-to-done and queue are STATUS columns only; they must not enter the ratio. Short of 10× is a STATUS number, not a license to skip identity (propose already says this). Worker with `KINUV_CHAIN_ID`/`--chain-id` in `{1,2,3,4}` runs **one** chain (`num_chains=1`), seed `11 + (chain_id-1)` (CPU receding used 11,12,13,14), writes `checkpoints/chain_{id}.npz` and `logs/chain_{id}.json`, does **not** loop 2–4, does **not** stack, does **not** call `write_nuts_product_plots` / `product_record` / `write_job_status_md`. Without chain-id, existing 4-loop CPU path unchanged. Flag name is chain **identity**, not `--n-chain` count. Unit test: chain-id=2 writes only `chain_2.npz` and does not import-call merge. If any GPU `chain_*.json` shows four elapsed entries or job `elapsed_s` ≈ 4× peers, 10× is void (mis-launch).

2. **`write_nuts_product_plots` default dest is still G3; 1-chain workers and an unscoped merge will copy into `2026-08-30-g3-nuts/`.** `src/kinuv/runner/plots.py` `artifact_dir=ARTIFACT_G3`. Worker uses `KINUV_ARTIFACT_DIR` else `artifact_dir_for_kind(kind)`; live `artifact_dir_for_kind` returns G3 for anything without `pa25`. `steal_latest("nuts-gpu")` is **True** today (`kind.py` only special-cases approaching). Four GPU jobs would retarget `KGAS066-latest` and race-write receding corners. Merge script does not exist; a defaulted `write_nuts_product_plots(rec, dest)` clobbers G3 even after kind.py is patched.

   **Bound:** `steal_latest("nuts-gpu")` is False; `artifact_dir_for_kind("nuts-gpu")` is `docs/reviews/artifacts/2026-09-02-gpu-nuts-smoke` (not G3, not leftover `pa25`). `point_latest` is not called. Merge is the only writer of product JSON/PNGs and **must pass** `artifact_dir=` explicitly. Do not rely on the function default. Unit test: `nuts-gpu` and `nuts-pa25` do not write G3; `nuts` still may; gpu does not call `point_latest`; `write_nuts_product_plots(..., artifact_dir=gpu_dest)` leaves a G3 sentinel untouched.

3. **STATUS patcher will hide `xgepg7qy` and point Next Step at G3 as soon as the first GPU chain exits.** `write_job_status_md` non-approaching branch: Phase `G3 066 NUTS {state}`, Next Step `Copy posteriors into docs/reviews/artifacts/2026-08-30-g3-nuts/`. Worker and watcher both call it. `patch_agent_run_status` correctly stops at `# Architecture mailbox`, but it **replaces** Phase / Next Step. Four 1-chain jobs plus merge would stamp G3 four extra times. Approaching `xgepg7qy` is still Running. Propose residual 6 names the hide; execute item 7 does not name a `nuts-gpu` branch or a “chain worker must not patch STATUS” rule.

   **Bound:** `nuts-gpu` 1-chain workers/watchers must not call `write_job_status_md` (or must no-op the mailbox patch). Merge/parent may patch Agent Run Status only after merge, must keep `xgepg7qy` Running in Phase/Next Step until that session is terminal, must not name `2026-08-30-g3-nuts` as Next Step, dest is `2026-09-02-gpu-nuts-smoke/`. Unit test: a `nuts-gpu` note does not contain `2026-08-30-g3-nuts` and does not drop the approaching session id if still Running. Do not `canfar delete` / poll `xgepg7qy`. Do not GPU `nuts-pa25`.

4. **jax-finufft CUDA source build can pull a newer jax; submit-host CUDA refuse can block the card; recovery pip is a one-way hole.** Stage 1 pins jax/jaxlib **0.11.1** CUDA12 + jax-finufft **1.3.1** + numpyro **0.21.0 `--no-deps`**, “do not silently upgrade jax.” Execute item 4 does not assert versions after install. `pyproject.toml` `nufft = ["jax>=0.4.30", "jax-finufft>=1.3"]` has no ceiling; `pip install jax-finufft` / source `pip install .` can move jax. G3 already required `jax.__version__ == "0.11.1"` and `BACKEND == "jax-finufft"` after numpyro. Runner lock “`--gpu` without CUDA devices in the **live** venv refuses” is lethal if “live” is the submit-host recovery interpreter (`jax.devices()` has no CUDA there by design). Entrypoint and `gpu_probe_canfar.sh` default `KINUV_VENV` to `/arc/home/thbrown/kinuv-venv-recovery`. `toby_sandbox/venvs/` does not exist yet.

   **Bound:** venv path is `/arc/projects/KILOGAS/analysis/toby_sandbox/venvs/kinuv-cuda` (project volume, not `$HOME`, not `/scratch`, not recovery). Create the prefix empty; never `pip` after `source` recovery. After every install/source-build, probe JSON must record `jax.__version__ == "0.11.1"`, `BACKEND == "jax-finufft"`, `has_cuda_device: true`. If jax moved or BACKEND is not jax-finufft: STATUS stop, keep CPU production, do not launch Stage 3. `--gpu` refuse is path-based: `KINUV_VENV` must be the cuda prefix, not recovery; do **not** call `jax.devices()` on the submit-host recovery python. CUDA device check is the GPU-session probe. Stage 3 `--image` is the same CUDA image that built the venv (`skaha/astroml-cuda:latest` or the astroflow fallback), not `DEFAULT_IMAGE` `skaha/astroml:latest`. `session_name` must include `c{N}` so four parallel creates do not share `kinuv-KGAS066-{sha6}-nuts-gpu`. `--run-id` must be `{KGASID}-{ts}-nuts-gpu-c{N}` (same-second `make_run_id` collision).

5. **Mixing fail must not become `sampler: nuts`; CPU stays default until identity AND mixing; this smoke is receding PA 199.73 only.** `product_record` already labels `nuts` only if autodiff **and** mixing; else `laplace_mh`. Worker maps mix fail to `COMPLETED_UNMIXED`. Merge must use that same function with `mixing_ok(..., rhat_max=1.01, ess_min=400.0, ess_tail_min=400.0)` on six names (live default `ess_min=200` is the tiny-mock bar, not DEC-067). 1-chain R_hat is not a mixing result. `launch_headless.py --gpu` default stays 0; cpu `headless_job_env` stays recovery + `JAX_PLATFORMS=cpu`. `pa_init_deg("nuts-gpu")` is official MAP **199.7298** (receding). Do not pass `--pa-init 25.2`. Do not start approaching on GPU.

   **Bound:** merge fail → state `COMPLETED_UNMIXED`, `sampler != "nuts"` (keep `laplace_mh` / unset). Do not retarget `KGAS066-latest`. Do not quote inner `dV/dr` or S2 16/50/84. Identity `|chi2 − 168675.6| < 1` at `s=0.5136098555284736` on CUDA is required before Stage 3; identity fail → no four-chain launch. CPU production NUTS remains the default until **both** identity and mixing pass.

## Comments

1. **major.** Lock 10× to `max(chain elapsed)+merge` vs 17440.032 s (≤1744.003 s). Queue is not in the ratio. Attack 1.
2. **major.** Each GPU job runs exactly one chain. `--chain-id` is identity 1–4, not `--n-chain` count. Seed `11+(chain_id-1)`. No in-job merge/plots/STATUS. Attack 1.
3. **major.** `nuts-gpu` must not steal latest or write G3. `write_nuts_product_plots` needs an explicit dest; default stays G3. Attack 2.
4. **major.** GPU chain workers/watchers must not hide `xgepg7qy` or point Next Step at `2026-08-30-g3-nuts/`. Attack 3.
5. **major.** After venv install, `jax.__version__==0.11.1` and `BACKEND==jax-finufft`. Do not pip into recovery. Durable path `.../toby_sandbox/venvs/kinuv-cuda`. No submit-host `jax.devices()` gate. Attack 4.
6. **major.** Mix fail → `COMPLETED_UNMIXED`, not `sampler: nuts`. CPU default until identity **and** mixing. Receding MAP PA 199.73 only; no GPU PA 25.2. Attack 5.
7. **minor.** `--gpu` without `--cpu` and `--memory` exits 2 (named in execute; still unenforced on disk). Stage 3 must pass `--image` of the CUDA builder image.
8. **minor.** Four parallel `--name` values must differ (`-c{N}`). `make_run_id` same-second collision needs `--run-id` or a chain suffix.

## Residual risks

1. No CUDA jax-finufft wheel for jax 0.11.1 → source build or no GPU NUTS. If source-build moves jax, stop. Keep CPU production.
2. MIG 1g.12gb (~12 GB) + x64 NUFFT OOM or kernel ≲2.65× (10× fail even with 4×1). Retry `--cpu 4 --memory 16` then STATUS; do not invent full-GPU quota.
3. Image cuBLAS vs pip jax CUDA mismatch (`dnm4sey1`).
4. Four GPU slots queue. Report run wall and submit-to-done separately.
5. Interactive agents must not block on sampling (DEC-067).
6. `xgepg7qy` still Running; do not rewrite Agent Run Status to hide it.
7. GPU NUTS mean leftover still structured; 16/50/84 still uncalibrated. Speed smoke is not a new science product.
8. NFS venv on `/arc` can be slow/fragile; JAX cache stays `/scratch`. Do not rsync the cache onto `/arc`.

## STATUS updates required

- `verdict` and `severity` as in the header (`accept`, `major`)
- `last_review_b`: this file
- Do not set `board: accepted` (parent tallies)
- Do not clear Agent Run Status for `xgepg7qy`
- Do not start G4
