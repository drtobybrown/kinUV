---
role: reviewer
seat: a
date: 2026-09-02
agent: review-a
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

# Review a: GPU NUTS smoke (CUDA venv + 4 parallel chains)

Do not read the other seat's review file. Do not implement.

Scope check: dedicated CUDA venv (not recovery), G1 identity `|chi2 − 168675.6| < 1` on CUDA, then four 1-chain GPU sessions whose run-wall vs serial CPU **17440 s** is the 10× metric. Existing ids only. No new `DEC-*`. Official MAP `kinuv-KGAS066-uvsign-map` read-only. `DEC-066-TARGET` still 066. Do not start G4. Do not interrupt `xgepg7qy`. Do not mutate `kinuv-venv-recovery`. Do not steal `KGAS066-latest`. Do not write `docs/reviews/artifacts/2026-08-30-g3-nuts/`. CPU NUTS stays production until identity **and** mixing. Reject-this-wave list (recovery pip, flexible GPU, `--gpu 0.1`, poll approaching, 007, MAP rewrite, G4, logit `[0.5, 15]`) stays rejected. Selected path is accept-eligible. Execute as typed still launches flexible+GPU, retargets latest, runs **16 sequential chains on 4 GPUs**, copies into G3, and overwrites Agent Run Status for `xgepg7qy`.

## Attacks / bounds

1. **GPU kind today is receding: `steal_latest` True, artifact dest G3, STATUS Next Step G3.** Live `src/kinuv/runner/kind.py`: `is_approaching_kind` is `"pa25" in kind`; `steal_latest` is `not is_approaching_kind`; `artifact_dir_for_kind` returns `ARTIFACT_G3_REL` (`docs/reviews/artifacts/2026-08-30-g3-nuts`) unless pa25. `tests/test_canfar_runner.py` `test_pa25_env_does_not_gpu_or_clobber_g3` only asserts `steal_latest("nuts-pa25") is False` and `steal_latest("nuts") is True`. A kind `nuts-gpu` stays green and **steals `KGAS066-latest`**. `scripts/launch_headless.py` calls `point_latest` **before** the dry-run return. Worker G3 guard is `kind == KIND_PA25 and ARTIFACT_G3_REL in str(artifact_dir)` only — gpu writes G3. `write_job_status_md` non-pa25 branch hardcodes Phase `G3 066 NUTS` and Next Step copy into `2026-08-30-g3-nuts/`, then `pending: []`. Four gpu chain jobs plus a venv-build that reused the NUTS entrypoint would each patch Agent Run Status and hide `xgepg7qy` Running. DEC-067-RUNNER literally copies PNGs+JSON into that G3 folder and points `{KGASID}-latest` at the newest run. Propose leaves both (same as pa25). Field Guide allows a STATUS one-liner; execute as typed does not name `write_job_status_md` or the dry-run steal.

   **Bound:** `steal_latest("nuts-gpu") is False` including `--dry-run`. `artifact_dir_for_kind("nuts-gpu")` is `docs/reviews/artifacts/2026-09-02-gpu-nuts-smoke` (not `ARTIFACT_G3`, not leftover `pa25`). Worker refuses if kind contains `gpu` and `ARTIFACT_G3_REL` is in `KINUV_ARTIFACT_DIR`. `write_job_status_md` for gpu/venv-build does **not** name G3, does **not** clear approaching, and does **not** replace Agent Run Status bullets that name `xgepg7qy` until that session is no longer Running. STATUS one-liner for the two DEC-067 leaves (G3 dest; latest symlink). Unit test: monkeypatch `ARTIFACT_G3` sentinel; gpu kind does not create or modify it; `point_latest` is not called.

2. **`--gpu 1` with cpu/memory default 0 submits flexible+GPU; `submit_headless` will still do it if only the CLI grows a check.** Live `launch_headless.py`: `--cpu` / `--memory` default **0**; `cpu = int(args.cpu) if int(args.cpu) > 0 else None` (same for memory); manifest `"flexible": cpu is None and memory is None`. Live `submit_headless`: `if cpu:` / `if memory:` / `if gpu:` independently. `--gpu 1` omits `--cpu` and `--memory` (DEC-067 CPU flexible path). `--gpu 1 --cpu 4` with `--memory 0` pins CPU and leaves RAM flexible. User forbade flexible on GPU (`docs/diagnostics/canfar-gpu.md`; DEC-067 item 2). Propose execute item 2 says launch refuses; it does not say `submit_headless(gpu=…)` without both pins raises. Partial pin must not count as pinned.

   **Bound:** refuse (exit 2 / raise) unless `gpu` is None **or** (`cpu > 0` **and** `memory > 0`). Both `launch_headless.py` and `submit_headless`. `--cpu 0` / `--memory 0` remain flexible and are illegal with `--gpu`. Unit test: `--gpu 1` alone, `--gpu 1 --cpu 4`, and `--gpu 1 --memory 16` all fail closed; `--gpu 1 --cpu 4 --memory 16` is the only legal GPU argv. `--gpu 0.1` stays rejected (`type=int` + CLI).

3. **`KINUV_VENV` is not on `--env`; entrypoint defaults recovery; a submit-host CUDA check would block the card.** Live `headless_job_env` has no `KINUV_VENV`. Live `scripts/canfar_entrypoint.sh`: `VENV="${KINUV_VENV:-/arc/home/thbrown/kinuv-venv-recovery}"`. Live `JAX_PLATFORMS` is already `cuda` when `gpu` is set. GPU job without `--env KINUV_VENV` sources recovery under `JAX_PLATFORMS=cuda` (probe `lyddomx5`: no CUDA backend) and still stole latest at submit. Propose runner lock: "`--gpu` without CUDA devices in the live venv refuses (DEC-067)". The **submit-host** live venv is recovery (CPU jax 0.11.1). Checking `jax.devices()` there refuses every GPU dispatch. DEC-067 means: do not request a GPU that the **worker** venv cannot use.

   **Bound:** (a) `headless_job_env(..., gpu=1)` sets `KINUV_VENV` to `/arc/projects/KILOGAS/analysis/toby_sandbox/venvs/kinuv-cuda` on `--env` (same path as `KINUV_RUN_ID`). CPU `gpu is None` stays recovery + `JAX_PLATFORMS=cpu`. (b) GPU env containing `kinuv-venv-recovery` is refused. Do not pip into recovery. (c) Submit host must **not** require local CUDA. Worker, after `source "${KINUV_VENV}"`, aborts before NUTS unless `jax.__version__ == "0.11.1"`, `jax.devices()` has CUDA, and `BACKEND == "jax-finufft"`. Probe JSON records those three. Silent jax upgrade to get a CUDA wheel is a stop, not a workaround. (d) Tests: gpu env has `JAX_PLATFORMS=cuda` and `KINUV_VENV` containing `kinuv-cuda`; cpu env has recovery + `cpu` and no cuda platform.

4. **Worker always loops `N_CHAINS = 4`; execute item 3's "without chain-id, 4-loop unchanged" on the GPU path is 16 chains / 4× wall.** Live `scripts/run_kgas066_nuts_headless.py`: `N_CHAINS = 4`; `for c in range(N_CHAINS): run_nuts_z6(..., rng_seed=11 + c, num_chains=1)`; then `np.stack`, `mixing_ok`, `write_nuts_product_plots`, `write_job_status_md`. No `--chain-id`. Entrypoint last line is `python …/run_kgas066_nuts_headless.py --run-id "${RUN_ID}"` plus optional `--pa-init` only — it does not forward a chain id. `launch_headless.py` has no `--chain-id`; `session_name` is `kinuv-KGAS066-{sha6}-{kind}` so four `--kind nuts-gpu` jobs collide on one CANFAR name; `make_run_id` is `{KGASID}-{ts}-{kind}` without `-c{N}`. Receding serial wall is the **sum** 4269+4446+4624+4102 ≈ **17440 s**. Four GPU jobs each running that loop: **16 chains**, wall ≈ 4× one-chain GPU time, 10× vs 17440 s is impossible. Propose Stage 3 wants 1 chain / session. Item 3 keeps CPU 4-loop as the no-id default. GPU + missing env therefore silently serializes four chains per card. After one chain the worker still merges, mixing-checks a `(1, 600, 8)` stack (default `mixing_ok` ess_min is **200**, not DEC-067 400), and four jobs race-write the same artifact dir. `scripts/merge_nuts_chains.py` does not exist.

   **Bound:** (a) `KINUV_CHAIN_ID` / `--chain-id` on `--env` (manifest is not delivery). Worker runs **one** chain (`num_chains=1`), seed `11 + (chain_id - 1)` if ids are 1–4 (receding used 11,12,13,14; typed `11 + chain_id` is 12–15). Writes `checkpoints/chain_{id}.npz`. Does **not** stack, does **not** call `mixing_ok`, does **not** call `write_nuts_product_plots`, does **not** copy into the shared artifact dest, does **not** patch Agent Run Status. (b) **GPU kind or `JAX_PLATFORMS=cuda` without a chain id is exit 2**, not the CPU 4-loop. CPU without chain-id may keep `range(4)`. (c) Entrypoint may keep argv = `RUN_ID` only if the worker reads `KINUV_CHAIN_ID` from env; optional `--chain-id` is belt-and-suspenders. (d) `make_run_id` / `--run-id` is `{KGASID}-{ts}-nuts-gpu-c{N}`; `session_name` includes `-c{N}` (DEC-OPS-AUTH 63-char cap). (e) Host `scripts/merge_nuts_chains.py` requires four npz, stacks `(4, 600, 8)`, then `mixing_ok(mix, rhat_max=1.01, ess_min=400.0, ess_tail_min=400.0)` on six sampled names — **not** `mixing_ok` default 200, not G3 tiny-mock 200. Fail → `COMPLETED_UNMIXED`, not `sampler: nuts`. Artifact dest `2026-09-02-gpu-nuts-smoke/` only. Unit test: gpu env without `KINUV_CHAIN_ID` fails; with id=1 the worker source does not execute `for c in range(N_CHAINS)` / does not import a merge; merge helper passes ess 400 explicitly.

5. **Stage 3 argv as typed uses CPU `DEFAULT_IMAGE`, no identity file gate, and `--memory 16` is not MIG VRAM.** Live `DEFAULT_IMAGE = "skaha/astroml:latest"`; `FALLBACK_IMAGE = "skaha/base-notebook:latest"`; launch falls back when submit fails and image equals default. Execute item 6: `launch_headless.py --kind nuts-gpu --gpu 1 --cpu 4 --memory 16 --chain-id {1..4}` — **no `--image`**. That is the CPU production image (`okzj0cod`). Execute item 4's venv-build, if it calls `launch_headless.py`, runs `canfar_entrypoint.sh` → NUTS worker (steal latest, 4-loop, STATUS). Stage 2 identity is a separate job; nothing stops Stage 3 before `timing.json` exists. Host `--memory 16` is RAM; probes ran on **H100 MIG 1g.12gb (~12 GB VRAM)**. Residual 2 "retry `--cpu 4 --memory 16` then STATUS" does not change device memory. `dnm4sey1` already hit image cuBLAS on MIG.

   **Bound:** (a) `--gpu` / kind `nuts-gpu` defaults image to `skaha/astroml-cuda:latest` (then `skaha/astroflow-cuda:latest`); **no** fallback to `skaha/astroml:latest` or `base-notebook`. (b) Venv-build is a probe/build script, **not** `launch_headless.py` / `run_kgas066_nuts_headless.py`. Creates `/arc/projects/KILOGAS/analysis/toby_sandbox/venvs/kinuv-cuda` only. (c) Stage 3 dispatch only after `docs/reviews/artifacts/2026-09-02-gpu-nuts-smoke/timing.json` records CUDA `|chi2 − 168675.6| < 1` on 881×95, `s=0.5136098555284736`, `NPZ_UV_SIGN=-1`, `jax==0.11.1`, finite six-axis `jax.grad(U)`. Identity fail → no 066 GPU NUTS; CPU production unchanged. Do not claim 10× from eval/s. (d) Device OOM on 1g.12gb is a STATUS stop or a full-GPU quota request; do not treat host `--memory` as VRAM. Run wall = `max(chain elapsed_s) + merge` vs **17440 s**; submit-to-done is a separate number (queue is not kernel time). Short of 10× is a STATUS number, not a license to skip identity or mixing.

## Comments

1. `major` -- `nuts-gpu` must not steal `KGAS066-latest` (including dry-run), must not write `2026-08-30-g3-nuts/`, and must not patch Agent Run Status over `xgepg7qy`. Worker G3 guard and `write_job_status_md` need a gpu branch. STATUS one-liner for the DEC-067 G3-dest and latest-symlink leaves. Unit tests fail closed.

2. `major` -- `--gpu` illegal unless both `--cpu > 0` and `--memory > 0`. Enforce in `submit_headless` and `launch_headless.py`. Partial pin is still flexible.

3. `major` -- `KINUV_VENV` on `--env` for gpu is `…/venvs/kinuv-cuda`, never recovery. CUDA/`jax==0.11.1`/`BACKEND` checks are on the worker after activate, not on the submit-host recovery venv. No silent jax upgrade.

4. `major` -- GPU without `KINUV_CHAIN_ID` exits 2 (not 4 sequential chains × 4 jobs = 16 chains / 4× wall). One chain, no merge, no shared artifact write, no STATUS patch. Unique run id and session name `-c{N}`. Merge script only, `mixing_ok(..., ess_min=400.0, ess_tail_min=400.0)` on six names.

5. `major` -- GPU image is astroml-cuda (not `DEFAULT_IMAGE` astroml / base-notebook fallback). Venv-build is not the NUTS launcher. Stage 3 only after CUDA identity `|chi2 − 168675.6| < 1` in `timing.json`. MIG 12 GB VRAM ≠ `--memory 16`.

6. `minor` -- Reject-this-wave stays: no recovery mutation, no G4/007, no MAP rewrite, no logit `[0.5, 15]`, no inner `dV/dr`, no S2 16/50/84. Official MAP unchanged. CPU NUTS remains the default until identity **and** mixing. Dispatch GPU NUTS only after the runner patch is on `origin/dev` (entrypoint `git pull` otherwise runs the unpatched 4-loop). Interactive agents must not block on sampling (DEC-067).

## Residual risks

1. No CUDA jax-finufft wheel for jax 0.11.1 → source build or no GPU NUTS. STATUS one-liner; keep CPU production. Carry-forward.

2. MIG 1g.12gb (~12 GB **VRAM**) + x64 NUFFT OOM or weak kernel speedup. Host `--memory` does not enlarge VRAM. STATUS stop or full-GPU quota; do not invent quota. Carry-forward; comment 5 is the lock.

3. Image cuBLAS vs pip jax CUDA mismatch (`dnm4sey1`). Carry-forward.

4. Four GPU slots queue. Report run wall vs submit-to-done separately. 10× uses run wall vs 17440 s. Carry-forward.

5. Interactive agents must not block on sampling (DEC-067). Carry-forward.

6. `xgepg7qy` still Running. Comment 1 is the STATUS-patch lock the propose residual named but execute did not.

7. GPU NUTS mean leftover still structured; 16/50/84 still uncalibrated. Speed smoke is not a new science product. Carry-forward.

8. **(new)** Entrypoint `git pull origin dev` plus dispatch-before-push runs the unpatched 4-loop on GPU. Comment 6; or `KINUV_SKIP_PULL=1` on the patched SHA.

9. **(new)** Four gpu sessions with one `session_name` collide at `canfar create`. Comment 4 `-c{N}` is the lock.

## STATUS updates required

- `verdict` and `severity` as in the header (`accept`, `major`)
- `last_review_a`: this file
- Do not set `board: accepted` (parent tallies)
- Do not clear Agent Run Status for `xgepg7qy`
- Do not start G4
