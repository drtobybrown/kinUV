---
role: orchestrator
date: 2026-08-21
agent: canfar-066-12-orch
canon_generation: 4
ids:
  - DEC-066-OSCMETRIC
  - DEC-066-VC
  - DEC-066-INFER
  - DEC-066-AGENTS
  - DEC-OPS-AUTH
verdict: propose
---

# Pickup: orchestrator build (residual Ω) — roles and state

**Read after** `AGENTS.md` → `field-guide/index.md` → `docs/architecture/STATUS.md`. Canon generation 4. You are picking up a **build**, not redesigning Gate 4 from zero. No new `DEC-*` id. No NUTS. No GPU. Do not create a `DEC-` id.

Repo: `https://github.com/drtobybrown/kinUV` branch **`dev`**. Clone on CANFAR: `/arc/projects/KILOGAS/analysis/toby_sandbox/kinUV`. Expected parent of this commit: `27b01d4` (absolute-Ω spec-failure doc). This commit adds residual-Ω code + this handoff.

## Roles (who does what)

| Role | Who | Authority | Must not |
|---|---|---|---|
| **User / tie-breaker** | Toby | Only person who may add a `DEC-*` stub. Authorized **orchestrator / build** on 2026-08-21 after the spec-failure propose. | — |
| **Orchestrator** | this chat (senior) | Sequence, review subagent diffs vs DEC/field-guide, start/stop campaigns, AIC Stage B only after λ. Writes mailbox + reviews. | Rubber-stamp; NUTS; new DEC ids; two campaigns at once |
| **Proposer (earlier)** | canfar-066-12-diagnosis | Closed *absolute* Ω: truth arctan already Ω>0.3. Doc: `2026-08-21-propose-gate4-spec-failure.md`. | Was not to implement residual Ω until user build dispatch |
| **Reviewer (handshake)** | vacant | Must challenge or ACK residual-Ω interpretation in a **later** turn (`DEC-066-AGENTS`). Orchestrator did not ACK own propose. | ACK in the same turn as proposing |
| **Impl subagent** | `d1dacd7d-3d31-4c8d-a7bf-a89745e2637e` | Residual Ω + recovery windows + tests. **Done. Orchestrator reviewed: accept.** | Campaign / real-vis Stage B |
| **Campaign subagent** | `290fb550-28fc-4202-aaed-5938fd59bd83` | Smoke n_mock=2 then 20×5 residual campaign then Stage B if λ∈ℝ. **Farmed 2026-08-21 ~14:33 UTC.** At handoff: **no `lambda-resid/` tree yet**; tmux `kinuv-map` still shows old `LAMBDA_NONE` prompt. Treat as **not started on disk**. | Resume absolute-Ω `campaign.json`; second copy of campaign |

Pickup agent **is the orchestrator**. Resume the campaign subagent only if it is healthy; otherwise run the campaign yourself in `tmux` `kinuv-map`.

## What is finished (do not redo)

**Stage A MAP (official kinematic product until Stage B JSON exists and AIC wins):**

`/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS066/kinuv-KGAS066-f47bc9-map/stage_a_map.json`

Δχ²=+26212.7, V₀=268.37 km/s, rₜ=0.5″ floor, vsys=8098.72, PA=381.86°, σ=11.66, flux=60.65, fit 881×95, s≈0.514. NPZ 997305244 bytes. Ico: `.../products/v1.3/original/by_galaxy/KGAS66/30kms/` (not `kinms_test`).

**Absolute-Ω Gate 4 (closed):** 20×5 at N=7 and N=8, `chosen_lambda=null`. Artifacts in-repo: `docs/reviews/artifacts/2026-08-21-gate4/`. Reports: `2026-08-21-report-gate4-stage-b.md`, `2026-08-21-propose-gate4-spec-failure.md`. Physics: noise-free truth max Ω=2.24 (N=7) / 1.65 (N=8) at Δv≈5.08. Unit test `test_truth_arctan_omega_exceeds_gate_at_oscmmetric_knots` **must stay**.

**Residual-Ω implementation (this commit, orchestrator-reviewed):**

- `omega_k` unchanged (absolute).
- `omega_residual(V, V_ref, Δv)` = `omega_k(V − V_ref)`.
- `run_stage_b_map.max_omega` = residual vs init arctan (`v_init`).
- `select_lambda_reg` from campaign uses **fixed** `RECOVERY_V0_KMS=10`, `RECOVERY_RT_ARCSEC=0.5`, not mock scatter.
- Checkpoint `"omega_mode": "residual"`; `_prior_checkpoint` **refuses** absolute-Ω JSON.
- pytest: `29 passed, 2 skipped` (`test_rotation`, `test_stage_b`, `test_campaign`, `test_map`). Files ≤400 lines.

Plan: `docs/reviews/2026-08-21-orchestrator-residual-omega.md`.

## What is not finished

1. Residual-Ω **smoke** (n_mock=2) — not on disk.
2. Residual-Ω **20×5 campaign** — not on disk. Target dir:

   `/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS066/kinuv-KGAS066-f47bc9-lambda-resid/`

   Do **not** write into `.../kinuv-KGAS066-f47bc9-lambda/` (absolute-Ω archive).

3. **Real-vis Stage B** — no `stage_b_map.json`. Run only if `chosen_lambda` is a float. AIC may keep Stage A; still write JSON. Copy `run_gate4.py` and point `OUT` at `lambda-resid`.

4. Handshake **review ACK** of residual-Ω as the ringing metric — still open (`next_role` after pickup work: reviewer, unless user says stay orchestrator).

## Pickup procedure (in order)

```bash
tmux has-session -t kinuv-map || tmux new -s kinuv-map
# inside pane:
cd /arc/projects/KILOGAS/analysis/toby_sandbox/kinUV
git fetch origin && git checkout dev && git pull origin dev
git log -3 --oneline
# expect residual-Ω commit on top of 27b01d4
pgrep -af 'run_gate4|calibrate_lambda'   # if live, attach; do not start a second job
ls /arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS066/kinuv-KGAS066-f47bc9-lambda-resid 2>/dev/null
```

Venv: `/arc/home/thbrown/kinuv-venv-recovery` (3.12, CPU JAX). Broken: `~/kinuv-venv` (conda python symlink). `PYTHONPATH=$PWD/src`. `OMP_NUM_THREADS=4`. `gh` for push: `/arc/home/thbrown/.local/bin/gh` (`/opt/conda/bin/gh` missing; do not `git config --global`).

Smoke then campaign:

```python
from pathlib import Path
from kinuv.infer.campaign import calibrate_lambda_reg
OUT = Path("/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS066/kinuv-KGAS066-f47bc9-lambda-resid")
calibrate_lambda_reg(n_mock=2, n_rings=7, smoke=False, out_dir=OUT / "smoke2")
# residual maxΩ at λ=0.01 must be ≪ 2.4; if still ~2.4, metric not wired — stop
calibrate_lambda_reg(smoke=False, n_mock=20, n_rings=7, out_dir=OUT)
```

N=8 recurse is already in `calibrate_lambda_reg` if `chosen is None`. If still None: **stop** (do not drop Ω). Densify λ only if Ω vs recovery conflict and you have budget (`{3,30}`). Then `run_gate4.py stage-b` only if λ chosen.

Ico abort if missing. No exponential SB.

## Env / ops

- Host: Skaha contributed CPU (`jax.devices()=[CpuDevice]`). Session `tmux` `kinuv-map`.
- HTTPS + restored `gh`; SSH GitHub often fails here.
- Cursor agent transcripts: `~/.cursor/projects/arc-home-thbrown/agent-transcripts/` (`290fb550…` campaign, `d1dacd7d…` impl).

## Forbidden (unchanged)

NUTS, dynesty, GPU, 066-9 XX+YY, rings on real vis with `chosen_lambda is None`, deleting `/arc` visibilities/Ico/cube, new `DEC-*` ids, using absolute-Ω `campaign.json` as residual resume.

## STATUS after this file

- `phase:` 066-12 (build: residual Ω → λ → Stage B iff λ)
- `last_propose:` this file
- Official product: **Stage A** until residual campaign + optional `stage_b_map.json`
- `next_role: proposer` (pickup orchestrator continues the build)
