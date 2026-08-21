---
role: proposer
date: 2026-08-19
agent: canfar-066-12
canon_generation: 4
ids:
  - DEC-066-OSCMETRIC
  - DEC-066-VC
  - DEC-066-INFER
  - DEC-OPS-AUTH
verdict: propose
---

# Handoff: Gate 4 λ_reg campaign stopped mid-run (066-12)

**Read this file after** `AGENTS.md` → `field-guide/index.md` → `docs/architecture/STATUS.md`. Canon generation 4. Do not create a `DEC-*` id. **No NUTS. No 066-9. No rings on real vis until `select_lambda_reg` returns a λ.** CPU only.

Previous session (same agent) was killed on user request at 2026-08-19 23:56 UTC. Campaign process is **stopped**. Do not assume it is still running.

## First actions

This tree is **not** on `origin/dev`. Clone HEAD is still `f47bc9a` (sha6 `f47bc9`). 066-12 lives as a **dirty working tree** plus a snapshot under results.

```bash
cd /arc/projects/KILOGAS/analysis/toby_sandbox/kinUV
git rev-parse HEAD   # expect f47bc9a40fc2ac7e37cba2413929a78ee04e5c51
git status -sb       # expect modified model.py, infer/__init__.py, rotation.py, test_rotation.py
                     # untracked: infer/campaign.py, infer/stage_b.py, tests/test_campaign.py, tests/test_stage_b.py
```

If the working tree is clean or missing those files, restore from

`/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS066/kinuv-KGAS066-f47bc9-lambda/uncommitted/`

(`tracked.diff` + copies of the new modules). SSH clone still fails on this host; HTTPS + `gh` works. Python 3.11 is absent; venv is `$HOME/kinuv-venv` (3.12, CPU jax). `PYTHONPATH=.../kinUV/src`.

Ico is **not** at `analysis/kinms_test`. Use

`/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/KGAS66_Ico_K_kms-1.fits`

Abort if missing (no exponential SB).

## Already done — do not redo

**Stage A MAP (official):** `/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS066/kinuv-KGAS066-f47bc9-map/stage_a_map.json`

Δχ²=+26212.7, V_0=268.37 km/s, r_t=0.5″ floor, vsys=8098.72, PA=381.86°, σ=11.66, (dx,dy)=(−0.104″,−0.059″), flux=60.65 Jy. Fit array 881×95, N=4, Δv=5.080, s=0.514.

**066-12 code (uncommitted):** optional `ring_vc` in `predict_vis`; `infer/stage_b.py` (`V_k` only, freeze nuisance); `infer/campaign.py` (kinematics-only mocks, sequential λ, early-exit). Tests: `pytest tests/test_rotation.py tests/test_stage_b.py tests/test_campaign.py tests/test_map.py` — 25 passed / 2 skipped. Do not grow files past 400 lines.

**Smoke:** 1 mock recovered V_0=200.7, r_t=3.02 (truth 200, 3″).

## Campaign state (killed here)

Dir: `/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS066/kinuv-KGAS066-f47bc9-lambda/`

- `campaign.json` checkpoint (chosen_lambda **null**)
- `campaign.log`, `smoke.log`, `run_gate4.py`
- seed **66**, N_rings **7**, n_mock **20**, λ grid `(0.01, 0.1, 1, 10, 100)`
- Stage A on mocks: ⟨V_0⟩=199.57±0.94 km/s (inject recovered)

| λ | n | ⟨max Ω⟩ | Ω<0.3 | ⟨V_0⟩_rings | ⟨r_t⟩ |
|---|---|---|---|---|---|
| 0.01 | 20 | 2.46 | 0% | 201.2 | 3.03 |
| 0.1 | 20 | 2.04 | 0% | 200.9 | 3.02 |
| 1 | 20 | 1.66 | 0% | 201.0 | 3.02 |
| 10 | 20 | 0.78 | 0% | 207.7 | 3.21 |
| 100 | **15/20** | **0.134** | **100% of 15** | **217.3** | **3.55** |

Killed during **λ=100 mock 14→15** (last checkpoint row: mock 14, nfev=495, ~3 min/eval at this λ). Mocks **15–19 at λ=100 are missing**.

`calibrate_lambda_reg` has **no resume**. A naive re-run repeats ~2.3 h. **Implement skip of completed (λ, mock) rows** using `campaign.json` + the same RNG seed (mocks are drawn before the λ loop). Then finish 5 Stage B jobs and call `select_lambda_reg`.

## Science already visible (do not invent a DEC)

Ω vs recovery is heading for **OSCMETRIC conflict**: λ=100 damps ringing (Ω≈0.13<0.3) but the arctan implied by rings is **high** (V_0≈217 vs 200, r_t≈3.55 vs 3.0) with scatter ~1.7 km/s, so “within 1σ of truth” will likely **fail**. λ≤10 never passes Ω. Criterion 3 (not biased *low* vs Stage A) is not the problem.

If `select_lambda_reg` returns `None` at 7 rings: **N_rings=8** once (ADR). If still `None`: **stop**. Do not drop Ω. Do not Stage B on real 066. Densify λ only if the three criteria conflict and the ADR allows it — that is not a new DEC if you stay inside OSCMETRIC §4.

Real Stage B (only if a λ is chosen): freeze nuisance at official Stage A MAP, fit `V_k`, AIC vs A (`aic_keep_stage_a`). Keep A unless B wins by more than `2(N_rings−2)`. Write `stage_b_map.json` next to Stage A. **No NUTS.**

## Hardware / env

CPU contributed session (`jax.devices()` = CpuDevice). Not GPU. `$HOME/kinuv-venv`. Paths in `kinuv.infer.campaign` are CANFAR. `nproc` in Cursor shells may show 1; `Cpus_allowed_list` is 0–191. Set `OMP_NUM_THREADS=4` etc. as before.

## Forbidden

- Restarting Stage A MAP on real vis
- NUTS / dynesty / GPU / 066-9 XX+YY
- Rings on real vis with `chosen_lambda is None`
- Committing unless the user asks (working tree is the live 066-12)
- New `DEC-*` ids

## STATUS updates required

- `last_propose:` this file
- `phase:` 066-12
- `next_role: proposer`
- Campaign **not** finished; official kinematic product is still Stage A
