---
role: proposer
date: 2026-08-21
agent: canfar-066-12-recovery
canon_generation: 4
ids:
  - DEC-066-OSCMETRIC
  - DEC-066-VC
  - DEC-066-INFER
  - DEC-OPS-AUTH
verdict: report
---

# Report: Gate 4 λ_reg complete; Stage B blocked (066-12)

**Read after** `AGENTS.md` → `field-guide/index.md` → `docs/architecture/STATUS.md`. Canon generation 4. No new `DEC-*`. **No NUTS. No 066-9.** Official kinematic product remains **Stage A MAP**.

This report is for **local diagnosis**: it ships the completed campaign JSON, Stage A MAP copy, and the 066-12 code that produced them.

## Verdict (one paragraph)

Stage A MAP on CANFAR is **VALID_COMPLETE** (Δχ²=+26212.7, V₀=268.4 km/s). Gate 4 (20 mocks × λ∈{0.01,0.1,1,10,100}) finished at **N_rings=7** and again at the ADR bump **N_rings=8**. In both cases `select_lambda_reg` returned **`None`**: only λ=100 satisfies the Ω ring-noise gate, but that λ biases the arctan recovery high (V₀≈217 vs truth 200; rₜ≈3.55 vs 3.0), so the 1σ recovery fraction fails. Per DEC-066-OSCMETRIC / handoff ADR: **do not drop Ω, do not densify λ without a new ADR, do not fit rings on real KGAS066.** `chosen.json` is `{"chosen_lambda": null}`. No `stage_b_map.json`.

## Timeline

| When (UTC) | Event |
|---|---|
| 2026-08-19 ~20:12 | Stage A MAP written under `kinuv-KGAS066-f47bc9-map/` |
| 2026-08-19 ~20:42 | Gate 4 smoke OK (V₀≈200.7 on 1 mock) |
| 2026-08-19 ~23:56 | Campaign killed mid λ=100 (15/20 mocks at N=7) |
| 2026-08-20 ~21:48 | MAP recovery session: existing `stage_a_map.json` validated; no MAP rerun |
| 2026-08-20 ~22:00 | Campaign resume: skip completed (λ,mock); finish mocks 15–19 at λ=100 |
| 2026-08-20 ~22:16 | N=7 complete → `select_lambda_reg` → **None** → ADR recurse N=8 |
| 2026-08-21 ~02:35 | N=8 complete → `select_lambda_reg` → **None** → **LAMBDA_NONE**; stop |

## Artifacts (CANFAR + this commit)

### On `/arc` (authoritative science tree)

```
/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS066/
  kinuv-KGAS066-f47bc9-map/
    stage_a_map.json          # official Stage A
    stage_a_map.log
    run_stage_a_map.py
    RECOVERY_NOTE.txt
  kinuv-KGAS066-f47bc9-lambda/
    campaign_n7_complete.json # frozen N=7 full 100 rows
    campaign_n8_complete.json # frozen N=8 full 100 rows
    campaign.json             # last write (= N=8)
    chosen.json               # {"chosen_lambda": null}
    campaign_resume.log
    run_gate4.py
    STAGE_B_STOP.txt
```

### Bundled in-repo for laptop pull (this commit)

```
docs/reviews/artifacts/2026-08-21-gate4/
  stage_a_map.json
  campaign_n7_complete.json
  campaign_n8_complete.json
  chosen.json
  STAGE_B_STOP.txt
  run_gate4.py
```

Reproduce `select_lambda_reg` offline (no visibilities needed):

```bash
cd kinUV && export PYTHONPATH=$PWD/src
python - <<'PY'
import json, numpy as np
from pathlib import Path
from kinuv.profiles.rotation import select_lambda_reg

def pick(path):
    d = json.loads(Path(path).read_text())
    lams, n = d["lambdas_tried"], d["n_mock"]
    v0_a = np.asarray(d["v0_stage_a"], float)
    om = np.zeros((len(lams), n)); v0 = np.zeros_like(om); rt = np.zeros_like(om)
    for r in d["rows"]:
        i, j = lams.index(r["lambda"]), r["mock"]
        om[i, j] = r["max_omega"]; v0[i, j] = r["v0_b"]; rt[i, j] = r["rt_b"]
    v0_sig = np.repeat(np.std(v0, 1, ddof=1)[:, None], n, axis=1)
    rt_sig = np.repeat(np.std(rt, 1, ddof=1)[:, None], n, axis=1)
    return select_lambda_reg(np.asarray(lams), om, v0, rt, v0_sigma=v0_sig, rt_sigma=rt_sig, v0_stage_a=v0_a)

base = "docs/reviews/artifacts/2026-08-21-gate4"
print("N7", pick(f"{base}/campaign_n7_complete.json"))
print("N8", pick(f"{base}/campaign_n8_complete.json"))
PY
```

Expected: `N7 None` / `N8 None`.

## Stage A (official product — unchanged)

Session / HEAD: `f47bc9a` (sha6 `f47bc9`). Fit array 881×95, N=4, Δv≈5.080 km/s, s≈0.514.

| Quantity | CANFAR MAP | Laptop reference |
|---|---|---|
| Δχ² vs V=0 | **+26212.7** | +26213 |
| V₀ (km/s) | **268.37** | 268 |
| vsys (radio, km/s) | 8098.72 | ≈8099 |
| PA (deg) | 381.86 ≡ 21.9 | 21.9 |
| σ_gas (km/s) | 11.66 | ≈11.7 |
| (dx, dy) (″) | (−0.104, −0.059) | (−0.10, −0.06) |
| flux (Jy) | 60.65 | ≈60.6 |
| rₜ (″) | **0.5 (floor)** | 0.5 floor |

Ico/cube on CANFAR: `/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/` (laptop `kinms_test` path absent). NPZ size 997305244 bytes.

## Gate 4 setup

- Truth inject: arctan (V₀=200 km/s, rₜ=3″) on real 066 uv + complex noise (`σ²=1/(s w)`).
- Frozen nuisance from Stage A seed geometry; flux 60.6 Jy; (dx,dy)=0 on mocks.
- Seed **66**, n_mock **20**, λ grid `(0.01, 0.1, 1, 10, 100)`.
- Stage B on mocks: fit `V_k` only; `ring_regulariser(λ)`; knots via `uniform_knot_radii` (DEC-066-OSCMETRIC).
- OSCMETRIC (`select_lambda_reg`): smallest λ with all three:
  1. `mean(Ω < 0.3) ≥ 0.95`
  2. joint 1σ recovery of (V₀, rₜ) vs truth in ≥68% of mocks
  3. not biased low vs Stage A: `mean(V₀_B) ≥ mean(V₀_A) − scatter(V₀_B)`

## Campaign tables

Columns: ⟨max Ω⟩, fraction Ω<0.3, ⟨V₀⟩_rings±1σ, ⟨rₜ⟩±1σ, fraction |V₀−200|≤σ_mock, fraction |rₜ−3|≤σ_mock.  
(Recovery gate uses the **joint** of the last two; those fractions are shown separately for diagnosis.)

### N_rings = 7 (`campaign_n7_complete.json`) → **None**

Stage A on mocks: ⟨V₀⟩ = 199.57 ± 0.97 km/s.

| λ | ⟨max Ω⟩ | Ω<0.3 | ⟨V₀⟩ | ⟨rₜ⟩ | 1σ V₀ | 1σ rₜ |
|---|---|---|---|---|---|---|
| 0.01 | 2.458 | 0% | 201.2±1.1 | 3.03±0.03 | 0.40 | 0.45 |
| 0.1 | 2.039 | 0% | 200.9±1.1 | 3.02±0.03 | 0.45 | 0.55 |
| 1 | 1.664 | 0% | 201.0±1.0 | 3.02±0.03 | 0.50 | 0.55 |
| 10 | 0.778 | 0% | 207.7±1.2 | 3.21±0.04 | 0.00 | 0.00 |
| 100 | 0.134 | **100%** | **217.6±1.7** | **3.56±0.05** | **0.00** | **0.00** |

### N_rings = 8 (`campaign_n8_complete.json`) → **None**

Same Stage A mock vector (RNG seed 66). Ω slightly lower at soft λ; λ=100 still the only Ω pass, still high recovery.

| λ | ⟨max Ω⟩ | Ω<0.3 | ⟨V₀⟩ | ⟨rₜ⟩ | 1σ V₀ | 1σ rₜ |
|---|---|---|---|---|---|---|
| 0.01 | 2.403 | 0% | 201.4±1.1 | 3.03±0.03 | 0.30 | 0.45 |
| 0.1 | 1.575 | 0% | 201.0±1.1 | 3.02±0.03 | 0.50 | 0.60 |
| 1 | 1.288 | 0% | 200.8±1.0 | 3.01±0.03 | 0.55 | 0.55 |
| 10 | 0.738 | 0% | 205.5±1.2 | 3.15±0.04 | 0.00 | 0.00 |
| 100 | 0.149 | **100%** | **216.6±1.4** | **3.55±0.04** | **0.00** | **0.00** |

Criterion 3 (not low vs Stage A) **passes** at all λ. The deadlock is **Ω vs recovery**.

## Code landed in this commit (066-12)

Working tree was uncommitted on `f47bc9a`; now recorded:

| Path | Role |
|---|---|
| `src/kinuv/infer/stage_b.py` | Ring MAP: `V_k` only, freeze nuisance; AIC vs Stage A |
| `src/kinuv/infer/campaign.py` | Mock campaign + **resume skip** of completed `(λ, mock)` rows |
| `src/kinuv/forward/model.py` | Optional `ring_vc` in `predict_vis` |
| `src/kinuv/profiles/rotation.py` | Campaign wrapper → `calibrate_lambda_reg` |
| `tests/test_stage_b.py`, `tests/test_campaign.py`, `tests/test_rotation.py` | Unit coverage |
| `docs/reviews/2026-08-19-handoff-gate4.md` | Mid-run handoff |
| this file | Final report |

Resume behaviour (required after the mid-λ=100 kill): load `campaign.json` if `n_rings`/`n_mock` match; reuse `v0_stage_a`; skip Stage B evals already present in `rows`; checkpoint after each new mock. Naive full re-run would repeat ~2–4 h of CPU.

Tests on CANFAR recovery venv (`~/kinuv-venv-recovery`, Python 3.12, CPU JAX):  
`pytest tests/test_rotation.py tests/test_stage_b.py tests/test_campaign.py tests/test_map.py` → **25 passed, 2 skipped**.

## What was not done (by design)

- No `stage_b_map.json` on real visibilities.
- No NUTS / dynesty / GPU / 066-9 XX+YY.
- No new DEC; no dropping Ω; no λ densification.
- No Stage A MAP rerun (already VALID_COMPLETE).

## Diagnosis notes for the next proposer

1. **Conflict is physical in the current metric**, not a resume bug: N=7 and N=8 agree qualitatively.
2. Soft λ recovers truth but rings oscillate (⟨Ω⟩≫0.3). Hard λ flattens rings (⟨Ω⟩≈0.13–0.15) but the implied arctan sits ~+17 km/s / +0.55″ high — outside mock 1σ by construction of the bias.
3. Options that require an **ADR / DEC change** (not this report): densify λ between 10 and 100; soften Ω or recovery fraction; change knot count / spacing; change recovery target (rings vs arctan projection); accept Stage A only and document Gate 4 as failed-open.
4. Until a λ is selected under OSCMETRIC as written: **keep Stage A**.

## Environment (CANFAR)

- Host: Skaha contributed CPU (`jax.devices()` = `CpuDevice`; no NVIDIA).
- Repo path used: `/arc/projects/KILOGAS/analysis/toby_sandbox/kinUV` at `f47bc9a` + dirty 066-12 (now this commit).
- Broken `~/kinuv-venv` (conda python symlink gone) → used `~/kinuv-venv-recovery` (3.12 + numpy/scipy/astropy/jax/jax-finufft).
- Session: `tmux` `kinuv-map`. `OMP_NUM_THREADS=4`.

## STATUS mailbox

- `phase:` 066-12
- `last_propose:` this file
- Official product: Stage A MAP only
- Next role: proposer (decide whether to reopen OSCMETRIC; do not silently Stage B)
