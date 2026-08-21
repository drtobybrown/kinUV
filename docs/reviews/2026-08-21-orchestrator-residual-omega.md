---
role: orchestrator
date: 2026-08-21
agent: canfar-066-12-orch
canon_generation: 4
ids:
  - DEC-066-OSCMETRIC
  - DEC-066-VC
  - DEC-066-INFER
verdict: propose
---

# Orchestrator path: residual ringing, then Stage B

User dispatch: switch proposer → **orchestrator**, **build phase**. Tie-break: implement the OSCMETRIC *question* (quantify **ringing**), not absolute concavity of V_c. No new `DEC-*` filename. No NUTS. No GPU.

## Why the last stop was right, and why we still build

Absolute `Ω(V_k)` on the true arctan is 1.65–3.15 at N=6–8 (Δv≈5.08). That is the turnover, not noise. User now authorizes the calibration that OSCMETRIC’s title asked for: **ringing = curvature of the residual relative to the smooth arctan used to initialise the rings.**

Same formula, same 0.3 / 95% gates, same λ grid, same 6–8 knots. `V_ref(r_k) = arctan samples at the Stage B init `(V₀, r_t)`` (mocks: truth 200/3″; real vis: official Stage A).

`Ω_k^ring = |Δ²(V − V_ref)| / Δv_chan`

Truth residual is identically 0. High λ (flatten toward a line) **fails** this gate. Low λ that stays on the arctan **passes** if wiggles stay under 0.3.

## Recovery (criterion 2)

Do **not** use mock-sample scatter as 1σ (shared uv ⇒ σ≈1 km/s, 1 km/s discretisation bias ⇒ coverage ~40% even when V₀=201). Field-guide Gate 4 is “injected V_c within **beam-scale** covariance.”

Fixed calibration window, still ≥68% of mocks:

- `|V₀ − 200| ≤ 10 km/s`
- `|r_t − 3″| ≤ 0.5″`

Criterion 3 (not low vs Stage A) unchanged.

Do not apply `|max Ω_abs − Ω_truth|` to old JSON: N=7 λ=0.1 only 70% within 0.3 of truth-max (need 95%). Residual Ω needs `V_k`. **Do not resume** `campaign.json` that stored absolute Ω.

## Sequence (orchestrator)

1. **Implement** (subagent): residual Ω in Stage B; campaign `select_lambda_reg` with σ_V=10, σ_rt=0.5; checkpoint `omega_mode=residual`; tests; files ≤400 lines.
2. **Review** (this chat, field-guide + DEC-066-VC/OSCMETRIC/INFER): no new DEC id; no NUTS; no exponential SB; no rings on real vis until λ chosen.
3. **Smoke** 1 mock × λ grid. If residual Ω at λ≤1 is not ≪0.3, adjust and go again.
4. **Campaign** 20×5 at N=7, new out dir. If `None`, N=8 once. If still `None`, densify λ only then (OSCMETRIC §2). Stop if still `None` (do not drop Ω).
5. **Stage B on real 066** only if λ chosen: freeze Stage A nuisance, AIC vs A. Write `stage_b_map.json` next to Stage A **or** keep A if AIC says so. No NUTS.

## Expected (not a gate)

Soft λ residual Ω small; `chosen_lambda` likely 0.01 or 0.1. Real Stage B may still **keep Stage A** (AIC `2(N−2)`). That is success of the procedure, not a crash.

## Forbidden

NUTS, GPU, 066-9, exponential Ico fallback, deleting `/arc` science files, new `DEC-*` ids, using absolute-Ω checkpoints as residual.
