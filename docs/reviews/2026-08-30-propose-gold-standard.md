---
role: proposer
date: 2026-08-30
agent: parent
canon_generation: 4
ids:
  - DEC-066-INFER
  - DEC-066-TARGET
  - DEC-066-INC
  - DEC-066-SB
  - DEC-066-REPO
  - DEC-066-SPECRESP
  - DEC-HIER-SELFUNC
verdict: propose
---

# Gold-standard inference and hard-target sequence (066 first)

## Scope

Architect decision after S1/S2 and the 2026-08-30 plot handoff. Existing ids only. No new `DEC-*`. Official MAP `kinuv-KGAS066-uvsign-map` stays the product until a later MAP card. This card is a **sequence**, not a JAX rewrite and not a 400-galaxy runner.

Human note: [`docs/diagnostics/gold-standard-roadmap.md`](../diagnostics/gold-standard-roadmap.md) (written on accept).

## Architect verdict (selected path)

Gold-standard NUTS + Talts SBC is **real**, and it is **blocked on autodiff**. The live likelihood is NumPy `predict_binned` + FINUFFT, FD Jacobian, ~0.45 s/eval. S2 `laplace_mh` mixed (`R_hat` ~1.00, `ESS` hundreds) and **Laplace SBC failed** 68/95. There is no NUTS posterior to calibrate.

**Do, in order, on 066 only:**

1. **MAP quality flags** (this card's code, small). Tag floor `r_t`, PA alias, `Delta_chi2` vs V=0, leftover-vs-velocity structure. 400-galaxy later must not quote a thin-disk arctan as truth when flags fire.
2. **JAX `predict_binned`** (next implementation wave, CPU). Move sky + Hann+bin + `chi2` onto XLA. NUFFT is already JAX. Keep tests on CPU. Do not provision GPU yet.
3. **Unconstrained coordinates** (with the JAX model). Logit/softplus on `r_t`, `gas_sigma`, flux, inclination-if-unfrozen, with Jacobian. This is for HMC, not cosmetics. Production box `(0.5, 15)` stays the science prior; the bijection is the sampler chart.
4. **NumPyro NUTS** on Stage A, 4 chains, two PA starts (205.2 and 25.2), report `R_hat` / `ESS`. Label `sampler: nuts`. Never label FD MH as NUTS.
5. **Talts SBC** on the exact kernel, `T=1`, O(100) draws once NUTS eval/s is known. Then a real-066 column that **does not** claim calibration under SB leftover.
6. **PSIS-LOO** Stage A vs Stage B only after (4). MAP AIC already prefers B (`Delta_chi2` vs A = +1373); LOO is the posterior version, not a reason to skip JAX.
7. **CANFAR GPU** only after a 066 NUTS smoke passes on CPU.

**Defer until user stubs (recommend, do not create):**

- TARGET: 066-like subset vs full ~400.
- Optional `h_z` (edge-on).
- Unfreeze `i` with catalogue/MaNGA prior (face-on `V sin i`).
- Warp / strip / KDC model classes.
- `DEC-HIER-SELFUNC` stays Phase 5. Partial pooling without calibrated galaxy posteriors and a selection function is not gold-standard.

**Reject this wave:**

- Finite-difference HMC/NUTS on GPU (tens of CPU-hours/chain already; GPU does not differentiate NumPy).
- emcee (DEC-066-REPO).
- Hierarchical NUTS over 400 galaxies.
- Quoting S2 Laplace intervals as 68/95.
- Restoring Ico as intrinsic SB to "fix" leftover.
- A survey runner.

## What changed / what was checked

- S1: vis recovers inject `r_t=0.25`, `gas_sigma=8`; CLEAN does not.
- S2: Laplace SBC fail is a **width** problem, not a MAP problem. MAP `Delta_chi2` vs V=0 = +35553 still stands.
- Official leftover `chi2` vs velocity is structured (SB/spirals). SBC on the exact mock kernel already failed; real-066 intervals would be worse if we pretended Laplace was NUTS.
- `DEC-066-TARGET` still locks **code** to 066. Survey-readiness is a checklist, not a runner.
- Hard 400-galaxy regimes (compact, low SNR, i extremes, warps, tails) need **model classes + flags**, not a faster sampler on the 066 thin disk.

## Hard-target taxonomy (flags, not a runner)

| Regime | Current 066 engine | Needed |
|---|---|---|
| Compact `r_t` << BMAJ | S1 shows vis can work if the box allows; production floor 0.5" will pin | Flag `r_t_at_floor`; mock box only when licensed |
| Low SNR | vis `chi2` still defined; MAP may not beat V=0 | Flag `delta_chi2_vs_zero`; do not sample if it fails |
| Edge-on | frozen thin disk, no `h_z` | user stub to allow `h_z`; else flag `high_i_thin_disk` |
| Face-on | `V sin i` with i frozen | user stub to unfreeze i; else flag `i_frozen_faceon` |
| Warp / strip / KDC | axisymmetric circular | flag `axisym_assumed`; do not auto-switch to rings as a warp model |

Stage B rings are a **rotation-curve** flexibility, not a strip/warp model.

## Rejected alternatives

- "Run NUTS now" on FD `chi2` — not NUTS; not GPU-shaped.
- Laplace temperature `T_nvis` as a substitute for SBC — stretches std by ~1.42; S2 already showed the Gaussian is too thin even at `T=1` on mocks.
- Hierarchical pooling from MAPs — point estimates plus failed Laplace widths are not exchangeable posteriors.
- uvkin KinMS/emcee as the gold standard — different stack; production vis MAP is kinUV.

## Residual risks

1. JAX rewrite is the long pole (forward model + Hann+bin + tests). Dual accept of this card does **not** finish it.
2. Even true NUTS CIs on real 066 will be overconfident if SB leftover stays (frozen Wiener Ico). Mock SBC tests the exact kernel only.
3. Bijection around a **bound** `r_t=0.5"` does not unstick a likelihood that wants 0.25" on the real galaxy.
4. GPU jax-finufft / CUDA images on Skaha can fail independently of a correct CPU JAX likelihood.

## Execute if accepted

1. Write [`docs/diagnostics/gold-standard-roadmap.md`](../diagnostics/gold-standard-roadmap.md) (this sequence, ASCII).
2. Point `docs/methodology.md` and `docs/diagnostics/survey-readiness.md` at it; list recommended user stubs (TARGET subset, `h_z`, unfreeze i, warp/strip).
3. Add `kinuv.infer.flags.map_quality_flags(stage_a_json) -> dict` (floor `r_t`, `delta_chi2`, PA vs 21.9 alias, i frozen). Unit test with the official Stage A numbers. Do not refit.
4. Commit and push. Do **not** start the JAX rewrite in this card. Next implementer wave after this accept is G1 JAX `predict_binned` on CPU, 066 only.

## STATUS updates required

- `next_role: board`
- `board: open`
- `last_propose:` this file
