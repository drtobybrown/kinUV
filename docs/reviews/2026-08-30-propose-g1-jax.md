---
role: proposer
date: 2026-08-30
agent: parent
canon_generation: 4
ids:
  - DEC-066-SPECRESP
  - DEC-066-INFER
  - DEC-066-TARGET
  - DEC-066-INC
  - DEC-066-GRID
  - DEC-066-WEIGHT
  - DEC-HIER-SELFUNC
verdict: propose
---

# G1: CPU JAX `predict_binned` (066 kernel)

## Scope

Next wave of the dual-accepted 066 kernel sequence ([`gold-standard-roadmap.md`](../diagnostics/gold-standard-roadmap.md)). G0 flags already landed (`36bfde3`). Existing ids only. No new `DEC-*`. Official MAP `kinuv-KGAS066-uvsign-map` stays read-only. This card is the JAX likelihood on CPU, not NUTS, not SBC, not hierarchical pooling, not a 400-galaxy runner.

This user directive restated Talts SBC, partial pooling, logit bijections, PSIS-LOO, hard-target models, CANFAR GPU, and a kernel-purge commit. Those are already sequenced or done. This card does **G1 only**.

## Architect verdict (selected path)

**Do this card:** move sky + Hann+bin + `chi2` onto XLA so `predict_binned` can be differentiated later. NUFFT is already `jax-finufft` but `nufft2_degrid` host-bounces through `np.asarray`. Production operator stays `kinuv.response.spectral.hann_then_bin`. Tests stay `JAX_PLATFORMS=cpu`. JAX compile cache on `/tmp` (or `XDG_CACHE_HOME` under `/tmp`), not NFS `/arc`.

**Hard gates (implementer decides pass/fail, records on STATUS):**

1. Identity: official Stage A vector, Hann+bin XX 881x95, `|chi2 - 168675.6| < 1`. Same `s`, same leftover identity as the plot folder. If `/arc` npz is missing, skip that test and still require a tiny numpy-vs-JAX identity (max abs vis error small enough that `chi2` cannot drift by 1 on the 066 array).
2. Operator: Gate 2 still asserts `hann_then_bin` before mock vis. `native_diagonal` still raises. Do not re-purge; hygiene is done (`331d787`).
3. Speed: after one warmup JIT, one `predict_binned` + `chi2` eval on the 066 fit array must beat S2 FD 0.329 eval/s. Identity is the reject-if-fail gate; speed miss is STATUS one-liner + continue only if identity holds.
4. No GPU session, no `canfar create --gpu`, no CUDA image.

**Defer (already decided; do not reopen):**

- G2 unconstrained chart. Do not logit `RT_BOUNDS_ARCSEC=(0.5, 15)` as a prior. MAP sits on the wall; G0 flags `r_t_at_floor`.
- G3 NumPyro NUTS (two Stage A runs, PA 205.2 and 25.2, 4 chains each, `R_hat` < 1.01, `ESS` > 200, label `sampler: nuts` only after autodiff).
- G4 Talts SBC, G5 PSIS-LOO on 066 vis cells.
- `DEC-HIER-SELFUNC` Phase 5. Unfreeze `i`. Add `h_z`. Warp/strip classes. TARGET subset.
- uvkin merge.

**Reject this wave:**

- Finite-difference HMC labeled NUTS.
- emcee.
- Hierarchical NUTS over 400.
- Quoting S2 Laplace 68/95.
- The requested commit subject `prepare Stage A NUTS engine` (false: this is JAX identity, not NUTS).
- Re-running the 2026-08-29 kernel purge as if it had not landed.
- Provisioning GPU because NUTS/SBC/400 were named in the user directive.

Hard targets stay G0 flags until the user adds stubs. Official 066 already fires `r_t_at_floor` and `leftover_chi2_structured`.

## What changed / what was checked

- G0: `kinuv.diagnostics.flags.map_quality_flags`; leftover vel span 0.355 > uv-binned span 0.115 on `docs/reviews/artifacts/2026-08-30-final-fit/leftover_chi2.npz`.
- `native_diagonal` raises; Gate 2 `tests/test_mock_recovery.py` asserts Hann+bin.
- Live `predict_binned` (`infer/map.py`) is NumPy `predict_vis` then `hann_then_bin`. `nufft2_degrid` returns host `numpy`.
- S2 FD: 0.329 eval/s. Laplace SBC failed 68/95. No NUTS posterior exists.

## Rejected alternatives

- "JAX the NUFFT only" — already JAX; the pole is sky + Hann+bin + `chi2` plus the host bounce.
- "Start NUTS on NumPy `chi2`" — not NUTS; S2 already mixed with FD.
- "Logit the production floor now" — G2; MAP at 0.5 arcsec sends unconstrained `r_t` to -inf.

## Residual risks

1. jax-finufft vs python-finufft fallback: identity must hold on the backend the CPU session actually uses. Do not require GPU jax-finufft.
2. JIT compile on first eval can exceed 0.329 s; the speed gate is **post-warmup**.
3. Official Stage A identity needs the CANFAR npz + Wiener Ico. Tiny-grid identity is the always-on test.
4. Even a correct JAX `chi2` does not calibrate leftover-vs-velocity (frozen Wiener Ico).

## Execute if accepted

1. JAX CPU `predict_binned` (sky + Hann+bin + `chi2`) with no host bounce on the vis path used by `log_like`. Keep NumPy as the identity reference. Do not add `i` or `h_z`.
2. Tests: tiny numpy-vs-JAX identity (always); official 066 `chi2` within 1 of 168675.6 when `/arc` paths exist; Gate 2 Hann+bin still passes. `JAX_PLATFORMS=cpu`. Cache under `/tmp`.
3. Write a small JSON timing summary (eval/s after warmup vs 0.329). Do not dump arrays into chat.
4. CHANGELOG + one STATUS line for identity and eval/s. Do not start G2/G3. Do not refit. Do not write the official MAP tree.
5. Commit and push `origin/dev`. Conventional subject about JAX identity, not NUTS.

Process (this propose, not G1 physics): mailbox `## Agent Run Status` block; `.cursor/notify.sh` + `hooks.json` posting the STATUS header to ntfy topic `kinuv_canfar_agent_thbrown` on agent stop. Fail open. No secrets.

## STATUS updates required

- `next_role: board`
- `board: open`
- `last_propose:` this file
