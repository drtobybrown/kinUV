# 066 kernel sequence (human)

Not a 400-galaxy runner. Not population inference (`DEC-HIER-SELFUNC` stays Phase 5). Official product: **`kinuv-KGAS066-uvsign-map`**. Propose: [`docs/reviews/2026-08-30-propose-gold-standard.md`](../reviews/2026-08-30-propose-gold-standard.md). Dual accept 2026-08-30 (major): leftover-vs-velocity is a G0 flag; flags live in `kinuv.diagnostics`; G1 is CPU JAX only.

## What is already true

- Vis Stage A beats V=0 (`Delta_chi2` = +35553). S1 recovered a sub-beam inject on Hann+bin; the CLEANed cube did not.
- Tiny-mock CPU NUTS exists (`sampler: nuts` on an inject, not 066). There is **no 066 NUTS posterior**. S2 was `laplace_mh`. Laplace 68/95 intervals **failed** SBC. Do not quote S2 or the tiny-mock 16/50/84 as calibrated 066 intervals.
- Leftover vis `chi2` vs velocity is more structured than leftover vs uv (spirals in a frozen Wiener Ico). `T_dof ~ 1` is not a calibration of that leftover.
- Official Stage A `r_t` sits on the L-BFGS box floor 0.5 arcsec. That box is not a science prior for HMC. A logit chart of `[0.5, 15]` with the MAP at the wall is not a later NUTS plan.

## Sequence (066 kernel)

| Wave | What | GPU? |
|---|---|---|
| G0 | MAP quality flags in `kinuv.diagnostics.flags` (floor `r_t`, `Delta_chi2`, PA vs 21.9, leftover-vs-velocity vs leftover-vs-uv, i frozen). 066 only. | no |
| G1 | JAX `predict_binned` on CPU (sky + Hann+bin + `chi2`; no host bounce through NUFFT). Identity: Stage A `chi2` within 1 of 168675.6. Tests CPU. Eval/s target vs S2 0.329. **Landed 2026-08-30:** `xla=True`, `chi2=168675.6`, 3.01 eval/s, tiny `jax.grad` vs FD. | no |
| G2 | Unconstrained chart + Jacobian on current Stage A names only. Do not add `i` or `h_z`. Do not logit the production floor as a prior. **Landed 2026-08-30:** `kinuv.infer.chart` log flux/gas_sigma/r_t, stable softplus `V_0`, identity PA/vsys/dx/dy; 8-vector JIT; official `r_t=0.5` finite `z`. | no |
| G3 | NumPyro NUTS. Receding init is official MAP PA **199.73°** (not L-BFGS seed 205.2°); approaching 25.2°. 4 chains. Label `sampler: nuts` only after autodiff. **Landed 2026-08-30:** JAX `U(z)`, frozen `(dx, dy)`, tiny-mock `sampler: nuts` (`R_hat<1.01`, ESS>200). 066 CPU NUTS skipped: projected 8.3 h > 2 h cap. No GPU. | no until 066 CPU smoke |
| G4 | Talts SBC on the exact kernel, `T=1`. Real-066 column with leftover caveat (not calibrated). | after G3 CPU smoke |
| G5 | PSIS-LOO Stage A vs B on **066 vis cells after Hann+bin**, only after NUTS. Not leave-one-galaxy. Hann correlation is a residual. MAP AIC preferring B is not a reason to skip JAX. | after G3 |
| later | user stubs: `h_z`, unfreeze `i`, warp/strip; TARGET subset | after models exist |
| Phase 5 | hierarchical pooling + selection function | after calibrated posteriors |

## Hard-target flags (066 now; quote rule later)

Software flags, not a dispatcher. Official 066 is expected to fire `r_t_at_floor` and `leftover_chi2_structured`. `Delta_chi2` vs V=0 will not fail. Do not quote inner `dV/dr` when `r_t` is on the floor. Do not treat Stage B rings as a warp.

Recommended user stubs (you add them): TARGET subset; optional `h_z`; unfreeze `i` with an optical prior; warp/strip/KDC classes.

## What we will not do

Call Laplace-MH NUTS. Quote S2 intervals as calibrated. Add `emcee`. Pool MAP point estimates. Restore Ico as intrinsic SB. Write a 400-galaxy runner. Provision GPU before a 066 CPU NUTS smoke. Unfreeze `i` or add `h_z` without a user DEC.
