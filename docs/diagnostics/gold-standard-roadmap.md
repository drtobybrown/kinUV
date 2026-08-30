# Gold-standard roadmap (human)

Architect sequence after S1/S2. Official 066 product remains **`kinuv-KGAS066-uvsign-map`**. This is not a 400-galaxy runner. Propose: [`docs/reviews/2026-08-30-propose-gold-standard.md`](../reviews/2026-08-30-propose-gold-standard.md).

## What is already true

- Vis Stage A beats V=0 (`Delta_chi2` = +35553). S1 recovered a sub-beam inject on Hann+bin; the CLEANed cube did not.
- There is **no NUTS posterior**. S2 was Laplace-preconditioned MH. Laplace 68/95% intervals **failed** simulation-based calibration on the exact mock kernel.
- Leftover vis `chi2` vs velocity is structured (spirals in a frozen Wiener Ico). That is model misspecification, not a noise-scale bug (`T_dof ~ 1`).

## What "gold standard" actually requires

Talts et al. (2018) SBC of credible intervals, HMC-friendly unconstrained parameters, non-centered funnels, and PSIS-LOO all need **posterior draws from the real likelihood with autodiff**. That means a JAX `predict_binned` (sky + Hann+bin + `chi2`), then NumPyro NUTS, **then** SBC. GPU on CANFAR helps only after that likelihood exists. Finite-difference "NUTS" on GPU is rejected.

## Sequence (066 first)

| Wave | What | GPU? |
|---|---|---|
| G0 | MAP quality flags (floor `r_t`, `Delta_chi2`, PA alias) | no |
| G1 | JAX likelihood on CPU; tests stay CPU | no |
| G2 | logit/softplus chart + Jacobian | no |
| G3 | NumPyro NUTS, 4 chains, two PA starts; `R_hat` / `ESS` | no until smoke passes |
| G4 | Talts SBC on mocks (`T=1`); real-066 column with leftover caveat | maybe |
| G5 | PSIS-LOO Stage A vs B | after G3 |
| later | user stubs: `h_z`, unfreeze `i`, warp/strip; then a TARGET subset | after models exist |
| Phase 5 | hierarchical pooling + selection function (`DEC-HIER-SELFUNC`) | after many real posteriors |

## Hard galaxies (the 400)

066 is the easy disk. Compact, faint, edge-on, face-on, tails, warps, and decoupled cores need **different physics**, not a faster 066 sampler. Until you add DEC stubs, the pipeline should **flag** those cases (do not quote a thin circular arctan as the answer) rather than silently generalize.

Recommended stubs (you add them; agents will not): TARGET subset; optional `h_z`; unfreeze `i` with an optical prior; warp/strip/KDC classes.

## What we will not do

Call Laplace-MH NUTS. Quote S2 intervals as calibrated. Add `emcee`. Pool 400 MAP point estimates and call it hierarchical Bayes. Restore Ico as the intrinsic SB to soak leftover. Write a 400-galaxy runner under the current TARGET lock.
