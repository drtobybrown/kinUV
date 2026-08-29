# 2026-08-29 S2 hybrid coverage

`sampler: laplace_mh` (not NUTS). Note: [`docs/diagnostics/s2-coverage.md`](../../../diagnostics/s2-coverage.md). Style: [`docs/diagnostics/plotting.md`](../../../diagnostics/plotting.md). Official MAP: `kinuv-KGAS066-uvsign-map`.

- `s2_mock_mcmc.json` — `R_hat`, `ESS`, eval/s, 68/95% intervals on the S1 inject
- `s2_sbc.json` — Laplace SBC hit rates (n=20; failed binomial 68/95)
- `s2_real_ci_table.json` / `.md` — unscaled vs `T_dof` vs `T_nvis`
