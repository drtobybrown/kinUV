# Senior-agent handoff (2026-08-30, after G1)

G1 JAX identity is landed. Do not treat this note as a license to run NUTS, logit the `r_t` floor, or corner S2 Laplace intervals.

## Last landed

- Official MAP (read-only): `/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS066/kinuv-KGAS066-uvsign-map/`
- Stage A: PA=199.73 deg, V0=267.7 km/s, `r_t` on 0.5 arcsec L-BFGS floor, `gas_sigma=12.05`, flux=70.46 Jy, `Delta_chi2` vs V=0 = +35553, `chi2=168675.6`, `s=0.5136`, XX 881x95, Hann+bin, `N=4`, `dv=5.080` km/s.
- Stage B: N=7, `lambda=0`, `chi2=167302` (Delta vs A = +1373). Quoted rotation curve stays Stage A arctan.
- S1: vis recovered inject `r_t=0.25`, `gas_sigma=8`, V0=250; CLEAN did not. Artifacts: `docs/reviews/artifacts/2026-08-29-s1-mock/`.
- S2: `sampler: laplace_mh` (not NUTS). Laplace SBC n=20 **failed** 68/95. Artifacts: `docs/reviews/artifacts/2026-08-29-s2/`.
- G0 flags: `kinuv.diagnostics.flags`. Official 066 fires `r_t_at_floor` and `leftover_chi2_structured`. PA=199.73 does not fire 21.9 alias.
- G1: `predict_binned(..., xla=True)` on CPU jax-finufft. Identity `|chi2-168675.6|<1`. Post-warmup 3.01 eval/s vs S2 0.329. Timing JSON: `docs/reviews/artifacts/2026-08-30-g1-jax/timing.json`.
- Human plots: `docs/reviews/artifacts/2026-08-30-final-fit/` (Stage A leftover + Stage B vs 10 km/s cube).

## Canon / process

Start: `AGENTS.md` → `field-guide/index.md` → `docs/architecture/STATUS.md` → `docs/reviews/BOARD.md`. Handshake: `DEC-066-AGENTS` (parent proposes; two independent reviewers; dual accept → execute). No new `DEC-*`. Do not overwrite the official MAP in place. Do not skip git hooks.

## Env defaults

- `JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`
- TMP / JAX cache: `/scratch/kinuv-$USER/$session` if writable, else `/tmp` (`docs/diagnostics/scratch.md`). Never high-frequency I/O on `/arc`.
- Operator: `kinuv.response.spectral.hann_then_bin`. `native_diagonal` raises.
- `NPZ_UV_SIGN=-1`. Ico east via `fits_image_east_north`. Frozen `i=43.86` deg. No `h_z`.
- Venv: `/arc/home/thbrown/kinuv-venv-recovery`. `PYTHONPATH=$PWD/src`. Branch `dev`. Push with `git -c credential.helper='!/arc/home/thbrown/.local/bin/gh auth git-credential'` (`/opt/conda/bin/gh` is missing).

## Forbidden (next agent)

- Label `laplace_mh` as NUTS. Quote S2 16/50/84 or Laplace 68/95 as calibrated.
- `plot_posterior_corner` on S2 JSON or interval tables. Plotter requires `sampler == "nuts"` and an 8-column draw array.
- G2 logit of `RT_BOUNDS_ARCSEC=(0.5, 15)` with MAP at 0.5 arcsec.
- G3 NumPyro / G4 SBC / G5 PSIS-LOO / GPU / 400-galaxy runner / unfreeze `i` / add `h_z` without a new dual-accepted propose.
- `DEC-HIER-SELFUNC` (Phase 5). Edit `DEC-066-TARGET`.

## Immediate next science card

G2 unconstrained chart on current Stage A names only (separate propose). Not this handoff. User plot surface remains `docs/reviews/artifacts/2026-08-30-final-fit/` until a later NUTS card.
