# Changelog

## 2026-08-30 — G1 CPU JAX `predict_binned`

Dual accept (major). `predict_binned(..., xla=True)` keeps JAX arrays from sky through NUFFT, Hann+bin, and `chi2`. NumPy path stays the identity reference. Official Stage A `chi2` matches 168675.6 at frozen `s=0.5136`. Post-warmup 3.01 eval/s vs S2 FD 0.329. Tiny-grid `jax.grad` vs FD. Tests set CPU/x64/`/tmp` cache. No NUTS, no GPU, no G2 logit of the `r_t` floor. Official MAP unchanged.

## 2026-08-30 — G0 MAP quality flags (066 kernel)

Dual accept (major) on the gold-standard sequence. Flags live in `kinuv.diagnostics.flags` (not `infer/`). Official 066 fires `r_t_at_floor` and leftover-vs-velocity vs leftover-vs-uv; PA=199.73 does not fire the 21.9 alias; `Delta_chi2` = +35553 records `beats_zero`. Roadmap rewritten as the 066 kernel sequence. No JAX, no NUTS, no GPU, no 400-galaxy runner. Official MAP unchanged.

## 2026-08-30 — Final-fit plot handoff

Dual accept (major). Regenerated Stage A leftover + Stage B D/M/R into `docs/reviews/artifacts/2026-08-30-final-fit/`. Leftover `chi2` = 168675.6. Official MAP not written. Matched cube stays in the artifact dir (not committed).

## 2026-08-30 — User reviews final plots; implementer owns gates

Physics stops are no longer user-blocking. The implementer decides each gate, notes it on STATUS, and continues. The human review surface is the final Data | Model | Residual + leftover `chi2` folder. Still forbidden: new `DEC-*`, in-place overwrite of `kinuv-KGAS066-uvsign-map`, calling Laplace-MH NUTS. Amended `DEC-066-AGENTS`.

## 2026-08-30 — Dual-review board; human methodology

User-directed handshake: the parent proposes; two independent reviewer sub-agents accept or reject (major/minor) on `docs/reviews/`. Dual accept → implement and execute the named stages with no third review. A user **build** command runs that loop end-to-end. Docs: `docs/methodology.md`, `docs/reviews/BOARD.md`, amended `DEC-066-AGENTS`. `build_licensed` stays false until that command. Official MAP unchanged. No new `DEC-*`.

## 2026-08-29 — S1/S2 land; kernel hygiene; diagnostic suite

### Production pointer

Official Stage A/B MAP remains `kinuv-KGAS066-uvsign-map`. S1 benchmark: `docs/reviews/artifacts/2026-08-29-s1-mock/`. S2 hybrid coverage: `docs/reviews/artifacts/2026-08-29-s2/` (`sampler: laplace_mh`, not NUTS). Image-plane check: `docs/reviews/artifacts/2026-08-28-stage-b-imaging/` (the 2026-08-27 folder is the pre-sign inverted-PA set).

### Deprecated kernels

- `native_diagonal` is gone from the mock and MAP paths. `kinuv.response.spectral.native_diagonal` now raises and points at `hann_then_bin`.
- `kinuv.likelihood` does not export `hann_then_bin` (that miss used to skip SPECRESP).
- Gate 2 (`tests/test_mock_recovery.py`) and S1 assert `kinuv.response.spectral.hann_then_bin` before generating vis.

### kinUV vs uvkin

Canonical Hann+bin, `NPZ_UV_SIGN=-1`, `fits_image_east_north`, and vis `chi2` live only in kinUV. uvkin is the KinMS/emcee science-matrix repo; see `docs/diagnostics/repos.md`. Tests already forbid uvkin imports in SPECRESP/vis/chi2 sources.

### Diagnostic plotters

- `kinuv.diagnostics.figures`: leftover `chi2` vs uv/velocity and PA/`gas_sigma`/`r_t` slices. ASCII labels. Style via `kinuv.diagnostics.style` (not viridis).
- `scripts/plot_fit_diagnostics.py` runs leftover (and optional imaging D/M/R). Preview dir `docs/reviews/artifacts/fit-diagnostics/` is gitignored.
- Guide: `docs/diagnostics/plotting.md`.

### What was not deleted

Stage A/B runners, `infer/campaign.py` (lambda-reg), historical reviews that mention `native_diagonal`, and the superseded 2026-08-27 imaging PNGs stay. No new `DEC-*`. No uvkin merge.
