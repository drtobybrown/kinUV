---
generation: 4
phase: 066-12
code_freeze: false
next_role: board
board: open
build_licensed: true
pending: []
last_propose: docs/reviews/2026-08-30-propose-g3-nuts.md
last_review: docs/reviews/2026-08-29-review-methodology.md
last_review_a: docs/reviews/2026-08-30-review-a-g2-chart.md
last_review_b: docs/reviews/2026-08-30-review-b-g2-chart.md
user_review: docs/reviews/artifacts/2026-08-30-final-fit/
open_questions: []
deadlocks: []
canon_generation: 4
---

## Agent Run Status

* **Phase:** G2 unconstrained chart landed on origin/dev (ee459af); G3 NumPyro NUTS proposed, board open
* **Last Action:** Re-verified `tests/test_g2_chart.py` (17 passed): official Stage A `|chi2-168675.6|<1` after chart roundtrip; finite z at `r_t=0.5`; no NumPyro in `chart.py`; `SAMPLER_NAME` stays `laplace_mh`. G3 propose written: `docs/reviews/2026-08-30-propose-g3-nuts.md`
* **Decisions Made:** No new G2 commit this turn; official MAP `kinuv-KGAS066-uvsign-map` unchanged; no NUTS run yet; no GPU; do not logit `RT_BOUNDS_ARCSEC=(0.5, 15)`; frozen i; no `h_z`
* **Blockers / Gates:** G3 execution blocked until dual review of `2026-08-30-propose-g3-nuts.md`; do not start NUTS or GPU this card
* **Next Step:** Dual review of G3 NumPyro NUTS propose — not executing NUTS

# Architecture mailbox

**2026-08-30 (G3 propose).** Autodiff `chi2(θ(z))` + CPU NumPyro NUTS. G2 already on `origin/dev` (`ee459af`, 17/17 this turn). Freeze `(dx, dy)` at MAP. Two PA runs (199.73 and 25.2), 4 chains. `sampler: nuts` only after autodiff. No logit of `[0.5, 15]`. No GPU. Official MAP unchanged. Propose: `docs/reviews/2026-08-30-propose-g3-nuts.md`.

**2026-08-30 (G2 verified).** Already executed (`ee459af`). Re-ran `tests/test_g2_chart.py`: 17 passed, including official `|chi2-168675.6|<1`. Host `log_prob_unconstrained` is not autodiff. Do not re-land the chart. Official MAP unchanged.

**2026-08-30 (spectral vsys).** User-directed: diagnose Stage B vs 10 km/s redshift. Root cause is vis-weighted MAP \(V_{\rm sys}\) (optical ~8323.6 km/s) vs CLEAN/catalogue (~8299.6), not radio/optical, `RESTFRQ`, `CRPIX3`, or Hann phase. Tests lock Hann impulse + m/s axis + rebin delta. Write path `SPECSYS=LSRK`. Spectra annotate \(\Delta v_{\rm M-D}\`. No silent fudge. Official MAP unchanged.

**2026-08-30 (G2 executed).** Dual accept (major). `kinuv.infer.chart` 8-vector log/softplus/identity maps; `jax.jit` type preservation; per-axis FD Jacobian; official `|chi2-168675.6|<1` after roundtrip. `log_prob_unconstrained` is host-only. No NumPyro, no NUTS label. Official MAP unchanged.

**2026-08-30 (G2 tally).** Dual accept (major): `review-a-g2-chart` and `review-b-g2-chart`. Execute: both-arm-finite softplus (no Python `if`); 8-vector JIT path; per-axis FD of `unconstrained_to_physical` (not chi2); no logit of `RT_BOUNDS`. Host `log_prob_unconstrained` is not autodiff. Do not start G3. Official MAP unchanged.

**2026-08-30 (G2 propose).** Unconstrained Stage A chart + Jacobian. log flux/gas_sigma/r_t; stable softplus V_0; identity PA/vsys/dx/dy. Do not logit `RT_BOUNDS_ARCSEC=(0.5, 15)`. No NumPyro, no NUTS label. Propose: `docs/reviews/2026-08-30-propose-g2-chart.md`. Official MAP unchanged.

**2026-08-30 (ops executed).** Dual accept (major) on `2026-08-30-propose-ops-scratch`. Scratch policy `docs/diagnostics/scratch.md`. Corner plotter refuses `laplace_mh` and S2 interval tables. Handoff: `docs/reviews/2026-08-30-handoff-senior.md`. No NUTS, no G2, official MAP unchanged.

**2026-08-30 (ops tally).** Dual accept (major): `review-a-ops-scratch` and `review-b-ops-scratch`. Provenance gate on corners; Composer edits only Agent Run Status; no vis checkpoints.

**2026-08-30 (ops propose).** User asked `/scratch` I/O, Composer 2.5 STATUS push, posterior corners, senior handoff. Propose: `docs/reviews/2026-08-30-propose-ops-scratch.md`. No NUTS; no S2 Laplace corners as 16/50/84 product. Official MAP unchanged.

**2026-08-30 (G1 executed).** Dual accept (major). JAX `predict_binned(..., xla=True)` stays on device through NUFFT/Hann/`chi2`. Official Stage A `chi2=168675.6` (same `s=0.5136`). Post-warmup 3.01 eval/s vs S2 FD 0.329. Tiny `jax.grad` vs FD. Timing: `docs/reviews/artifacts/2026-08-30-g1-jax/timing.json`. No G2/G3/GPU. Official MAP unchanged.

**2026-08-30 (G1 tally).** Dual accept (major): `review-a-g1-jax` and `review-b-g1-jax`. Execute: XLA sky+NUFFT+Hann+`chi2` (no host bounce); tiny `jax.grad` vs FD; frozen `s`; x64; `/tmp` cache; official `|chi2-168675.6|<1` when npz exists. Do not start G2/G3/GPU. Official MAP unchanged.

**2026-08-30 (G1 propose).** User gold-standard/hygiene/GPU dump mapped onto G1 only. Propose: `docs/reviews/2026-08-30-propose-g1-jax.md`. ntfy: `kinuv_canfar_agent_thbrown`. Official MAP unchanged.

**2026-08-30 (G0 executed).** Dual accept (major) on `2026-08-30-propose-gold-standard`. `kinuv.diagnostics.flags.map_quality_flags`: leftover-vs-velocity vs leftover-vs-uv, `r_t_at_floor`, PA vs 21.9, `beats_zero`. Official 066 fires leftover structure and the `r_t` floor. Roadmap rewritten as the 066 kernel sequence. Methodology + survey-readiness point at it. No JAX / NUTS / GPU this card. Next wave is G1 CPU JAX `predict_binned` (separate propose). Official MAP unchanged.

**2026-08-30 (gold-standard tally).** Dual accept (major): `review-a` and `review-b` on `2026-08-30-propose-gold-standard`. Execute G0 flags + rewrite roadmap. Do not start G1.

**2026-08-30 (gold-standard propose).** Architect sequence: no fake NUTS; JAX likelihood then NumPyro then Talts SBC; GPU after CPU NUTS smoke; hierarchical and 400-galaxy runner deferred; hard targets get flags until user stubs. Propose: `docs/reviews/2026-08-30-propose-gold-standard.md`. Official MAP unchanged.

**2026-08-30 (tally).** Dual accept (major): `review-a` and `review-b` on `2026-08-30-propose-final-plots`. Execute plots only from official Stage A/B MAP; no refit. Official MAP read-only.

**2026-08-30 (gates).** User: not babysitting; implementer decides each gate; human reviews **final fit plots** only. Physics stops are judgment, not a user ACK. Official MAP unchanged.

**2026-08-30 (handshake).** User: methodology is good; relax stage stops. Parent proposes; two independent sub-agents accept/reject (major/minor) on `docs/reviews/`. Dual accept → implement/execute with no third review. User **build** runs that loop through licensed stages. Human science: `docs/methodology.md`. Board: `docs/reviews/BOARD.md`. Amended `DEC-066-AGENTS`. `build_licensed: true` (user reviews plots at the end). Official MAP unchanged (`kinuv-KGAS066-uvsign-map`).

**2026-08-29 (hygiene).** `native_diagonal` raises; vis SPECRESP / `NPZ_UV_SIGN` / Ico east / `chi2` stay in kinUV only (`docs/diagnostics/repos.md`). Standard leftover + slice plotters: `kinuv.diagnostics.figures` and `scripts/plot_fit_diagnostics.py`. Guide: `docs/diagnostics/plotting.md`. Changelog: `CHANGELOG.md`. Official MAP unchanged (`kinuv-KGAS066-uvsign-map`). No new DEC.

**2026-08-29 (S2 results).** Hybrid executed (`sampler: laplace_mh`, not NUTS). Mock MH: `R_hat` 1.000-1.004, `ESS` 876-1757, accept 0.613, eval/s 0.329. Laplace SBC n=20 fails binomial 68/95 (rate68 `v0_kms`/`r_t` = 0.10; `pa_deg` = 0.30). Real-066 `T_dof = 1.0077`, `T_nvis = 2.0154`; width ratios 1.004 and 1.420. Artifacts: `docs/reviews/artifacts/2026-08-29-s2/`. Note: `docs/diagnostics/s2-coverage.md`. Official MAP unchanged (`kinuv-KGAS066-uvsign-map`).

**2026-08-29 (S2).** User licensed hybrid coverage (not autodiff NUTS): `laplace_mh` on the S1 inject plus ~20 Laplace SBCs; real-066 CI table uses `T_dof = chi2 / (2 n_vis)` and sensitivity `T_nvis = chi2 / n_vis`. Propose: `docs/reviews/2026-08-29-propose-s2.md`. Stage A only. Official MAP unchanged (`kinuv-KGAS066-uvsign-map`).

**2026-08-29 (S1).** ACK executed. Vis Stage A recovered inject `r_t=0.25` arcsec, `gas_sigma=8` km/s on Hann+bin 881x95; CLEAN-beam M1 slope 95 vs truth 237 km/s/arcsec, M2 56 vs 8. Leftover `chi2=168676` matches the official MAP. Artifacts: `docs/reviews/artifacts/2026-08-29-s1-mock/`. Note: `docs/diagnostics/s1-mock.md`. No NUTS.

**2026-08-29 (review).** ACK with four mods (Hann+bin assert, i freeze / no `h_z`, Stage A only, XX `s`). Review: `docs/reviews/2026-08-29-review-methodology.md`. Official MAP unchanged.

**2026-08-28.** Official MAP `kinuv-KGAS066-uvsign-map`: Stage A PA=199.73 deg, V0=267.7 km/s, Delta_chi2 vs V=0 = +35553 (`chi2=168676`); Stage B N=7 lambda=0 `chi2=167302` (Delta vs A = +1373). Figures: `docs/reviews/artifacts/2026-08-28-stage-b-imaging/`. Keep `f47bc9-map` (PA=21.9 deg) as the pre-sign vis-winner.

Native preview: `s≈0.77`. **Fit array (066-6):** `n_row=881`, `n_chan=95`, `dv=5.080 km/s`, `N=4`, `s=0.514`.

## 066 npz (local inventory)

- Local: `/Users/thbrown/kilogas/DR1/visibilities/KILOGAS066.npz` (native 43240×1920)
- CANFAR: `/arc/projects/KILOGAS/analysis/toby_sandbox/visibilities/KILOGAS066.npz`
- Ico / vis-trim: `/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/`
- Image-plane Stage B diagnostics: `.../KGAS66/10kms/`
- YAML `obs_freq_range` clips the receding side — do not use it as the trim
