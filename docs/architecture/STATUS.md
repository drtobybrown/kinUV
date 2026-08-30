---
generation: 4
phase: 066-12
code_freeze: false
next_role: proposer
board: accepted
build_licensed: true
pending: []
last_propose: docs/reviews/2026-08-30-propose-g1-jax.md
last_review: docs/reviews/2026-08-29-review-methodology.md
last_review_a: docs/reviews/2026-08-30-review-a-g1-jax.md
last_review_b: docs/reviews/2026-08-30-review-b-g1-jax.md
user_review: docs/reviews/artifacts/2026-08-30-final-fit/
open_questions: []
deadlocks: []
canon_generation: 4
---

## Agent Run Status

* **Phase:** G1 CPU JAX `predict_binned` (landed)
* **Last Action:** Dual accept (major) executed; identity `chi2=168675.6`; 3.01 eval/s vs S2 0.329
* **Decisions Made:** XLA sky+NUFFT+Hann+`chi2`; NumPy identity kept; no GPU/NUTS/G2
* **Blockers / Gates:** G1 gates passed (`jax-finufft`, x64, tiny `jax.grad` vs FD)
* **Next Step:** G2 unconstrained chart (separate propose); do not logit the 0.5" `r_t` floor

# Architecture mailbox

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
