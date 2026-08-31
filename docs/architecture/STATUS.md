---
generation: 4
phase: 066-12
code_freeze: false
next_role: implementer
board: accepted
build_licensed: true
pending: ["KGAS066-20260831T194009Z-nuts headless sd3ckpf2"]
last_propose: docs/reviews/2026-08-30-propose-g3-nuts.md
last_review: docs/reviews/2026-08-29-review-methodology.md
last_review_a: docs/reviews/2026-08-30-review-a-g3-nuts.md
last_review_b: docs/reviews/2026-08-30-review-b-g3-nuts.md
user_review: docs/reviews/artifacts/2026-08-30-final-fit/
open_questions: []
deadlocks: []
canon_generation: 4
---

## Agent Run Status

* **Phase:** G3 066 NUTS relaunched on CANFAR headless (DEC-067-RUNNER)
* **Last Action:** Session `sd3ckpf2` left Running. Stopped teeing tqdm onto `/arc`; watcher no longer appends full CANFAR dumps
* **Decisions Made:** Chain `npz` (kB draws) still scratch-then-/arc after each chain. JAX cache, vis, cubes, and per-sample stdout stay off `/arc`
* **Blockers / Gates:** Waiting on `.../kinuv_runs/KGAS066-latest/.trigger_complete`; mixing still `R_hat<1.01` and ESS>400 before `sampler: nuts` on 066 JSON
* **Next Step:** On sentinel, copy posteriors into `docs/reviews/artifacts/2026-08-30-g3-nuts/`, 6D corner, STATUS. Do not block this chat on the chain. Official MAP unchanged

# Architecture mailbox

**2026-08-31 (do not balloon /arc).** User: do not copy scratch onto `/arc` wholesale. Durable on `/arc` is status, `run.log`, overwrite-copied `worker.log`, and kB chain-draw `npz`. No JAX cache, vis, cubes, or per-sample tqdm tee. Watcher overwrites `canfar-*.txt`; does not append dumps into `platform.log`. Live `sd3ckpf2` kept Running (entrypoint already in flight). Official MAP unchanged. Do not start G4.

**2026-08-31 (checkpoints).** `on109zo9` died after chain 1 (`savez` appended `.npz` onto `.tmp`). Dual checkpoint scratch→`/arc` via file handle. Relaunch `sd3ckpf2` flexible `KGAS066-20260831T194009Z-nuts`. Official MAP unchanged. Do not start G4.

**2026-08-31 (flexible + dated runs).** User: 64 GB fights the scheduler; flexible grows to 32 GB. Run dir `{KGASID}-{YYYYMMDDTHHMMSSZ}-nuts` plus `KGAS066-latest`. Killed `ckhi0px1` (pinned 64 GB). Session `on109zo9` flexible. Official MAP unchanged. Do not start G4.

**2026-08-31 (project runs).** User: all work ends on `/arc/projects/KILOGAS/analysis/toby_sandbox`, not `$HOME`. Jobs run on ephemeral `/scratch` and checkpoint logs/draws to `toby_sandbox/kinuv_runs`. `m7pd3tib` Failed (1-chain stitch `(6,1)`); relaunch `ckhi0px1` 8 CPU / 64 GB. Official MAP unchanged. Do not start G4.

**2026-08-31 (job logs).** User: platform `canfar logs` expire in ~1 hour. Persist each job onto `/arc/home/thbrown/kinuv_runs/<run_id>/`. `h2dlc07f` vanished without a product (likely OOM under flexible ≤32 GB). Relaunch `m7pd3tib` 8 CPU / 64 GB, image `skaha/astroml:latest`, no `--gpu`. Official MAP unchanged. Do not start G4.

**2026-08-30 (DEC-067-RUNNER).** User: relax 7200 s interactive cap for batch; jobs > 15 min go to CANFAR headless. Session `h2dlc07f` Running, image `skaha/astroml:latest`, flexible CPU/RAM, no `--gpu` (recovery venv is CPU jax-finufft). Manifest: `/arc/home/thbrown/kinuv_runs/kgas066-nuts/`. 4×600 at MAP PA 199.73. Do not start G4. Official MAP unchanged.

**2026-08-30 (G3 executed).** Dual accept (major). JAX `U(z)=0.5(chi2+shift_prior_const)-log|J|`; sampled-name `float()` off the XLA vis path; `(dx, dy)` frozen at MAP host floats. Tiny-mock 4-chain NUTS mixed (`R_hat<1.01`, ESS>200) → `sampler: nuts`. 066 `jax.grad` 0.434 s; projected 29845 s > 7200 s cap → no 066 `sampler: nuts`, no GPU. numpyro 0.21.0 `--no-deps` keeps jax 0.11.1 / jax-finufft. Artifacts: `docs/reviews/artifacts/2026-08-30-g3-nuts/`. Receding init is MAP 199.73, not seed 205.2. Do not quote S2 16/50/84 or inner `dV/dr`. Official MAP unchanged. Do not start G4.

**2026-08-30 (G3 tally).** Dual accept (major): `review-a-g3-nuts` and `review-b-g3-nuts`. Execute: U not 2U; six-axis grad not flux-only; freeze (dx, dy) as host floats (no live shift_prior in U); 8-col draws; mixing on six names; pin numpyro without upgrading jax; `sampler: nuts` only after autodiff (066 only after mixing). No GPU. Official MAP unchanged.

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
