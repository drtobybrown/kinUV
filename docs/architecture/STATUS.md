---
generation: 4
phase: 066-12
code_freeze: false
next_role: reviewer
pending: []
last_propose: docs/reviews/2026-08-29-propose-s2.md
last_review: docs/reviews/2026-08-29-review-methodology.md
open_questions: []
deadlocks: []
canon_generation: 4
---

# Architecture mailbox

**2026-08-29 (hygiene).** `native_diagonal` raises; vis SPECRESP / `NPZ_UV_SIGN` / Ico east / `chi2` stay in kinUV only (`docs/diagnostics/repos.md`). Standard leftover + slice plotters: `kinuv.diagnostics.figures` and `scripts/plot_fit_diagnostics.py`. Guide: `docs/diagnostics/plotting.md`. Changelog: `CHANGELOG.md`. Official MAP unchanged (`kinuv-KGAS066-uvsign-map`). No new DEC.

**2026-08-29 (S2 results).** Hybrid executed (`sampler: laplace_mh`, not NUTS). Mock MH: `R_hat` 1.000-1.004, `ESS` 876-1757, accept 0.613, eval/s 0.329. Laplace SBC n=20 fails binomial 68/95 (rate68 `v0_kms`/`r_t` = 0.10; `pa_deg` = 0.30). Real-066 `T_dof = 1.0077`, `T_nvis = 2.0154`; width ratios 1.004 and 1.420. Artifacts: `docs/reviews/artifacts/2026-08-29-s2/`. Note: `docs/diagnostics/s2-coverage.md`. Official MAP unchanged (`kinuv-KGAS066-uvsign-map`).

**2026-08-29 (S2).** User licensed hybrid coverage (not autodiff NUTS): `laplace_mh` on the S1 inject plus ~20 Laplace SBCs; real-066 CI table uses `T_dof = chi2 / (2 n_vis)` and sensitivity `T_nvis = chi2 / n_vis`. Propose: `docs/reviews/2026-08-29-propose-s2.md`. Stage A only. Official MAP unchanged (`kinuv-KGAS066-uvsign-map`).

**2026-08-29 (S1).** ACK executed. Vis Stage A recovered inject \(r_t=0.25″\), \(\sigma=8\) km/s (\(\Delta V_0=+0.19\), \(\Delta\sigma=-0.11\)) on Hann+bin 881×95; CLEAN-beam M1 slope 95 vs truth 237 km/s/″, M2 56 vs 8 km/s. 3DBarolo unavailable. Leftover \(\chi^2=168676\) matches the official MAP. Artifacts: `docs/reviews/artifacts/2026-08-29-s1-mock/`. Note: `docs/diagnostics/s1-mock.md`. No NUTS.

**2026-08-29 (review).** ACK with four mods: S1 must track \(i\) freeze / no \(h_z\) and PA–\(\sigma\) covariance; assert `Hann+bin` (`kinuv.response.spectral.hann_then_bin`) before mock vis; Stage A only (no \(\lambda=0\) rings); XX-only empirical \(s\) in the likelihood. Licensed: S1 mock + leftover \(\chi^2\). No NUTS. Review: `docs/reviews/2026-08-29-review-methodology.md`. Official MAP unchanged (`kinuv-KGAS066-uvsign-map`).

**2026-08-29 (propose).** Methodology propose (no new DEC): mission 10× is vis posteriors + sub-beam \(V_c\) vs CLEAN, not NUFFT vs KinMS clouds. Scorecard: 066 vis MAP is the right likelihood domain; S1/S2 not shown (`r_t` floor, no NUTS). Survey checklist only: `docs/diagnostics/survey-readiness.md`. Official MAP unchanged (`kinuv-KGAS066-uvsign-map`). No NUTS that turn.

**2026-08-28.** 180° image-plane PA was the CASA Fourier sign (`NPZ_UV_SIGN = -1`), not a disk-frame PA domain bug and not a conjugated npz export: FT of the WCS-true CLEAN cube onto the npz matches visibilities only with that sign. Ico `CDELT1<0` was placed mirrored on ImageGrid; `fits_image_east_north` flips NAXIS1 before Wiener. Official MAP `kinuv-KGAS066-uvsign-map`: Stage A PA=199.73°, V₀=267.7 km/s, Δχ² vs V=0 = **+35553** (χ²=168676); Stage B N=7 λ=0 χ²=167302 (Δ vs A = +1373), AIC prefers B. Keep `f47bc9-map` (PA=21.9°) as the pre-sign vis-winner. Figures: `docs/reviews/artifacts/2026-08-28-stage-b-imaging/`. No NUTS. No new DEC id.

**2026-08-27.** Image-plane Stage B vs 10 km/s cube: `docs/diagnostics/stage-b-vs-imaging.md`, runner `scripts/plot_stage_b_vs_imaging.py`, figures `docs/reviews/artifacts/2026-08-27-stage-b-imaging/` (inverted PA; superseded 2026-08-28). Not a new fit. No NUTS. No new DEC id.

**2026-08-24.** User: best vis fit; Ω 0.3 not a veto. **Stage B on real 066 is done** (`kinuv-KGAS066-f47bc9-map/stage_b_map.json`): N=7, λ=0, χ²=176879 vs Stage A 178016 (Δ=+1136), AIC prefers B. Grid: `.../lambda-resid/vis_fit/`. Stage A MAP kept as the arctan product. No NUTS. No new DEC id.

Native preview: `⟨w|V|²⟩≈2.59`, `s≈0.77`. **Fit array (066-6):** `n_row=881`, `n_chan=95`, `Δv=5.080 km/s`, `N=4`, `s=0.514`. Replica was 881×125 (wider buffer).

## 066 npz (local inventory)

- Local: `/Users/thbrown/kilogas/DR1/visibilities/KILOGAS066.npz` (native 43240×1920)
- CANFAR: `/arc/projects/KILOGAS/analysis/toby_sandbox/visibilities/KILOGAS066.npz`
- Ico / vis-trim: `/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/` (`KGAS66_Ico_K_kms-1.fits`, sibling clipped cube). Laptop `kinms_test` path is absent on `/arc`.
- Image-plane Stage B diagnostics: `.../KGAS66/10kms/` (`KGAS66_clipped_cube.fits`, `KGAS66_mask_cube.fits`).
- YAML `obs_freq_range` clips the receding side — do not use it as the trim
