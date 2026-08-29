---
role: proposer
date: 2026-08-29
agent: canfar-066-methodology
canon_generation: 4
ids:
  - DEC-066-TARGET
  - DEC-066-VC
  - DEC-066-OSCMETRIC
  - DEC-066-INFER
  - DEC-066-POL
  - DEC-066-ZEROMODEL
  - DEC-066-SB
  - DEC-066-GRID
  - DEC-HIER-SELFUNC
verdict: propose
---

# Mission 10× is vis posteriors and sub-beam V_c, not NUFFT vs KinMS clouds

**Read after** `AGENTS.md` → field-guide → `STATUS.md`. Canon generation 4. No new `DEC-*` id. No ACK in this turn. No NUTS in this turn. Official MAP remains `kinuv-KGAS066-uvsign-map`.

## Scope

Existing ids only. This propose **defines the 10× metric** against the image-plane bottleneck (CLEAN covariance, beam smearing, missing flux near \(u=v=0\)) and **scores the 066 MAP** against it. It does not open TARGET, OSCMETRIC, baryons, or non-circular. Reviewer must challenge or ACK; user is the only one who may add stubs listed in residual risks.

## What changed / what was checked

User mission (2026-08-29): 10× is three axes — Science (exact vis inference, sub-beam \(\sigma\) and \({\rm d}V/{\rm d}r\), baryons/halo, asymmetric gas), Systems (\(10^5\)–\(10^6\) evals in minutes, no per-galaxy knobs), Pipeline (plug-in kinematics, differentiable HMC). Sample \(N\approx 400\). [`PLAN.md`](../PLAN.md) Phase-1 FINUFFT-vs-clouds numbers are engine hygiene, not that leap.

Generation-4 ADRs still lock **code** to KGAS066 ([DEC-066-TARGET](../decisions/DEC-066-TARGET.md)). Official vis MAP (2026-08-28, after CASA `NPZ_UV_SIGN=-1` and Ico `CDELT1<0` → +x east):

- Stage A: PA=199.73°, \(V_0=267.7\) km/s, \(r_t=0.5″\) **floor**, \(\sigma=12.05\) km/s, flux=70.46 Jy, \(\Delta\chi^2\) vs \(V=0\) = **+35553** (\(\chi^2=168676\), \(\chi^2_0=204228\)).
- Stage B: N=7, \(\lambda=0\), \(\chi^2=167302\) (\(\Delta\) vs A = +1373), AIC prefers B, `max_omega≈10.2`. User 2026-08-24: Ω 0.3 not a vis-fit veto. [OSCMETRIC](../decisions/DEC-066-OSCMETRIC.md) absolute Ω vs truth curvature remains a spec failure ([2026-08-21 propose](2026-08-21-propose-gate4-spec-failure.md)).
- Fit array: 881×95, \(N=4\), \(\Delta v=5.080\) km/s, \(s=0.514\). Eval ~0.45–0.6 s. XX-only ([DEC-066-POL](../decisions/DEC-066-POL.md) still unused).
- Image-plane check (not χ²): mask-sum M0 28480 vs data 27864 K km/s. Figures: `docs/reviews/artifacts/2026-08-28-stage-b-imaging/`.
- No NUTS. Mock recovery ([`tests/test_mock_recovery.py`](../../tests/test_mock_recovery.py)) exists with loose tolerances; may still use `native_diagonal`.

χ²/\(n\) ≈ \(167302/(881\times 95)\approx 2.0\) after empirical \(s\): leftover is **model misspecification**, not CLEAN.

## The 10× metric (formal)

Compare against **image-plane kinematic fitting** (3DBarolo / KinMS-on-CLEAN-cube / moment-1 rings), not against bilinear degridding. A survey claim needs all three axes. A 066 paper may claim Science 10× on **S1+S2 only**.

### Axis 1 — Science

- **S1 Sub-beam recovery.** On mocks that use **real 066 \((u,v,\nu)\)**, recover intrinsic `gas_sigma` and inner \({\rm d}V/{\rm d}r\) at \(r<0.5\,{\rm BMAJ}\) inside a pre-registered window (\(|\Delta V_0|<10\) km/s, \(|\Delta\sigma|<2\) km/s) while a 3DBarolo/KinMS fit to the **CLEANed mock cube** is biased by the restoring beam. **066 MAP does not show this:** \(r_t\) is on the Stage A 0.5″ bound; Stage B knots start at 0.65″.
- **S2 Exact Bayesian vis likelihood.** Posterior from ungridded visibilities (NUTS/HMC or calibrated Laplace); mock coverage ≥68% in ≥68% of draws. MAP \(\Delta\chi^2\) vs \(V=0\) is necessary and **not sufficient**. **Not started** ([DEC-066-INFER](../decisions/DEC-066-INFER.md) order: NUTS after MAP beats zero *and* mocks recover).
- **S3 Baryon vs DM.** Unbiased central \(V_c\) then stars+gas+halo. kinUV DR product is \(V_c(r)\); γ is post-hoc ([PLAN.md](../PLAN.md) Q4 deferred). **No claim from this MAP.**
- **S4 Diffuse / asymmetric.** Disk plus tail/extraplanar/outflow without CLEAN missing-flux bowls. Circular thin disk + frozen Wiener Ico only. **Blocked** until a user DEC.

### Axis 2 — Systems

- **Y1** \(\ge 10^5\) likelihood evals in **minutes** on one aggregated galaxy (NUTS-scale). CPU first; GPU is the same product faster.
- **Y2** MAP without hand-set \(\lambda\), PA box, or channel window (two-start PA, cube-trim, empirical \(s\)).
- **Y3** Cost 400 MAP nights vs 400 NUTS weeks after TARGET expands.

**Measured:** ~2 eval/s CPU. \(10^5\) evals ≈ 12–17 **hours**, not minutes. Systems 10× is eval/s × chain length, not FINUFFT vs clouds.

### Axis 3 — Pipeline

- **P1** One degridder; swap arctan / rings / later warp-bar.
- **P2** Differentiable operator (JAX) so HMC is not a rewrite. Production MAP is scipy L-BFGS + finite-difference Jacobian.
- **P3** Catalogue ingest (npz, Ico, cube window, \(b/a\), PA seed). Keep `NPZ_UV_SIGN=-1` and `fits_image_east_north`.

## Scorecard

| Gate | Mission 10× | 066 now |
|------|-------------|---------|
| S1 | Inner \({\rm d}V/{\rm d}r\) and \(\sigma\) inside 0.5 BMAJ, vis unbiased vs CLEAN-biased cube | Fail: \(r_t\) floor; no inject-vs-CLEAN table |
| S2 | Calibrated vis posteriors | Fail: MAP only |
| S3 | Halo vs stars at sub-kpc | Fail: deferred; do not quote γ |
| S4 | Tails/outflows in vis | Fail: circular + frozen Ico |
| Y1 | \(10^5\) evals in minutes | Fail: hours on CPU |
| Y2 | No per-galaxy knobs | Partial: two-start PA + cube-trim; 066 paths hardcoded; OSCMETRIC unusable as written |
| P1–P3 | Modular differentiable survey ingest | Fail: one galaxy, FD Jacobian |

**Verdict:** UV is the correct likelihood domain. The engine is no longer the limiter. **We have not produced science image-plane codes cannot.** Matching the 10 km/s cube (PA~200°, M0 within ~2%) is expected if the model is a rotating disk; it is not a UV-only win unless S1 shows the cube fit is biased where vis is not.

Explicitly **not** 10×: FINUFFT vs KinMS shot noise (already ~7× vs the 0.139 mJy coherent floor); χ² vs \(V=0\) on one galaxy; GPU as a substitute for S1.

## Rejected alternatives

- Call Phase-1 transform accuracy the mission 10× — contradicts the image-plane bottleneck (covariance, smearing, missing flux).
- Treat “model looks like the cube” as S1 — an image-plane fitter can match CLEAN too.
- Reopen the λ-grid inside OSCMETRIC — truth arctan already has \(\Omega>0.3\); densifying λ cannot pass recovery and Ω together.
- NUTS or 400-galaxy runner in this turn — INFER waits on S1 mock recovery on the **Hann+bin** path; TARGET forbids 007/452 until that mock and MAP-vs-zero (zero is passed).
- Create a `DEC-*` for residual-Ω, survey TARGET, baryons, or non-circular — field-guide stop; recommend only (below).

## What is executable after ACK (066 only)

1. **S1 mock:** inject steep inner \(V_c\) (\(r_t\) well below BMAJ) + known \(\sigma\) on real 066 uv, Hann+bin; fit vis Stage A; CLEAN the mock; 3DBarolo/KinMS-on-cube; table inner slope and \(\sigma\) vs truth. UV win = cube \(\sigma\) high / inner slope shallow, vis inside the window.
2. **Leftover χ²** of the real MAP vs baseline length and vs velocity. Spiral M0 leftover is SB misspecification, not CLEAN.
3. **NUTS on Stage A only** if S1 recovers ([DEC-066-INFER](../decisions/DEC-066-INFER.md)): \(\hat R<1.01\), ESS>200 on PA, vsys, flux. Report eval/s honestly (Y1 gap).
4. **POL:** XX+YY re-export if one job; document √2 if still deferred.

Do not put CLEAN in the likelihood. Do not mix 066-6/7/8 in one commit if code lands.

Survey-readiness (no runner): [`docs/diagnostics/survey-readiness.md`](../diagnostics/survey-readiness.md).

## Residual risks

1. S1 may **fail**: visibilities may not constrain \(r_t\) inside the Ico beam even with a correct kernel; then UV fitting does not beat CLEAN on the quantity that matters, and the circular+frozen-Ico model is the next limiter (χ²/\(n\)≈2).
2. Stage B λ=0 is unregularised rings (`max_omega≈10`). Using it as the S1 fitter without a user residual-Ω DEC repeats the OSCMETRIC conflict.
3. Mock recovery today may not be the Hann+bin operator; S1 is invalid if the inject uses `native_diagonal`.
4. XX-only is √2 sensitivity left on the table, not 10×, but it biases any “exact” posterior width.
5. Frozen Wiener Ico cannot represent spirals; vis χ² leftover can look like a kinematic failure when it is SB.
6. Opening N≈400, baryons, warps, or GPU without TARGET/S3/S4 stubs forks the architecture (DEC-066-AGENTS).
7. [`DEC-HIER-SELFUNC`](../decisions/DEC-HIER-SELFUNC.md) still deferred; population γ without it is selection-biased.

**User stubs if you want the rest of the mission (do not implement here):** residual-Ω (excess vs Stage A, not absolute Ω); TARGET expansion to a regular-disk Ico+npz subset; baryon mass-model boundary; optional second SB / non-circular component.

## STATUS updates required

- `next_role: reviewer`
- `pending: []`
- `last_propose:` this file
- Official product unchanged: `kinuv-KGAS066-uvsign-map`
- `deadlocks:` none
