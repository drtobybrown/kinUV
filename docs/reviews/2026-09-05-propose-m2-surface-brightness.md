---
role: proposer
date: 2026-09-05
agent: parent
canon_generation: 4
ids:
  - DEC-066-SB
  - DEC-066-VIS
  - DEC-066-AGENTS
  - DEC-066-INC
  - DEC-066-PA
  - DEC-066-VC
  - DEC-066-ZEROMODEL
verdict: propose
recommended_new_id: DEC-066-SB-m2
---

# Visibility-native m=2 SB on axisymmetrised 30 km/s Ico

## Scope

Existing DEC ids only. **No new `DEC-*` file this card.** Recommend `DEC-066-SB-m2` for the user to stub if they want it in the INDEX. Until then m=2 is an **opt-in** path in [`src/kinuv/forward/sb.py`](../../src/kinuv/forward/sb.py). Do **not** change `load_sb_template` default, `BMAJ_ICO_ARCSEC`, or [`DEC-066-SB.md`](../decisions/DEC-066-SB.md). Official MAP `kinuv-KGAS066-uvsign-map` stays read-only. `DEC-066-INC` stays frozen. \(V_c(R)\) stays axisymmetric (`DEC-066-VC`). Visibility window stays the 30 km/s vis cube (`DEC-066-VIS`).

User licensed **m=2 surface brightness only**. Do not unfreeze \(i\). Do not start a new 066 NUTS. Do not start G4. Do not interrupt 007 `b1mqxsov` `xkytxih1` `y5tspgit` `zq1olquy`. `quote_inner_slope: false`. Do not push `main`.

\[
I(R,\phi)=I_0(R)\bigl[1+a_2(R)\cos\bigl(2(\phi-\phi_2)\bigr)\bigr],
\quad a_2(R)=A\exp\bigl(-(R-2.5'')^2/(2\times(1.5'')^2)\bigr).
\]

\(I_0(R)\) is the **azimuthal average** of the official 30 km/s Ico Wiener template (not the full 2-D Ico, not exponential). Free: \(A\in[0,0.8]\), \(\phi_2\in[0,\pi)\). Frozen: \(R_2=2.5''\), \(\sigma=1.5''\).

## What changed / what was checked

- Leftover gate on 066 is **SB-dominated**. Official 2-D Ico is already non-axisymmetric. Axisymmetrising it **discards** that structure, so \(A=0\) will have \(\chi^2\) worse than 168675.6. The m=2 family does **not** contain the official template.
- `los_velocity` uses `cos θ = xg/R` after `sky_to_galaxy`. Disk \(\phi=\mathrm{atan2}(y_g,x_g)\) matches that \(x_g\). \(\phi=0\) is the receding major axis.
- Production `sky_cube` is `flux × I_template × Gaussian(v−v_los)`. m=2 multiplies the spatial template only.

## Rejected alternatives

- Full 2-D Ico × m=2 (double-counts CLEAN/arm structure).
- Exponential \(I_0\) (leaves DEC-066-SB).
- Constant \(a_2(R)\) (no central-5″ node).
- Free \(R_2,\sigma\), unfreezing \(i\), non-axisymmetric \(V_c\), 066 NUTS.
- Single-start L-BFGS on periodic \(\phi_2\).
- Writing `docs/decisions/DEC-066-SB-m2.md` (user only).

## Residual risks

1. \(A=0\) is not the official MAP. Report three \(\chi^2\): (1) 2-D Ico identity, (2) \(A=0\), (3) best \((A,\phi_2)\). (3)−(2) is the m=2 gain; (3)−(1) is leftover vs locked SB. Not a nested 3σ test. S2 SBC failed.
2. Morphology can still leak into M1 via spatial weighting of \(v_{\mathrm{los}}\). M1 invariance is a named gate, not assumed.
3. 720 `predict_binned` evals on 881×95. Wall-clock hours if serial; implementer may JAX/batch but must not change the likelihood.
4. Outer \(I_0(R)\) can still be mask-limited. Floor: drop annuli with \(n_{\mathrm{finite}}<8\); hold last finite mean.
5. 007 chains still running. Do not steal `KGAS066-latest`.

## Execute if accepted

1. Opt-in `axisymmetrise_template` and `apply_m2` in `sb.py`. \(\phi=\mathrm{atan2}(y_g,x_g)\) after `sky_to_galaxy` at MAP PA 199.73° and \(i=\arccos(0.721)\). No `atan2(east, north)`. Finite/positive pixels only in elliptical annuli; \(n<8\) dropped; last finite mean held outside. Renormalise \(\int I\,d\Omega=1\).
2. Tests (no FITS): \(A=0\) equals \(I_0\) to \(10^{-6}\); \(A>0\) centroid move \(<0.01''\); \(I\ge 0\) for \(A\le 0.8\); \(\phi_2\) and \(\phi_2+\pi\) identical.
3. Identity: official 2-D Ico at MAP \(\theta\) still \(|\chi^2-168675.6|<1\) on 881×95, \(s=0.5136098555284736\). Fail → STATUS, do not quote m=2 \(\Delta\chi^2\).
4. Script `scripts/analysis/fit_m2_sb_at_map.py`. Grid 36 × \(\phi_2\in[0,\pi)\) by 20 × \(A\in[0,0.8]\), then L-BFGS-B polish with \(\phi_2\) wrapped. If polish loses, keep the grid point. Artifacts `docs/reviews/artifacts/2026-09-05-kgas066-m2-sb/`. Prefer JSON+PNG; new tree `kinuv-KGAS066-m2-map` only if a refit JSON is written (kinematics copied, not overwritten official).
5. Plots: leftover χ² for (1)(2)(3); M0 D/M/R in the central 5″ (K km/s); M1 invariance (median \(|\Delta M1|\) vs M1 error). If M1 fails, STATUS one-liner, do not claim decoupling.
6. Commit/push `origin/dev` after propose, tally, execute. Do not merge to `main`. Official MAP unchanged. No G4.

## STATUS updates required

- `next_role: board`
- `board: open`
- `last_propose:` this file
- Keep `pending: ["b1mqxsov", "xkytxih1", "y5tspgit", "zq1olquy"]`
