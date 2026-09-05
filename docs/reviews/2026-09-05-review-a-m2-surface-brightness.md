---
role: reviewer
seat: a
date: 2026-09-05
agent: review-a
canon_generation: 4
ids:
  - DEC-066-SB
  - DEC-066-VIS
  - DEC-066-AGENTS
  - DEC-066-INC
  - DEC-066-PA
  - DEC-066-VC
  - DEC-066-ZEROMODEL
verdict: accept
severity: major
propose: docs/reviews/2026-09-05-propose-m2-surface-brightness.md
---

# Review a: visibility-native m=2 SB on axisymmetrised 30 km/s Ico

Do not read the other seat's review file. Do not implement.

Scope check: existing DEC ids only. No `DEC-066-SB-m2.md`. Do not edit `DEC-066-SB.md`. `load_sb_template` signature and default stay the official 2-D Wiener Ico. `BMAJ_ICO_ARCSEC = 1.30` stays. Official MAP `kinuv-KGAS066-uvsign-map` stays read-only. `DEC-066-INC` stays frozen. \(V_c(R)\) stays axisymmetric (`DEC-066-VC`). Visibility window stays `DEC-066-VIS`. No new 066 NUTS. No G4. Do not interrupt 007 `b1mqxsov` `xkytxih1` `y5tspgit` `zq1olquy`. Do not steal `KGAS066-latest`. `quote_inner_slope: false`. Leftover gate stays **SB-dominated**.

Execute-as-typed cannot steal the official MAP, write a DEC file, unfreeze \(i\), start G4/NUTS, or touch the four 007 sessions **if** `load_sb_template` is not given \(A,\phi_2\) kwargs, identity is the unmodified 2-D Ico, and `kinuv-KGAS066-m2-map` is never pointed as official (Comments 1 and 5). Accept + major: those comments are execute obligations, not optional polish.

## Attacks / bounds

1. **The three \(\chi^2\) numbers stop a false nested test. They do not stop a false MAP.** Propose residual 1 is right that \(A=0\) is not official: `sky_cube` multiplies the full 2-D Wiener stamp (`model.py`), and axisymmetrising discards CLEAN/arm structure the official MAP was fit against. \(\chi^2_{A=0}\) must be worse than 168675.6. The m=2 family does not contain the official template. DEC-066-ZEROMODEL’s \(\Delta\chi^2\) is vs \(V=0\), no parameters. S2 Laplace SBC failed 68/95. Leftover SB already moves official \(\chi^2\) by **1373** (Stage B 167302 vs Stage A 168676) and NUTS-mean vs MAP by **1189**. A 1-dof \(\chi^2_1\) tail of 9 is not this card.

   Naming only (3)−(2) and (3)−(1) hides the axisymmetrisation tax (2)−(1). If (3)<(1) at frozen \(\theta\), that is a leftover note, not a license to replace production SB or write a competing MAP. Execute 4’s optional tree `kinuv-KGAS066-m2-map` is the over-read: this card does not refit kinematics.

   **Bound:** Report all three \(\chi^2\) and **three** deltas: (2)−(1) axisymmetrisation tax, (3)−(2) m=2 gain, (3)−(1) leftover vs locked 2-D Ico. Print them next to leftover 1373 and vs-\(V=0\) +35553. Do not print “3σ” or “nested”. Do not call (3) a MAP. If (3)<(1): STATUS one-liner “m=2 beat 2-D Ico at frozen \(\theta\)”; still do not change `load_sb_template` or the official tree. Identity `|chi2-168675.6|<1` on unmodified 2-D Ico; fail → do not quote any m=2 \(\Delta\chi^2\) and do not write `kinuv-KGAS066-m2-map`.

2. **Execute prose locks \(\phi=\mathrm{atan2}(y_g,x_g)\). The named tests do not.** `los_velocity` sets \(\cos\theta=x_g/R\) after `sky_to_galaxy` (`model.py` 62–73). `rotate_by_pa` puts +x on the receding major axis; `test_receding_major_axis_is_redshifted` already locks that sky point. \(\phi=\mathrm{atan2}(y_g,x_g)\) is the unique two-argument angle with \(\cos\phi=x_g/R\), so \(\phi=0\) is the receding major axis. A constant \(\phi\) offset is absorbed by free \(\phi_2\). Sky \(\mathrm{atan2}(\mathrm{east},\mathrm{north})\) is **not** a constant offset: `incline` stretches the minor axis by \(1/\cos i\approx 1.387\) at \(i=\arccos(0.721)\). Propose tests (\(A=0\), centroid, \(I\ge 0\), \(\phi_2\sim\phi_2+\pi\)) all pass under sky \(\phi\).

   **Bound (missing test, no FITS):** `apply_m2` must call `sky_to_galaxy` at PA \(199.73^\circ\) and \(i=\arccos(0.721)\), then \(\phi=\mathrm{atan2}(y_g,x_g)\). Pixel \((x_g,y_g)=(1,0)\) has \(\phi=0\); \((0,1)\) has \(\phi=\pi/2\). No `atan2(east, north)`. Synthetic exponential \(I_0\), \(A=0.5\), \(\phi_2=0\): after minimizing \(\phi_{2,\mathrm{sky}}\) to match the galaxy-\(\phi\) map in \(L^2\), \(\max|\Delta I|/\max(I_0)>0.02\). That proves \(\phi_2\) cannot absorb the convention.

3. **Wiener already wrote Ico NaN corners to zero. `n_{\mathrm{finite}}` then biases \(I_0\).** 30 km/s Ico is 135² with **1709** finite and **16516** NaN (empty-corner \(n=0\)). `wiener.py` `_finite_zero` and `_apply_mask` set non-finite / off-mask pixels to **0.0** before the unit-integral stamp. `place_template_on_grid` sees those zeros, not NaNs. Propose residual 4’s floor `n_finite<8` counts zeros as members. An annulus with 4 positive pixels and 96 mask zeros has \(n_{\mathrm{finite}}=100\), mean \(\approx 0.04\,I_{\mathrm{true}}\), and the floor never trips. Renormalise \(\int I\,d\Omega=1\) then boosts the inner disk. \(A=0\) \(\chi^2\) is no longer “axisymmetrised Ico”; it is mask-diluted Ico. DEC-066-SB: do not inpaint.

   Execute “finite/**positive**” is the right membership test. Residual 4’s `n_finite` is not.

   **Bound (missing test, no FITS):** membership is `(I > 0) & isfinite`. Never `nan_to_num` before the annulus mean. \(n<8\) is \(n_{\mathrm{positive}}\). Compact Gaussian plus a zero (or NaN) frame: \(I_0(R<3'')\) matches the Gaussian-alone profile to \(10^{-6}\). A control that `nan_to_num`s the frame and `nanmean`s must fail that bound. Hold last **positive** mean outside.

4. **720-eval grid + “if polish loses, keep the grid” blocks a worse polish. It does not block skipping the grid.** Propose rejects single-start L-BFGS on periodic \(\phi_2\). Execute 4 is a \(36\times 20\) grid then one L-BFGS-B from the grid winner with \(\phi_2\) wrapped. \(\phi_2\) step \(\pi/36\approx 5^\circ\) is fine for a single \(\cos 2(\phi-\phi_2)\) basin (half-width \(\pi/4\)). `DEC-066-AGENTS` lets the implementer decide **physics gates**, not replace a named search because 720 `predict_binned` evals on 881×95 are “hours if serial” (residual 3). A wall-clock judgment that substitutes two L-BFGS starts is the rejected alternative. Box-bounded L-BFGS on \(\phi_2\in[0,\pi)\) is also wrong at the identified edges \(\phi_2\sim 0\sim\pi\).

   **Bound:** the 36×20 grid is mandatory. \(\phi_2=\mathrm{linspace}(0,\pi,36,\mathrm{endpoint=False})\). Polish starts only at the grid winner. If \(\chi^2_{\mathrm{polish}}\ge\chi^2_{\mathrm{grid}}\), keep the grid point. Do not replace the grid with L-BFGS starts. JAX/batch may change wall-clock only, not the likelihood. JSON records both \(\chi^2\) values.

5. **Opt-in functions in `sb.py` are safe. Wiring them through `load_sb_template` is a steal, including \(A=0\).** Official \(\theta\) was fit against the 2-D Wiener stamp from `load_sb_template` → `place_template_on_grid`. An \(A=0\) default that axisymmetrises first returns \(I_0\), not official Ico, and moves production \(\chi^2\) off 168675.6 while looking like a no-op. Adding `A`/`phi2` kwargs with defaults is the same steal. Execute 4’s `kinuv-KGAS066-m2-map` tree is a new `DEC-066-OPS-AUTH`-shaped MAP name for a frozen-\(\theta\) SB scan. `DEC-066-AGENTS` forbids in-place overwrite of `kinuv-KGAS066-uvsign-map` and forbids a new DEC id.

   **Bound:** `load_sb_template(grid, ico_path=None)` signature and body stay. No \(A,\phi_2\), axisym, or m=2 kwargs. Identity \(\chi^2\) calls that unmodified function. `axisymmetrise_template` / `apply_m2` are new opt-in helpers; only `scripts/analysis/fit_m2_sb_at_map.py` calls them. Do not edit `DEC-066-SB.md`. Do not write `DEC-066-SB-m2.md`. Do not change `BMAJ_ICO_ARCSEC`. Prefer artifact JSON+PNG under `docs/reviews/artifacts/2026-09-05-kgas066-m2-sb/`. If a tree `kinuv-KGAS066-m2-map` is written: diagnostic copy of frozen kinematics, `steal_latest` false, dest not G3, not `KGAS066-latest`, not official. No re-MAP of \(\theta\). No unfreeze \(i\). No 066 NUTS. No G4. Pending 007 ids untouched.

## Comments

1. `major` — Three \(\chi^2\) plus (2)−(1), (3)−(2), (3)−(1). Quote leftover 1373 and vs-\(V=0\) +35553. Not 3σ, not a MAP. (3)<(1) does not unlock production. Identity fail → no m=2 quote, no m2-map tree. Attack 1.

2. `major` — Unit test: \(\phi=\mathrm{atan2}(y_g,x_g)\) after `sky_to_galaxy`; \(\phi=0\) on \(+x_g\); sky \(\mathrm{atan2}\) residual after best \(\phi_2\) exceeds \(0.02\max I_0\). Attack 2.

3. `major` — Annulus membership is positive and finite, not `n_finite` after Wiener zero-fill. Unit test: Gaussian + zero/NaN frame matches Gaussian-alone \(I_0(R<3'')\) to \(10^{-6}\); `nan_to_num`+`nanmean` control fails. Do not inpaint. Attack 3.

4. `major` — 36×20 grid is the search. Polish only from the grid winner; keep the grid if polish loses. Do not substitute L-BFGS starts. Attack 4.

5. `major` — Do not change `load_sb_template` default or signature. Do not edit `DEC-066-SB.md` or draft `DEC-066-SB-m2.md`. Official MAP read-only. No `KGAS066-latest` steal. No 066 re-MAP, no 066 NUTS, no G4, no 007 interrupt (`b1mqxsov` `xkytxih1` `y5tspgit` `zq1olquy`). `DEC-066-INC` frozen. `quote_inner_slope: false`. Attack 5.

6. `minor` — M1 invariance remains a named gate (propose residual 2). If median \(|\Delta\mathrm{M1}|\) exceeds the M1 error, STATUS one-liner, do not claim SB/M1 decoupling. Central 5″ M0 stays K km/s.

## Residual risks

1. \(A=0\) is not official. (3)−(1) is leftover vs locked SB, not an SB-model selection. S2 SBC failed 68/95. Propose residual 1, tightened (Comment 1).

2. Morphology can still leak into M1 via spatial weighting of \(v_{\mathrm{los}}\). Propose residual 2. Comment 6.

3. 720 `predict_binned` evals on 881×95. JAX/batch must not change the likelihood or skip the grid. Propose residual 3, tightened (Comment 4).

4. Outer \(I_0(R)\) is mask-limited **and** Wiener-zero-filled. Residual 4’s `n_finite` is the omitted failure mode. Comment 3. **(new)**

5. 007 chains still running. Do not steal `KGAS066-latest`. Propose residual 5.

6. `load_sb_template` remains ADR-wrong on empty-corner \(K=(0.02)^2\). Out of scope to fix. This card must not rewrite the wrapper to “fix” SB via m=2. **(new)**

## STATUS updates required

- `verdict: accept`, `severity: major`
- `last_review_a:` this file
- Do not set `board: accepted` (parent tallies)
- Keep `pending: ["b1mqxsov", "xkytxih1", "y5tspgit", "zq1olquy"]`
