---
role: reviewer
seat: b
date: 2026-09-05
agent: review-b
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

# Review b: visibility-native m=2 SB on axisymmetrised 30 km/s Ico

Do not read the other seat's review file. Do not implement.

Execute-as-typed cannot steal `kinuv-KGAS066-uvsign-map`, write `docs/decisions/DEC-066-SB-m2.md`, change `load_sb_template` default / `BMAJ_ICO_ARCSEC` / `DEC-066-SB.md`, unfreeze \(i\), start G4, or interrupt 007 `b1mqxsov` `xkytxih1` `y5tspgit` `zq1olquy`. Accept is only that integrity claim. The majors below must be fixed during execute. `recommended_new_id` stays a recommendation; the user stubs DEC ids.

`I_0` = azimuthal mean of official 30 km/s Ico, not the 2-D template and not exponential. Free \(A\in[0,0.8]\), \(\phi_2\in[0,\pi)\). Frozen \(R_2=2.5''\), \(\sigma=1.5''\). Opt-in only. \(V_c(R)\) stays axisymmetric. Visibility window stays the 30 km/s vis cube. `quote_inner_slope: false`. Do not push `main`.

## Attacks / bounds

1. **M1 invariance is not operational as typed.** Execute item 5 says “M1 invariance (median \(|\Delta M1|\) vs M1 error). If M1 fails, STATUS one-liner, do not claim decoupling.” That gate can be made to pass by choosing the cube, the mask, and the error after the fact. Production `masked_moments` (`src/kinuv/diagnostics/imaging.py`) returns `(m0, m1, m2)` and **no** per-pixel M1 uncertainty. S1’s cube estimator is `sky_cube` → restoring beam of the **10 km/s** imaging header → major-axis M1/M2 (`docs/diagnostics/s1-mock.md`); leftover / identity live on the **30 km/s** vis cube. Opening `KGAS66_mom1.fits` (either `10kms/` or `30kms/`) to invent \(\sigma_{M1}\) is a second image-plane likelihood (already rejected). Dividing \(|\Delta M1|\) by M2 (\(\sim 10\)–\(50\) km/s) or by a CLEAN formal error makes any SB-induced leak look small. “Median vs error” without naming (i) which two maps, (ii) which cube, (iii) which mask, (iv) which \(\sigma\) is not a gate.

   **Bound:** M1 leak is **model–model** at frozen MAP \(\theta\), not data–model and not a CLEAN mom1 residual. Build three `sky_cube`s on the production vis `ImageGrid` (same native axis as `predict_binned`): (1) official 2-D `load_sb_template`, (2) \(A=0\), (3) best \((A,\phi_2)\). Optional restoring-beam match uses the **30 km/s** Ico header only — do not open a `mom1.fits`. Mask: finite \(M0>0.05\times\mathrm{peak}\) **and** sky \(R<5''\) (the \(a_2\) node). Report median \(|M1_{(3)}-M1_{(2)}|\) and median \(|M1_{(3)}-M1_{(1)}|\) on that mask, in km/s. Pass if both medians are \(<2\) km/s (binned \(\Delta v=5.080\) km/s; do not divide by M2 or an invented error). Fail → STATUS one-liner, do not claim decoupling. Do not convert \(\Delta M1\) into inflow, warp, or an \(i\) shift (`DEC-066-INC` stays frozen; \(i\) is not taken from moment-1).

2. **Unit-integral after m=2 conserves broadband DC amplitude, not phase.** \(I\leftarrow I/\int I\,d\Omega\) forces \(V(0,0)=\sum I\) real and equal to the free flux after the spectral Gaussian integrates, **if** \(I\) stays real. The 066 npz has **no** zero-spacing; leftover vs baseline is “not a classic missing-flux bowl at \(u=v=0\)” (`s1-mock.md`). The incomplete Ico mask (1709 finite pixels, not a filled ellipse) makes \(\int I_0(R)\,a_2(R)\cos 2(\phi-\phi_2)\,d\Omega\neq 0\): a monopole (absorbed by the renormalisation) **and** a dipole. The dipole is a centroid shift. `flux_weighted_centroid` (`wiener.py`) plus the DEC-066-SB gate already exist: \(|\langle x\rangle|<0.01''\) absolute. A shifted real cube still has \(\mathrm{Im}\,V(0,0)=0\), but \(\arg V\) on the shortest sampled baselines is a phase ramp \(2\pi(u\Delta x+v\Delta y)\) — the same operator as `(dx, dy)` / `fourier_shift`. Renormalising does not cancel that ramp. `sky_cube` applies `fourier_shift` **after** the template; a centroid that moved inside the template is an extra shift on top of MAP `(dx, dy)`.

   **Bound:** After `apply_m2` and **before** `fourier_shift`, run `flux_weighted_centroid` vs \(I_0\) on the same grid/mask. Fail if the hypot shift is \(\ge 0.01''\) (same number as `CENTROID_TOL_ARCSEC` and the propose’s \(A>0\) test). Report the pre-norm monopole \(\int I_0 a_2\cos\,d\Omega/\int I_0\). Assert the cube is real: \(|\mathrm{Im}\sum_{\mathrm{pix},\nu} I|<10^{-9}\,|\mathrm{Re}\sum I|\). Do not claim “flux conservation” for \(\arg V\). Do not touch official `(dx, dy)`.

3. **Annulus floor \(n=8\) is slightly tight on a 0.1″ grid and too tight on the production vis grid.** Propose residual 4 / execute 1: drop elliptical annuli with \(n_{\mathrm{finite}}<8\); hold last finite mean **outside**. \(\Delta R\) and the **inner** hole are unnamed. DEC-066-GRID production cell is **not** YAML 0.1″ (that was the uvkin header-override bug). `image_grid_from_uv(305e3)` with `CHOOSE_MARGIN=1.2` gives \(\mathrm{cell}=1/(2\cdot 1.2\cdot 305\,\mathrm{k}\lambda)/\mathrm{ARCSEC\_TO\_RAD}\approx 0.282''\), FoV \(25.9''\), \(n\approx 92^2\). One-cell galaxy-plane annuli on a sky grid: \(n(R)\approx 2\pi R\,\Delta R\cos i/\mathrm{cell}^2\) with \(\cos i=0.721\). For \(\Delta R=\mathrm{cell}\): \(n\ge 8\) only at \(R\gtrsim 0.18''\) on 0.1″ and at \(R\gtrsim 0.50''\) on 0.282″. The production hole is the entire \(r_t\) floor / inner Ico beam. On the 0.4″ Ico stamp, \(n\ge 8\) only at \(R\gtrsim 0.71''\). `exponential_r_scale`’s \(n<8\) is an lstsq cut over \(1.5\)–\(8''\), not a per-annulus SB floor. Holding the last finite mean **outside** does not fill the nucleus; \(I_0(0)=0\) steals nuclear SB and reweights \(v_{\mathrm{los}}\) (Attack 1).

   **Bound:** Axisymmetrise **after** `place_template_on_grid` on the production vis `ImageGrid` from `image_grid_for_vis` (DEC-066-GRID), not a 0.1″ test canvas and not the raw 0.4″ Ico stamp. Name \(\Delta R\) (\(\ge\) one cell). Inner dropped annuli: single nuclear aperture (all finite/positive pixels with \(R<R_{\mathrm{first}}\) where \(n\ge 8\)) held **inward**; last finite mean held **outward**. Artifact: \(n(R)\) and \(I_0(R)\). Do not leave \(I_0=0\) for \(R<R_{\mathrm{first}}\).

4. **Identity \(|\chi^2-168675.6|<1\) is arm (1) only: 30 km/s vis + production `load_sb_template`, not axisym \(I_0\).** Propose already says \(A=0\) is not the official MAP and will be worse than 168675.6. The failure mode is running that bound on arm (2) (fails by construction → “do not quote \(\Delta\chi^2\)” guts the card) or “fixing” it with exponential fallback / a 10 km/s Ico / a new \(K\). `ICO_FITS` in `sb.py` is still the laptop path; missing file → `exponential_template`, not the Wiener Ico. Last card locked 30 km/s: identity \(\chi^2_{30}=168675.596\) (\(|\Delta|=0.004\)) through `load_sb_template` \(K=(0.02)^2\) (empty-corner \(n=0\)). Vis window is DEC-066-VIS (881×95), not the 10 km/s imaging cube.

   **Bound:** Identity = official 2-D Ico at MAP \(\theta\) on the 30 km/s vis cube, \(s=0.5136098555284736\), Hann+bin, `NPZ_UV_SIGN=-1`, `load_sb_template(grid, ico_path=<CANFAR 30kms KGAS66_Ico_K_kms-1.fits>)`. Not axisym \(I_0\), not `exponential_template`, not `ico_to_template` empty-corner \(K\), not 10 km/s Ico. Fail → STATUS one line, do not quote any m=2 \(\Delta\chi^2\), do not re-MAP, do not touch official MAP. Report three raw \(\chi^2\) as propose residual 1. \(\Delta\chi^2\) vs \(V=0\) (`DEC-066-ZEROMODEL`) on arm (1) stays \(+35553\) within 1; \(A=0\) is not a new zero model.

5. **\(\Delta\ln L=-\frac12\Delta\chi^2\) is not calibrated on this leftover SB surface.** S2 Laplace SBC failed 68/95 (`docs/diagnostics/s2-coverage.md`; `laplace_mh`, not NUTS). Leftover-vs-velocity is True at Stage B; gate is **SB-dominated**. The m=2 family does not contain the official template. \((3)-(2)\) is a two-parameter SB tweak on a misspecified leftover, not a nested LR. \(\sqrt{\Delta\chi^2}\) / AIC / “\(N\sigma\)” will be copied as a detection.

   **Bound:** Artifact JSON/README quote **raw** \(\chi^2_{(1)},\chi^2_{(2)},\chi^2_{(3)}\) and the two differences only. No \(\Delta\ln L\), no \(\sqrt{\Delta\chi^2}\), no AIC \(p\), no 16/50/84, no “nested 3σ”. `intervals_calibrated: false`. `quote_inner_slope: false`. Do not start a 066 NUTS. Do not unfreeze \(i\). Do not write `docs/decisions/DEC-066-SB-m2.md`. Prefer JSON+PNG under `docs/reviews/artifacts/2026-09-05-kgas066-m2-sb/`. Do **not** write `kinuv-KGAS066-m2-map` this card (kinematics are not refit; \(A,\phi_2\) at frozen MAP \(\theta\)). Official tree read-only. 007 pending four ids untouched. No G4.

## Comments

1. `major` — M1 gate is model–model at MAP \(\theta\) on `sky_cube` (30 km/s native axis; optional 30 km/s Ico restoring beam only). Mask: \(M0>0.05\times\mathrm{peak}\) and sky \(R<5''\). Pass = both median \(|\Delta M1|_{(3)-(2)}\) and \(|(3)-(1)|\) \(<2\) km/s. No `mom1.fits`. No M2-as-error. Fail → STATUS, no decoupling claim. Attack 1.

2. `major` — Post-m=2 centroid vs \(I_0\) \(<0.01''\) (`flux_weighted_centroid`, DEC-066-SB). Report pre-norm monopole. Assert \(\mathrm{Im}\sum I\approx 0\). Unit-integral is not a phase-conservation claim. Do not retune official `(dx, dy)`. Attack 2.

3. `major` — Axisymmetrise on the production vis `ImageGrid` (`image_grid_for_vis`, \(\sim 0.282''\)), not 0.1″ and not the 0.4″ Ico stamp. Name \(\Delta R\). Nuclear aperture held inward; last finite mean held outward. Artifact \(n(R)\), \(I_0(R)\). Attack 3.

4. `major` — Identity \(|\chi^2-168675.6|<1\) is arm (1) only: 30 km/s vis, production `load_sb_template` + CANFAR 30 km/s `ico_path`, frozen \(s\), 881×95. Not axisym \(I_0\). Fail → no m=2 \(\Delta\chi^2\). ZEROMODEL vs \(V=0\) on arm (1) stays +35553 within 1. Attack 4.

5. `major` — Raw \(\chi^2\) only. No \(\Delta\ln L=-\frac12\Delta\chi^2\), no \(\sqrt{\Delta\chi^2}\), no AIC / \(N\sigma\). No `DEC-066-SB-m2.md`. No `load_sb_template` default change. No `kinuv-KGAS066-m2-map` tree. No 066 NUTS. No unfreeze \(i\). No G4. Pending 007 ids untouched. Official MAP read-only. Attack 5.

6. `minor` — Tests (no FITS) as typed: \(A=0\) equals \(I_0\) to \(10^{-6}\); \(A>0\) centroid \(<0.01''\); \(I\ge 0\) for \(A\le 0.8\); \(\phi_2\equiv\phi_2+\pi\). \(\phi=\mathrm{atan2}(y_g,x_g)\) after `sky_to_galaxy` at PA \(199.73^\circ\), \(i=\arccos(0.721)\). No `atan2(east, north)`.

7. `minor` — 720 `predict_binned` evals may JAX/batch; likelihood operator, \(s\), and Hann+bin stay identical. Do not steal `KGAS066-latest`.

## Residual risks

1. \(A=0\) is not the official MAP. Three \(\chi^2\); \((3)-(2)\) is m=2 gain, \((3)-(1)\) is leftover vs locked SB. Propose 1. Tightened: no LR / \(\sigma\) conversion (comment 5). (new)

2. Morphology can leak into M1. Propose 2. Tightened: cube / mask / 2 km/s bound (comment 1). (new)

3. 720 evals, hours if serial. Propose 3.

4. Outer \(I_0(R)\) mask-limited. Propose 4. The omitted failure is the **inner** \(n<8\) hole on the production \(\sim 0.282''\) grid, not the outer hold. Comment 3. (new)

5. 007 chains still running. Propose 5.

6. Incomplete Ico mask gives m=2 a dipole; unit-integral does not kill the phase ramp. Comment 2. (new)

7. `ICO_FITS` laptop default → exponential fallback if the script omits CANFAR `ico_path`. Comment 4. (new)

## STATUS updates required

- `verdict: accept`, `severity: major`
- `last_review_b:` this file
- Do not set `board: accepted` (parent tallies)
- Keep `pending: ["b1mqxsov", "xkytxih1", "y5tspgit", "zq1olquy"]`
