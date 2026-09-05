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
  - DEC-066-ZEROMODEL
verdict: accept
severity: major
propose: docs/reviews/2026-09-05-propose-ico-10-vs-30-and-figure-closeout.md
---

# Review a: official 10 vs 30 Ico probe + S3 figure closeout

Do not read the other seat's review file. Do not implement.

Scope check: existing DEC ids only. No `DEC-066-SB-v2`. `DEC-066-INC` stays frozen. Official MAP `kinuv-KGAS066-uvsign-map` stays read-only. Production `src/kinuv/forward/sb.py` default and `BMAJ_ICO_ARCSEC = 1.30` stay. Visibility window stays `DEC-066-VIS` (30 km/s vis cube). No new 066 NUTS. No G4. Do not interrupt 007 `b1mqxsov` `xkytxih1` `y5tspgit` `zq1olquy`. `quote_inner_slope: false` on real-066 artifacts. Figure D stays vis/cube χ² slices, not a sampler. Leftover gate stays **SB-dominated**.

Both candidate FITS exist. Headers match the propose table (this review measured them):

| Map | shape | CDELT2 | BMAJ × BMIN | BPA | BUNIT | finite |
|---|---|---|---|---|---|---|
| 30 km/s | 135² | 0.4000″ | 1.2948″ × 1.1795″ | −18.30° | K km/s | 1709 |
| 10 km/s | 180² | 0.3000″ | 1.0400″ × 0.9532″ | −44.85° | K km/s | 2871 |

HISTORY (not in the propose table): both `tclean` `restoration=True` `restoringbeam=common` `pbcor=True` `imsize=[540,540]` `cell=0.1arcsec`. **30 km/s `weighting=natural`**, threshold 0.795 mJy, `width=30.0km/s`, date 2025-04-10. **10 km/s `weighting=briggs` robust=0.5**, threshold 1.476 mJy, `width=10.0km/s`, date 2025-03-27. Same sky FOV (54″). Same `CRVAL`. Different CLEAN.

Execute-as-typed cannot steal the official MAP, rewrite `sb.py`, draft SB-v2, unfreeze \(i\), start G4, or touch the four 007 sessions **if** the `unless both` clause is treated as archive-only (Comment 1). Accept + major: those comments are execute obligations, not optional polish.

## Attacks / bounds

1. **Official 10 km/s Ico is the right product class and is still not a clean \(\Delta v\) A/B.** Same `BTYPE=Ico`, not cube M0 — that part of the propose is correct. HISTORY breaks the \(\Delta v\)-only claim. Beam solid-angle ratio \(\theta_{\mathrm{maj}}\theta_{\mathrm{min}}\) is **1.541**. Cell-area ratio \((0.4/0.3)^2\) is **1.778**. Finite-mask ratio 2871/1709 is **1.680**. BPA differs by **26.5°**. Weighting is **natural vs Briggs**. CLEAN threshold is **0.80 vs 1.48 mJy**. Two tclean dates. Propose residual 1 names beam/cell/stamp/mask/BPA/CLEAN residual and omits weighting and threshold. A “10 km/s loses” (or wins) result may be Briggs vs natural, not channel width.

   **Bound:** Artifact note and STATUS one-liner must say **“10 km/s Ico product (Briggs, 1.04″, 0.3″, 2871 pix)”**, never “10 km/s channels” or “\(\Delta v\) A/B”. Record the six confounders above as numbers, not a footnote. Do not interpret either sign of \(\Delta\chi^2\) as a channel-width result. Do not derive M0 from `KGAS66_clipped_cube.fits`.

2. **\(\Delta\chi^2 < -9\) on \(B>50\,\mathrm{k}\lambda\) is not a 3\(\sigma\) convention on this 881×95 vis \(\chi^2\).** DEC-066-ZEROMODEL’s \(\Delta\chi^2\) is \(\chi^2_{\mathrm{zero}}-\chi^2_{\mathrm{MAP}}\) vs \(V=0\), no parameters. A 1-dof \(\chi^2_1\) tail of 9 is a nested-model story this swap is not: two non-nested templates, different beam/mask/weighting, \(\theta\) frozen at the 30 km/s MLE (Attack 5). S2 Laplace SBC failed 68/95 on every reported parameter (`docs/diagnostics/s2-coverage.md`); this likelihood’s \(\chi^2\) surface is not a calibrated Gaussian. Leftover SB already moves official \(\chi^2\) by **1373** (Stage B 167302 vs Stage A 168676) and NUTS-mean vs MAP by **1189**. Nine units is \(9/1373 \approx 0.007\) of the leftover SB scale.

   Native 066 uv (43240 rows, 224.3 GHz): **66%** of rows have \(B>50\,\mathrm{k}\lambda\); median \(B \approx 66\,\mathrm{k}\lambda\); 50 kλ is a **4.13″** fringe, not “long baseline”. Beam-scale \(k=1/\theta_{\mathrm{BMAJ}}\) is **159 kλ** (30 km/s) and **198 kλ** (10 km/s), where only **13%** and **6%** of native rows live. \(B>50\,\mathrm{k}\lambda\) is most of the array, where both restoring beams are still large.

   **Bound:** Do not print “3\(\sigma\)” on this gate. Report \(n_{\mathrm{row}}\), \(n_{\mathrm{vis}}\), and \(\chi^2\) for the split actually used; define \(B=\sqrt{u_\lambda^2+v_\lambda^2}\) **per visibility** via `vis_uv_wavelengths` at that channel’s frequency (not one row-mean). Also report the same split at **\(B>150\,\mathrm{k}\lambda\)** (30 km/s beam scale). Quote \(\Delta\chi^2_{10-30}\) next to leftover 1373 and vs-\(V=0\) +35553. Convention (a) may stay as typed for the lock logic; it is not evidence.

3. **`ico_to_template` / `place_template_on_grid` accept a 180² 0.3″ stamp. The production wrapper and the tests do not prove it.** `wiener.py` takes `cell_arcsec`, header `BMAJ`/`BMIN`/`BPA`, `mask=isfinite` (not 1709), `pad_n=default_pad_n(max(ny,nx))` → `max(2N,512)` so 180 → 512. `place_template_on_grid` resamples from the passed cell; 0.4″ appears only in docstrings. That path is clean **if the script calls it that way**.

   `load_sb_template` does **not** follow DEC-066-SB empty-corner \(K\). It passes `sigma_empty=0.02*nanmax(|data|)` so \(K=4\times10^{-4}\) on every map. `tests/test_template.py` hardcodes `NPIX=135`, `CELL=0.4`, `_mask_like_ico(..., n_pix=1709)`, beam 1.30×1.18, BPA −18.3. The optional live test asserts `shape==(135,135)` and `finite==1709` and uses the same 0.02-peak \(K\). `ICO_FITS` in `sb.py` is the Mac path, not the CANFAR 30 km/s product. Propose execute 2 names `ico_to_template` + empty-corner \(K\); a “production path” via `load_sb_template` silently drops empty-corner and can still apply 30 km/s test habits.

   **Bound (missing test):** Script calls `ico_to_template` with header `BMAJ`/`BMIN`/`BPA`/`CDELT`, `mask=None` (isfinite), `sigma_empty=None` (empty-corner on Jy), `pad_n=None`. Do not pass 1709, 0.4″, or `BMAJ_ICO_ARCSEC`. Do not call `load_sb_template` for this probe. Unit test, no FITS required: 180² 0.3″ Gaussian + header-like beam 1.04″×0.95″, BPA −44.8°; `default_pad_n(180)==512`; returned `cell_arcsec==0.3`; mask sum is the finite count of the input, not 1709; `place_template_on_grid` onto a vis `ImageGrid` conserves \(\int I\,d\Omega\) to \(10^{-4}\). Explicit CANFAR paths only.

4. **“No \(P(k)\) amplification” is not operational.** Execute 3 marks \(k=1/\theta_{\mathrm{beam}}\) per map. Execute 5(b) says “\(P(k)\) at \(k>1/\theta_{\mathrm{beam}}\) is not higher on the 10 km/s template” with no ratio, no estimator, no shared \(k\) grid, no bin list. Per-map cuts are 0.772 vs 0.962 cycles/arcsec — different ranges. Existing `azimuthal_pk` in `scripts/run_s3_advanced_diagnostics.py` divides by the **high-\(k\) floor** (`centres > 0.55 kmax`, `kmax=0.45/cell`). After that normalisation, extra high-\(k\) power is invisible. Different cells (0.4 vs 0.3) change `kmax` (1.125 vs 1.50). That function must not be the gate.

   **Bound:** After both Wiener stamps are on the **same** vis `ImageGrid` (unit \(\int I\,d\Omega\)), compute raw (not floor-normalised) azimuthal \(P(k)\) on one \(k\) grid. Let \(R(k)=P_{10}(k)/P_{30}(k)\). (b) holds only if **median \(R(k) \le 1.2\)** on bins with \(k > 1/\theta_{30}=0.772\) arcsec⁻¹ **and** \(k < 1/(2\times 0.4″)\) (30 km/s Nyquist). Report \(R\) at \(k=1/\theta_{30}\) and \(k=1/\theta_{10}\) as numbers, not a Boolean. If any high-\(k\) bin is empty, (b) fails (lock 30 km/s). Do not use `BMAJ_ICO_ARCSEC` as \(\theta_{\mathrm{beam}}\).

5. **Conditional \(\chi^2\) at official MAP \(\theta\) cannot fire fairly for a narrower-beam template.** Official \(\theta\) (flux, PA, \(V_{\mathrm{sys}}\), \(\sigma_{\mathrm{gas}}\), \(dx,dy\), \(V_0\), \(r_t\)) is the 30 km/s MLE. Templates are unit-normalised; frozen flux is the 30 km/s amplitude. A 1.04″ restoring beam keeps Wiener gain out to ~198 kλ; extra high-\(k\) shape at the wrong flux/\((dx,dy)\) **raises** \(\chi^2_{10}\). Gate (a) is then biased against 10 km/s. It can still go negative from Briggs/mask/threshold (Attack 1), which is the unfair fire. DEC-066-SB robustness (Ico vs exponential \(V_c\) within 1\(\sigma\)) is not this test and is not licensed here.

   **Bound:** (a) and (b) never unlock production. Even if both fire, archive the note and **lock DEC-066-SB at 30 km/s**. Do not edit `sb.py`, `BMAJ_ICO_ARCSEC`, `DEC-066-SB`, or the official MAP tree. Allowed extra number (not a gate): per-template closed-form `_optimal_flux` \(\chi^2\) at frozen kinematics. If that sign-flips \(\Delta\chi^2\) vs frozen-flux, the note says flux-confounded; (a) still did not fire. No re-MAP. No 066 NUTS.

## Comments

1. `major` — `unless both` is archive-only. Lock 30 km/s in all cases this card. Do not rewrite `sb.py` / `BMAJ_ICO_ARCSEC` / `DEC-066-SB`. Do not draft SB-v2. Official MAP read-only. Attack 5.

2. `major` — Label “10 km/s Ico product”. Record natural-vs-Briggs, beam-area 1.541, cell-area 1.778, mask 1.680, BPA 26.5°, thresholds 0.80 vs 1.48 mJy. Not a \(\Delta v\) A/B. Attack 1.

3. `major` — Do not call \(\Delta\chi^2=-9\) “3\(\sigma\)”. Per-visibility \(B\) via `vis_uv_wavelengths`. Report \(n_{\mathrm{vis}}\) in the split, leftover 1373, vs-\(V=0\) +35553, and a \(B>150\,\mathrm{k}\lambda\) column. S2 SBC failure stays in the note. Attack 2.

4. `major` — Probe uses `ico_to_template` + empty-corner \(K\), not `load_sb_template`’s \(K=4\times10^{-4}\). Unit test: 180² 0.3″ stamp, pad 512, cell 0.3″, mask ≠ 1709, flux conservation \(10^{-4}\). Explicit CANFAR paths. Attack 3.

5. `major` — \(P(k)\) gate is median \(R(k)\le 1.2\) on a common ImageGrid, common \(k\), \(k>1/\theta_{30}\) and below 30 km/s Nyquist, raw (not S3 floor-normalised) \(P(k)\). Empty high-\(k\) bin → (b) fails. Attack 4.

6. `major` — Conditional \(\theta\) only. Optional optimal-flux \(\chi^2\) is diagnostic, not an unlock. No re-MAP, no 066 NUTS, no G4, no 007 interrupt (`b1mqxsov` `xkytxih1` `y5tspgit` `zq1olquy`). `DEC-066-INC` frozen. `quote_inner_slope: false`. Attack 5.

7. `minor` — Phase 3: re-render only for clipping / overlap / honesty. Figure D caption stays “χ² slices”, not NUTS / MCMC / Laplace posterior. Real moments stay K km/s and km/s. SKA panel stays an analytic volume integral. Write only under `docs/reviews/artifacts/2026-09-05-kgas066-ico-10-vs-30/` and in-place S3 honesty edits. Do not merge `dev` → `main`.

## Residual risks

1. Natural vs Briggs (and 0.80 vs 1.48 mJy) can dominate any \(\Delta\chi^2\) or \(P(k)\) ratio. Propose residual 1 did not name weighting or threshold. **(new)** Comments 2 and 5.

2. Frozen-θ \(\chi^2_{10}\) is biased high for a narrower-beam template. A “10 km/s loses” note will be over-read as “\(\Delta v\) does not matter”. Comments 1 and 6. Tightened from propose residual 2.

3. \(\Delta\chi^2=-9\) on \(B>50\,\mathrm{k}\lambda\) is a convention, not coverage. S2 SBC failed 68/95. Leftover SB scale is 1373. Propose residual 3, tightened (Comment 3).

4. Vis spectral window stays 30 km/s (propose residual 4). Template swap cannot test a 10 km/s vis cube.

5. `load_sb_template` remains ADR-wrong (\(K=4\times10^{-4}\)) after this card. Out of scope to fix the wrapper. Comment 4 keeps the probe off that path. **(new)**

6. 10 km/s pad 180 → 512 and centroid 0.01″ absolute are fine (propose residual 5). Carry forward.

## STATUS updates required

- `verdict: accept`, `severity: major`
- `last_review_a:` this file
- Do not set `board: accepted` (parent tallies)
- Keep `pending: ["b1mqxsov", "xkytxih1", "y5tspgit", "zq1olquy"]`
