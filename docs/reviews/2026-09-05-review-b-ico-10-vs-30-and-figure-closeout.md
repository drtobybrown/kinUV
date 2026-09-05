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
  - DEC-066-ZEROMODEL
verdict: accept
severity: major
propose: docs/reviews/2026-09-05-propose-ico-10-vs-30-and-figure-closeout.md
---

# Review b: official 10 vs 30 Ico probe + S3 figure closeout

Do not read the other seat's review file. Do not implement.

Headers on both FITS match the propose table. Canon 30 km/s: 135², CDELT 0.400″, BMAJ×BMIN 1.295″×1.180″, BPA −18.30°, BUNIT `K km/s`, `tclean` `width="30.0km/s"` `restoration=True` `restoringbeam=common` `pbcor=True`, finite 1709. Candidate 10 km/s: 180², CDELT 0.300″, 1.040″×0.953″, BPA −44.85°, same BUNIT / tclean flags except `width="10.0km/s"`, finite 2871. Both stamps are 54″ on a side. Official MAP `kinuv-KGAS066-uvsign-map` stays read-only. `DEC-066-INC` stays frozen. No new `DEC-*` id. Do not start G4. Do not interrupt 007 `b1mqxsov` `xkytxih1` `y5tspgit` `zq1olquy`. Do not call Figure D NUTS.

Execute-as-typed cannot steal the official MAP, rewrite `sb.py`’s default path, draft `DEC-066-SB-v2`, unfreeze \(i\), start G4, or touch the 007 sessions. Accept is only that integrity claim. The majors below must be fixed during execute.

## Attacks / bounds

1. **ADR contradiction: DEC-066-SB empty-corner \(K\) is undefined on both v1.3 Ico products.** `empty_corner_rms` (12% corner boxes) sees **zero** finite pixels on both stamps (30 km/s: 16516 NaN, empty-corner \(n=0\); 10 km/s: 29529 NaN, \(n=0\)). `ico_to_template` with `k_wiener=None` and `sigma_empty=None` therefore **raises** (`empty-corner rms unavailable`). Production `load_sb_template` already knows this: it passes `sigma_empty=0.02 * nanmax(|I|)` so \(K=(0.02)^2=4\times10^{-4}\) on every map. Propose execute item 2 names “production `ico_to_template` (… empty-corner \(K\))” — that path is not the official MAP path and does not run on these FITS. A NaN→0 fill to “make empty-corner work” yields \(\sigma\approx0\), \(K\approx0\), and a near-inverse filter (taper only). Peaks already differ (30 km/s \(I_{\mathrm{peak}}=38.96\) K km/s; 10 km/s \(43.17\)). Bound: Wiener **both** arms through `load_sb_template(grid, ico_path=…)` (header beam / isfinite mask / `0.02*peak` \(K\)). Do not call `empty_corner_rms` on these stamps. Do not edit `wiener.py` or `sb.py` to invent a \(K\). Report \(K\), \(\sigma_{\mathrm{empty}}\) used, and \(I_{\mathrm{peak}}\) for both. 30 km/s arm identity: `|chi2-168675.6|<1` at official MAP \(\theta\) on 881×95, frozen \(s=0.5136098555284736\), `NPZ_UV_SIGN=-1`. That identity is the load-bearing proof the 30 km/s arm **is** the official template, not a new \(K\).

2. **Tighter bound: \(B>50\,\mathrm{k}\lambda\) is inside both restoring-beam tapers; leftover \(B\) is metres.** `vis_uv_wavelengths` is `NPZ_UV_SIGN * u_m * \nu / c` (shape `(n_row, n_chan)`). Sign cancels in \(B=\sqrt{u_\lambda^2+v_\lambda^2}\). `leftover_chi2` in `s1.py` returns `hypot(u_m, v_m)` in **metres** (50 kλ at 224.3 GHz is **66.8 m**, not 50 m and not 50 000 m). Copying leftover `baseline_m` into a 50 kλ cut is a unit error. Analytic \(|\tilde B|=0.05\) (DEC-066-SB taper) from header BMAJ: **146 kλ** (30 km/s) and **182 kλ** (10 km/s). At 50 kλ, \(|\tilde B|_{30}=0.70\) and \(|\tilde B|_{10}=0.80\). 066 \(u_{\max}\approx305\,\mathrm{k}\lambda\), so \(B>50\,\mathrm{k}\lambda\) mixes (i) jointly supported 50–146 kλ, (ii) 10 km/s-only 146–182 kλ, (iii) both-tapered \(>182\,\mathrm{k}\lambda\). Gate (a) as typed can fire on beam support, not \(\Delta v\). Gate (b) \(k>1/\theta_{\mathrm{BMAJ}}\) is **past** both tapers (\(1/1.295''=159\,\mathrm{k}\lambda\), \(1/1.040''=198\,\mathrm{k}\lambda\)). Bound: \(B_{\mathrm{row}}=\mathrm{hypot}(u_\lambda,v_\lambda)\) from `vis_uv_wavelengths` at one stated frequency (binned / `nu.max()`, named in the artifact). Do not use metre baselines. Report \(\chi^2\) on \(B<50\), \(50\)–\(146\), \(146\)–\(182\), and \(>182\,\mathrm{k}\lambda\). Load-bearing long-baseline \(\Delta\chi^2\) for the lock is the **jointly supported** bin \(50\)–\(146\,\mathrm{k}\lambda\), still vs the 1-dof convention \(-9\). Mark \(P(k)\) with both \(1/\theta_{\mathrm{BMAJ}}\) **and** \(|\tilde B|=0.05\) for that map.

3. **512-pad does not silently fail; a literal 1709-pixel / 0.4″ DEC-066-SB step 6 would.** `default_pad_n(180)=max(360,512)=512`. `ico_to_template` masks `isfinite`, not a hardcoded 1709. `place_template_on_grid` reads the template cell. The silent fail is not the pad (propose residual 5). It is applying DEC-066-SB’s “then apply the 1709-pixel mask” / “Ico CDELT is 0.4″” text onto the 180² 0.3″ stamp, or feeding the Ico-cell Wiener stamp to `predict_binned` without `place_template_on_grid`. `test_live_ico_optional` still asserts `shape==(135,135)` and `finite==1709`. Bound: no 1709 / 0.4″ / 1.30″ constants on the 10 km/s arm. \(\chi^2\) path is `load_sb_template` → `place_template_on_grid` → `predict_binned`. Add a script-level (or test) gate: 10 km/s `NAXIS=180`, cell \(0.3''\), finite 2871, `default_pad_n>=360`, centroid gate still \(0.01''\) absolute.

4. **Figure closeout: named honesty items are on disk; Figure A adjoint is not in the Phase 3 list.** `quote_inner_slope: false` on real-066 JSON; Figure D JSON is \(r_t\)–\(V_0\) \(\chi^2\) slices (not draws); SKA ledger already says analytic volume integral, not a vis simulation, not KGAS066. Missing from execute item 6: Figure A kinUV column is a **script-local type-1 DFT adjoint**, units adjoint-arbitrary, not Kelvin, not production NUFFT, not a CASA residual (`advanced_diagnostics/README.md`). A re-render that drops that caption is an honesty defect. `s3_table.json` still has “NUTS mean comparator” notes — do not promote those to a Figure D / real-066 inner-scale quote. Real moment colourbars stay **K km/s** (M0) and **km/s** (M1 as \(v-v_{\mathrm{sys}}\)). Do not relabel Figure D as MCMC / posterior / NUTS. Leftover `plot_leftover_chi2` xlabel is still “radio velocity”; if that PNG is in the ten, fix or leave with a STATUS one-liner — do not silently convert the leftover operator.

5. **Lock 30 km/s without a new DEC id.** `DEC-066-SB` already names the 30 km/s restored Ico (HISTORY `width=30`, 1709 finite). Lock = keep that ADR. Record the probe as artifact README + STATUS one-liner only. Do **not** edit `DEC-066-SB.md`. Do not draft `DEC-066-SB-v2`. Gate-fire \(\neq\) production swap: even if both (a) and (b) pass, this card still must not change `sb.py` `ICO_FITS` / `BMAJ_ICO_ARCSEC=1.30` or steal `kinuv-KGAS066-uvsign-map`. A fired gate is a note recommending a user-stubbed future ADR, not an in-card default change.

## Comments

1. `major` — Empty-corner \(K\) is unavailable (\(n=0\) finite in all four corners on both FITS). Use production `load_sb_template(..., ico_path=)` \(K=(0.02)^2\). Do not NaN-fill to force `empty_corner_rms`. Do not edit `src/kinuv/forward/sb.py` or `src/kinuv/template/wiener.py`. Report \(K\), \(\sigma\) used, \(I_{\mathrm{peak}}\) for both. Attack 1.

2. `major` — 30 km/s arm identity `|chi2-168675.6|<1` at official MAP \(\theta\) on 881×95 before any 10-vs-30 \(\Delta\chi^2\) is quoted. Same \(s\), Hann+bin, `NPZ_UV_SIGN=-1`. Fail → STATUS one line, do not lock-from-garbage, do not re-MAP, do not touch official MAP. Attack 1.

3. `major` — \(B=\mathrm{hypot}(*\mathrm{vis\_uv\_wavelengths})\) at a named frequency; never leftover `baseline_m`. Split \(\chi^2\) on \(B<50\), \(50\)–\(146\), \(146\)–\(182\), \(>182\,\mathrm{k}\lambda\). Lock gate (a) uses the jointly supported \(50\)–\(146\,\mathrm{k}\lambda\) bin and \(\Delta\chi^2=\chi^2_{10}-\chi^2_{30}<-9\). Mark \(P(k)\) with \(1/\theta_{\mathrm{BMAJ}}\) and \(|\tilde B|=0.05\). Attack 2.

4. `major` — Do not apply 1709 / 0.4″ / 1.30″ onto the 10 km/s stamp. `predict_binned` sees ImageGrid templates from `load_sb_template` / `place_template_on_grid`, not the Ico-cell Wiener stamp. Script gate: 180², 0.3″, finite 2871, pad ≥360. Attack 3.

5. `major` — Lock is artifact + STATUS only. Do not edit `DEC-066-SB.md`. Do not draft `DEC-066-SB-v2`. Do not change `sb.py` default or `BMAJ_ICO_ARCSEC` even if the gate fires. Official MAP read-only. No 066 re-MAP. No new NUTS. `DEC-066-INC` frozen. No G4. Pending 007 ids untouched. Attack 5.

6. `major` — Phase 3: keep Figure D as vis/cube \(\chi^2\) slices (not NUTS / not MCMC). Real-066 `quote_inner_slope: false`. Real M0 colourbar **K km/s**, M1 **km/s**. SKA panel stays the analytic volume integral. If Figure A is re-rendered, keep the type-1 adjoint / not-Kelvin caption. Attack 4.

7. `minor` — Vis spectral window stays the 30 km/s `DEC-066-VIS` cube (propose residual 4). Label the candidate “10 km/s Ico product”. \(\Delta\chi^2\) vs V=0 (`DEC-066-ZEROMODEL`) on the 30 km/s identity arm should stay +35553 within 1; do not invent a new zero model.

8. `minor` — `ICO_FITS` in `sb.py` is still the laptop path. The script must pass both CANFAR `ico_path`s. Leaving default is exponential-fallback, not a 30 km/s lock.

## Residual risks

1. Same product class, not a clean \(\Delta v\) A/B (beam, cell, stamp, finite mask, BPA, CLEAN residual). Propose 1. Tightened: \(K\) itself is not a second confounder **if** both arms use \(0.02\times\mathrm{peak}\) (identical \(K=4\times10^{-4}\)). It **is** the confounder if execute follows empty-corner as typed. Comment 1. (new)

2. \(\chi^2\) is conditional at official MAP \(\theta\). Propose 2.

3. \(\Delta\chi^2=-9\) is a 1-dof 3\(\sigma\) convention. S2 SBC failed 68/95. Propose 3. Tightened: apply it on 50–146 kλ, not all \(B>50\,\mathrm{k}\lambda\). Comment 3. (new)

4. Vis remain 30 km/s channels. Propose 4.

5. Pad 180→512 is safe; centroid 0.01″ is absolute. Propose 5. The omitted failure mode is NaN corners / 1709-mask literalism, not the pad. Comments 1 and 4. (new)

6. Figure A adjoint caption can vanish on re-render. Comment 6. (new)

7. A fired lock-gate still cannot change production this card (no user DEC stub). Comment 5. (new)

## STATUS updates required

- `verdict: accept`, `severity: major`
- `last_review_b:` this file
- Do not set `board: accepted` (parent tallies)
