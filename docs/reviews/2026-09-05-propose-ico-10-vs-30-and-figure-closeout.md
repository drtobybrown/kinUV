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
  - DEC-066-ZEROMODEL
verdict: propose
---

# Official 10 vs 30 Ico probe + S3 figure closeout

## Scope

Existing DEC ids only. **No new `DEC-*` id.** Licensed this card: **Phase 1** (Ico 10 vs 30 diagnostic) + **Phase 3** (publication figure honesty polish). Do not draft `DEC-066-SB-v2`. Do not unfreeze \(i\) (`DEC-066-INC` stays frozen). Do not start a new 066 NUTS. Do not start G4. Do not interrupt 007 NUTS `b1mqxsov` `xkytxih1` `y5tspgit` `zq1olquy`. Official MAP `kinuv-KGAS066-uvsign-map` stays read-only. `quote_inner_slope: false` on all real-066 artifacts. Visibility likelihood stays `DEC-066-VIS`. Production default in [`src/kinuv/forward/sb.py`](../../src/kinuv/forward/sb.py) stays the 30 km/s Ico. `BMAJ_ICO_ARCSEC = 1.30` stays.

User 2026-09-05: the 10 km/s candidate is the official pipeline Ico, not a cube moment-0.

| Map | Path | Stamp | Cell | Beam | Finite pix |
|---|---|---|---|---|---|
| Canon 30 km/s Ico | `/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/KGAS66_Ico_K_kms-1.fits` | 135² | 0.4″ | 1.295″ × 1.180″, BPA −18.3° | 1709 |
| Candidate 10 km/s Ico | `/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/10kms/KGAS66_Ico_K_kms-1.fits` | 180² | 0.3″ | 1.040″ × 0.953″, BPA −44.8° | 2871 |

Both are `BUNIT=K km/s`, `tclean` restored. Vis spectral window stays the 30 km/s vis cube; only the SB template swaps. Label the candidate **“10 km/s Ico product”**, not canon, not cube M0.

## What changed / what was checked

- Phase 0 snapshot `dbabde9` is on `origin/main` and `origin/dev`. This card continues on `dev`.
- v1.3 `10kms/` contains `KGAS66_Ico_K_kms-1.fits` (same product class as 30 km/s). Do not derive M0 from `KGAS66_clipped_cube.fits`.
- `ico_to_template` / `load_sb_template(..., ico_path=)` already take header `BMAJ`/`BMIN`/`BPA`/`CDELT`. Do not apply the 30 km/s 1709-pixel / 0.4″ constants onto the 10 km/s stamp.
- 066 leftover_gate stays **SB-dominated**. Real-066 inner slope stays unquoted.

## Rejected alternatives

- Cube-M0 as the 10 km/s template (different product class).
- Re-MAP or new 066 NUTS on the 10 km/s template.
- Drafting `DEC-066-SB-v2` this card.
- Unfreezing \(i\), \(m=2\) SB, G4, stealing `KGAS066-latest`, interrupting 007 chains.
- Changing production `sb.py` default or `BMAJ_ICO_ARCSEC`.

## Residual risks

1. Same product class (Ico), not a clean \(\Delta v\)-only A/B. Restoring beam, cell, stamp, finite mask, BPA, and CLEAN residual all differ. A “10 km/s loses” result may be beam/mask, not channel width.
2. \(\chi^2\) is **conditional** at official MAP \(\theta\). A 10 km/s template at 30 km/s \(\theta\) is not an optimized 10 km/s model.
3. Gate \(\Delta\chi^2=-9\) is a 1-dof 3\(\sigma\) **convention** on the long-baseline split only. S2 SBC failed 68/95. Not a calibrated coverage claim.
4. Vis data remain 30 km/s channels. Template swap does not change the spectral window.
5. 10 km/s Ico pad is 180 → ≥360 (512 still safe). Centroid gate 0.01″ is absolute.

## Execute if accepted

1. Script [`scripts/analysis/compare_ico_10_vs_30.py`](../../scripts/analysis/compare_ico_10_vs_30.py). Artifacts [`docs/reviews/artifacts/2026-09-05-kgas066-ico-10-vs-30/`](artifacts/2026-09-05-kgas066-ico-10-vs-30/).
2. Wiener both Ico maps with production `ico_to_template` (header beam, empty-corner \(K\), taper \(|\tilde B|<0.05\), pad ≥2× NAXIS).
3. Spatial \(P(k)\) of each deconvolved SB. Mark \(k=1/\theta_{\mathrm{beam}}\) using **that map's** header BMAJ.
4. At official MAP \(\theta\) only: `predict_binned` with each template. Report total \(\chi^2\) and the split on rows with \(B>50\,\mathrm{k}\lambda\) via `vis_uv_wavelengths`. **No re-MAP.**
5. **Gate (implementer decides, do not pause):** lock DEC-066-SB at 30 km/s unless **both** (a) long-baseline \(\Delta\chi^2=\chi^2_{10}-\chi^2_{30}<-9\) and (b) \(P(k)\) at \(k>1/\theta_{\mathrm{beam}}\) is not higher on the 10 km/s template. Archive the note. Do not draft `DEC-066-SB-v2`.
6. Phase 3: audit the ten existing PNGs under [`docs/reviews/artifacts/2026-09-05-kgas066-s3-image-benchmark/`](artifacts/2026-09-05-kgas066-s3-image-benchmark/) against [`docs/diagnostics/plotting.md`](../diagnostics/plotting.md). Re-render only for clipping, overlap, or honesty defects. Figure D stays Laplace/vis-cube \(\chi^2\) slices, not NUTS. Real moment colorbars stay **K km/s** and **km/s**. SKA panel stays an analytic volume integral. No KinMS rerun. No official MAP rewrite.
7. Commit and push `origin/dev` after propose, tally, probe, and closeout. Do not merge `dev` → `main` this card.

## STATUS updates required

- `next_role: board`
- `board: open`
- `last_propose:` this file
- Keep `pending: ["b1mqxsov", "xkytxih1", "y5tspgit", "zq1olquy"]`
