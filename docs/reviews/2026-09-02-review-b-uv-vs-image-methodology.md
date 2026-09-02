---
role: reviewer
seat: b
date: 2026-09-02
agent: review-b
canon_generation: 4
ids:
  - DEC-066-TARGET
  - DEC-066-INFER
  - DEC-066-VC
  - DEC-066-ZEROMODEL
  - DEC-067-RUNNER
verdict: accept
severity: major
propose: docs/reviews/2026-09-02-propose-uv-vs-image-methodology.md
---

# Review b: UV vs image-plane methodology note

Do not read the other seat's review file. Do not implement.

Scope check: docs-only restating of landed S1 + leftover SB-dominated gate on KGAS066 is the right card. Existing ids only. No new `DEC-*`. Official MAP `kinuv-KGAS066-uvsign-map` stays read-only. `DEC-066-TARGET` still 066. No G4, no GPU, no 007, no KinMS/emcee/3DBarolo install, no logit of `[0.5, 15]`, no MAP rewrite, no type-1 adjoint. Canon leftover JSON (`docs/reviews/artifacts/2026-09-02-kgas066-leftover-and-modes/comparison.json`): Stage A MAP `168675.59555208942`, receding NUTS-mean `167486.7639374534`, Stage B `167302.18673431588`, `leftover_gate: SB-dominated`, `quote_inner_slope: false`, `intervals_calibrated: false`. Approaching session `xgepg7qy` is Running; this card must not poll it (DEC-067-RUNNER). That is accept-eligible. Execute as typed still has eight holes: Track S can become a second image-plane likelihood off `mom1.fits`; the S3 table can grow an uncalibrated 16/50/84 column; Section C citations can land as a harmonic plan; `docs/architecture/notes/` is off the INDEX rank; the uvkin KinMS notebook is a live import; the leftover dirty-residuals README does not open with the sentence the propose claims; execute names no test; STATUS/Field Guide mailbox can hide the Running NUTS job; “mode synthesis after mixing” can wait on tqdm.

## Attacks / bounds

1. **Track S “inventory v1.3 FITS as the CLEAN comparator” becomes a second image-plane likelihood if execute writes a KinMS-like residual cube from `mom1.fits`.** Propose Track S row 4: inventory `/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/` (`10kms/` Stage B vs imaging; `30kms/` Ico / vis-trim) “as the CLEAN comparator already used. Do not fit them.” On disk those trees contain `KGAS66_mom1.fits` (and mom0/mom2, clipped cubes) under both `10kms/` and `30kms/`. `src/kinuv/diagnostics/imaging.py` already says the module “is the image-plane comparison, not a second fit.” `docs/diagnostics/s1-mock.md` cube estimator is `sky_cube` → restoring beam of the 10 km/s cube → major-axis M1/M2; 3DBarolo was not on PATH. Nothing in execute items 1–5 forbids opening those FITS, subtracting a model mom1, or writing a residual cube into the leftover tree.

   **Bound:** inventory = path listing in the notes file (and leftover README if needed). Do not open `KGAS66_mom1.fits` to fit, ring, or residual. Do not write a new FITS cube, mom1 residual, or KinMS-like model cube this card. Do not call `KinMS.generate_cube`. The CLEAN comparator is the S1 restoring-beam M1/M2 table plus the already-plotted leftover D/M/R PNGs. Vis `chi2 = s * sum w |ΔV|^2` on 881×95 remains the likelihood (DEC-066-ZEROMODEL).

2. **PI S3 “parameter uncertainties” vs this card’s S3 table: refuse is correct; the table still needs a no-quantile column bound.** Propose refuses 16/50/84 (S2 Laplace SBC failed 68/95; leftover structured; `intervals_calibrated: false`). Landed `comparison.json` has no uncertainty keys. Receding `docs/reviews/artifacts/2026-08-30-g3-nuts/summary.json` has mixing (`R_hat` ≤ 1.004, ESS ≥ 889) and `"intervals_calibrated": false` and a note “16/50/84 not calibrated”; it has **no** 16/50/84 values. `kgas066_nuts.json` holds draws from which an implementer can compute quantiles in one numpy call. Methodology S3 (2026-08-29) is a different S3: baryon vs DM / halo vs stars — also not this card.

   **Bound:** the notes S3 table has **no** uncertainty column, no 16/50/84, no ± copied from NUTS quantiles, no “NUTS errors 3–5× tighter than image-plane.” Name the table “vis vs restoring-beam / CLEAN (landed)” or similar. Do not title it methodology-S3 (baryons) and do not title it PI-S3 (calibrated parameter uncertainties). Quote Δχ², `r_t` wall vs NUTS-mean 0.224″, Stage B vis gap +185, leftover uv/vel spans, and S1 M1 94.7 vs truth 236.7 only.

3. **Section C citing Franx 1994 / Schoenmakers 1997 / Trachternach 2008 / DiskFit can land as a planned harmonic decomposition.** Propose: cite those papers only as *why we are not adding those terms this wave*; leftover-vs-velocity at Stage B (uv span 0.093, vel span 0.335) is SB-dominated, not an `s_1`/`c_3` detection, not an inflow-rate formula. DEC-066-VC quoted `V_c` is Stage A arctan then 6–8 rings; no Fourier non-circular names. Gold-standard: do not treat Stage B rings as a warp. `comparison.json` already has `rings_are_not_a_warp: true`. `src/kinuv/**` has no `s_1`, `c_3`, Franx, or \(\dot{M}\) symbol.

   **Bound:** Section C may name those papers in a “not this wave” sentence. No equation for \(\dot{M}_{\rm inflow}\). No `s_1`/`c_3` amplitude, no harmonic expansion, no DiskFit parameter vector, no planned-term stub that a later implementer can copy into `kinuv.forward`. If leftover-vs-velocity stays True at Stage B, the gate stays **SB-dominated** (frozen Wiener Ico). Do not convert ΔM1 into mass inflow.

4. **`docs/architecture/notes/` is a new tree; DEC-066-INDEX rank does not include it.** INDEX (highest first): `docs/decisions/DEC-*.md` → `field-guide/index.md` → `docs/architecture/STATUS.md` → `docs/reviews/` → `PLAN.md` → Cursor plans. Glob of `docs/architecture/notes/**` is empty today. A file named `kinematic-methodology-review.md` under `architecture/` will be read as sitting next to STATUS, i.e. as an ADR. DEC-066-AGENTS: human-facing science is `docs/methodology.md`; Field Guide is an 80-line OS (on disk: 54 lines). Execute item 4 also patches the Field Guide mailbox.

   **Bound:** notes are **not** ADRs. Do not add `architecture/notes` to INDEX. Do not create a `DEC-*`. `docs/methodology.md` remains the human science page; execute item 3 may add a “Where to look” row pointing at the notes file and **must** keep `user_review` on `docs/reviews/artifacts/2026-09-02-kgas066-leftover-and-modes/` (not the notes file). Field Guide mailbox: one line that leftover + PA 25.2 is still in flight and this card is S1/methodology restating; do not paste the essay into `field-guide/index.md`; stay ≤80 lines.

5. **uvkin KinMS notebook exists and imports KinMS; kinUV execute must not run it.** Path `/arc/projects/KILOGAS/analysis/toby_sandbox/uvkin/kinMS_kgas66_example.ipynb` exists. Cell source is `"from kinms import KinMS"`. `docs/diagnostics/repos.md`: uvkin is the legacy KinMS + UVfit/emcee repo; kinUV likelihood is `chi2 = s * sum w |d-m|^2`. `tests/test_forward.py` `test_no_uvkin_or_kinms_import` already rglob-bans `from kinms` / `import kinms` / `from uvkin` / `import uvkin` in `src/kinuv/**` and `scripts/*.py`. Propose execute item 1 says “Do not import KinMS to write the file” but does not forbid `pip install kinms`, `jupyter nbconvert` of that notebook, or copying cells into `scripts/`.

   **Bound:** citing the uvkin path as “uvkin has a KinMS notebook; not a kinUV likelihood” is allowed. Execute must not run that notebook, must not `pip install kinms`, must not `import kinms`, and must not copy notebook cells into `src/` or `scripts/`.

6. **Leftover dirty-residuals README does not say what the propose claims as its first sentence.** Propose §A: “First sentence of that subsection: the leftover dirty-residuals README already forbids selling those PNGs as vis inversions.” Propose leftover dual-accept bullet: “dirty-residuals first-sentence bound.” Propose reject-list: folder `*kinms*` whose README does not open “Not KinMS.” Actual first lines of `docs/reviews/artifacts/2026-09-02-kgas066-leftover-and-modes/dirty-residuals/README.md`:

   ```
   # Restoring-beam / CLEAN-matched residuals (not a cube fitter)

   S1 already showed vis Stage A recovers inject \(r_t=0.25''\) where the CLEANed cube does not (M1 slope 94.7 vs truth 236.7 km/s/arcsec; M2 56 vs 8). 3DBarolo was not on PATH. That table: [`docs/diagnostics/s1-mock.md`](../../../../diagnostics/s1-mock.md).

   The three-way moments/spectra in the parent folder are CLEAN-matched cubes of vis models. They are not inverse Fourier transforms of residual visibilities (type-1 NUFFT is not implemented). Captions: restoring-beam / dirty residual, not a second likelihood.
   ```

   The H1 forbids a cube fitter, not vis inversion. Vis-inversion is paragraph 3, not the first sentence. The file never says “Not KinMS.” Folder name is `dirty-residuals/` (no `kinms`), which is why the leftover first-sentence “Not KinMS.” rule did not fire.

   **Bound:** this card either (a) prepends one line to **that** README — `Not KinMS. Not a vis inversion.` — as the first sentence after the H1, or (b) drops the claim that the README already forbids vis-inversion reading as its first sentence. Do not create a `*kinms*` folder. Notes §A must quote the README as it will stand after execute, not the overclaim.

7. **Execute names no test. The Python import ban does not cover a notes file.** Propose execute 1–5: create notes, S3 table, methodology pointer, STATUS/Field Guide/CHANGELOG, commit. No `tests/`. Import ban already exists (`test_forward.py` rglob on `src/kinuv` + `scripts`). A markdown posterior table named “KinMS” would pass that test.

   **Bound:** this card is docs-only: no new Python under `src/kinuv/` or `scripts/`. If the implementer adds any `.py`, the existing import ban must still pass. Add a grep gate (tiny test or execute-time `rg`) that `docs/architecture/notes/2026-09-02-kinematic-methodology-review.md` does **not** contain a KinMS posterior table (no column header `KinMS` as a fitter, no 16/50/84 quoted as intervals, no \(\dot{M}_{\rm inflow}\) equation). If execute truly adds no new code, the grep is a STATUS one-liner that the notes file was scanned; do not skip it.

8. **Opening this board while leftover approaching NUTS is still implementer-in-flight (`xgepg7qy`).** STATUS Agent Run Status now: Phase “leftover identity landed; approaching NUTS Running (`xgepg7qy`)”; Next Step “Wait for `xgepg7qy`.” DEC-067-RUNNER: worker/watcher patch Agent Run Status; they do not rewrite Architecture mailbox history. Propose execute item 4: “Patch STATUS: board tally, mailbox one-liner … `xgepg7qy` still Running.” Propose STATUS updates: “Do not clear Agent Run Status for `xgepg7qy`.” A mailbox rewrite that sets Phase to a docs-only methodology card will hide the Running job and can send `nuts-pa25` plots to the wrong dest.

   **Bound:** this card must not rewrite Agent Run Status into a docs-phase that drops `xgepg7qy` Running, leftover + PA 25.2, or dest `leftover-and-modes/pa25/`. Mailbox may add a one-liner that the methodology notes card landed. Do not retarget `KGAS066-latest`. Do not overwrite `2026-08-30-g3-nuts/`. Do not start G4. Do not set `board: accepted` in this review file.

9. **“Mode synthesis after mixing” must not become an execute item that waits on tqdm.** Propose: mode synthesis (receding χ² 167487 vs approaching) is a STATUS line **after** mixing (`R_hat≤1.01`, ESS bulk/tail >400 on six names). Execute items 1–5 do not name that wait, but residual risk 5 (“do not sit on tqdm”) is not an execute numbered stage. DEC-067: interactive agents must not block on the sampling loop.

   **Bound:** execute writes the notes file from landed receding/leftover JSON only. Do not poll `xgepg7qy`. Do not wait on mixing, tqdm, or chain 1/800. Do not pre-write a two-mode topology. Approaching vs receding comparison is a later STATUS line after mixing, not this card.

## Comments

1. `major` — Track S inventory is a path list. No residual cube / mom1 fit / KinMS-like FITS from v1.3 `KGAS66_mom1.fits`. Attack 1.

2. `major` — S3 table: no uncertainty column, no 16/50/84, no NUTS quantile copy. Do not collide with methodology S3 (baryons). Attack 2.

3. `major` — Section C: no \(\dot{M}_{\rm inflow}\) equation, no `s_1`/`c_3` amplitude. Leftover gate stays SB-dominated. Attack 3.

4. `major` — `docs/architecture/notes/` are not ADRs; INDEX unchanged; `methodology.md` stays human science; Field Guide ≤80 lines, no essay paste. Attack 4.

5. `major` — Do not run `/arc/projects/KILOGAS/analysis/toby_sandbox/uvkin/kinMS_kgas66_example.ipynb`, do not `pip install kinms`, do not copy cells into `scripts/`. Attack 5.

6. `major` — dirty-residuals README first lines do not say “Not KinMS.” and do not forbid vis inversion in the first sentence. This card: one-line fix on **that** README, or drop the first-sentence claim. Attack 6.

7. `major` — Docs-only (no new `src/`/`scripts/` Python) plus a grep gate that the notes file is not a KinMS posterior table. Attack 7.

8. `major` — Do not rewrite Agent Run Status to hide `xgepg7qy` Running. Mailbox one-liner only. Attack 8.

9. `major` — Mode synthesis is not an execute wait. Do not poll tqdm. Attack 9.

## Residual risks

1. A notes file under `docs/architecture/` can still be cited as if it outranked `methodology.md` even with the INDEX bound. Next card must not treat it as an ADR.

2. PI can tie-break and demand a live cube fitter or calibrated 16/50/84. That is a later propose, still outside `src/kinuv` and `scripts/`, with its own dual review.

3. Approaching `xgepg7qy` may fail-to-mix toward PA~200°. This card must not pre-write two-mode topology (attack 9). Fail-to-mix is a 180° result, not a stack.

4. Interactive follow-up may still poll the chain. STATUS Next Step stays “wait for `xgepg7qy`”; do not sit on tqdm.

5. S1 cube M1 94.7 used 12 major-axis inner pixels (`s1-mock.md` caveats). Restating that number is allowed; treating it as a 3DBarolo/KinMS posterior is not.

6. G4 Talts SBC would fail for physics while leftover-vs-velocity is True. Do not start G4.

## STATUS updates required

- `verdict: accept`, `severity: major` as in the header
- `last_review_b:` this file (`docs/reviews/2026-09-02-review-b-uv-vs-image-methodology.md`)
- Do not set `board: accepted` (parent tallies)
- Do not clear Agent Run Status for `xgepg7qy` (keep Running; leftover + PA 25.2 in flight)
- Do not start G4
- Official MAP unchanged
