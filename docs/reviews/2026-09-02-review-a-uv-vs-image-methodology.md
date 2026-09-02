---
role: reviewer
seat: a
date: 2026-09-02
agent: review-a
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

# Review a: UV vs image-plane methodology note (S1 restating; no KinMS fitter)

Do not read the other seat's review file. Do not implement.

Scope check: docs-only restating of landed 066 evidence (S1 vis vs restoring-beam; leftover SB-dominated gate; G2 chart + receding NUTS mixing). Existing ids only. No new `DEC-*`. Official MAP `kinuv-KGAS066-uvsign-map` stays read-only. `DEC-066-TARGET` still 066. No KinMS / 3DBarolo / emcee fitter. No type-1 / `F^{-1}{ΔV}` claim. No 007. No G4. No G5. No GPU. No MAP rewrite. Do not logit `[0.5, 15]`. Do not quote inner `dV/dr`. Do not quote `s_1`/`c_3`. Do not poll `xgepg7qy` (DEC-067-RUNNER). Canon chi2 is landed JSON, not a chat summary: Stage A MAP `168675.59555208942`; receding NUTS-mean `167486.7639374534`; Stage B N=7 λ=0 `167302.18673431588` (`docs/reviews/artifacts/2026-09-02-kgas066-leftover-and-modes/comparison.json`; leftover_gate `SB-dominated`; `quote_inner_slope: false`; `intervals_calibrated: false`). Propose table (168675.6 / 167486.8 / 167302.2) matches that file. Selected path (Track L notes + Track S matrix from landed products) is accept-eligible. Execute as typed can still sell a KinMS comparison from a notes header, treat `docs/architecture/notes/` as an ADR, copy a stale warmup fraction, and turn NUTS `r_t` 0.224″ into an inner slope.

## Attacks / bounds

1. **Execute items 1–2 still let the notes file grow a KinMS S3 header, and the leftover dirty-residuals README does not open with the sentence the propose cites.** On disk, `docs/reviews/artifacts/2026-09-02-kgas066-leftover-and-modes/dirty-residuals/README.md` is:

   - H1: `Restoring-beam / CLEAN-matched residuals (not a cube fitter)`
   - First body sentence: S1 inject recovery (`r_t=0.25''`; M1 94.7 vs 236.7; M2 56 vs 8; 3DBarolo not on PATH).
   - Paragraph 2: CLEAN-matched cubes, not inverse FT of residual vis (type-1 NUFFT not implemented).

   The first sentence is **not** `Not KinMS.` Paragraph 2 forbids vis inversions; the first sentence does not. Propose section A says that README “already forbids selling those PNGs as vis inversions” and tells the notes subsection to open with that claim. Copying the claim is a product lie about the first sentence. Reject-this-wave only blocks a folder named `*kinms*` whose README does not open `Not KinMS.` The notes path is `docs/architecture/notes/2026-09-02-kinematic-methodology-review.md` (no `kinms` in the folder name). Execute item 2 forbids a KinMS *parameter column* but not an S3 header, H1, or table title `KinMS` / `vs KinMS` / `3DBarolo`. Execute item 1 allows “any KinMS mention” if it cites the uvkin notebook. `tests/test_forward.py` `test_no_uvkin_or_kinms_import` rglob `src/kinuv/**/*.py` and `scripts/*.py` only (banned: `from kinms`, `import kinms`, `from uvkin`, `import uvkin`). Markdown under `docs/` is untested. A literature subagent asked to “ground in Franx / KinMS” can `pip install kinms` and still leave that test green. This propose has no literature-subagent execute path.

   **Bound:** Parent writes the notes file in this chat. Do not launch literature subagents. Do not `pip install` / import / run KinMS, emcee, or 3DBarolo to draft citations. Notes H1, YAML, and S3 headers: `restoring-beam / CLEAN cube (S1)`, not `KinMS`, `3DBarolo`, or `emcee`. No KinMS parameter column (`V_t`, clouds, `inc` as a fit, etc.). Filename stays `2026-09-02-kinematic-methodology-review.md` (no `kinms` token). Any KinMS mention is one sentence: uvkin notebook `/arc/projects/KILOGAS/analysis/toby_sandbox/uvkin/kinMS_kgas66_example.ipynb` + “not a kinUV likelihood; not executed from kinUV.” Notes section A **first sentence** (not a later paragraph, not a citation of the leftover README) is: `Not KinMS. These PNGs are CLEAN-matched cubes of vis models, not vis inversions.` Patch leftover `dirty-residuals/README.md` so its first body sentence is the same forbid; keep paragraph 2. Do not claim the leftover README already opened with that sentence.

2. **`docs/architecture/notes/` is not in DEC-066-INDEX and will be read as rank-3 architecture.** INDEX rank: ADRs > Field Guide > `docs/architecture/STATUS.md` > `docs/reviews/` > `PLAN.md`. `docs/architecture/` today is the mailbox (`STATUS.md`). A sibling `notes/` next to STATUS is a new class. Execute item 3 points methodology.md “Where to look” at that file. Human science is `docs/methodology.md` (DEC-066-AGENTS); methodology currently points “Your review folder” at `docs/reviews/artifacts/2026-08-30-final-fit/`, which is not the live leftover surface (`STATUS` `user_review` is `2026-09-02-kgas066-leftover-and-modes/`). Residual risk 1 says “no new DEC-*; methodology stays the human science page” but does not rank the notes file or retarget the stale plot pointer. Field Guide mailbox is leftover + PA 25.2; this card must not overwrite that with a notes essay.

   **Bound:** Notes open with a rank banner: not an ADR; if this file disagrees with a `DEC-*`, the DEC wins; rank below STATUS. Do not put it under `docs/decisions/`. Do not copy it into the Field Guide. methodology.md “Where to look” may add one row labeled `S1 restating (not ADR)` pointing at the notes file **and** must retarget “Your review folder” to `docs/reviews/artifacts/2026-09-02-kgas066-leftover-and-modes/`. Do not replace leftover plots as the human surface. No new `DEC-*`.

3. **“Cite Franx/Schoenmakers only as why we are not adding terms this wave” is a later `s_1` stub.** DEC-066-VC is Stage A arctan then 6–8 rings; no harmonic m=1. DEC-066-OSCMETRIC is curvature on circular `V_c`, not Fourier terms. Landed leftover README already says “Do not add \(s_1\)/\(c_3\)” with no “this wave.” comparison.json leftover_gate `SB-dominated`; `rings_are_not_a_warp: true`. Gold-standard later stubs are user-added `h_z` / unfreeze `i` / warp/strip — not a notes-file license for `s_1`. Propose reject-list forbids quoting leftover as `s_1`/`c_3`; section C still names those symbols as terms deferred to a later wave.

   **Bound:** Section C may name Franx (1994) / Schoenmakers (1997) once, as papers whose terms are **not** in DEC-066-VC. Sentence form: leftover-vs-velocity at Stage B is frozen Wiener Ico (uv span 0.093, vel span 0.335), not a harmonic detection; adding those terms needs a user DEC stub, not this notes file. Do not write a stub, a parameter table, a “next wave” sentence that licenses the symbols, or an inflow-rate formula. Do not quote `s_1`/`c_3` except in the negative.

4. **Heartbeat `~13/800` will stale; copying it into the notes file is a product lie.** Propose-time snapshot: `xgepg7qy` Running, chain 1 warmup ~13/800, `KINUV_PA_INIT=25.2`. That fraction is already wrong when the notes file is read tomorrow. Approaching mixing is not in. DEC-067: interactive agents must not poll.

   **Bound:** Notes must not quote warmup step counts, `~13/800`, tqdm, or any live chain fraction. Section B: approaching mixing is not in; do not invent it; do not pre-write a two-mode topology. STATUS may keep “`xgepg7qy` Running” without the fraction. Do not poll. Mode synthesis stays a STATUS line after mixing (`R_hat≤1.01`, ESS bulk/tail >400 on six names).

5. **S3 row “NUTS `r_t` mean 0.224 left the wall” plus `V_0` is an inner-slope quote.** comparison.json `nuts_mean.params.r_t_arcsec = 0.22392216472996415`, `v0_kms = 254.9834109292598`, `quote_inner_slope: false` at all three vis points. Propose forbids quoting inner `dV/dr` and S2 / uncalibrated 16/50/84. Execute item 2 still lists “receding NUTS vs MAP Δχ² and `r_t` wall.” `V_0/r_t` or arctan′ at 0.25 BMAJ from that 0.224″ is inner `dV/dr`. S1 (`docs/diagnostics/s1-mock.md`) may restate inject M1 94.7 vs truth 236.7 km/s/arcsec — that is the cube comparator on a known inject, not real-066.

   **Bound:** S3 NUTS row may state `r_t` mean 0.224″ left the L-BFGS 0.5″ wall and Δχ² = −1189 vs MAP. Must not compute or quote `V_0/r_t`, arctan′(r), km/s/arcsec, or copy the S1 inner-slope column onto the NUTS or Stage B rows. S1 row restates `s1-mock.md` vis `r_t` 0.254 vs truth 0.25 and restoring-beam M1/M2 only. No 16/50/84. `intervals_calibrated: false`. Quoted `V_c` stays Stage A arctan.

6. **G4 / 007 are rejected in prose; leftover structured still fails G4 for physics.** Gold-standard G4 is Talts SBC after G3 CPU smoke **and** leftover caveat. Receding G3 landed (`sd3ckpf2`). `leftover_chi2_structured: true` at MAP / NUTS-mean / Stage B. G4 on this kernel would fail for physics. Propose reject-list names G4 Talts SBC, G5 PSIS-LOO, KGAS007 G3 draft. Execute boundary: “No 007. No G4.” No execute item drafts 007. TARGET answer is KGAS066 only until a user stub.

   **Bound:** This card does not start G4 or G5 and does not draft KGAS007 G3. TARGET remains 066. Leftover structured is a G4 physics stop, not a STATUS one-liner that quietly unlocks SBC.

## Comments

1. `major` -- Notes first sentence is `Not KinMS. These PNGs are CLEAN-matched cubes of vis models, not vis inversions.` Patch leftover dirty-residuals README first body sentence to the same forbid. S3 headers = restoring-beam / CLEAN (S1), not KinMS. Parent writes the markdown; no literature subagents; no `pip install kinms`. Attack 1.

2. `major` -- Notes rank banner: not an ADR; DEC wins; below STATUS. methodology “Where to look” adds a not-ADR notes row **and** retargets the human review folder to `2026-09-02-kgas066-leftover-and-modes/`. Do not replace leftover as the user surface. Attack 2.

3. `major` -- Franx/Schoenmakers only as “not in DEC-066-VC”; leftover-vs-velocity is SB Ico, not a later `s_1` stub. Attack 3.

4. `major` -- Notes must not quote `~13/800` or any warmup fraction. Approaching mixing not in. Do not poll `xgepg7qy`. Attack 4.

5. `major` -- S3 may report NUTS `r_t` 0.224″ left the wall; must not quote inner `dV/dr` from that number. `quote_inner_slope` stays false. Attack 5.

6. `minor` -- Reject-this-wave stays: no live KinMS/3DBarolo/emcee, no type-1/`F^{-1}{ΔV}`, no 16/50/84, no NUTS-vs-image “3–5× tighter,” no logit `[0.5, 15]`, no unfreeze `i`, no `h_z`, no MAP rewrite, no G4/G5, no 007, no GPU, no retarget `KGAS066-latest`. Canon chi2 from `comparison.json`. CHANGELOG + Field Guide mailbox: leftover + PA 25.2 still in flight; this card is S1/methodology restating. Official MAP unchanged. Attack 6.

## Residual risks

1. A notes file under `docs/architecture/` is still readable as policy. Comment 2 is the rank lock; INDEX is not rewritten. Carry-forward from the propose, tightened.

2. PI can tie-break and demand a live cube fitter. Later propose, still outside `src/kinuv` and `scripts/`. Import ban remains `src/kinuv/**` and `scripts/*.py`. Carry-forward.

3. Approaching job may fail-to-mix. This card must not pre-write two-mode topology. Carry-forward. Comment 4.

4. Interactive follow-up may still poll `xgepg7qy`. STATUS Next Step stays wait; do not sit on tqdm. Carry-forward.

5. Real-066 16/50/84 remain uncalibrated (S2 Laplace SBC failed 68/95; leftover structured). Notes must say so. Carry-forward.

6. **(new)** Leftover dirty-residuals README first sentence was not landed as `Not KinMS.` Comment 1 patches it this card; until then, citing that README as already opening with the forbid is false.

7. **(new)** G4 remains a physics fail while leftover-vs-velocity is True. Do not treat receding NUTS mixing as an SBC unlock. Comment 6.

## STATUS updates required

- `verdict` and `severity` as in the header (`accept`, `major`)
- `last_review_a`: this file
- Do not set `board: accepted` (parent tallies)
- Do not clear Agent Run Status for `xgepg7qy`
- Do not start G4
- Official MAP unchanged
