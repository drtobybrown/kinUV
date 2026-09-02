---
role: proposer
date: 2026-09-02
agent: parent
canon_generation: 4
ids:
  - DEC-066-TARGET
  - DEC-066-INFER
  - DEC-066-VC
  - DEC-066-ZEROMODEL
  - DEC-067-RUNNER
verdict: propose
---

# UV vs image-plane methodology note (S1 restating; no KinMS fitter)

## Scope

PI 2026-09-02 mandate: advance past CLEAN-cube kinematics, benchmark vis NUTS against image-plane fitting, ground the architecture in literature, then 007.

This card is **docs-only on KGAS066**. Existing ids only. No new `DEC-*`. Official MAP `kinuv-KGAS066-uvsign-map` stays read-only. `DEC-066-TARGET` still 066. Leftover + approaching card stays in flight (`xgepg7qy`); this card does not retarget it, does not stack PA modes, and does not block on the sampling loop (DEC-067-RUNNER).

Canon numbers (landed JSON, already on the leftover card):

| Product | chi2 | leftover-vs-velocity |
|---|---|---|
| Stage A MAP | 168675.6 | True |
| Receding NUTS mean | 167486.8 (Δ=−1189 vs MAP) | True |
| Stage B N=7 λ=0 | 167302.2 (gap vs NUTS +185) | True |

Post-leftover gate is **SB-dominated** (frozen Wiener Ico). `quote_inner_slope` False. Quoted `V_c` stays Stage A arctan. 16/50/84 not calibrated (S2 Laplace SBC failed 68/95). Do not quote inner `dV/dr`. Do not quote `s_1`/`c_3`. Do not convert ΔM1 into \(\dot{M}_{\rm inflow}\).

## Architect verdict (selected path)

**Track L — literature note, not ADRs.**

Write `docs/architecture/notes/2026-09-02-kinematic-methodology-review.md` with three sections that **restate landed evidence**. They must not open Fourier non-circular terms, unfreeze `i`, add `h_z`, or claim a type-1 adjoint (none exists; NUFFT is type-2 degrid only).

- **A (interferometry):** Why `chi2 = s * sum w |ΔV|^2` on the 881×95 Hann+bin array is the likelihood, and why CLEAN cubes (Briggs, restoring beam, off-diagonal image covariance) are a diagnostic. Dirty residual here means `data.vis − model` and CLEAN-matched D/M/R already in `docs/reviews/artifacts/2026-09-02-kgas066-leftover-and-modes/dirty-residuals/`. First sentence of that subsection: the leftover dirty-residuals README already forbids selling those PNGs as vis inversions. Do not claim `F^{-1}{ΔV}`.
- **B (HMC / chart):** G2 unconstrained `log(r_t)` (finite at the 0.5″ MAP; **not** a logit of `[0.5, 15]`) plus receding NUTS mixing (`sd3ckpf2`, R_hat≤1.004, ESS≥889, `V_0`–`r_t` corr 0.87). Do not claim step-size / mass-matrix “guarantees” convergence. Approaching mixing is **not** in; do not invent it.
- **C (ISM kinematics):** Diagnostic already on disk: leftover-vs-velocity vs leftover-vs-uv at Stage B (uv span 0.093, vel span 0.335). That gate is SB-dominated, not a Franx/Schoenmakers `s_1`/`c_3` detection, not an inflow-rate formula. Cite those papers only as *why we are not adding those terms this wave*.

**Track S — S3 comparison matrix from landed products, not a new cube fitter.**

Same notes file (or a subsection of the leftover README). Rows that exist:

1. S1 inject on real 066 uv (`docs/diagnostics/s1-mock.md`): vis recovered `r_t=0.254″` vs truth 0.25″; restoring-beam M1 slope 94.7 vs truth 236.7 km/s/arcsec; M2 56 vs 8. 3DBarolo was **not on PATH**. That is the image-plane breakdown number. It is not a KinMS posterior.
2. Receding NUTS vs Stage A MAP: Δχ²=−1189; `r_t` mean 0.224″ left the L-BFGS 0.5″ wall; leftover still structured. Mixing pass. Intervals **not** calibrated.
3. Stage B rings vs NUTS-mean: vis gap +185; leftover-vs-velocity still True → SB leftover, not missing circular `V_c`.
4. Official v1.3 imaging root `/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/` (`10kms/` is the Stage B vs imaging cube; `30kms/` is Ico / vis-trim). Inventory those FITS as the CLEAN comparator already used. Do not fit them.

**KinMS:** do **not** install, import, or run KinMS / emcee / 3DBarolo in kinUV. Dual-accept this morning (`review-a`/`review-b` leftover) already bound that: no `from kinms` in `src/kinuv/**` or `scripts/*.py`; no folder named `kinms` sold as a posterior. Historical uvkin example `/arc/projects/KILOGAS/analysis/toby_sandbox/uvkin/kinMS_kgas66_example.ipynb` stays in uvkin (`docs/diagnostics/repos.md`). Citing it as “uvkin has a KinMS notebook” is allowed. Executing it from kinUV is not.

**Approaching NUTS:** heartbeat at propose time: session `xgepg7qy` Running, chain 1 warmup ~13/800, `KINUV_PA_INIT=25.2`, dest `leftover-and-modes/pa25/`, `KGAS066-latest` still receding. Interactive agents must not poll until completion. Mode synthesis (receding χ² 167487 vs approaching) is a STATUS line **after** mixing (`R_hat≤1.01`, ESS bulk/tail >400 on six names). Fail-to-mix toward PA~200° is a 180° result; do not stack or flip.

**Reject this wave:**

- Live KinMS / 3DBarolo / emcee fit to the v1.3 cube.
- Folder `*kinms*` whose README does not open “Not KinMS.”
- Type-1 / `F^{-1}{ΔV}` claim (no adjoint on disk).
- Quote leftover as `s_1`/`c_3` or \(\dot{M}_{\rm inflow}\).
- Quote inner `dV/dr` or S2 / uncalibrated 16/50/84.
- Claim NUTS credible intervals are 3–5× tighter than image-plane (S2 failed; leftover structured).
- Poll `xgepg7qy` in this chat.
- G4 Talts SBC. G5 PSIS-LOO. KGAS007 G3 draft. GPU. MAP rewrite. Logit `[0.5, 15]`. Unfreeze `i`. Add `h_z`. New `DEC-*`.
- Retarget `KGAS066-latest`. Overwrite `2026-08-30-g3-nuts/`.

Human review surface remains `docs/reviews/artifacts/2026-09-02-kgas066-leftover-and-modes/` plus the notes file. Not a mid-gate chat.

## What changed / what was checked

- Leftover identity: `docs/reviews/artifacts/2026-09-02-kgas066-leftover-and-modes/comparison.json` (chi2 168675.596 / 167486.764 / 167302.187; leftover_gate `SB-dominated`).
- S1 table: `docs/diagnostics/s1-mock.md` (vis vs restoring-beam, not KinMS).
- Methodology 10×: vis posteriors + sub-beam `V_c` vs CLEAN, **not** NUFFT vs KinMS clouds (`docs/reviews/2026-08-29-propose-methodology.md`; ACK `2026-08-29-review-methodology.md`).
- Leftover dual-accept major: no KinMS import; dirty-residuals first-sentence bound; `quote_inner_slope` False while leftover-vs-velocity.
- `tests/test_forward.py` KinMS import ban on `src/kinuv/**` and `scripts/*.py`.
- v1.3 KGAS66: `10kms/` and `30kms/` CLEAN products (Ico, mom1, clipped cubes). No KinMS posterior JSON in that tree.
- uvkin KinMS notebook exists and `from kinms import KinMS`; that is uvkin, not kinUV.
- `xgepg7qy` Running (warmup chain 1). DEC-067: do not block.
- Gold-standard G4 still after G3 CPU smoke **and** leftover caveat; leftover still structured → G4 would fail for physics. TARGET still 066.

## Rejected alternatives

- “Execute KinMS on the official cube because the PI named it” — contradicts this morning’s dual-accept, the import ban, and methodology 10× (S1 already scored CLEAN-beam bias). A user tie can reopen KinMS later; this propose does not.
- “Poll `xgepg7qy` until mixed, then write the note” — DEC-067. Docs do not need the approaching chain.
- “Draft KGAS007 G3 now so 007 is ready” — leftover kernel is SB-misspecified; TARGET still 066; gold-standard dual-accept: MAP vs V=0 does not unlock a survey runner.
- “Literature note that recommends adding `s_1` because leftover-vs-velocity is True” — that flag is the Wiener Ico gate, the opposite license.
- “Quote receding NUTS 16/50/84 as the calibrated vis posterior vs KinMS errors” — S2 failed; leftover structured; `intervals_calibrated: false`.

## Residual risks

1. A notes file under `docs/architecture/` can be read as a new ADR. Bound: no new `DEC-*`; notes cite S1/leftover/G2/G3 only; methodology.md stays the human science page.
2. Naming KinMS in the title or an S3 column will be sold as a KinMS posterior. Bound: table column is “restoring-beam / CLEAN cube (S1)”, not “KinMS”.
3. PI can tie-break and demand a live cube fitter. That is a **later** propose with its own dual review, still outside `src/kinuv` and `scripts/`.
4. Approaching job may fail-to-mix. This card must not pre-write a two-mode topology.
5. Interactive follow-up may still poll the 5 h chain. STATUS Next Step stays “wait for `xgepg7qy`”; do not sit on tqdm.

## Execute if accepted

Boundary: one methodology notes file + S3 table from landed JSON. No sampler. No KinMS. No 007. No G4. No MAP write.

1. Create `docs/architecture/notes/2026-09-02-kinematic-methodology-review.md` with sections A/B/C as bounded above. Any KinMS mention: uvkin notebook path + “not a kinUV likelihood.” Do not import KinMS to write the file.
2. S3 table in that file: S1 vis vs restoring-beam; receding NUTS vs MAP Δχ² and `r_t` wall; Stage B vis gap + leftover spans; v1.3 `10kms/` as the CLEAN product already plotted. No KinMS parameter column. No 16/50/84. No inner `dV/dr`. No \(\dot{M}_{\rm inflow}\).
3. Point methodology.md “Where to look” at the notes file **and** keep `user_review` on the leftover plot folder. Do not replace leftover as the human surface.
4. Patch STATUS: board tally, mailbox one-liner (docs card; KinMS not run; 007 not opened; `xgepg7qy` still Running). Field Guide mailbox: leftover + PA 25.2 in flight; this card is the S1/methodology restating; do not start G4. CHANGELOG. Official MAP unchanged.
5. Commit and push `origin/dev`. Conventional subject. Do not skip hooks. Do not commit `.cursor/` deletions.

## STATUS updates required

- `next_role: board`
- `board: open`
- `last_propose:` this file
- Do not clear Agent Run Status for `xgepg7qy`
- Do not start G4
