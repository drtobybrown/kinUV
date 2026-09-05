---
role: proposer
date: 2026-09-05
agent: parent
canon_generation: 4
ids:
  - DEC-066-TARGET
  - DEC-066-INFER
  - DEC-066-INC
  - DEC-066-PA
  - DEC-066-VC
  - DEC-066-ZEROMODEL
  - DEC-067-RUNNER
verdict: propose
---

# Live 066 S3, literature note, user-licensed KGAS007 NUTS

## Scope

Existing DEC ids only. **No new `DEC-*` id.** User 2026-09-05 (this chat) is the TARGET stub: after dual accept, amend [`DEC-066-TARGET`](../decisions/DEC-066-TARGET.md) so KGAS007 is a second official galaxy; Field Guide `TARGET` line becomes `KGAS066 + KGAS007`. User also waives DEC-066-INFER’s 007 mock-recovery clause **for this card only**. Official `kinuv-KGAS066-uvsign-map` stays read-only. Receding NUTS `sd3ckpf2` stays the 066 sampling product. Approaching stays closed (`pa25/failure.md`).

Four execute tracks if the board accepts. No GPU. No G4. No G5. `quote_inner_slope: false`. `intervals_calibrated: false`. Do not quote inner `dV/dr`. Do not form `V_0/r_t` from the uncalibrated 066 NUTS **mean** `r_t=0.2239″`. Leftover gate on 066 stays **SB-dominated**. Do not write G3. Do not steal `KGAS066-latest`.

**Path isolation:** no new writes under `/arc/home/thbrown/`. Isolated Barolo/KinMS env is `/arc/projects/KILOGAS/analysis/toby_sandbox/external_fitters/` only. Patches: `/arc/projects/KILOGAS/analysis/toby_sandbox/patches/`. Recovery venv may be **read** as `KINUV_VENV` for jax; do not pip into it.

Canon (disk, not chat):

| Product | PA (deg) | V_0 (km/s) | r_t (arcsec) | chi2 |
|---|---|---|---|---|
| 066 Stage A MAP | 199.73 | 267.7 | 0.5 (L-BFGS box) | 168675.6 |
| 066 receding NUTS mean | 200.05 | **255** | **0.2239 mean** | 167486.8 |
| 007 Stage A MAP | 151.6 | 196 | 0.5 floor | 122071 (Δ vs V=0 = +6212) |

007 leftover: `leftover_chi2_structured: false` (uv span 0.737 > vel 0.214). That is not the 066 SB-dominated gate.

## Track A — live image-plane S3

Isolated env at `toby_sandbox/external_fitters/` (never recovery, never `$HOME`). Install `bbarolo` and `kinms` there only. Runner: `kinUV/external/run_image_benchmarks.py` (not `src/`, not `scripts/`). Cube: `/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/10kms/`. Products: `docs/reviews/artifacts/2026-09-05-kgas066-s3-image-benchmark/live_fitters/`. Rebuild `s3_table.json`. README first body sentence: vis χ² is the fit. If tools still miss: STATUS one-liner and keep S1. No `from kinms` under `src/` or `scripts/`.

## Track B — literature note

[`docs/architecture/notes/2026-09-05-literature-synthesis-vis-vs-cubes.md`](../architecture/notes/2026-09-05-literature-synthesis-vis-vs-cubes.md). Not an ADR. CLEAN/beam covariance; Barolo/KinMS inner-beam coupling; visibility forward-modeling. Ground in landed S1 (vis r_t 0.254″ vs CLEAN M1 94.7 vs 236.7) and leftover **SB-dominated**. Do **not** claim 0.224″ is a published 066 inner scale.

## Track C — amend TARGET then 007 NUTS

After accept only: amend `DEC-066-TARGET` (existing id). Do not edit `geometry.py` 066 ba/PA. 007 still uses catalogue/MAP **overrides** (`i=28.9°`, PA init **151.6°**, vsys radio from the 007 MAP).

New kind **`nuts-kgas007` only**. Live `steal_latest("nuts")` is True — `--kind nuts` on 007 is forbidden (retargets `KGAS066-latest`, writes G3). Kind must: `steal_latest` False; artifact `docs/reviews/artifacts/2026-09-05-kgas007-nuts/` (not G3); `session_name` uses `--galaxy KGAS007`; entrypoint execs a 007 worker. Product tree: `/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS007/kinuv-KGAS007-nuts/`. Init from the landed 007 MAP JSON. Wavelength→metre loader. Frozen `i_rad`. 4×1-chain flexible CPU (`DEC-067-RUNNER`). `sampler: nuts` only after autodiff + mix (`R_hat≤1.01`, ESS>400); else `nuts_unmixed`. No 007 G4.

## Rejected alternatives

- `--kind nuts` for 007 — steals `KGAS066-latest` and writes G3.
- Pip kinms/bbarolo into `kinuv-venv-recovery` or `$HOME`.
- Quote 0.224″ / `V_0/r_t` as science.
- Start G4. GPU. New `DEC-*` id. Edit `geometry.py` 066 seeds.
- Write `/arc/home/thbrown/external_fitters/` or `/arc/home/thbrown/patches/`.

## Residual risks

1. conda-forge `bbarolo` / `kinms` missing on the image. S3 then stays S1.
2. 007 NUTS without an S1 inject (user waiver). Reviewers may still reject on INFER.
3. `session_name` / entrypoint still hardcode KGAS066 unless patched.
4. 007 leftover is not SB-dominated; do not copy the 066 leftover_gate string onto 007 NUTS products.
5. HTTPS push may fail; patches go to `toby_sandbox/patches/`.
6. Real-066 16/50/84 stay uncalibrated. Do not start G4.

## Execute if accepted

1. Amend `DEC-066-TARGET` + Field Guide TARGET line. No new DEC id. No `geometry.py` 066 seed edits.
2. Isolated env + `external/run_image_benchmarks.py`; live_fitters; rebuild S3 table.
3. Literature note (not ADR).
4. `nuts-kgas007` kind + 007 worker + entrypoint/`session_name`; unit tests on steal_latest; dispatch 4×1 CPU; watcher; merge later.
5. Commit after propose, tally, and each deliverable. Push or `format-patch` to `toby_sandbox/patches/`. Official 066 MAP unchanged. Do not start G4.

## STATUS updates required

- `next_role: board`
- `board: open`
- `last_propose:` this file
