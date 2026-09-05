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
revised: after reject (review-b) + accept-major (review-a)
---

# Live 066 S3, literature note, user-licensed KGAS007 NUTS (revise)

## Scope

Existing DEC ids only. **No new `DEC-*` id.** User 2026-09-05 is the TARGET stub: after dual accept, amend [`DEC-066-TARGET`](../decisions/DEC-066-TARGET.md) so KGAS007 is a second official galaxy; Field Guide `TARGET` line becomes `KGAS066 + KGAS007`. Do not drop 066 MAP/zero-model gates. Do not add 452, γ, GPU. Do not amend INC / PA / VC / INFER / OPS-AUTH / RUNNER.

User waives DEC-066-INFER’s 007 mock-recovery clause **for this card only**. Do not amend `DEC-066-INFER`. Official `kinuv-KGAS066-uvsign-map` stays read-only. Receding NUTS `sd3ckpf2` stays the 066 product. Approaching stays closed.

No GPU. No G4. No G5. `quote_inner_slope: false` (hard write on 007 products). `intervals_calibrated: false`. Do not quote inner `dV/dr`. Do not form `V_0/r_t`. 066 leftover_gate stays **SB-dominated**. Do **not** copy that string onto 007.

**Path isolation:** no new writes under `/arc/home/thbrown/`. Fitters: `/arc/projects/KILOGAS/analysis/toby_sandbox/external_fitters/` only. Patches: `/arc/projects/KILOGAS/analysis/toby_sandbox/patches/`. Recovery venv **read** only.

Canon (disk):

| Product | PA (deg) | V_0 (km/s) | r_t (arcsec) | chi2 |
|---|---|---|---|---|
| 066 Stage A MAP | 199.73 | 267.7 | 0.5 box | 168675.6 |
| 066 receding NUTS mean | 200.05 | **255** | **0.2239 mean** | 167486.8 |
| 007 Stage A MAP | 151.6 | 196 | 0.5 floor | 122070.76 (Δ vs V=0 = +6211.63) |

007 leftover: `leftover_chi2_structured: false` (uv 0.737 > vel 0.214).

## Revise locks (why review-b rejected)

Live `kind.py` is a two-way switch (`pa25` vs else). The string `nuts-kgas007` today: `steal_latest` True, artifact G3, PA 199.73. `make_potential` has no `i_rad` → 066 43.9°. `merge_nuts_chains.py` defaults G3 / `kgas066_nuts.json` / official 066 MAP. Entrypoint always execs the 066 worker. Dispatch before those patches is the steal.

## Track A — live image-plane S3

Isolated env at `toby_sandbox/external_fitters/` only. Runner: `kinUV/external/run_image_benchmarks.py`. Cube: `KGAS66/10kms/`. Products: `…/live_fitters/`. README first body sentence: vis χ² is the fit. PATH miss → keep S1. No `from kinms` under `src/` or `scripts/`. No pip into recovery or `$HOME`.

## Track B — literature note

[`docs/architecture/notes/2026-09-05-literature-synthesis-vis-vs-cubes.md`](../architecture/notes/2026-09-05-literature-synthesis-vis-vs-cubes.md). Not an ADR. Ground in S1 (vis r_t 0.254″ vs CLEAN M1 94.7 vs 236.7) and 066 leftover **SB-dominated**. Do **not** claim 0.2239″ is a published inner scale.

## Track C — 007 NUTS (execute order is tests → then dispatch)

After accept only:

1. Amend existing TARGET + Field Guide TARGET line. New 007 NUTS tree only. Do not rewrite `kinuv-KGAS007-stage-a-map` in place. NUTS JSON: `galaxy: KGAS007`, `dec_066_target_amended: true`, `infer_007_s1_waived: true` / `infer_mock_recovery: waived-this-card`, `quote_inner_slope: false`, `intervals_calibrated: false`.

2. Patch [`kind.py`](../../src/kinuv/runner/kind.py): `steal_latest("nuts-kgas007") is False` including dry-run; `steal_latest("nuts")` stays True; `artifact_dir_for_kind` → `docs/reviews/artifacts/2026-09-05-kgas007-nuts/` (not G3, not leftover `pa25`); `pa_init_deg` → **151.6**. Tests green **before** any `canfar create`. `--kind nuts` on a 007 launch is **exit 2**. `--galaxy KGAS007` required for this kind (default galaxy + 007 kind is exit 2). `point_latest` is not called.

3. `session_name` uses `--galaxy` (`kinuv-KGAS007-{sha6}-nuts-kgas007-c{N}`, 63-char cap). STATUS one-liner: DEC-067 items 3–4 (G3 copy / `kinuv-KGAS066-…` session) are **left** for 007 isolation; do not amend DEC-067.

4. New worker [`scripts/run_kgas007_nuts_headless.py`](../../scripts/run_kgas007_nuts_headless.py). Entrypoint execs it **only** when `KINUV_KIND=nuts-kgas007`. Default/`nuts` still the 066 worker. Loader is the landed 007 wavelength→metre path (`uv_stored: wavelengths`; refuse `u_m`; refuse `load_kgas066` and any 066 vis/MAP/Ico path). Init from 007 MAP JSON: `i_rad` frozen **0.5044** (28.9°), PA **151.6°**, vsys and `(dx, dy)` from that file. Do not edit `geometry.py` 066 ba/PA.

5. Close `i_rad` into `make_potential` / `predict_binned` / leftover (optional kwarg, default None = 066 `inclination_rad()` so 066 NUTS is unchanged). Before dispatch: MAP-θ identity `|chi2-122070.76|<1` on 956×66.

6. Do **not** run live `merge_nuts_chains.py` defaults on 007. 007 merge must pass `--artifact-dir docs/reviews/artifacts/2026-09-05-kgas007-nuts`, refuse dest containing `2026-08-30-g3-nuts`, read the **007** MAP for `(dx, dy)` / PA, write `kgas007_nuts.json` (never `kgas066_nuts.json`), `kind: nuts-kgas007`. `write_nuts_product_plots` / leftover must pass `artifact_dir=` (function default is still G3). `write_job_status_md` must not name G3 or “copy into G3” for this kind. Product tree: `/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS007/kinuv-KGAS007-nuts/`. Durable runs: `toby_sandbox/kinuv_runs/KGAS007-{ts}-nuts-kgas007-c{N}/`.

7. `sampler: nuts` only after autodiff + four finite chains + mix (`R_hat≤1.01`, ESS>400). Else `nuts_unmixed`. **Never** `laplace_mh` on this kind; autodiff fail → STATUS stop. Hard-set `quote_inner_slope: false` even if `r_t` leaves 0.5″. Do not copy `leftover_gate: SB-dominated` onto 007.

8. Dispatch 4×1 flexible CPU only after steps 2–7 tests are green. Image `skaha/astroml:latest`. `KINUV_VENV` read recovery jax. No G4.

## Rejected alternatives

- `--kind nuts` for 007. Dispatch before tests. Live merge defaults on 007 shards.
- Pip into recovery or `$HOME`. Quote 0.224″ / `V_0/r_t`. G4. GPU. New `DEC-*`. Edit `geometry.py` 066 seeds. Amend INFER. Write G3 or steal `KGAS066-latest`.

## Residual risks

1. conda-forge `bbarolo` / `kinms` missing. S3 stays S1.
2. 007 NUTS without S1 (user waiver, this card only). Not an INFER amend.
3. HTTPS push may fail; patches → `toby_sandbox/patches/`.
4. Real-066 16/50/84 stay uncalibrated. Do not start G4.

## Execute if accepted

1. Amend TARGET + Field Guide TARGET line only.
2. Isolated S3 + literature note (Tracks A–B).
3. Kind + `i_rad` on `make_potential` + 007 worker + entrypoint/`session_name` + merge refuse-G3. Unit tests green.
4. MAP-θ identity. Then dispatch 4×1. Watcher. Merge with explicit 007 dest only.
5. Commit after propose, tally, each deliverable. Push or format-patch under `toby_sandbox/patches/`. Official 066 MAP unchanged. Do not start G4.

## STATUS updates required

- `next_role: board`
- `board: open`
- `last_propose:` this file
