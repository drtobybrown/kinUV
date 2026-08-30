---
role: proposer
date: 2026-08-30
agent: parent
canon_generation: 4
ids:
  - DEC-066-INFER
  - DEC-066-SPECRESP
  - DEC-066-REPO
  - DEC-066-TARGET
verdict: propose
---

# Ops: /scratch I/O, STATUS sync, corner plotter, senior handoff

## Scope

G1 JAX identity is landed (`b67365f`). This card is **process + a plotter**, not G2/G3. Existing ids only. Official MAP `kinuv-KGAS066-uvsign-map` stays read-only. No NUTS run. No GPU. `DEC-066-TARGET` still 066.

This host has local `/scratch` (6.1 T free) distinct from NFS `/arc`. JAX compile cache today sits under `/tmp`. User asked for TMPDIR on `/scratch`, regular checkpoints off NFS, Composer 2.5 STATUS push, Bayesian corner plots with 16/50/84 lines, and a senior-agent handoff prompt.

## Architect verdict (selected path)

**Do this card:**

1. Document CANFAR I/O: high-frequency writes (JIT cache, TMP, MCMC checkpoints, run logs) go to `/scratch` (else `/tmp` or `/dev/shm`). Sync **summaries** (JSON, last checkpoint) to `/arc` or the git artifact dir on a timer, not every eval. Never loop unbuffered vis arrays over `/arc/projects` or `/arc/home`.
2. Prefer `/scratch/kinuv-$USER/` for `TMPDIR` and `JAX_COMPILATION_CACHE_DIR` when `/scratch` exists; keep `/tmp` fallback (G1 already used `/tmp`). Tests `setdefault` that path in `conftest.py`.
3. Keep the 8-10 line `## Agent Run Status` block. After each landed stage, a Composer 2.5 (`composer-2.5-fast`) subagent may refresh that block, commit `STATUS.md` (+ AGENTS mailbox one-liner if needed), and push `origin/dev`. Parent still writes physics mailbox lines. Fail-open ntfy stays.
4. Add `plot_posterior_corner` in `kinuv.diagnostics.figures` (style guide, ASCII names, inward ticks). **Refuse** unless `sampler == "nuts"`. 1-D 16/50/84 lines only on a NUTS draw array. Stage A names: `flux`, `pa_deg`, `vsys_kms`, `gas_sigma_kms`, `dx_arcsec`, `dy_arcsec`, `v0_kms`, `r_t_arcsec`. Do not import seaborn. Unit test: `laplace_mh` raises; synthetic NUTS draws write a PNG.
5. Write `docs/reviews/2026-08-30-handoff-senior.md` plus the chat handoff prompt. Workspace is a stable G1 checkpoint.

**Reject this wave:**

- Plotting S2 `laplace_mh` draws as a "Bayesian posterior" with 16/50/84 credible intervals. Laplace SBC failed 68/95. Label remains `laplace_mh`, not NUTS.
- Starting G2 logit of `RT_BOUNDS_ARCSEC=(0.5, 15)` (MAP sits on the wall; G0 `r_t_at_floor`).
- Starting G3 NumPyro, G4 SBC, GPU, a 400-galaxy runner, unfreeze `i`, `h_z`.
- High-frequency checkpoint of native vis cubes onto `/arc`.
- New `DEC-*`. Overwriting the official MAP tree.

Human plot folder stays `docs/reviews/artifacts/2026-08-30-final-fit/` until a later NUTS card writes corners there. This card may write a **fixture** PNG under `docs/reviews/artifacts/2026-08-30-ops/` (synthetic nuts only) or gitignore preview.

## What changed / what was checked

- `/scratch` exists and is local (not the Ceph `/arc` overlay).
- G1 identity: official Stage A `chi2=168675.6`, 3.01 eval/s vs S2 0.329, `JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`, cache was `/tmp`.
- No NUTS posterior exists. S2 artifacts: `docs/reviews/artifacts/2026-08-29-s2/` (`sampler: laplace_mh`).
- Plotting guide: DejaVu Sans (not a second serif stack), ASCII labels, `apply_style()`, no viridis.

## Rejected alternatives

- "Corner the S2 JSON intervals" — those p16/p84 are Laplace-MH, SBC-failed; not a product posterior.
- "NUTS now that JAX exists" — G2 chart first; do not logit the production floor.
- TMPDIR on `/arc/home` — that is the NFS stall mode this card is for.

## Residual risks

1. `/scratch` is node-local. A preempted Skaha session loses unsynced checkpoints. Sync JSON to git/`/arc` on a timer is the resume path.
2. Composer 2.5 STATUS-only commits must not rewrite science mailbox history or skip hooks.
3. A NUTS corner later will still overstate real-066 leftover-vs-velocity (G0 `leftover_chi2_structured`).

## Execute if accepted

1. `docs/diagnostics/scratch.md` (ASCII). Point AGENTS.md, field-guide, survey-readiness.
2. `conftest.py` + G1 timing script: `TMPDIR` / JAX cache under `/scratch/kinuv-$USER` if `/scratch` is writable, else `/tmp`.
3. `plot_posterior_corner` + unit tests (refuse `laplace_mh`; accept `nuts` fixture). No S2 product PNG.
4. Handoff note `docs/reviews/2026-08-30-handoff-senior.md`.
5. CHANGELOG. Commit and push. Do not start G2/G3. Do not refit.

## STATUS updates required

- `next_role: board`
- `board: open`
- `last_propose:` this file
