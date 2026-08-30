---
role: reviewer
seat: a
date: 2026-08-30
agent: review-a
canon_generation: 4
ids:
  - DEC-066-INFER
  - DEC-066-SPECRESP
  - DEC-066-REPO
  - DEC-066-TARGET
verdict: accept
severity: major
propose: docs/reviews/2026-08-30-propose-ops-scratch.md
---

# Review a: Ops /scratch I/O, STATUS sync, corner plotter, senior handoff

Do not read the other seat's review file. Do not implement.

Scope check: process + a refuse-`laplace_mh` plotter, not G2/G3. Official MAP `kinuv-KGAS066-uvsign-map` stays read-only (no refit, no write). No NUTS run. No GPU. No new `DEC-*`. `DEC-066-TARGET` still 066. Live sampler stays `SAMPLER_NAME = "laplace_mh"` in `src/kinuv/infer/posterior.py`; do not relabel it NUTS. Do not plot S2 Laplace 16/50/84 as a calibrated posterior. Do not logit `RT_BOUNDS_ARCSEC=(0.5, 15)`. In-repo face is DejaVu Sans via `apply_style()`, not a second serif stack. `/scratch` is local on this host. That is accept-eligible. Execute as typed can still ship a string-gated corner that draws S2 `p16`/`p84` under `sampler="nuts"`, rsync an XLA cache onto NFS, or let a Composer/handoff chat start G2.

## Attacks / bounds

1. **`sampler == "nuts"` is a caller string, not provenance.** Architect item 4: refuse unless `sampler == "nuts"`; 1-D 16/50/84 only on a NUTS draw array; unit test `laplace_mh` raises and synthetic NUTS writes a PNG. Live code is `SAMPLER_NAME = "laplace_mh"` (`posterior.py`); `MhResult.sampler` defaults to that. There is no NUTS posterior. S2 `docs/reviews/artifacts/2026-08-29-s2/s2_mock_mcmc.json` already stores `sampler: laplace_mh` plus `p16`/`p84`/`median` on the eight Stage A names (`flux`, `pa_deg`, `vsys_kms`, `gas_sigma_kms`, `dx_arcsec`, `dy_arcsec`, `v0_kms`, `r_t_arcsec`) and already mixed: `R_hat` 1.000-1.004, `ESS` 876-1757 (accept 0.613, 0.329 eval/s). Laplace SBC n=20 failed binomial 68/95 (`rate68` `v0_kms`/`r_t` = 0.10; `pa_deg` = 0.30). G3 will reuse the same `R_hat` < 1.01 and `ESS` > 200 gates. Mixing stats therefore cannot tell `laplace_mh` from NUTS; S2 already passes them. A kwarg `sampler="nuts"` plus draws built from those `p16`/`p84` (or from `MhResult.samples`) is exactly "corner the S2 JSON intervals", which this card rejects. The synthetic-NUTS PNG test is the first object in the repo labeled `nuts`. `scripts/plot_fit_diagnostics.py` is leftover `chi2` plus optional imaging; a `--corner` flag there is a product path onto S2. **Bound:** `plot_posterior_corner` takes one metadata object that owns both `sampler` and the draw array. Refuse unless `sampler` on that object is exactly `"nuts"` (not `"NUTS"`, not `"laplace_mh"`, not `SAMPLER_NAME`, not missing). Refuse an intervals/`p16`/`p84` table with no draws. Refuse `docs/reviews/artifacts/2026-08-29-s2/` paths. Do not accept `R_hat`/`ESS` as a substitute for the label. Draws are `(n_draw, n_param)` or `(n_chain, n_draw, n_param)` on `PARAM_NAMES`; 16/50/84 lines are `np.quantile` of those draws, not S2 JSON keys. Unit tests: `laplace_mh` raises; a `p16`/`p84` dict raises; `sampler="nuts"` with S2-shaped intervals raises; synthetic draws write a PNG only under a pytest `tmp_path` (or gitignore). If a fixture PNG is committed under `docs/reviews/artifacts/2026-08-30-ops/`, the figure title must say `synthetic` and `not 066`. Do not write `2026-08-30-final-fit/`. Do not add `--corner` to `plot_fit_diagnostics.py`. `apply_style()` first (DejaVu Sans, inward ticks, `COLOUR` tokens). ASCII names. No seaborn. No viridis. No new serif stack. 2-D panels if any are scatter/density without a 68/95 claim.

2. **"Sync summaries to `/arc` or git on a timer" can move the wrong bytes.** Architect item 1-2: high-frequency writes to `/scratch` (else `/tmp` / `/dev/shm`); sync JSON / last checkpoint to `/arc` or the git artifact dir on a timer; never loop unbuffered vis over `/arc/projects` or `/arc/home`. G1 already set `JAX_COMPILATION_CACHE_DIR=/tmp/kinuv-jax-cache` and `XDG_CACHE_HOME=/tmp/kinuv-xdg` in `tests/conftest.py` (and `scripts/time_g1_jax.py`) so pytest does not stall on NFS `/arc`. `/scratch` on this host is local (same preemption class as `/tmp`: node-local; a new Skaha node does not see the old tree). An XLA compile cache is machine-specific. Rsync of `JAX_COMPILATION_CACHE_DIR` to `/arc` or `git add` of checkpoints on a timer is the NFS stall this card exists to stop, plus repo bloat. `plot_fit_diagnostics.py` already binds `SCRATCH` to `docs/reviews/artifacts/fit-diagnostics/` (gitignored preview). That name is not filesystem `/scratch`. `DEC-066-REPO`: vis stay in kinUV; do not dump native cubes as "checkpoints". **Bound:** one writable root `/scratch/kinuv-$USER/` (mode 0700) when `/scratch` exists and is writable, else `/tmp/kinuv-$USER/`. Put `TMPDIR`, `JAX_COMPILATION_CACHE_DIR`, and `XDG_CACHE_HOME` under that root. `setdefault` only after the existence/writable check; do not fail pytest when `/scratch` is absent (laptop / CI). Change both `conftest.py` and `time_g1_jax.py`. Keep `JAX_PLATFORMS=cpu` and `JAX_ENABLE_X64=1`. Never sync the JAX cache, TMP, vis arrays, or native cubes to `/arc` or git. Timer sync is small JSON only (timing, last MCMC metadata). Do not `git add` on a timer. Do not retarget `plot_fit_diagnostics.SCRATCH` at `/scratch`. Do not write `/arc/home` as `TMPDIR`. This card does not start a sampler, so "MCMC checkpoints" is docs for a later G3 card, not a run now.

3. **Composer 2.5 STATUS push + senior handoff can start G2 or rewrite the board.** Architect item 3 and 5: keep the 8-10 line `## Agent Run Status` block; a Composer 2.5 (`composer-2.5-fast`) subagent may refresh it, commit `STATUS.md`, and push `origin/dev`; write `docs/reviews/2026-08-30-handoff-senior.md`. `DEC-066-AGENTS`: parent proposes; dual review; parent tallies; parent implements and commits after each stage. Reviewers do not implement. Neither agent creates a `DEC-*`. `board: accepted` is a parent tally, not a STATUS-refresh side effect. AGENTS mailbox today: G1 landed (`chi2=168675.6`, 3.01 eval/s); next G2 is a **separate propose**. A handoff that says "JAX exists, plot Bayesian 16/50/84, workspace is stable" is a license to logit `r_t` or launch NumPyro. Official Stage A sits on `r_t` = 0.5 arcsec (`RT_BOUNDS_ARCSEC` floor); G0 already flags `r_t_at_floor`. A 16/50/84 line on a wall-piled `r_t` is not an interior interval (S2 real-066 table already hugs 0.5). **Bound:** Composer 2.5 may edit only the `## Agent Run Status` block. It must not edit YAML `board`, `next_role`, `last_propose`, `last_review_a`, `last_review_b`, `pending`, `deadlocks`. It must not write review files, create `DEC-*`, skip hooks, or push while `board: open`. Parent still writes physics mailbox lines. ntfy stays fail-open, no secrets. Handoff must name G1 as the last landed wave and list forbidden: no NUTS run; no `sampler: nuts` on a product; no G2 logit of `(0.5, 15)`; no G3/G4/GPU; no unfreeze `i` / `h_z`; official MAP read-only; S2 stays `laplace_mh`; do not quote Laplace 68/95. Next science card is a new propose, not this handoff.

## Comments

1. `major` -- Corner gate is provenance, not `sampler="nuts"` as a kwarg. Same object as the draws. Refuse `laplace_mh`, `SAMPLER_NAME`, S2 paths, and `p16`/`p84` tables. `R_hat` / `ESS` do not distinguish S2 from NUTS. Synthetic PNG: title `synthetic` / `not 066`; pytest `tmp_path` preferred. No `--corner` on `plot_fit_diagnostics.py`. `apply_style()` / DejaVu Sans / ASCII / no seaborn.

2. `major` -- `/scratch/kinuv-$USER` (0700) if writable, else `/tmp`. Move `TMPDIR` + JAX cache + `XDG_CACHE_HOME` together. Do not sync the XLA cache or vis to `/arc` or git. Timer sync = small JSON only. `conftest.py` and `time_g1_jax.py`. Tests stay green without `/scratch`. Do not confuse `plot_fit_diagnostics.SCRATCH` with `/scratch`.

3. `major` -- Composer 2.5 edits only the Agent Run Status block. No YAML board fields. No push while the board is open. Handoff forbids G2/G3/NUTS/`sampler: nuts`/logit floor/official MAP write.

4. `minor` -- Keep `JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`. Do not start G2/G3. Do not refit. Do not write `kinuv-KGAS066-uvsign-map`. Point AGENTS / field-guide / survey-readiness at `docs/diagnostics/scratch.md` (ASCII). CHANGELOG + one STATUS line after execute.

5. `minor` -- If the plotter later sees real NUTS draws piled on `r_t` = 0.5, do not draw 16/50/84 as an interior interval. Not this wave: no logit, no G2 chart.

## Residual risks

1. `/scratch` is node-local. Preempt or a new Skaha node drops unsynced files. Resume is the small JSON copy on `/arc` or git, not the JAX cache.

2. A later NUTS corner on real 066 still overstates leftover-vs-velocity (G0 `leftover_chi2_structured`). `T_dof = 1.0077` is not that leftover. Do not quote S2 Laplace 68/95.

3. Composer 2.5 can still draft a STATUS line that reads like a physics decision. Parent owns mailbox science.

4. This accept does not retag `kinuv-KGAS066-uvsign-map`. Read-only. `SAMPLER_NAME` stays `laplace_mh`.

## STATUS updates required

- `verdict` and `severity` as in the header (`accept`, `major`)
- `last_review_a`: this file
- Do not set `board: accepted` (parent tallies)
