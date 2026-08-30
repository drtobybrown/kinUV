---
role: reviewer
seat: b
date: 2026-08-30
agent: review-b
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

# Review b: Ops /scratch I/O, STATUS sync, corner plotter, senior handoff

Do not read the other seat's review file. Do not implement.

Accept because this card is process plus a plotter, not G2/G3: no NUTS run, no GPU, no 400-galaxy runner, `DEC-HIER-SELFUNC` stays Phase 5, official MAP `kinuv-KGAS066-uvsign-map` stays read-only, `DEC-066-TARGET` stays 066, and S2 `laplace_mh` is not a 16/50/84 product. G1 identity is already landed (`b67365f`). The card as written can still ship a corner helper that plots S2 JSON intervals under a `nuts` kwarg, a Composer 2.5 commit that rewrites the physics mailbox or skips hooks, and a scratch recipe that dumps native vis cubes. Those are implementer-must-fix bounds, not a re-propose.

## Attacks / bounds

1. **`sampler == "nuts"` is not a provenance gate.** Committed S2 mock MCMC is `docs/reviews/artifacts/2026-08-29-s2/s2_mock_mcmc.json`: `"sampler": "laplace_mh"`, Stage A names with `p16`/`median`/`p84` (example `r_t_arcsec` p16=0.253 p84=0.262; `v0_kms` p16=250.11 p84=250.61). Laplace SBC failed 68/95 (STATUS: rate68 `v0_kms`/`r_t` = 0.10). Live `SAMPLER_NAME` is still `laplace_mh`. Propose execute is "unit test: `laplace_mh` raises; synthetic NUTS draws write a PNG." That passes if a Python string `sampler="laplace_mh"` raises, while still accepting (a) precomputed `intervals` from that JSON, (b) a draw array with `sampler="nuts"` overwritten by the caller, (c) `sampler="NUTS"`. There is no NUTS posterior. **Bound:** `plot_posterior_corner` in `kinuv.diagnostics.figures` requires exact `sampler == "nuts"` and a 2-D float `draws` with `shape[1] == 8` in Stage A order `flux`, `pa_deg`, `vsys_kms`, `gas_sigma_kms`, `dx_arcsec`, `dy_arcsec`, `v0_kms`, `r_t_arcsec`. Do not take p16/p50/p84 as input. Raise on any other sampler string, including `laplace_mh`, missing sampler, and interval-only dicts. Unit test must load the committed S2 JSON, assert `sampler == "laplace_mh"`, and assert the plotter raises on that object and on its `intervals`. Do not import seaborn. Call `apply_style()` (DejaVu Sans, inward ticks already in `style.py`). 1-D 16/50/84 lines from the draw array only. Fixture PNG only under `docs/reviews/artifacts/2026-08-30-ops/`; ASCII title `synthetic nuts fixture; not 066; not laplace_mh`. Never write `docs/reviews/artifacts/2026-08-30-final-fit/`. Do not start a NUTS run to obtain draws.

2. **Composer 2.5 STATUS push can skip hooks or rewrite the physics mailbox.** Propose: after each landed stage, `composer-2.5-fast` may refresh the 8-10 line `## Agent Run Status` block, commit `STATUS.md` (+ AGENTS one-liner), and push `origin/dev`. Residual 2 is a wish, not a gate. DEC-066-AGENTS: the repo is the mailbox; parent writes physics; do not skip the board. A fast model asked to "update STATUS" can rewrite Architecture mailbox history, YAML `canon_generation` / `open_questions` / `deadlocks`, or `git commit --no-verify`. **Bound:** Composer edits only the `## Agent Run Status` section up to the next ATX heading `# Architecture mailbox`. Parent still writes physics mailbox lines. `git add` those paths only. `git commit` with hooks on: no `--no-verify`, no `--no-gpg-sign`, no `HUSKY=0`, no amend, no rebase, no force push. If `git diff --cached` touches Architecture mailbox bullets or physics, abort the commit. Fail-open ntfy stays. Do not let Composer start G2/G3/GPU.

3. **Node-local `/scratch` plus unbuffered vis is a silent preempt hole, and `/dev/shm` OOMs.** Native 066 vis is 43240 x 1920 complex128 ~ 1.33 GB (STATUS inventory). Post-warmup G1 is 3.01 eval/s; writing that cube every eval is ~4 GB/s even on local disk. Hann+bin 881 x 95 complex128 is ~1.3 MB; still not a per-eval product. Propose forbids looping vis over `/arc` but names `/scratch` (else `/tmp` or `/dev/shm`) for "MCMC checkpoints" and JIT. `/scratch` is node-local: preempt drops unsynced state. `/dev/shm` is RAM. This card does not run NUTS, so there is no chain to resume; the doc will be copied into G3. **Bound:** high-frequency writes are parameter vectors, RNG, chain metadata, logs, JAX compile cache (disposable). Never serialize native vis or 881x95 model cubes every eval on `/scratch`, `/tmp`, `/dev/shm`, or `/arc`. Sync JSON summaries (last theta, `chi2`, `nfev`) to `/arc` or the git artifact dir on a named timer (60 s or every N evals, whichever comes first). Do not rsync the JAX cache onto NFS. Scratch prefix `/scratch/kinuv-$USER/` if writable, else `/tmp`; never `/arc`. If a session id env exists, nest under it so two Skaha sessions on one node do not clobber TMP. `conftest.py` `setdefault` `TMPDIR` and `JAX_COMPILATION_CACHE_DIR` **before** jax import; keep G1 `JAX_PLATFORMS=cpu` and `JAX_ENABLE_X64=1`. Test: cache path is not under `/arc`. Do not re-run a 400-galaxy runner. Do not provision GPU.

Carry-forward (this execute must not reopen): do not logit `RT_BOUNDS_ARCSEC=(0.5, 15)` (official MAP on the L-BFGS floor; G0 `r_t_at_floor`). GPU only after a 066 CPU NUTS smoke (`R_hat` < 1.01, `ESS` > 200, `sampler: nuts`). G0 `leftover_chi2_structured` still stands: a later NUTS corner overstates real-066 leftover-vs-velocity. `DEC-HIER-SELFUNC` stays Phase 5. Do not unfreeze `i` or add `h_z`. Do not call `laplace_mh` NUTS.

## Comments

1. **major.** Corner plotter: exact `sampler == "nuts"` plus 8-column Stage A `draws`; refuse `laplace_mh`, missing sampler, and p16/p50/p84-only input. Unit test loads `s2_mock_mcmc.json` and raises. Synthetic fixture PNG only; not the final-fit folder; not a NUTS run.

2. **major.** Composer 2.5 may edit `## Agent Run Status` only. Hooks on. No `--no-verify`, no amend, no mailbox rewrite. Parent writes physics. Abort if the cached diff leaves that block.

3. **major.** Checkpoints are kB-MB state, not 1.33 GB native vis, including on `/scratch` and `/dev/shm`. Named 60 s / N-eval JSON sync to `/arc` because preempt drops `/scratch`. Cache never `/arc`. `conftest` sets scratch-or-`/tmp` before jax import. No GPU. No 400-galaxy runner.

4. **major.** Do not start G2, G3 NumPyro, G4 SBC, GPU, unfreeze `i`, add `h_z`, or logit `[0.5, 15]`. `DEC-HIER-SELFUNC` stays Phase 5. Official MAP read-only. Handoff note must repeat that sequence; it must not tell the next agent to run NUTS or corner S2.

5. **minor.** Keep NumPy `predict_binned` as the G1 identity reference. Do not refit. Do not regenerate leftover plots. ASCII labels via `apply_style()`; no seaborn; no viridis.

6. **minor.** JAX compile cache on `/scratch` is disposable after preempt; do not treat a cache miss as a physics stop. Post-warmup eval/s is not a reason to sync the cache to NFS.

## Residual risks

1. `/scratch` is node-local. A preempted Skaha session loses unsynced checkpoints and the JAX cache (propose residual 1, tightened: only JSON on `/arc` resumes; cache recompile is expected).

2. Composer 2.5 can still mis-summarize the 8-10 line block even if the mailbox is untouched. Parent reads STATUS before the next propose.

3. A NUTS corner later will still overstate real-066 leftover-vs-velocity (G0 `leftover_chi2_structured`; propose residual 3). No NUTS posterior exists.

4. Fixture 16/50/84 lines are synthetic. They must not be quoted as 066 or as calibrated S2 coverage.

5. Two sessions on one node sharing `/scratch/kinuv-$USER/` can clobber TMP unless a session id is nested (attack 3).

6. `r_t` remains on the L-BFGS floor. G2 must not logit `[0.5, 15]` as a prior. This card does not run G2.

## STATUS updates required

- `verdict` and `severity` as in the header (`accept`, `major`)
- `last_review_b`: this file
- Do not set `board: accepted` (parent tallies)
