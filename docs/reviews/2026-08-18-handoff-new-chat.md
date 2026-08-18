---
role: proposer
date: 2026-08-18
agent: chat-A
canon_generation: 4
ids: []
verdict: propose
---

# Handoff: pick up in a new chat

**Read this file after** `AGENTS.md` → `field-guide/index.md` → `docs/architecture/STATUS.md`. Cursor plans without `canon_generation: 4` are stale. Do not create a `DEC-*` id.

Repo: `/Users/thbrown/kilogas/bin/kinUV/kinUV`  
Remote: `https://github.com/drtobybrown/kinUV` (private), default branch **`dev`**.  
Old Cursor branch `cursor/two-agent-ssot` was deleted.

## New-chat first actions

1. Handshake (files above). `STATUS.next_role: proposer`. `code_freeze: false` for **066-6/7**. MAP is **066-8** after those APIs. No NUTS.
2. `git checkout dev && git pull`. HEAD at handoff write was `1ec99dc` (Wave 1 complete). **Uncommitted on `dev` (commit these, they are the 066-6 mailbox):**
   - `AGENTS.md`, `field-guide/index.md`, `docs/architecture/STATUS.md`
   - `docs/reviews/2026-08-18-dispatch-066-6.md` (MAP path)
   - this file
   - Leave untracked: `benchmarks/`, `environment.yml`
3. Check whether [066-6 likelihood](04a8f326-5b9c-42f0-81f8-45ee736f04ed) and [066-7 forward+mocks](e3ed8b93-050a-417a-93d6-fa40b4f6f126) have finished. If yes: review, merge onto `dev` (expect `__init__.py` conflicts), pytest in conda env `kinuv` (`/Users/thbrown/bin/mambaforge/envs/kinuv`), push `origin/dev`. If still running, wait; do not duplicate their work.
4. Then dispatch **066-8-map** (L-BFGS on real 066, Δχ² vs V=0, `(dx,dy)` Fourier-shift then PB). Not 066-9/10/11 until MAP beats zero and mocks recover.

## What landed on `dev` (Wave 1)

| Component | Feature commit | Notes |
|---|---|---|
| 066-1 DFT + `@requires` | `b4ab50f` | Gaussian + thin-ring J0 < 1e-7 |
| mailbox → `dev` | `87fd2e3` | renamed off `cursor/two-agent-ssot` |
| 066-3 geometry | `0bbdb2c` | `i=arccos(0.721)`, PA seed 205.2° |
| 066-4 rings | `b976d19` | arctan; solid-body inner / flat outer; no 20×5 campaign |
| 066-5 NUFFT T2 | `3c34344` | jax-finufft vs DFT **5.1e-10**; Nyquist rejects 0.4″; ~92² @ 0.283″ |
| 066-2 Wiener+PB | `45d7ea8` | restoring-beam Wiener; stationary PB; Fourier shift for PB gate only |
| Wave 1 mailbox | `1ec99dc` | 52 tests passed in `kinuv` env |

Package layout: `src/kinuv/{constants,decisions,geometry,template,response,profiles,transforms}`. Tests: `PYTHONPATH=src /Users/thbrown/bin/mambaforge/envs/kinuv/bin/python -m pytest tests/ -q` (system python lacks jax-finufft).

## Path to best fit (user: “you choose”)

**Native source + replica operator + local CPU.** GPU/CANFAR is not the 066-8 job.

- Local npz `/Users/thbrown/kilogas/DR1/visibilities/KILOGAS066.npz` = CANFAR `/arc/projects/KILOGAS/analysis/toby_sandbox/visibilities/KILOGAS066.npz` (997305244 bytes). **Native 43240×1920**, N=1, `|Δv|=1.270 km/s` radio vs rest CO. Keys: `u_m,v_m,vis,weights,freqs,time,baseline,phase_dir_rad`.
- There is **no** stored 881×125 replica; uvkin aggregated at fit time. Do not fit 1920 native channels (≈7 s/eval vs 0.5 s GRID cap).
- **Operator:** trim to Ico **cube** velocity 8034–8536 km/s (`KGAS66_clipped_cube.fits`, VOPT) + 3 native guards; Hann **model** `[0.25,0.5,0.25]` on native+guards; software bin **N=4**; time 30 s, uv 10 m. Data are already correlator-Hann’d — do not Hann data.
- **Do not** trim with YAML `obs_freq_range: [224.148, 224.506]` GHz — it clips the receding side. `spectral_trim_from_imaging_cube: true` is the replica intent.
- Native line-free preview `⟨w|V|²⟩≈2.59`, `s≈0.77`. Remeasure `s` on the **fit** array (DEC-066-WEIGHT). Not YAML 0.5, not `12/29`.
- Escalate to CANFAR only if **aggregated** likelihood eval > ~0.5 s.

Ico: `/Users/thbrown/kilogas/analysis/kinms_test/kgas066/KGAS66_Ico_K_kms-1.fits`.

## In-flight workers (launched from `1ec99dc`, isolated worktrees)

**066-6** — Hann+bin likelihood. Load/trim/aggregate; `hann_then_bin`; empirical `s`; `chi2` / `chi2_zero`. Do not import uvkin (copy operators from `kilogas/analysis/uvkin/src/uv_aggregate.py` as evidence). Commit message `feat(066-6): …`.

**066-7** — forward model + mock recovery (flux, PA, vsys, 0.3″ `(dx,dy)`). Template × Gaussian LOS → Fourier shift → PB → T2. Stage A arctan. Subsample uv; do not NUFFT full 1920×43240 in tests. Real-data MAP is 066-8.

Merge protocol (same as Wave 1): combine `src/kinuv/__init__.py` exports; pytest in `kinuv` env; mailbox line in STATUS; push `origin/dev`. One component id per commit.

## 066-8 when 066-6/7 APIs exist

L-BFGS MAP on real aggregated 066. Parameters: flux, PA, vsys, σ, `(dx,dy)` (prior σ=0.5″, ±2″), Stage A arctan then rings if AIC. `(dx,dy)`: image Fourier shift then `A` at phase centre — **no** vis ramp after PB (DEC-066-SHIFT). Report **Δχ² vs V=0**, not reduced χ². Stop if MAP loses to zero (no dynesty/NUTS).

Later: 066-9 XX+YY; 066-10 NUTS if gate 2–3 pass; 066-11 one CANFAR job `kinuv-KGAS066-{sha6}-map`.

## Traps

- ADRs > Field Guide > STATUS > PLAN.md > `~/.cursor/plans/*` (architecture plan SHIFT still says vis ramp; ADR wins).
- `code_freeze` Field Guide line tracks 066-6/7 now; do not write 066-8 until 066-6/7 merged unless user says so.
- Do not import uvkin/KinMS. Standalone kinUV.
- Files >400 lines Python: `BLOATED:`.
- XX-only √2 still on the table (066-9).

## Tests / env

`conda env kinuv` has jax 0.10.2 + jax-finufft. CADC cert `~/.ssl/cadcproxy.pem` valid through 2026-09-17 (not needed for local MAP).
