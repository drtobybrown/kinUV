---
role: reviewer
seat: a
date: 2026-08-30
agent: review-a
canon_generation: 4
ids:
  - DEC-066-INFER
  - DEC-066-TARGET
  - DEC-066-REPO
verdict: accept
severity: major
propose: docs/reviews/2026-08-30-propose-gold-standard.md
---

# Review a: gold-standard inference sequence (066 first)

Do not read the other seat's review file. Do not implement.

Scope check: sequence and G0 flags only. Official MAP stays `kinuv-KGAS066-uvsign-map` (read-only; no refit). No JAX rewrite this card. No emcee (`DEC-066-REPO`). No FD-HMC labeled NUTS. No 400-galaxy runner. S2 stays `laplace_mh`; Laplace SBC fail 68/95 is a width fact, not a NUTS posterior. That is accept-eligible. The execute list still ships a G0 that cannot see leftover-vs-velocity, and the parent already wrote the roadmap before dual accept.

## Attacks / bounds

1. **Execute writes the roadmap before dual accept.** Propose Scope says the human note `docs/diagnostics/gold-standard-roadmap.md` is "written on accept." Execute item 1 is "Write" that file. The file is already on disk and the propose already links it. `DEC-066-AGENTS` is propose then dual accept then execute. Pre-writing the named deliverable is execute, not propose. The on-disk G0 row is also thinner than Architect verdict item 1: it lists floor `r_t`, `Delta_chi2`, PA alias and drops leftover-vs-velocity. **Bound:** do not treat the on-disk roadmap as accepted. After tally, rewrite G0 to match the bound in attack 2. Do not grow G1-G7 in this card.

2. **G0 flags omit leftover-vs-velocity and cannot compute it from `stage_a_json`.** Architect verdict item 1 tags floor `r_t`, PA alias, `Delta_chi2` vs V=0, **and leftover-vs-velocity structure**, so a later survey cannot quote a thin-disk arctan when flags fire. Execute item 3 is `map_quality_flags(stage_a_json) -> dict` with only floor `r_t`, `delta_chi2`, PA vs 21.9 alias, i frozen. Official Stage A JSON cannot see leftover vs channel. Official leftover (`docs/reviews/artifacts/2026-08-30-final-fit/leftover_chi2.json`, sum 168675.6) is the documented misspec: flat-ish vs uv, structured vs velocity (SB / frozen Wiener Ico). S2 already said real-066 intervals stay over-narrow after `T_dof` because of that leftover. A G0 that fires `r_t_at_floor` and `i_frozen` on official 066 and stays silent on leftover-vs-velocity misses the failure mode STATUS and S2 already recorded. Official `chi2_chan` (95 chans) max/mean = 1.20, std/mean = 0.069; that ratio alone is a weak gate. The flag is the **comparison** leftover-vs-velocity vs leftover-vs-uv (bowl), not a new fit. **Bound:** `map_quality_flags` must take leftover `chi2_chan` / `chi2_row` (or the leftover json+npz). Official 066 must fire leftover-vs-velocity. Unit test: official Stage A numbers plus that leftover; PA=199.73 does not fire the 21.9 alias; a 21.9 fixture does. Do not refit. Do not overwrite `kinuv-KGAS066-uvsign-map`.

3. **JAX wave is not sized.** G1 is "move sky + Hann+bin + `chi2` onto XLA" with no eval/s, no `nfev` budget, no CPU-smoke gate. S2 FD was 0.329 eval/s, 6005 `nfev`, wall ~100 min for mock MH + SBC. G3 is 4 chains and two PA starts with no warmup, no draw count, no `R_hat` / `ESS` numbers. Field Guide gate 5 already has `R_hat` / `ESS` (S2 used `R_hat` < 1.01 and `ESS` > 200). G4 "O(100) once NUTS eval/s is known" sizes SBC after G3, so G1 has no exit. **Bound (this card):** do not start JAX. Next wave must state a CPU eval/s target vs 0.329 and reuse S2 `R_hat` / `ESS` gates before G4. Do not provision GPU. Do not call FD MH NUTS.

4. **TARGET lock vs "400 reality" language.** `DEC-066-TARGET` is KGAS066 only. MAP vs V=0 already passed (`Delta_chi2` = +35553); that does not unlock a survey runner. The card is correct that this is flags, not a runner, and it rejects hierarchical NUTS over 400 and a survey runner. Architect verdict item 1 and the on-disk roadmap section "Hard galaxies (the 400)" still talk as if a 400-galaxy product is the next sentence. Flags on 066 are in scope. A 400-galaxy later clause is not a TARGET amendment. **Bound:** G0 ships 066 flags only. No subset dispatcher. No GPU. No replica. Do not edit `DEC-066-TARGET`. Recommended stubs stay user-owned (`h_z`, unfreeze i, warp/strip, TARGET subset).

## Comments

1. `major` — G0 leftover-vs-velocity is required, not optional prose. Execute item 3 as typed cannot implement Architect verdict item 1. Ingest leftover `chi2_chan` / `chi2_row`. Official 066 must fire the leftover flag. Roadmap G0 must list it.

2. `major` — Roadmap-on-disk is pre-execute. After dual accept, align that file to this bound; do not start G1.

3. `minor` — Two PA starts (205.2 and 25.2) plus 4 chains is ambiguous (8 chains vs two 4-chain runs). Record as two Stage A NUTS runs, label `sampler: nuts` only after autodiff. Carry S2 `R_hat` / `ESS` numbers into the G3 card. Not this card.

4. `minor` — `DEC-066-INFER` already allows NUTS after MAP vs V=0 and vsys/PA/flux mock recover (S1 did). This card correctly blocks NUTS until JAX. Do not read that as a license to run FD HMC this wave.

5. `minor` — Point `docs/methodology.md` and `docs/diagnostics/survey-readiness.md` at the roadmap only after G0 matches attack 2. Do not quote S2 Laplace 68/95. Do not add emcee.

## Residual risks

1. Handshake already leaked: human docs can cite a pre-accept sequence. Parent must not treat on-disk G0 as closed.

2. Even true NUTS CIs on real 066 stay overconfident if leftover vs velocity stays (frozen Wiener Ico). Mock SBC tests the exact kernel only. S2 Laplace SBC failed 68/95 on that kernel.

3. Bijection around bound `r_t` = 0.5 arcsec does not unstick a likelihood that wanted 0.25 arcsec on S1. Floor flag is not a slope recovery.

4. GPU jax-finufft / CUDA images on Skaha can fail after a correct CPU JAX likelihood. Out of scope here.

5. Official MAP remains the product until a later MAP card. This accept does not retag `kinuv-KGAS066-uvsign-map`.

## STATUS updates required

- `verdict` and `severity` as in the header (`accept`, `major`)
- `last_review_a`: this file
- Do not set `board: accepted` (parent tallies)
