---
role: proposer
date: 2026-09-06
agent: parent
canon_generation: 4
ids:
  - DEC-066-TARGET
  - DEC-066-INFER
  - DEC-066-INC
  - DEC-066-AGENTS
  - DEC-067-RUNNER
verdict: propose
---

# KGAS007 NUTS chain-2 relaunch and four-shard merge

## Scope

Existing DEC ids only. **No new `DEC-*` id.** Leftover execute of the accepted 007 NUTS card (`nuts-kgas007`), not new physics. Official MAP `kinuv-KGAS066-uvsign-map` stays read-only. `steal_latest("nuts-kgas007")` stays False. Dest stays `docs/reviews/artifacts/2026-09-05-kgas007-nuts/` (not G3, not leftover `pa25`). `DEC-066-INC` stays frozen at 007 `i_rad=0.5044`. `quote_inner_slope: false`. Do not amend DEC-067 items 3–4. Do not start G4. Do not start a 066 NUTS. Do not push `main`.

Licensed this card: relaunch **chain 2 only**, wait for a valid sentinel + uncorrupted `chain_2.json`, then merge four finite shards. Do not merge three shards as the official product.

## What changed / what was checked

- c1 `b1mqxsov`, c3 `y5tspgit`, c4 `zq1olquy` are `SUCCEEDED` (600×6 each, `.trigger_complete` present). Leave those run dirs read-only.
- c2 `xkytxih1` (`KGAS007-20260905T141746Z-nuts-kgas007-c2`) `CRASHED` at startup: four workers raced `write_json` on shared `results/KILOGAS007/kinuv-KGAS007-nuts/status.json` (`os.replace` of a missing `.tmp`). No samples. Keep that dir as evidence; **do not reuse it**.
- Live worker already writes `PRODUCT/status_c{N}.json` when `KINUV_CHAIN_ID` is set. A single-chain relaunch does not hit the four-writer race.
- MAP-θ identity still holds: 007 χ² = 122070.76 on 956×66 at `i_rad=0.5044`; official 066 χ² = 168675.60.
- `KGAS066-latest` still points at the receding 066 run.

## Rejected alternatives

- Merge 1/3/4 as `sampler: nuts` (mix gate requires four finite shards).
- Reuse the crashed `xkytxih1` run dir.
- `--kind nuts` (exit 2 on 007; would steal G3 / 066 worker).
- Patch `write_json` this card unless the new c2 dies the same way.
- New 066 NUTS, G4, unfreeze \(i\), steal `KGAS066-latest`.

## Residual risks

1. NFS `write_json` tmp race can still hit shared `PRODUCT/{identity,summary,kgas007_nuts}.json` if two writers overlap. Single-chain relaunch does not overlap live siblings.
2. If the new c2 crashes again, stop and report. Do not invent a third kind. Do not merge 1/3/4 as official NUTS.
3. `chain_2.json` can exist as truncated JSON from an incomplete write. Merge must not proceed on a parse failure or a shape other than `[600, 6]`.
4. S2 SBC failed 68/95. `intervals_calibrated: false`. Do not quote inner \(dV/dr\).

## Execute if accepted

1. Dry-run, then launch **exactly** this argv (no `--kind nuts`, no extra flags that change kind/galaxy/chain):

```bash
python scripts/launch_headless.py \
  --galaxy KGAS007 --kind nuts-kgas007 \
  --chain-id 2 --skip-pull
```

New `run_id` (`KGAS007-{utc}-nuts-kgas007-c2`). Flexible CPU. `--skip-pull` so the job does not revert to an old worker. `point_latest` is not called.

2. STATUS mailbox: replace `xkytxih1` in YAML `pending` with the **new session id only**. Do not put `b1mqxsov` `y5tspgit` `zq1olquy` back in `pending`. Agent Run Status must mark c1/c3/c4 complete and only the new c2 as running. Append the new session/run to `dispatch.json` + README. Keep crashed `xkytxih1` in the README as crashed evidence. Commit+push that mailbox/dispatch edit.

3. Wait. Before `merge_nuts_chains.py`: the new c2 dir must have `.trigger_complete` **and** `logs/chain_2.json` must parse as JSON with `z6_shape` exactly `[600, 6]`. Reject truncated/corrupted JSON. Crashed `xkytxih1` stays out of the merge list.

4. Merge with the **new** c2 run id (not the crashed dir):

```bash
python scripts/merge_nuts_chains.py \
  KGAS007-20260905T141720Z-nuts-kgas007-c1 \
  <NEW-c2-run-id> \
  KGAS007-20260905T141754Z-nuts-kgas007-c3 \
  KGAS007-20260905T141801Z-nuts-kgas007-c4 \
  --kind nuts-kgas007 \
  --artifact-dir docs/reviews/artifacts/2026-09-05-kgas007-nuts \
  --map-json /arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS007/kinuv-KGAS007-stage-a-map/stage_a_map.json
```

`sampler: nuts` only if four finite shards and \(\hat{R}\le 1.01\), ESS/tail > 400; else `COMPLETED_UNMIXED`. Expect `kgas007_nuts.json`, `summary.json`, `wall.json`. Human plots in the same artifact dir.

5. Mailbox close: clear `pending` if merge wrote the product. Phase = 007 NUTS merged (or unmixed). AGENTS + Field Guide: 007 product path; do not start G4. CHANGELOG one-liner. Commit+push `origin/dev`. Do not merge to `main`.

## STATUS updates required

- `next_role: board`
- `board: open`
- `last_propose:` this file
- Keep `pending` as the four original ids until execute replaces `xkytxih1` with the new session
