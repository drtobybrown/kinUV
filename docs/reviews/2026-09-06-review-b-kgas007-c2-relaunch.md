---
role: reviewer
seat: b
date: 2026-09-06
agent: review-b
canon_generation: 4
ids:
  - DEC-066-TARGET
  - DEC-066-INFER
  - DEC-066-INC
  - DEC-066-AGENTS
  - DEC-067-RUNNER
verdict: accept
severity: major
propose: docs/reviews/2026-09-06-propose-kgas007-c2-relaunch.md
---

# Review b: KGAS007 NUTS chain-2 relaunch and four-shard merge

Do not read the other seat's review file. Do not implement.

Execute-as-typed cannot steal `KGAS066-latest`, write G3, use `--kind nuts`, merge three shards as `sampler: nuts`, or relaunch all four chains. Launch argv, `steal_latest("nuts-kgas007") is False`, entrypoint `*kgas007*` branch, and merge `--kind nuts-kgas007` dest/map refuses were verified in source. On-disk c1/c3/c4 are finite 600×6; crashed `xkytxih1` has no samples. Accept the leftover execute. Implementer must close the mailbox and merge gates below during execute — not treat the propose checklist as a substitute for code.

## Attacks / bounds

1. **Watcher will clear YAML `pending` before the four-shard merge (omitted residual).** Execute step 1 is exactly `launch_headless.py --galaxy KGAS007 --kind nuts-kgas007 --chain-id 2 --skip-pull` (no `--no-watch`). `launch_headless.main` starts `watch_headless.py` whenever submit succeeds. On `.trigger_complete`, the watcher calls `write_job_status_md` with the shard `status.json`. The single-chain worker writes `state=SUCCEEDED`, `sampler=pending_merge`, `mixing_pass=False`. `write_job_status_md` then sets `pending: []` for any `SUCCEEDED` / `COMPLETED_UNMIXED` and rewrites Phase to `007 NUTS SUCCEEDED`. That fires as soon as chain 2 lands, not after merge. Propose step 2 wants `pending = [<NEW>]` only while c2 runs; step 5 clears `pending` only if merge wrote the product. Execute-as-typed violates that mailbox contract automatically. Blockers text is also wrong-galaxy: `mixing failed; do not treat 066 JSON as calibrated NUTS`. Bound: `--no-watch` on this relaunch, **or** `write_job_status_md` must leave `pending` untouched when `sampler == pending_merge` / single-chain 007; do not rewrite Phase as product-complete until merge.

2. **Merge preflight is a human checklist; `merge_nuts_chains.py` does not enforce it (missing gate + test).** `_load_chain` reads `checkpoints/chain_{id}.npz` and uses `json.loads` only for `elapsed_s`. It does not require `.trigger_complete`, does not read `z6_shape`, and does not assert `z6.shape == (600, 6)`. `nargs` accepts 3 dirs (`need at least 3`). `mix_pass` needs `n_kept == 4`, so three shards cannot become `sampler: nuts` (`product_record` → `nuts_unmixed`). The script will still write `kgas007_nuts.json` / `summary.json` / `wall.json` into the official 007 artifact dir as `COMPLETED_UNMIXED`. Propose residual 3 and execute step 3 are the only refuse for truncated JSON or a short draw. `tests/test_canfar_runner.py` has G3 / 066-MAP refuses; there is no test that a 007 merge without four triggers or with `z6_shape != [600, 6]` exits before writing the product. Bound: for `--kind nuts-kgas007`, after the existing dest/map refuses, require exactly four run dirs, each with `.trigger_complete`, parseable `logs/chain_{id}.json` whose `z6_shape` is exactly `[600, 6]`, and `npz` `z6.shape == (600, 6)`; `JSONDecodeError` or any other shape is a hard refuse (no artifact write). Add that test. Do not treat a 3-shard `COMPLETED_UNMIXED` file on `docs/reviews/artifacts/2026-09-05-kgas007-nuts/` as the official 007 product.

3. **Propose overstates the live worker; tighter crash bound.** Crash traceback (`xkytxih1` `crash.log` / `worker.log`, 2026-09-05T14:19:52Z) is `write_json` at `run_kgas007_nuts_headless.py:268` replacing `…/kinuv-KGAS007-nuts/status.json.tmp` → `status.json`. PRODUCT on disk is only `identity_chi2.json` and `status.json` (`chain_id: 4`, c4 run id). There are no `status_c{1,3,4}.json`. The three SUCCEEDED workers wrote the same shared `status.json`; c2 lost the `os.replace` race. Current source writes `PRODUCT/status_c{N}.json` when `KINUV_CHAIN_ID` is set (lines 268–285). That split is a post-crash working-tree change, not what 973f99 ran. `--skip-pull` will pick up the current file, so a single-chain relaunch does not re-enter the four-writer PRODUCT race **if** execute uses this tree. `write_json` tmp names remain deterministic (`path.suffix + ".tmp"`). Heartbeat and main both call `write_status` on the run-dir `status.json` after compile; c1/c3/c4 survived that window (~3478 / 3823 / 4209 s). Do not patch `write_json` this card unless the new c2 dies `FileNotFoundError` on a `.tmp` again; then unique tmp (pid/thread) before a second relaunch. Do not invent a third kind.

On-disk (this seat, not the propose):

| run | session | state | trigger | `z6_shape` / npz | notes |
|---|---|---|---|---|---|
| `…T141720Z-…-c1` | `b1mqxsov` | SUCCEEDED | yes | `[600, 6]` / `(600, 6)` finite | `point_latest: false`; dest 007 artifacts; PA 151.60 |
| `…T141746Z-…-c2` | `xkytxih1` | CRASHED | no | no `chain_2.json`, empty `checkpoints/` | startup `status.json.tmp` race; keep as evidence |
| `…T141754Z-…-c3` | `y5tspgit` | SUCCEEDED | yes | `[600, 6]` / `(600, 6)` finite | |
| `…T141801Z-…-c4` | `zq1olquy` | SUCCEEDED | yes | `[600, 6]` / `(600, 6)` finite | |

`KGAS066-latest` → `KGAS066-20260831T194009Z-nuts` (`sd3ckpf2`, receding). No `KGAS007-latest`. Manifests: `kind=nuts-kgas007`, `point_latest: false`, `KINUV_SKIP_PULL=1`. Artifact dest `2026-09-05-kgas007-nuts` (not G3, not `pa25`). c1 identity χ² = 122070.7626 at `i_rad=0.50440015` on 956×66. `dispatch.json` `merge_when_complete` still lists the crashed c2 run id — rewrite it to the new c2 before anyone can run that argv.

Hard-constraint check on execute-as-typed: CLI is `--kind nuts-kgas007` (not `nuts`); `refuse_007_kind` exits 2 on `--kind nuts` for 007; `steal_latest` is False so `point_latest` is not called; entrypoint dispatches `run_kgas007_nuts_headless.py`; merge argv names four dirs, `--kind nuts-kgas007`, 007 dest, 007 Stage A JSON (G3 dest and 066 MAP refuse in script). Chain 2 only. Official 066 MAP unread as a write target. No G4.

`DEC-066-INFER` still has the waived 007 mock-recovery precondition from the accepted leftover card; this propose does not restore it. Not a reject of a c2 relaunch.

## Comments

1. **major.** Execute step 1 must not let the default watcher clear `pending` or mark 007 NUTS complete on the shard sentinel. Add `--no-watch` (does not change kind/galaxy/chain) **or** gate `write_job_status_md` so `sampler=pending_merge` / single-chain 007 does not set `pending: []` and does not rewrite Phase as product-complete. After launch, YAML `pending` is the new session id only — not `b1mqxsov` `y5tspgit` `zq1olquy`, and not `[]` until merge writes the product.

2. **major.** Before `merge_nuts_chains.py` writes any 007 product JSON, enforce four run dirs, four `.trigger_complete` files, parseable `logs/chain_{N}.json` with `z6_shape == [600, 6]`, and `npz` shape `(600, 6)`. Reject truncated JSON. Crashed `xkytxih1` stays out of the list. Do not write `kgas007_nuts.json` from three shards into `docs/reviews/artifacts/2026-09-05-kgas007-nuts/`. Add a unit test. Host-side eyeballing is not the gate.

3. **major.** `sampler: nuts` only if four finite shards and \(\hat{R}\le 1.01\), ESS/tail > 400; else `COMPLETED_UNMIXED` / `nuts_unmixed`. Do not label a 3-kept merge `sampler: nuts`. `intervals_calibrated: false`. `quote_inner_slope: false`.

4. **minor.** Current worker writes `status_c{N}.json`; the crashed job did not. `--skip-pull` must run this tree. If the new c2 dies on `FileNotFoundError` for a `.tmp`, patch `write_json` (unique tmp) then relaunch c2 only. Do not patch `write_json` speculatively this card.

5. **minor.** Rewrite `dispatch.json` + README merge argv to the **new** c2 run id; keep `xkytxih1` as crashed evidence. The stale `merge_when_complete` still names `KGAS007-20260905T141746Z-nuts-kgas007-c2` (missing `chain_2.npz` → exit). Do not reuse that dir.

6. **minor.** New c2 uses `rng_seed = 12` (`11 + (c-1)`). Leave c1/c3/c4 read-only. Official MAP `kinuv-KGAS066-uvsign-map` read-only. Dest stays `docs/reviews/artifacts/2026-09-05-kgas007-nuts/`. Do not start G4. Do not start a 066 NUTS. Do not push `main`.

## Residual risks

1. NFS `write_json` race on any shared path (PRODUCT or run-dir `status.json.tmp` vs heartbeat). Single-chain + `status_c2.json` avoids the four-writer PRODUCT collision; intra-process run-dir race remains. If it recurs, stop and report.

2. If the new c2 crashes again, do not invent a third kind; do not merge 1/3/4 as official NUTS.

3. `chain_2.json` can be truncated; merge must refuse parse failure or any `z6_shape` other than `[600, 6]` (comment 2).

4. S2 SBC failed 68/95. `intervals_calibrated: false`. Do not quote inner \(dV/dr\).

5. Watcher / `write_job_status_md` can still clobber Agent Run Status with shard-complete language if comment 1 is skipped.

6. `DEC-066-INFER` mock-recovery remains waived for 007; 16/50/84 on the merged product are not calibrated.

## STATUS updates required

- `verdict: accept`, `severity: major` (this file)
- `last_review_b: docs/reviews/2026-09-06-review-b-kgas007-c2-relaunch.md`
- Do not set `board: accepted` (parent tallies)
- After launch (implementer): `pending: ["<NEW_SESSION>"]` only; drop `xkytxih1`; do not keep `b1mqxsov` `y5tspgit` `zq1olquy`
- After merge writes the product: `pending: []`
