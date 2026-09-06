---
role: reviewer
seat: a
date: 2026-09-06
agent: review-a
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

# Review a: KGAS007 NUTS chain-2 relaunch

Do not read the other seat's review file. Do not implement.

Execute-as-typed does **not** steal `KGAS066-latest`, write G3, use `--kind nuts`, merge three shards as `sampler: nuts`, or relaunch all four chains. CLI is exactly `--galaxy KGAS007 --kind nuts-kgas007 --chain-id 2 --skip-pull`. `steal_latest("nuts-kgas007")` is False; dest is `docs/reviews/artifacts/2026-09-05-kgas007-nuts/`; entrypoint routes `*kgas007*` to `run_kgas007_nuts_headless.py` before the 066 worker. Official 066 MAP `stage_a_map.json` mtime 28 Aug, unread-write. `KGAS066-latest` → `KGAS066-20260831T194009Z-nuts`. No `KGAS007-latest`. No G4.

On-disk (not proposer-trusted):

| shard | session | state | trigger | `z6` | wall |
|---|---|---|---|---|---|
| c1 `…T141720Z-…-c1` | `b1mqxsov` | SUCCEEDED | 2026-09-05T15:19:19Z | npz+JSON `(600, 6)` finite | 3478 s |
| c2 `…T141746Z-…-c2` | `xkytxih1` | CRASHED | absent | no npz, no `identity_chi2` | — |
| c3 `…T141754Z-…-c3` | `y5tspgit` | SUCCEEDED | 2026-09-05T15:25:05Z | npz+JSON `(600, 6)` finite | 3823 s |
| c4 `…T141801Z-…-c4` | `zq1olquy` | SUCCEEDED | 2026-09-05T15:31:48Z | npz+JSON `(600, 6)` finite | 4209 s |

c2 `crash.log` / `worker.log`: `KINUV_CHAIN_ID=2`, `KINUV_KIND=nuts-kgas007`, `KINUV_SKIP_PULL=1`, dest not G3; died at first `write_json` of shared `results/KILOGAS007/kinuv-KGAS007-nuts/status.json` (`os.replace` ENOENT on `status.json.tmp`), RSS 183 MB, no samples. Keep that dir out of the merge list.

## Attacks / bounds

1. **Missing merge gate (code + test).** Propose step 3 is honor-system. `scripts/merge_nuts_chains.py` `_load_chain` requires only `checkpoints/chain_{id}.npz`. It does not require `.trigger_complete`. If `logs/chain_{id}.json` is missing, `elapsed_s` becomes NaN and merge **continues**. If the JSON exists but is truncated, `json.loads` raises `JSONDecodeError` (not a clean `z6_shape` reject). The JSON `z6_shape` field is never read. The npz `z6` array is never asserted to be exactly `(600, 6)`. `test_kgas007_merge_refuses_g3_and_066_map` only hits dest/map refuses with dummy dirs `a b c` (three names). No test refuses a missing sentinel, truncated JSON, `z6_shape != [600, 6]`, npz shape ≠ `(600, 6)`, or the crashed `xkytxih1` dir. `docs/reviews/artifacts/2026-09-05-kgas007-nuts/dispatch.json` `merge_when_complete` and README still name `KGAS007-20260905T141746Z-nuts-kgas007-c2`. Copy-paste of that list is a missing-npz exit, not a silent 3-chain `sampler: nuts` — but it is also not the typed four-shard merge.

   **Bound (execute must nail before `merge_nuts_chains.py`):** new c2 dir has `.trigger_complete` **and** `logs/chain_2.json` parses **and** `z6_shape` is exactly `[600, 6]` **and** `checkpoints/chain_2.npz` `z6.shape == (600, 6)` and finite. Crashed `xkytxih1` dir stays out. Add a unit test (or a tiny pre-merge checker the merge script calls) for those four predicates. Do not treat JSON metadata as a substitute for the npz shape.

2. **Tighter physical box the propose omitted.** c1/c3/c4 are `chain_physically_ok` (boxes flux ≤ 1e4, gas_σ ≤ 200, V_0 ≤ 400, |r_t| ≤ 15). Unconstrained `z6` max ~13555 is the vsys identity axis (MAP `vsys_kms=13554.48`), not an approaching-style explosion. Physical medians already agree:

   | | PA (deg) | V_0 (km/s) | r_t (″) |
   |---|---|---|---|
   | c1 | 151.62 (150.61–152.60) | 194.9 (189.6–201.6) | 0.481 (0.410–0.558) |
   | c3 | 151.59 (150.74–152.60) | 194.9 (188.0–200.8) | 0.483 (0.388–0.558) |
   | c4 | 151.61 (150.48–152.66) | 194.8 (189.8–201.5) | 0.481 (0.407–0.557) |

   **Bound:** if the new c2 is non-finite, fails `chain_physically_ok`, or lands outside PA ∈ [150.4, 152.7] or V_0 ∈ [188, 202], do **not** force `sampler: nuts` by averaging a discrepant shard with these three. `COMPLETED_UNMIXED` or stop. r_t medians 0.48″ already left the Stage A 0.5″ floor (same pattern as 066); `quote_inner_slope: false`; do not quote inner \(dV/dr\). Expect wall ~1.0–1.2 h (c4 = 4209 s).

3. **`write_json` tmp basename is still shared; residual 1 named the wrong files.** Crash was `…/kinuv-KGAS007-nuts/status.json.tmp` → `status.json` with `KINUV_CHAIN_ID=2` already set (`worker.log` line 12; traceback then-line 268). Live PRODUCT still has only `status.json` (chain_id 4, 14:20Z) and `identity_chi2.json` — **no** `status_c{N}.json`. So the Sep-5 jobs wrote the shared file; the `status_c{N}` split is current-tree only. A `--skip-pull` relaunch will write `PRODUCT/status_c2.json` and should not overlap completed siblings. `write_json` still uses deterministic `path.with_suffix(path.suffix + ".tmp")` with no pid. Propose residual 1 lists `identity,summary,kgas007_nuts` and omits the file that actually crashed (`status.json`) and the file the relaunch will write (`status_c2.json`). Double-launch of chain 2 re-races that tmp.

## Comments

1. **major.** Before merge, enforce (code or test, not a STATUS sentence) `.trigger_complete` + parseable `logs/chain_2.json` with `z6_shape == [600, 6]` + npz `z6.shape == (600, 6)` finite. Rewrite `dispatch.json` / README merge argv to the **new** c2 `run_id`. Keep `xkytxih1` as crashed evidence only. Do not merge three shards into the official artifact as the product.

2. **major.** Apply the physical box in Attacks/bounds §2 to the new c2 before labeling `sampler: nuts`. Four finite + \(\hat{R}\le 1.01\) + ESS/tail > 400 is necessary, not sufficient, if c2 is a different mode.

3. **major.** Preflight `results/KILOGAS007/kinuv-KGAS007-nuts/` has no `*.json.tmp`. Launch chain 2 **once**. Confirm the new worker writes `status_c2.json`, not `status.json`. If the new c2 dies with the same `os.replace` ENOENT, patch `write_json` to a pid-unique tmp **this card** (propose already allows that). Do not invent a third kind.

4. **minor.** Propose wording “`DEC-066-INC` stays frozen at 007 `i_rad=0.5044`” is sloppy. INC’s answer is 43.9° for 066. 007 uses the catalogue/MAP override 28.9° / `i_rad=0.5044` without rewriting INC (`DEC-066-TARGET`). Keep `i_rad=0.5044` on the worker; do not edit `DEC-066-INC.md`.

5. **minor.** `--dry-run` still `write_manifest` / `write_status` (SUBMITTING) before return. Real launch gets a new timestamp. Delete or ignore the orphan dry-run dir. `point_latest` is not called (`steal_latest` False); existing test covers that.

6. **minor.** Single-chain worker returns before `write_job_status_md` (`run_kgas007_nuts_headless.py` ~429–445). It will not patch YAML `pending`. Execute step 2 is mandatory: after launch, `pending` is the **new** session id only — do not keep `b1mqxsov` `y5tspgit` `zq1olquy` in `pending`.

## Residual risks

1. NFS `write_json` ENOENT on a deterministic `{name}.tmp` if two writers share a PRODUCT path (`status_c2.json` on a retry; `identity_chi2.json` if chain 1 is ever re-run). Single-chain relaunch does not overlap live siblings.
2. New c2 crashes again → stop and report. Do not merge 1/3/4 as official `sampler: nuts`. Do not reuse `xkytxih1`.
3. `chain_2.json` can be truncated; merge must not proceed on parse failure or shape other than `[600, 6]` / npz `(600, 6)`.
4. S2 SBC failed 68/95. `intervals_calibrated: false`. Do not quote inner \(dV/dr\). r_t already off the 0.5″ MAP floor on c1/c3/c4.
5. `DEC-066-INFER` mock-recovery waiver is leftover of the accepted 007 NUTS card (`DEC-066-TARGET` “this card only”). Do not amend INFER. Do not treat this relaunch as a new waiver for later 007 work. MAP Δχ² vs V=0 is +6211.63 (`chi2_map=122070.76`, `chi2_zero=128282.39`).
6. 007 Stage A JSON still says `diagnostic_only: true` / `dec_066_target_amended: false` / “No 007 NUTS”. Leave that tree read-only. Do not overwrite it.
7. DEC-067 items 3–4 stay 066-only (G3 copy / `kinuv-KGAS066-…` session). Merge default `--artifact-dir` is still G3; execute-as-typed overrides. Do not run live merge defaults.

## STATUS updates required

- `verdict: accept`, `severity: major` (this file)
- `last_review_a:` `docs/reviews/2026-09-06-review-a-kgas007-c2-relaunch.md`
- Do **not** set `board: accepted` (parent tallies)
- Keep YAML `pending: ["b1mqxsov", "xkytxih1", "y5tspgit", "zq1olquy"]` until execute replaces `xkytxih1` with the new session id only
- After launch: `pending` = new id only; Agent Run Status marks c1/c3/c4 complete and only the new c2 as running
- Official MAP read-only. No G4. No `--kind nuts`.
