# KGAS007 CPU NUTS (kind `nuts-kgas007`)

MAP-θ identity passed: χ² = 122070.76 on 956×66 at `i_rad=0.5044`. Official 066 MAP-θ χ² = 168675.60 at default inclination. `quote_inner_slope: false`. `steal_latest: false`. Dest is this directory, not G3.

Four flexible CPU chains (`--skip-pull`, image `skaha/astroml:latest`):

| chain | session | run_id |
|---|---|---|
| 1 | `b1mqxsov` | `KGAS007-20260905T141720Z-nuts-kgas007-c1` |
| 2 | `xkytxih1` | `KGAS007-20260905T141746Z-nuts-kgas007-c2` |
| 3 | `y5tspgit` | `KGAS007-20260905T141754Z-nuts-kgas007-c3` |
| 4 | `zq1olquy` | `KGAS007-20260905T141801Z-nuts-kgas007-c4` |

`KGAS066-latest` still points at the receding 066 run. Official MAP `kinuv-KGAS066-uvsign-map` was not written. DEC-067 items 3–4 (G3 copy / `kinuv-KGAS066-…` session) are left as 066-only.

Merge after four sentinels (do not use live merge defaults):

```bash
python scripts/merge_nuts_chains.py \
  KGAS007-20260905T141720Z-nuts-kgas007-c1 \
  KGAS007-20260905T141746Z-nuts-kgas007-c2 \
  KGAS007-20260905T141754Z-nuts-kgas007-c3 \
  KGAS007-20260905T141801Z-nuts-kgas007-c4 \
  --kind nuts-kgas007 \
  --artifact-dir docs/reviews/artifacts/2026-09-05-kgas007-nuts \
  --map-json /arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS007/kinuv-KGAS007-stage-a-map/stage_a_map.json
```

`sampler: nuts` only if \(\hat{R}\le 1.01\) and ESS > 400 on four finite chains. Do not start G4.
