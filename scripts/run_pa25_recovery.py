#!/usr/bin/env python3
"""CANFAR/local: c1–c3 diagnostic merge + approaching L-BFGS. No NUTS."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PA25 = (
    REPO
    / "docs/reviews/artifacts/2026-09-02-kgas066-leftover-and-modes/pa25"
)


def main() -> int:
    merge = [
        sys.executable,
        str(REPO / "scripts/merge_nuts_chains.py"),
        "KGAS066-20260902T170918Z-nuts-pa25-c1",
        "KGAS066-20260902T170918Z-nuts-pa25-c2",
        "KGAS066-20260902T170918Z-nuts-pa25-c3",
        "--chain-ids",
        "1,2,3",
        "--artifact-dir",
        str(PA25 / "c1c3-diagnostic"),
        "--pa-init",
        "25.2",
    ]
    print("merge", merge, flush=True)
    subprocess.run(merge, check=False)
    map_cmd = [sys.executable, str(REPO / "scripts/run_approaching_map.py")]
    print("map", map_cmd, flush=True)
    rc = subprocess.run(map_cmd, check=False).returncode
    _write_failure()
    return rc


def _write_failure() -> None:
    merge_sum = {}
    map_sum = {}
    ms = PA25 / "c1c3-diagnostic" / "summary.json"
    ap = PA25 / "approaching-map" / "summary.json"
    if ms.is_file():
        merge_sum = json.loads(ms.read_text())
    if ap.is_file():
        map_sum = json.loads(ap.read_text())
    winner = map_sum.get("winner", {})
    body = f"""# Approaching PA 25.2 failure (066)

Official MAP `kinuv-KGAS066-uvsign-map` was not written. Receding G3 remains
the only `sampler: nuts` product. Do not start G4. Do not quote inner dV/dr
or S2 16/50/84.

## Four-chain merge already on disk (`pa25/`)

Written 2026-09-03T13:10Z as `COMPLETED_UNMIXED`. **Do not overwrite.**

| Lie on disk | Fact |
|---|---|
| `sampler: laplace_mh` | Autodiff NUTS merge that failed mix. Label leak in `product_record`. |
| `leftover_chi2_structured: false` | Leftover was not measured. Official leftover-vs-velocity is True. |

c1/c3 PA ~15°, c2 ~64°, c4 exploded (flux ~1e262). R_hat(PA) ~22. Not a mode.

## Official two-start (already discarded approaching)

`stage_a_map.json` message: PA=205.2 Δχ²=35552.7, PA=25.2 Δχ²=4260.2.
Approaching start χ² ≈ 199968 (gap vs official MAP **+31292.5**).
DEC-066-PA is receding-side PA, seed 205.2°.

## c1–c3 diagnostic

See `c1c3-diagnostic/`. Expect `sampler: nuts_unmixed`, leftover key omitted,
`mixing_pass: false`. Three chains cannot mint `sampler: nuts`.

Landed: sampler={merge_sum.get("sampler")}, n_kept={merge_sum.get("n_kept")},
mixing_pass={merge_sum.get("mixing_pass")}.

## Approaching L-BFGS (new tree only)

See `approaching-map/`. Winner χ²={winner.get("chi2_map")},
Δχ² vs V=0={winner.get("delta_chi2")},
Δ vs official MAP={winner.get("delta_vs_official_map")},
PA={winner.get("params", {}).get("pa_deg")}.
Leftover bit from arrays: {map_sum.get("leftover_chi2_structured")}.

Quoted NUTS-median χ² uses `r_t=0.5` arcsec only (`chain_medians_rt_clamped`).
Raw `r_t~6e-4` medians are under `approaching-map/raw/` and are unphysical.

## Gate

No approaching NUTS this card (dual accept major). If a later MAP ever
competes (Δ vs official MAP ≥ −200 and Δχ² vs V=0 ≥ 35000), new propose.
Do not stack modes. `KGAS066-latest` stays receding.
"""
    (PA25 / "failure.md").write_text(body)


if __name__ == "__main__":
    raise SystemExit(main())
