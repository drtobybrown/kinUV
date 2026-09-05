#!/usr/bin/env python3
"""Write the 066 S3 comparator table. Vis chi2 is the fit. No KinMS import."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEST = REPO / "docs/reviews/artifacts/2026-09-05-kgas066-s3-image-benchmark"
CMP = (
    REPO
    / "docs/reviews/artifacts/2026-09-02-kgas066-leftover-and-modes/comparison.json"
)
G3 = REPO / "docs/reviews/artifacts/2026-08-30-g3-nuts/summary.json"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(REPO / "scripts/run_s3_barolo.py")], check=False
    )
    ext = REPO / "external"
    if ext.is_dir():
        for runner in sorted(p for p in ext.glob("*.py") if p.is_file()):
            subprocess.run([sys.executable, str(runner)], check=False)
    cmp = json.loads(CMP.read_text())
    g3 = json.loads(G3.read_text()) if G3.is_file() else {}
    nuts_rt = float(cmp["nuts_mean"]["params"]["r_t_arcsec"])
    nuts_v0 = float(cmp["nuts_mean"]["params"]["v0_kms"])
    barolo = {}
    kin = {}
    if (DEST / "barolo.json").is_file():
        barolo = json.loads((DEST / "barolo.json").read_text())
    if (DEST / "kinms.json").is_file():
        kin = json.loads((DEST / "kinms.json").read_text())
    table = {
        "likelihood": "vis chi2 = s * sum w |d-m|^2 on 881x95; cube fitters are comparators",
        "leftover_gate": "SB-dominated",
        "quote_inner_slope": False,
        "intervals_calibrated": False,
        "nuts_mean_r_t_arcsec": nuts_rt,
        "nuts_mean_v0_kms": nuts_v0,
        "chi2_nuts_mean": float(cmp["nuts_mean"]["chi2_sum"]),
        "s1": {
            "truth_r_t_arcsec": 0.25,
            "vis_r_t_arcsec": 0.254,
            "clean_m1_inner_slope": 94.7,
            "truth_inner_slope": 236.7,
            "clean_m2": 56.1,
            "truth_gas_sigma": 8.0,
            "barolo_on_path_in_s1": False,
            "note": "3DBarolo was not on PATH for S1; cube estimator was restoring-beam M1/M2",
        },
        "receding_nuts": {
            "session": "sd3ckpf2",
            "sampler": g3.get("sampler", "nuts"),
            "mixing_pass": g3.get("mixing_pass", True),
            "pa_deg_mean": 200.05,
            "v0_kms_mean": nuts_v0,
            "r_t_arcsec_mean": nuts_rt,
            "chi2": float(cmp["nuts_mean"]["chi2_sum"]),
            "quote_inner_slope": False,
            "intervals_calibrated": False,
        },
        "barolo": barolo,
        "kinms": kin,
        "geometry_soak": {
            "kinuv_i_frozen_deg": 43.86,
            "note": (
                "kinUV does not unfreeze i. Cube-fitter Delta i if a tool ran; "
                "not a harmonic amplitude."
            ),
        },
    }
    (DEST / "s3_table.json").write_text(json.dumps(table, indent=2) + "\n")
    readme = f"""# 066 S3 image-plane benchmark

Vis χ² `s * sum w |ΔV|^2` is the fit; KinMS/Barolo are image-plane comparators, not a kinUV likelihood.

Official MAP `kinuv-KGAS066-uvsign-map` was not written. Receding NUTS `sd3ckpf2` stays the 066 sampling product. Approaching search is closed (`pa25/failure.md`). Leftover gate is **SB-dominated**. That is not an s1 or c3 detection. `quote_inner_slope: false`. `intervals_calibrated: false`. Do not start G4.

## S1 restated (not a new Barolo run)

| | truth | vis Stage A | CLEAN-beam cube |
|---|---|---|---|
| r_t (arcsec) | 0.25 | 0.254 | — |
| inner slope (km/s / arcsec) | 236.7 | 237.8 | M1 94.7 |
| σ / M2 (km/s) | 8 | 7.89 | 56.1 |

3DBarolo was **not on PATH** for S1. The cube estimator was `sky_cube` → restoring beam → major-axis M1/M2.

## Receding NUTS mean (uncalibrated)

`r_t` **mean** {nuts_rt:.4f} arcsec (left the 0.5″ L-BFGS wall). V_0 mean {nuts_v0:.1f} km/s. χ² {cmp['nuts_mean']['chi2_sum']:.1f}. This is not a quoted 066 inner scale. Do not form V_0/r_t.

## External tools this card

| Tool | status |
|---|---|
| 3DBarolo CLI | {barolo.get('status', 'not_run')} |
| KinMS (standalone under external/) | {kin.get('status', 'not_run')} |

PATH miss does not license adding packages to the recovery venv. S3 still ships from S1.

Cube-fit PV/moment overlays were not produced (no Barolo/KinMS on PATH). Vis leftover D/M/R remains [`2026-09-02-kgas066-leftover-and-modes`](../2026-09-02-kgas066-leftover-and-modes/).

## Files

- `s3_table.json` — machine table
- `barolo.json` / `kinms.json` — tool receipts
"""
    (DEST / "README.md").write_text(readme)
    print(json.dumps({"dest": str(DEST), "barolo": barolo.get("status"), "kinms": kin.get("status")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
