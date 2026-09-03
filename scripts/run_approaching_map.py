#!/usr/bin/env python3
"""Diagnostic approaching-only Stage A L-BFGS. Does not write official MAP."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kinuv.infer.chart import PARAM_NAMES  # noqa: E402
from kinuv.infer.map import (  # noqa: E402
    _lbfgs_one_start,
    image_grid_for_vis,
    predict_binned,
)
from kinuv.infer.nuts import physical_sampled_from_z6  # noqa: E402
from kinuv.infer.seeds import RT_BOUNDS_ARCSEC, stage_a_seeds  # noqa: E402
from kinuv.forward.sb import load_sb_template  # noqa: E402
from kinuv.io.vis import load_kgas066  # noqa: E402
from kinuv.likelihood.chi2 import chi2  # noqa: E402
from kinuv.runner.kind import ARTIFACT_PA25  # noqa: E402
from kinuv.runner.plots import write_leftover_at_params  # noqa: E402

MAP = Path(
    "/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS066/"
    "kinuv-KGAS066-uvsign-map/stage_a_map.json"
)
NPZ = Path("/arc/projects/KILOGAS/analysis/toby_sandbox/visibilities/KILOGAS066.npz")
ICO = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/"
    "KGAS66_Ico_K_kms-1.fits"
)
CUBE = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/"
    "KGAS66_clipped_cube.fits"
)
RUNS = Path("/arc/projects/KILOGAS/analysis/toby_sandbox/kinuv_runs")
RT_CLAMP = 0.5
OFFICIAL_PA = 199.72980072503037
RECEDING_DCHI = 35552.65225039818
RECEDING_CHI2 = 168675.59555208942


def _params_from_result(r) -> dict:
    return {n: float(getattr(r, n)) for n in PARAM_NAMES}


def _result_dict(r) -> dict:
    return {
        "pa_start_deg": float(r.pa_start_deg),
        "params": _params_from_result(r),
        "chi2_map": float(r.chi2_map),
        "chi2_zero": float(r.chi2_zero),
        "delta_chi2": float(r.delta_chi2),
        "delta_vs_official_map": float(r.chi2_map) - RECEDING_CHI2,
        "nfev": int(r.nfev),
        "success": bool(r.success),
        "message": str(r.message),
    }


def _median_clamped(run: Path, chain_id: int, official: dict) -> dict:
    npz = run / "checkpoints" / f"chain_{chain_id}.npz"
    z6 = np.load(npz)["z6"]
    if z6.ndim == 3:
        z6 = z6[0]
    p8 = physical_sampled_from_z6(
        z6[None, ...], official["dx_arcsec"], official["dy_arcsec"]
    )[0]
    med = {n: float(np.median(p8[:, i])) for i, n in enumerate(PARAM_NAMES)}
    clamped = dict(med)
    clamped["r_t_arcsec"] = RT_CLAMP
    return {"median_raw": med, "median_rt_clamped": clamped}


def _chi2_at(data, tmpl, grid, params) -> float:
    model = predict_binned(data, params, tmpl, grid, xla=True)
    return float(chi2(data.vis, model, data.weights, data.s))


def main() -> int:
    dest = ARTIFACT_PA25 / "approaching-map"
    dest.mkdir(parents=True, exist_ok=True)
    official = json.loads(MAP.read_text())
    data = load_kgas066(NPZ, cube_path=CUBE if CUBE.is_file() else None)
    grid = image_grid_for_vis(data)
    tmpl = load_sb_template(grid, ico_path=ICO if ICO.is_file() else None)
    extra = {"r_t_arcsec": RT_BOUNDS_ARCSEC}

    starts = []
    # Replay official two-start loser (catalogue seeds, PA=25.2).
    starts.append(("catalogue_25.2", stage_a_seeds(pa_deg=25.2), 25.2))
    map_theta = {n: float(official[n]) for n in PARAM_NAMES}
    seed_map = dict(map_theta)
    seed_map["pa_deg"] = 25.2
    seed_map["r_t_arcsec"] = RT_CLAMP
    starts.append(("map_theta_pa_25.2", seed_map, 25.2))

    runs = []
    for name, seed, pa in starts:
        rec = _lbfgs_one_start(
            data, tmpl, grid, seed, 0.0, 80, pa, extra_bounds=extra
        )
        payload = _result_dict(rec)
        payload["start_name"] = name
        runs.append(payload)

    winner = min(runs, key=lambda r: r["chi2_map"])
    win_p = winner["params"]
    leftover = write_leftover_at_params(
        win_p, dest, data=data, tmpl=tmpl, grid=grid
    )
    leftover["note"] = (
        "Approaching-only diagnostic MAP leftover. Not official MAP. "
        "Do not quote inner dV/dr."
    )
    (dest / "leftover_chi2.json").write_text(
        json.dumps(leftover, indent=2) + "\n"
    )

    med_rows = []
    for c in (1, 2, 3):
        run = RUNS / f"KGAS066-20260902T170918Z-nuts-pa25-c{c}"
        pack = _median_clamped(run, c, official)
        chi2_clamp = _chi2_at(data, tmpl, grid, pack["median_rt_clamped"])
        raw_dir = dest / "raw"
        raw_dir.mkdir(exist_ok=True)
        raw_payload = {
            "chain": c,
            "median_raw": pack["median_raw"],
            "unphysical_rt_below_lbfgs_box": pack["median_raw"]["r_t_arcsec"] < 0.5,
        }
        (raw_dir / f"chain_{c}_median_raw.json").write_text(
            json.dumps(raw_payload, indent=2) + "\n"
        )
        med_rows.append(
            {
                "chain": c,
                "pa_deg": pack["median_rt_clamped"]["pa_deg"],
                "vsys_kms": pack["median_rt_clamped"]["vsys_kms"],
                "v0_kms": pack["median_rt_clamped"]["v0_kms"],
                "gas_sigma_kms": pack["median_rt_clamped"]["gas_sigma_kms"],
                "r_t_arcsec": RT_CLAMP,
                "chi2_rt_clamped_0.5": chi2_clamp,
                "delta_vs_official_map": chi2_clamp - RECEDING_CHI2,
            }
        )

    c0 = float(runs[0]["chi2_zero"]) if runs else float("nan")
    out = {
        "official_map_readonly": str(MAP),
        "receding_chi2": RECEDING_CHI2,
        "receding_delta_chi2": RECEDING_DCHI,
        "official_two_start_message": official.get("message"),
        "starts": runs,
        "winner": winner,
        "leftover_chi2_structured": leftover.get("leftover_chi2_structured"),
        "chain_medians_rt_clamped": med_rows,
        "note": (
            "Diagnostic only. DEC-066-PA receding remains the product. "
            "Quoted chi2 at NUTS medians uses r_t=0.5 arcsec only. "
            "Do not start G4. Official MAP unchanged."
        ),
        "chi2_zero": c0,
    }
    (dest / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in out if k != "starts"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
