#!/usr/bin/env python3
"""Stage A+B MAP after the npz (u,v) sign fix. No NUTS.

``NPZ_UV_SIGN = -1`` makes vis-fitted receding PA match the 10 km/s cube
(~202° E of N). Do not write into ``kinuv-KGAS066-f47bc9-map`` (historical
vis-winner at PA=21.9° before the sign). Do not write into
``kinuv-KGAS066-pa201-map`` (collapsed lock-PA run).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from kinuv.forward.sb import load_sb_template
from kinuv.infer.map import image_grid_for_vis, run_stage_a_map
from kinuv.infer.stage_b import nuisance_from_params, run_stage_b_map
from kinuv.io.vis import load_kgas066
from kinuv.transforms.dft import NPZ_UV_SIGN

NPZ = Path("/arc/projects/KILOGAS/analysis/toby_sandbox/visibilities/KILOGAS066.npz")
ICO = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/"
    "KGAS66_Ico_K_kms-1.fits"
)
CUBE = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/"
    "KGAS66_clipped_cube.fits"
)
OUT = Path(
    "/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS066/"
    "kinuv-KGAS066-uvsign-map"
)
IMAGING_RECEDING_DEG = 201.9
PA_ALIAS_DEG = 21.9


def _jsonable(obj) -> dict:
    src = asdict(obj) if not isinstance(obj, dict) else obj
    out = {}
    for k, v in src.items():
        if isinstance(v, tuple):
            out[k] = [float(x) for x in v]
        elif isinstance(v, (np.floating, float)):
            out[k] = float(v)
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        elif isinstance(v, (np.bool_, bool)):
            out[k] = bool(v)
        else:
            out[k] = v
    return out


def _pa_near(pa_deg: float, target_deg: float, half_deg: float = 40.0) -> bool:
    d = abs((float(pa_deg) - float(target_deg) + 180.0) % 360.0 - 180.0)
    return d < half_deg


def main() -> int:
    if abs(float(NPZ_UV_SIGN) + 1.0) > 1e-12:
        raise SystemExit(f"NPZ_UV_SIGN must be -1, got {NPZ_UV_SIGN}")
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_kgas066(NPZ, cube_path=CUBE)
    grid = image_grid_for_vis(data)
    tmpl = load_sb_template(grid, ico_path=ICO)
    rec_a = run_stage_a_map(data, template=tmpl, grid=grid)
    payload_a = _jsonable(rec_a)
    (OUT / "stage_a_map.json").write_text(json.dumps(payload_a, indent=2) + "\n")
    print("Stage A", json.dumps(payload_a, indent=2), flush=True)
    if rec_a.delta_chi2 <= 0.0:
        raise SystemExit("MAP_LOSES_TO_ZERO")
    if rec_a.v0_kms < 1.0:
        raise SystemExit("MAP_STILL_COLLAPSED")
    pa = float(rec_a.pa_deg) % 360.0
    if _pa_near(pa, PA_ALIAS_DEG):
        raise SystemExit(f"PA still on approaching side: {pa:.2f} deg")
    if not _pa_near(pa, IMAGING_RECEDING_DEG):
        raise SystemExit(f"PA not on imaging receding side: {pa:.2f} deg")

    rec_b = run_stage_b_map(
        data,
        nuisance_from_params(payload_a),
        tmpl,
        grid,
        lam_reg=0.0,
        v0_init=rec_a.v0_kms,
        rt_init=rec_a.r_t_arcsec,
        n_rings=7,
        chi2_stage_a=rec_a.chi2_map,
    )
    payload_b = _jsonable(rec_b)
    payload_b["delta_chi2_vs_a"] = float(rec_a.chi2_map - rec_b.chi2_map)
    (OUT / "stage_b_map.json").write_text(json.dumps(payload_b, indent=2) + "\n")
    print("Stage B", json.dumps(payload_b, indent=2), flush=True)
    (OUT / "NOTE.txt").write_text(
        f"NPZ_UV_SIGN={NPZ_UV_SIGN}. MAP PA={rec_a.pa_deg:.2f}°. "
        f"Δχ² vs V=0 = {rec_a.delta_chi2:.1f}. "
        f"vis χ² A={rec_a.chi2_map:.1f} B={rec_b.chi2_map:.1f}. "
        f"Historical vis-winner (wrong uv sign) was PA=21.9° Δχ²=26213.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
