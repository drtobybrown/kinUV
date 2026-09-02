#!/usr/bin/env python3
"""Three-way vis leftover at Stage A MAP, receding NUTS mean, and Stage B rings.

Not a likelihood. Not a second plotter: leftover_chi2 + plot_leftover_chi2
and plot_stage_b_vs_imaging --model-label. Image-plane D/M/R is a CLEAN-matched
cube, not an inverse FT of residual vis. Do not quote inner dV/dr.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kinuv.diagnostics.figures import plot_leftover_chi2
from kinuv.diagnostics.flags import leftover_velocity_structured, map_quality_flags
from kinuv.diagnostics.s1 import (
    CANFAR_CUBE_30,
    CANFAR_ICO,
    CANFAR_NPZ,
    MAP_DIR,
    assert_hann_bin_operator,
    leftover_chi2,
)
from kinuv.forward.sb import load_sb_template
from kinuv.infer.map import image_grid_for_vis, predict_binned as stage_a_predict
from kinuv.infer.posterior import PARAM_NAMES
from kinuv.infer.stage_b import nuisance_from_params, predict_binned as stage_b_predict
from kinuv.io.vis import load_kgas066

CHI2_STAGE_A = 168675.59555208942
CHI2_NUTS_MEAN = 167486.7639374534
CHI2_STAGE_B = 167302.18673431588
S_FROZEN = 0.5136098555284736
ARTIFACT = REPO / "docs/reviews/artifacts/2026-09-02-kgas066-leftover-and-modes"
NUTS_MEAN_JSON = (
    REPO / "docs/reviews/artifacts/2026-08-30-g3-nuts/nuts_mean_params.json"
)
NUTS_CUBE = Path(
    "/arc/projects/KILOGAS/analysis/toby_sandbox/kinuv_runs/"
    "KGAS066-20260831T194009Z-nuts/plots/stage_a_nuts_mean.fits"
)
CUBE_10 = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/10kms/"
    "KGAS66_clipped_cube.fits"
)
MASK_10 = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/10kms/"
    "KGAS66_mask_cube.fits"
)


def _params8(rec: dict) -> dict[str, float]:
    return {n: float(rec[n]) for n in PARAM_NAMES}


def _dump_leftover(dest: Path, data, model, *, label: str, expect: float, extra: dict) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    b_m, per_row, vel, per_chan = leftover_chi2(data, model)
    total = float(np.sum(per_row))
    if abs(total - float(expect)) >= 1.0:
        raise SystemExit(
            f"{label} leftover chi2_sum={total:.4f} != identity {expect:.4f}"
        )
    shape = tuple(int(x) for x in np.asarray(data.vis).shape)
    if shape != (881, 95):
        raise SystemExit(f"{label} vis shape {shape} != (881, 95)")
    if abs(float(data.s) - S_FROZEN) > 1.0e-9:
        raise SystemExit(f"{label} s={data.s} != {S_FROZEN}")
    structured = leftover_velocity_structured(b_m, per_row, per_chan)
    flags = map_quality_flags(
        extra.get("flag_rec", {"r_t_arcsec": 0.5, "pa_deg": 199.73, "delta_chi2": 1.0}),
        leftover_npz={
            "baseline_m": b_m,
            "chi2_row": per_row,
            "chi2_chan": per_chan,
        },
    )
    summary = {
        "label": label,
        "chi2_sum": total,
        "chi2_identity": float(expect),
        "leftover_chi2_structured": bool(structured),
        "leftover_uv_span": flags["leftover_uv_span"],
        "leftover_vel_span": flags["leftover_vel_span"],
        "quote_inner_slope": flags["quote_inner_slope"],
        "rings_are_not_a_warp": True,
        "n_row": 881,
        "n_chan": 95,
        "s": float(data.s),
        **extra.get("json", {}),
    }
    (dest / "leftover_chi2.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez(
        dest / "leftover_chi2.npz",
        baseline_m=b_m,
        chi2_row=per_row,
        vel_kms=vel,
        chi2_chan=per_chan,
    )
    plot_leftover_chi2(b_m, per_row, vel, per_chan, dest / "leftover_chi2.png")
    return summary


def _imaging(out: Path, geom: Path, cube: Path, label: str) -> None:
    if not (CUBE_10.is_file() and MASK_10.is_file() and cube.is_file()):
        return
    sys.path.insert(0, str(REPO / "scripts"))
    from plot_stage_b_vs_imaging import main as imaging_main

    imaging_main(
        [
            "--data-cube",
            str(CUBE_10),
            "--mask-cube",
            str(MASK_10),
            "--stage-a",
            str(geom),
            "--model-cube",
            str(cube),
            "--out-dir",
            str(out),
            "--matched-fits",
            str(out / "model_on_10kms.fits"),
            "--model-label",
            label,
        ]
    )


def main() -> int:
    from kinuv.scratch import apply_scratch_env

    apply_scratch_env()
    kernel = assert_hann_bin_operator()
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    cube30 = CANFAR_CUBE_30 if CANFAR_CUBE_30.is_file() else None
    ico = CANFAR_ICO if CANFAR_ICO.is_file() else None
    data = load_kgas066(CANFAR_NPZ, cube_path=cube30)
    grid = image_grid_for_vis(data)
    tmpl = load_sb_template(grid, ico_path=ico)

    a_rec = json.loads((MAP_DIR / "stage_a_map.json").read_text())
    b_rec = json.loads((MAP_DIR / "stage_b_map.json").read_text())
    n_rec = json.loads(NUTS_MEAN_JSON.read_text())
    a_params = _params8(a_rec)
    n_params = _params8(n_rec)
    nuis = nuisance_from_params(a_params)

    map_dir = ARTIFACT / "stage-a-map"
    nuts_dir = ARTIFACT / "nuts-mean-receding"
    b_dir = ARTIFACT / "stage-b-rings"

    a_model = stage_a_predict(data, a_params, tmpl, grid, xla=True)
    a_sum = _dump_leftover(
        map_dir,
        data,
        a_model,
        label="Stage A MAP",
        expect=CHI2_STAGE_A,
        extra={
            "flag_rec": {
                "r_t_arcsec": a_params["r_t_arcsec"],
                "pa_deg": a_params["pa_deg"],
                "delta_chi2": float(a_rec.get("delta_chi2", 35552.65)),
            },
            "json": {"params": a_params},
        },
    )
    n_model = stage_a_predict(data, n_params, tmpl, grid, xla=True)
    n_sum = _dump_leftover(
        nuts_dir,
        data,
        n_model,
        label="NUTS-mean Stage A",
        expect=CHI2_NUTS_MEAN,
        extra={
            "flag_rec": {
                "r_t_arcsec": n_params["r_t_arcsec"],
                "pa_deg": n_params["pa_deg"],
                "delta_chi2": float(a_rec.get("delta_chi2", 35552.65)),
            },
            "json": {"params": n_params},
        },
    )
    b_model = stage_b_predict(
        data,
        nuis,
        tmpl,
        grid,
        r_knots_arcsec=b_rec["r_knots_arcsec"],
        v_knots_kms=b_rec["v_knots_kms"],
    )
    b_sum = _dump_leftover(
        b_dir,
        data,
        b_model,
        label="Stage B rings",
        expect=CHI2_STAGE_B,
        extra={
            "flag_rec": {
                "r_t_arcsec": float(b_rec.get("r_t_recovered", 0.5)),
                "pa_deg": a_params["pa_deg"],
                "delta_chi2": float(b_rec.get("delta_chi2", 1.0)),
            },
            "json": {
                "n_rings": int(b_rec["n_rings"]),
                "lam_reg": float(b_rec["lam_reg"]),
                "v0_recovered": float(b_rec["v0_recovered"]),
                "r_t_recovered": float(b_rec["r_t_recovered"]),
            },
        },
    )

    (map_dir / "geom.json").write_text(json.dumps(a_params, indent=2) + "\n")
    (nuts_dir / "geom.json").write_text(json.dumps(n_params, indent=2) + "\n")
    _imaging(map_dir, map_dir / "geom.json", MAP_DIR / "stage_a_model_cube.fits", "Stage A MAP")
    _imaging(
        nuts_dir,
        nuts_dir / "geom.json",
        NUTS_CUBE,
        "NUTS-mean Stage A",
    )
    _imaging(
        b_dir,
        MAP_DIR / "stage_a_map.json",
        MAP_DIR / "stage_b_model_cube.fits",
        "Stage B rings",
    )

    gap = float(n_sum["chi2_sum"] - b_sum["chi2_sum"])
    still_sb = bool(b_sum["leftover_chi2_structured"])
    comparison = {
        "pipeline_kernel": kernel,
        "s": S_FROZEN,
        "n_row": 881,
        "n_chan": 95,
        "stage_a_map": a_sum,
        "nuts_mean": n_sum,
        "stage_b": b_sum,
        "delta_chi2_nuts_minus_map": float(n_sum["chi2_sum"] - a_sum["chi2_sum"]),
        "delta_chi2_nuts_minus_stage_b": gap,
        "leftover_gate": "SB-dominated" if still_sb else "leftover-vs-velocity cleared at Stage B",
        "rings_are_not_a_warp": True,
        "quote_inner_slope": False,
        "intervals_calibrated": False,
        "note": (
            "Quoted V_c stays Stage A arctan. Image-plane moments are CLEAN-matched "
            "cubes, not inverse FT of residual vis. Do not quote inner dV/dr. "
            "Official MAP unchanged. Do not start G4."
        ),
    }
    (ARTIFACT / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps({k: comparison[k] for k in comparison if k not in {"stage_a_map", "nuts_mean", "stage_b"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
