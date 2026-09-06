#!/usr/bin/env python3
"""Generate the KGAS007 Stage A MAP used to initialize its NUTS run."""

from __future__ import annotations

import json
import math
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kinuv.targets import get_target

TARGET = get_target("KGAS007")
DEST = Path(
    "/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS007/"
    "kinuv-KGAS007-stage-a-map"
)
ARTIFACT = (
    REPO / "docs/reviews/artifacts/2026-09-05-kgas007-stage-a-map"
)
VIS = Path(
    "/arc/projects/KILOGAS/analysis/toby_sandbox/visibilities/KILOGAS007.npz"
)
ICO = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS7/30kms/"
    "KGAS7_Ico_K_kms-1.fits"
)
CUBE = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS7/30kms/"
    "KGAS7_clipped_cube.fits"
)

# Named search only. Do not glob 066 or invent an npz.
SEARCH = [
    VIS,
    Path("/arc/projects/KILOGAS/analysis/toby_sandbox/visibilities/KGAS007.npz"),
    Path("/arc/projects/KILOGAS/analysis/toby_sandbox/visibilities/KILOGAS7.npz"),
    Path("/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS7/30kms"),
    Path("/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS007/30kms"),
    Path("/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/../KGAS7"),
]


def _exists(p: Path) -> bool:
    return p.is_file() or p.is_dir()


def _inventory() -> dict:
    hits = {str(p): _exists(p) for p in SEARCH}
    vis = next(
        (
            p
            for p in SEARCH
            if p.is_file() and p.suffix == ".npz" and "066" not in p.name
        ),
        None,
    )
    ico = ICO if ICO.is_file() else None
    if ico is None:
        for p in SEARCH:
            if p.is_dir():
                for cand in p.glob("*Ico*.fits"):
                    if "66" not in cand.name and "066" not in cand.name:
                        ico = cand
                        break
    cube = CUBE if CUBE.is_file() else None
    cat = TARGET.inference_overrides()
    inv = {
        "galaxy": "KGAS007",
        "diagnostic_only": True,
        "dec_066_target_amended": False,
        "searched": hits,
        "vis": str(vis) if vis else None,
        "ico": str(ico) if ico else None,
        "cube": str(cube) if cube else None,
        "catalogue": TARGET.source,
        "catalogue_overrides": cat,
        "sampler": "map",
        "note": (
            "KGAS007 Stage A MAP provenance. Target metadata is owned by "
            "kinuv.targets. Official KGAS066 products are untouched."
        ),
    }
    return inv, vis, ico, cube, cat


def _stop(inv: dict, status: str) -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    inv["status"] = status
    (DEST / "inventory.json").write_text(json.dumps(inv, indent=2) + "\n")
    print(json.dumps(inv, indent=2))
    return 0


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    inv, vis, ico, cube, cat = _inventory()
    (DEST / "inventory.json").write_text(json.dumps(inv, indent=2) + "\n")
    if vis is None:
        return _stop(inv, "stopped_no_vis")
    if ico is None:
        return _stop(inv, "stopped_no_ico")
    if cube is None:
        return _stop(inv, "stopped_no_007_cube")
    if cat is None:
        return _stop(inv, "stopped_no_007_catalogue")

    from kinuv.forward.sb import load_sb_template
    from kinuv.infer.map import MAXITER_STAGE_A, _lbfgs_one_start, image_grid_for_vis
    from kinuv.infer.seeds import (
        PA_AMBIGUITY_DEG,
        PA_BOUND_HALF_DEG,
        VSYS_BOUND_HALF_KM_S,
        stage_a_seeds,
    )
    from kinuv.io.vis import (
        VisData,
        load_target_vis,
        optical_to_radio_kms,
    )
    from kinuv.runner.plots import write_leftover_at_params

    def _load_007(path: Path, cube_path: Path, cat_row: dict) -> tuple[VisData, dict]:
        """Load current ms2kinuv output or the retained historical 007 NPZ."""
        phase = np.radians([cat_row["ra_deg"], cat_row["dec_deg"]])
        return load_target_vis(path, cube_path=cube_path, phase_dir_rad=phase)

    i_rad = math.radians(float(cat["i_deg"]))
    pa_seed = float(cat["pa_deg"])
    vsys_radio = float(optical_to_radio_kms(cat["vsys_optical_kms"]))
    pa_starts = (pa_seed, pa_seed - PA_AMBIGUITY_DEG)
    extra = {
        "vsys_kms": (
            vsys_radio - VSYS_BOUND_HALF_KM_S,
            vsys_radio + VSYS_BOUND_HALF_KM_S,
        ),
        "pa_deg": (pa_seed - PA_BOUND_HALF_DEG, pa_seed + PA_BOUND_HALF_DEG),
    }

    print(
        json.dumps(
            {
                "status": "loading_vis",
                "vis": str(vis),
                "i_deg": cat["i_deg"],
                "pa_starts": list(pa_starts),
                "vsys_radio_kms": vsys_radio,
            }
        ),
        flush=True,
    )
    data, load_meta = _load_007(vis, cube, cat)
    print(json.dumps({"status": "vis_loaded", **load_meta, "n_row": int(data.vis.shape[0]), "n_chan": int(data.vis.shape[1]), "s": float(data.s)}), flush=True)
    grid = image_grid_for_vis(data)
    tmpl = load_sb_template(grid, ico_path=ico)

    runs = []
    for pa in pa_starts:
        seeds = stage_a_seeds(pa_deg=pa)
        seeds["vsys_kms"] = vsys_radio
        rec = _lbfgs_one_start(
            data,
            tmpl,
            grid,
            seeds,
            0.0,
            MAXITER_STAGE_A,
            pa,
            extra_bounds=extra,
            i_rad=i_rad,
            xla=True,
        )
        payload = asdict(rec)
        payload["pa_start_deg"] = float(pa)
        runs.append(payload)
        print(
            json.dumps(
                {
                    "pa_start_deg": float(pa),
                    "chi2_map": rec.chi2_map,
                    "delta_chi2": rec.delta_chi2,
                    "pa_deg": rec.pa_deg,
                }
            ),
            flush=True,
        )

    winner = max(runs, key=lambda r: r["delta_chi2"])
    params = {
        "flux": winner["flux"],
        "pa_deg": winner["pa_deg"],
        "vsys_kms": winner["vsys_kms"],
        "gas_sigma_kms": winner["gas_sigma_kms"],
        "dx_arcsec": winner["dx_arcsec"],
        "dy_arcsec": winner["dy_arcsec"],
        "v0_kms": winner["v0_kms"],
        "r_t_arcsec": winner["r_t_arcsec"],
    }
    leftover = write_leftover_at_params(
        params, DEST, data=data, tmpl=tmpl, grid=grid, i_rad=i_rad
    )
    leftover["note"] = (
        "KGAS007 Stage A MAP leftover; sampler is map. Target metadata is "
        "owned by kinuv.targets. Official KGAS066 products are untouched. "
        "Do not quote inner dV/dr."
    )
    leftover["galaxy"] = "KGAS007"
    leftover["diagnostic_only"] = True
    leftover["dec_066_target_amended"] = False
    leftover["sampler"] = "map"
    leftover["i_deg_frozen"] = float(cat["i_deg"])
    (DEST / "leftover_chi2.json").write_text(json.dumps(leftover, indent=2) + "\n")

    product = {
        "galaxy": "KGAS007",
        "diagnostic_only": True,
        "dec_066_target_amended": False,
        "sampler": "map",
        "i_deg_frozen": float(cat["i_deg"]),
        "i_rad_frozen": i_rad,
        "catalogue_source": TARGET.source,
        "catalogue_overrides": cat,
        "vsys_seed_radio_kms": vsys_radio,
        "pa_starts_deg": list(pa_starts),
        "vis": str(vis),
        "ico": str(ico),
        "cube": str(cube),
        "load": load_meta,
        "n_row": winner["n_row"],
        "n_chan": winner["n_chan"],
        "dv_kms": winner["dv_kms"],
        "n_bin": winner["n_bin"],
        "s": winner["s"],
        "chi2_map": winner["chi2_map"],
        "chi2_zero": winner["chi2_zero"],
        "delta_chi2": winner["delta_chi2"],
        "nfev": sum(int(r["nfev"]) for r in runs),
        "success": winner["success"],
        "optimiser_ran": winner["optimiser_ran"],
        "message": (
            f"{winner['message']}; starts "
            + ", ".join(
                f"PA={r['pa_start_deg']:.1f} Δχ²={r['delta_chi2']:.1f}" for r in runs
            )
        ),
        "pa_start_deg": winner["pa_start_deg"],
        **params,
        "starts": runs,
        "note": (
            "KGAS007 Stage A MAP provenance. Target metadata is owned by "
            "kinuv.targets. The official KGAS066 MAP is untouched."
        ),
    }
    (DEST / "stage_a_map.json").write_text(json.dumps(product, indent=2) + "\n")
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    for name in ("stage_a_map.json", "inventory.json", "leftover_chi2.json", "leftover_chi2.png"):
        src = DEST / name
        if src.is_file():
            shutil.copy2(src, ARTIFACT / name)
    inv["status"] = "map_written"
    inv["delta_chi2"] = winner["delta_chi2"]
    inv["chi2_map"] = winner["chi2_map"]
    (DEST / "inventory.json").write_text(json.dumps(inv, indent=2) + "\n")
    summary = {
        "status": "map_written",
        "galaxy": product["galaxy"],
        "diagnostic_only": product["diagnostic_only"],
        "sampler": product["sampler"],
        "chi2_map": product["chi2_map"],
        "delta_chi2": product["delta_chi2"],
        "pa_deg": product["pa_deg"],
        "i_deg_frozen": product["i_deg_frozen"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
