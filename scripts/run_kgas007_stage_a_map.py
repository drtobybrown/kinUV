#!/usr/bin/env python3
"""KGAS007 Stage A MAP diagnostic. Does not amend DEC-066-TARGET. No NUTS."""

from __future__ import annotations

import json
import math
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEST = Path(
    "/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS007/"
    "kinuv-KGAS007-stage-a-map"
)
ARTIFACT = (
    REPO / "docs/reviews/artifacts/2026-09-05-kgas007-stage-a-map"
)
CATALOGUE_YAML = Path(
    "/arc/projects/KILOGAS/analysis/toby_sandbox/uvkin/config/uvkin_settings.yaml"
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
    CATALOGUE_YAML,
]


def _exists(p: Path) -> bool:
    return p.is_file() or p.is_dir()


def _read_kgas007_catalogue(path: Path) -> dict | None:
    """Parse the KGAS007 block only. Refuse 066 ba/PA/vsys fallbacks."""
    if not path.is_file():
        return None
    in_block = False
    got: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("  KGAS007:"):
            in_block = True
            continue
        if in_block and line.startswith("  KGAS") and not line.startswith("  KGAS007"):
            break
        if not in_block:
            continue
        raw = line.split("#", 1)[0].strip()
        if ":" not in raw:
            continue
        key, val = raw.split(":", 1)
        key = key.strip()
        val = val.strip()
        if key == "pa_init":
            got["pa_deg"] = float(val)
        elif key == "inc_init":
            got["i_deg"] = float(val)
        elif key == "vsys":
            got["vsys_optical_kms"] = float(val)
        elif key == "ra_deg":
            got["ra_deg"] = float(val)
        elif key == "dec_deg":
            got["dec_deg"] = float(val)
    need = ("pa_deg", "i_deg", "vsys_optical_kms")
    if any(k not in got for k in need):
        return None
    if abs(got["i_deg"] - 43.9) < 0.2 or abs(got["pa_deg"] - 205.2) < 0.05:
        return None
    if abs(got["vsys_optical_kms"] - 8299.563) < 1.0:
        return None
    return got


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
    cat = _read_kgas007_catalogue(CATALOGUE_YAML)
    inv = {
        "galaxy": "KGAS007",
        "diagnostic_only": True,
        "dec_066_target_amended": False,
        "searched": hits,
        "vis": str(vis) if vis else None,
        "ico": str(ico) if ico else None,
        "cube": str(cube) if cube else None,
        "catalogue": str(CATALOGUE_YAML) if cat else None,
        "catalogue_overrides": cat,
        "sampler": "map",
        "note": (
            "TARGET stays KGAS066 only. This tree is a diagnostic. "
            "No 007 NUTS. Official 066 MAP untouched."
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

    sys.path.insert(0, str(REPO / "src"))
    from kinuv.constants import C_LIGHT_M_S, freq_to_velocity_kms
    from kinuv.forward.sb import load_sb_template
    from kinuv.infer.map import MAXITER_STAGE_A, _lbfgs_one_start, image_grid_for_vis
    from kinuv.infer.seeds import (
        PA_AMBIGUITY_DEG,
        PA_BOUND_HALF_DEG,
        VSYS_BOUND_HALF_KM_S,
        stage_a_seeds,
    )
    from kinuv.io.vis import (
        N_BIN,
        N_GUARD,
        TRIM_MARGIN_NATIVE,
        UV_BIN_M,
        VisData,
        _extend_axis,
        _trim_and_guard_indices,
        bin_uv_plane,
        cube_vopt_window_kms,
        optical_to_radio_kms,
    )
    from kinuv.likelihood.chi2 import empirical_s
    from kinuv.response.spectral import bin_channels
    from kinuv.runner.plots import write_leftover_at_params

    def _load_007(path: Path, cube_path: Path, cat_row: dict) -> tuple[VisData, dict]:
        """007 npz is (u,v) in wavelengths, no time/baseline. Do not invent those keys."""
        z = np.load(path, mmap_mode="r")
        if any(k not in z.files for k in ("u", "v", "vis", "weights", "freqs")):
            raise KeyError(f"{path} missing u/v/vis/weights/freqs")
        if "u_m" in z.files:
            raise RuntimeError("unexpected u_m on 007; refuse 066 copy")
        freqs_all = np.asarray(z["freqs"], dtype=np.float64).ravel()
        f_ref = float(np.mean(freqs_all))
        u_m = np.asarray(z["u"], dtype=np.float64) * C_LIGHT_M_S / f_ref
        v_m = np.asarray(z["v"], dtype=np.float64) * C_LIGHT_M_S / f_ref
        vel_all = freq_to_velocity_kms(freqs_all)
        v_lo_opt, v_hi_opt = cube_vopt_window_kms(cube_path)
        v_lo_line = float(optical_to_radio_kms(v_lo_opt))
        v_hi_line = float(optical_to_radio_kms(v_hi_opt))
        i0, i1, g0, g1, dv_native, extra_lo, extra_hi, dvel = _trim_and_guard_indices(
            vel_all,
            v_lo_line,
            v_hi_line,
            margin=int(TRIM_MARGIN_NATIVE),
            n_guard=int(N_GUARD),
        )
        sl = slice(i0, i1 + 1)
        vis = np.asarray(z["vis"][:, sl], dtype=np.complex128)
        weights = np.asarray(z["weights"][:, sl], dtype=np.float64)
        freqs_trim = freqs_all[sl]
        vel_trim = vel_all[sl]
        freqs_native = freqs_all[g0 : g1 + 1]
        vel_native = vel_all[g0 : g1 + 1]
        freqs_native, vel_native = _extend_axis(
            freqs_native, vel_native, extra_lo, extra_hi, dvel
        )
        u_m, v_m, vis, weights = bin_uv_plane(u_m, v_m, vis, weights, float(UV_BIN_M))
        vis_b, w_b, vel_b, freqs_b, _ = bin_channels(
            vis, weights, vel_trim, freqs_trim, int(N_BIN)
        )
        dv_kms = (
            float(np.median(np.abs(np.diff(vel_b))))
            if vel_b.size > 1
            else float(N_BIN) * dv_native
        )
        line_free = (vel_b < v_lo_line) | (vel_b > v_hi_line)
        s = empirical_s(vis_b, w_b, line_free)
        ra = float(cat_row.get("ra_deg", 0.0))
        dec = float(cat_row.get("dec_deg", 0.0))
        data = VisData(
            u_m=u_m,
            v_m=v_m,
            vis=vis_b,
            weights=w_b,
            freqs=freqs_b,
            vel=vel_b,
            freqs_native=freqs_native,
            vel_native=vel_native,
            n_bin=int(N_BIN),
            dv_kms=dv_kms,
            s=s,
            phase_dir_rad=np.array([np.radians(ra), np.radians(dec)], dtype=np.float64),
            line_free_mask=line_free,
            n_guard=int(N_GUARD),
            weights_native=weights,
            v_lo_line=v_lo_line,
            v_hi_line=v_hi_line,
        )
        meta = {
            "npz_keys": list(z.files),
            "uv_stored": "wavelengths",
            "uv_ref_hz": f_ref,
            "time_average": False,
            "uv_bin_m": float(UV_BIN_M),
        }
        return data, meta

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
        "KGAS007 diagnostic Stage A leftover. TARGET stays KGAS066 only. "
        "sampler is map. No 007 NUTS. Official 066 MAP untouched. "
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
        "catalogue_source": str(CATALOGUE_YAML),
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
            "Diagnostic new tree only. TARGET stays KGAS066 only. "
            "No 007 NUTS even if MAP beats V=0. Official MAP untouched."
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
