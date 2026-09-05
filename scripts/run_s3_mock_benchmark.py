#!/usr/bin/env python3
"""Controlled S3 mock: known arctan truth on KGAS066 template; kinUV vs KinMS."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

from kinuv.diagnostics.s1 import (
    CANFAR_CUBE_10,
    CANFAR_ICO,
    CANFAR_NPZ,
    S1_RT_BOUNDS_ARCSEC,
    add_xx_noise,
    dirty_cube_from_truth,
    inject_vis,
    inner_slope_arctan,
    params_from_map,
    r_eval_arcsec,
    vis_recovery_table,
)
from kinuv.infer.map import image_grid_for_vis, predict_binned, run_stage_a_map
from kinuv.forward.sb import load_sb_template
from kinuv.infer.seeds import vsys_seed_radio_kms
from kinuv.io.vis import load_kgas066, radio_to_optical_kms

REPO = Path(__file__).resolve().parents[1]
DEST = REPO / "docs/reviews/artifacts/2026-09-05-kgas066-s3-image-benchmark"
MOCK = DEST / "mock_controlled"
FITTERS = Path("/arc/projects/KILOGAS/analysis/toby_sandbox/external_fitters")

MOCK_TRUTH = {
    "flux": 70.0,
    "pa_deg": 200.0,
    "vsys_kms": vsys_seed_radio_kms(),
    "gas_sigma_kms": 10.0,
    "dx_arcsec": 0.0,
    "dy_arcsec": 0.0,
    "v0_kms": 250.0,
    "r_t_arcsec": 0.25,
    "i_deg": 43.8,
}


def _cube_rms(cube_k: np.ndarray, mask3d: np.ndarray) -> float:
    m = mask3d & np.isfinite(cube_k)
    if not np.any(m):
        return 0.05
    return float(np.nanmedian(np.abs(cube_k[~m]))) if np.any(~m) else 0.05


def _add_cube_noise(cube_k, mask3d, rms, rng):
    out = np.asarray(cube_k, dtype=np.float64).copy()
    noise = rng.normal(0.0, rms, out.shape)
    noise = np.where(mask3d, noise, 0.0)
    return out + noise


def _inner_slope(v0, rt):
    return inner_slope_arctan(v0, rt, r_eval_arcsec())


def _run_kinms_mock(
    cube_path: Path, mask_path: Path, work: Path, init: dict, *, sb_cube: Path | None = None
) -> dict:
    venv_py = FITTERS / "venv" / "bin" / "python"
    worker = REPO / "external" / "_kinms_best_worker.py"
    work.mkdir(parents=True, exist_ok=True)
    bounds = {
        "v0_kms": (180.0, 320.0),
        "r_t_arcsec": (0.05, 2.0),
        "pa_deg": (185.0, 220.0),
        "i_deg": (35.0, 55.0),
        "vsys_optical_kms": (8250.0, 8400.0),
        "gas_sigma_kms": (5.0, 30.0),
    }
    cfg = {
        "cube": str(cube_path),
        "mask": str(mask_path),
        "work": str(work),
        "init": init,
        "bounds": bounds,
        "n_clouds": 100000,
        "n_clouds_de": 50000,
        "flux_floor_frac": 0.001,
        "phase_center": [init.get("dx_arcsec", 0.0), init.get("dy_arcsec", 0.0)],
        "deproject_inclouds": True,
    }
    if sb_cube is not None:
        cfg["sb_cube"] = str(sb_cube)
    (work / "fit_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    proc = subprocess.run(
        [str(venv_py), str(worker), str(work / "fit_config.json")],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO / "external"),
    )
    out = work / "kinms_fit_result.json"
    if not out.is_file():
        return {
            "status": "failed",
            "returncode": proc.returncode,
            "stderr": (proc.stderr or "")[-1500:],
        }
    return json.loads(out.read_text())


def main(argv=None) -> int:
    MOCK.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(66)

    data = load_kgas066(CANFAR_NPZ, cube_path=CANFAR_CUBE_10)
    grid = image_grid_for_vis(data)
    tmpl = load_sb_template(grid, ico_path=CANFAR_ICO)
    ico_hdr = fits.getheader(CANFAR_ICO)
    img_hdr = fits.getheader(CANFAR_CUBE_10)
    mask3d = np.asarray(
        fits.getdata(CANFAR_CUBE_10.parent / "KGAS66_mask_cube.fits"), dtype=np.float64
    ) > 0.5

    truth = dict(MOCK_TRUTH)
    truth["vsys_kms"] = float(vsys_seed_radio_kms())

    model_vis = predict_binned(
        data, truth, tmpl, grid, i_rad=np.radians(truth["i_deg"])
    )
    noisy_vis = add_xx_noise(model_vis, data.weights, data.s, rng)
    np.savez(
        MOCK / "mock_vis.npz",
        u_m=data.u_m,
        v_m=data.v_m,
        vis=noisy_vis,
        weights=data.weights,
        freqs=data.freqs_native,
        s=data.s,
        vel=data.vel,
    )

    matched, v_data, dv, _ = dirty_cube_from_truth(
        truth, tmpl, grid, data.freqs_native, ico_hdr, img_hdr
    )
    rms = _cube_rms(matched, mask3d)
    mock_cube_k = _add_cube_noise(matched, mask3d, rms, rng)
    mock_path = MOCK / "mock_cube.fits"
    truth_path = MOCK / "mock_cube_truth.fits"
    out_hdr = img_hdr.copy()
    out_hdr["BUNIT"] = "K"
    out_hdr["ORIGIN"] = "controlled arctan mock; beam convolved + noise"
    fits.PrimaryHDU(data=mock_cube_k.astype(np.float32), header=out_hdr).writeto(
        mock_path, overwrite=True
    )
    out_hdr["ORIGIN"] = "controlled arctan mock; beam convolved, no noise"
    fits.PrimaryHDU(data=matched.astype(np.float32), header=out_hdr).writeto(
        truth_path, overwrite=True
    )

    from dataclasses import replace

    mock_data = replace(data, vis=np.asarray(noisy_vis, dtype=np.complex128))
    kinuv_rec = run_stage_a_map(
        mock_data,
        template=tmpl,
        grid=grid,
        maxiter=80,
        rt_bounds_arcsec=S1_RT_BOUNDS_ARCSEC,
    )
    kinuv_table = vis_recovery_table(truth, kinuv_rec)
    kinuv_out = {
        "status": "ran",
        "truth": truth,
        "fitted": params_from_map(kinuv_rec),
        "recovery": kinuv_table,
        "inner_slope_fit_kms_per_arcsec": _inner_slope(kinuv_rec.v0_kms, kinuv_rec.r_t_arcsec),
        "mock_inner_slope_recovered": bool(
            abs(float(kinuv_rec.r_t_arcsec) - truth["r_t_arcsec"]) < 0.08
        ),
        "quote_inner_slope": True,
        "note": "mock only; inner slope recovery is licensed on synthetic truth",
    }
    (MOCK / "kinuv_mock.json").write_text(json.dumps(kinuv_out, indent=2) + "\n")

    kinms_init = {
        "v0_kms": float(truth["v0_kms"]),
        "r_t_arcsec": 0.5,
        "pa_deg": float(truth["pa_deg"]),
        "i_deg": float(truth["i_deg"]),
        "vsys_optical_kms": float(radio_to_optical_kms(truth["vsys_kms"])),
        "gas_sigma_kms": float(truth["gas_sigma_kms"]),
        "dx_arcsec": float(truth.get("dx_arcsec", 0.0)),
        "dy_arcsec": float(truth.get("dy_arcsec", 0.0)),
    }
    mask_path = CANFAR_CUBE_10.parent / "KGAS66_mask_cube.fits"
    kinms_fit = _run_kinms_mock(
        mock_path, mask_path, MOCK / "kinms_mock", kinms_init, sb_cube=truth_path
    )
    fitted = kinms_fit.get("fitted") or {}
    kinms_out = {
        "status": "ran" if kinms_fit.get("success") else "failed",
        "truth": truth,
        "fitted": fitted,
        "inner_slope_fit_kms_per_arcsec": _inner_slope(
            fitted.get("v0_kms", float("nan")),
            fitted.get("r_t_arcsec", float("nan")),
        )
        if fitted
        else float("nan"),
        "mock_inner_slope_recovered": False,
        "quote_inner_slope": True,
        "sb_mode": kinms_fit.get("sb_mode", "inClouds_from_M0"),
        "chi2_cube": kinms_fit.get("chi2_cube"),
        "fit": kinms_fit,
    }
    if fitted:
        rt_ratio = float(fitted["r_t_arcsec"]) / float(truth["r_t_arcsec"])
        kinms_out["r_t_inflation_factor"] = rt_ratio
        kinms_out["mock_inner_slope_recovered"] = bool(rt_ratio >= 2.5)
    (MOCK / "kinms_mock.json").write_text(json.dumps(kinms_out, indent=2) + "\n")

    summary = {
        "truth": truth,
        "truth_inner_slope_kms_per_arcsec": _inner_slope(truth["v0_kms"], truth["r_t_arcsec"]),
        "kinuv_mock": kinuv_out,
        "kinms_mock": kinms_out,
        "mock_cube": str(mock_path),
        "mock_vis": str(MOCK / "mock_vis.npz"),
        "quote_inner_slope_real_data": False,
    }
    (MOCK / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
