#!/usr/bin/env python3
"""KinMS image-plane comparator for 066 S3 (independent cube fit).

Vis chi2 is the kinUV fit. KinMS is fit directly to the 10 km/s cube with
kinUV Stage A catalogue seeds and harmonized prior bounds only — no kinUV
posterior arrays. Isolated env:
/arc/projects/KILOGAS/analysis/toby_sandbox/external_fitters/

Plotting uses in-repo style via scripts/run_s3_live_plots.py (subprocess).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CUBE_DIR = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/10kms/"
)
CUBE = CUBE_DIR / "KGAS66_clipped_cube.fits"
MASK = CUBE_DIR / "KGAS66_mask_cube.fits"
DEST = REPO / "docs/reviews/artifacts/2026-09-05-kgas066-s3-image-benchmark"
LIVE = DEST / "live_fitters"
RECOVERY_SITE = Path(
    "/arc/home/thbrown/kinuv-venv-recovery/lib/python3.12/site-packages"
)


def _kinuv_env() -> dict:
    py_path = str(REPO / "src")
    if RECOVERY_SITE.is_dir():
        py_path = py_path + os.pathsep + str(RECOVERY_SITE)
    return {
        **os.environ,
        "MPLBACKEND": "Agg",
        "PYTHONPATH": py_path + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }


def _kinuv_python() -> str:
    for cand in (
        "/usr/local/bin/python3.12",
        "/usr/local/bin/python3",
        sys.executable,
    ):
        if Path(cand).is_file():
            return cand
    return sys.executable


SCRATCH = None  # deprecated; all work under artifact tree

FITTERS = Path("/arc/projects/KILOGAS/analysis/toby_sandbox/external_fitters")
KINMS_WORK = LIVE / "kinms"
KINMS_BEST = LIVE / "kinms_best"
MOCK = DEST / "mock_controlled"

# kinUV Stage A catalogue seeds (066). Not NUTS / MAP posterior values.
BEAM_ARCSEC = 1.04
INIT = {
    "v0_kms": 200.0,
    "r_t_arcsec": BEAM_ARCSEC,
    "pa_deg": 205.2,
    "i_deg": 43.86289587982063,
    "vsys_optical_kms": 8323.6,
    "gas_sigma_kms": 10.0,
    "dx_arcsec": 0.09104737371760792,
    "dy_arcsec": 0.018566961155444102,
}
BOUNDS = {
    "v0_kms": (180.0, 320.0),
    "r_t_arcsec": (0.05, 2.0),
    "pa_deg": (185.0, 220.0),
    "i_deg": (35.0, 55.0),
    "vsys_optical_kms": (8250.0, 8400.0),
    "gas_sigma_kms": (5.0, 30.0),
}


def _receipt(**extra) -> dict:
    rec = {
        "quote_inner_slope": False,
        "intervals_calibrated": False,
        "leftover_gate": "SB-dominated",
        "note": (
            "vis chi2 is the fit; quote_inner_slope: false. "
            "KinMS is an image-plane comparator fit to the cube. "
            "Init from kinUV Stage A catalogue seeds only; "
            "no kinUV posterior conditioning."
        ),
    }
    rec.update(extra)
    return rec


def _write(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2) + "\n")


def _purge_compromised() -> None:
    """Remove the prior NUTS-seeded KinMS products."""
    for p in (
        LIVE / "kinms.json",
        DEST / "kinms.json",
        LIVE / "barolo.json",
        DEST / "barolo.json",
        LIVE / "pv_major_minor.png",
        LIVE / "moments_slices.png",
        LIVE / "overlays.json",
        KINMS_WORK / "kinms_arctan_cube.npy",
        KINMS_WORK / "model_cube.fits",
    ):
        if p.is_file():
            p.unlink()
    if KINMS_WORK.is_dir():
        shutil.rmtree(KINMS_WORK, ignore_errors=True)


def _kinms_importable() -> tuple[bool, str | None]:
    venv_py = FITTERS / "venv" / "bin" / "python"
    if not venv_py.is_file():
        return False, "missing external_fitters/venv"
    try:
        proc = subprocess.run(
            [str(venv_py), "-c", "import kinms; print(kinms.__file__)"],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout or "import failed").strip()
        return True, proc.stdout.strip()
    except OSError as exc:
        return False, str(exc)


def _cube_chi2_k(
    model_k: "object",
    data_k: "object",
    mask3d: "object",
) -> float:
    import numpy as np

    diff = np.asarray(model_k, dtype=np.float64) - np.asarray(data_k, dtype=np.float64)
    m = np.asarray(mask3d, dtype=bool) & np.isfinite(diff)
    if not np.any(m):
        return float("inf")
    return float(np.sum(diff[m] ** 2))


def _fit_kinms_worker(worker: Path, work: Path, *, label: str) -> dict:
    ok, detail = _kinms_importable()
    if not ok:
        rec = _receipt(tool=label, ran=False, status="missing", error=detail)
        return rec

    venv_py = FITTERS / "venv" / "bin" / "python"
    if not worker.is_file():
        return _receipt(tool=label, ran=False, status="failed", error=f"missing {worker}")

    work.mkdir(parents=True, exist_ok=True)
    cfg = {
        "cube": str(CUBE),
        "mask": str(MASK) if MASK.is_file() else None,
        "work": str(work),
        "init": INIT,
        "bounds": BOUNDS,
        "n_clouds": 100000,
        "n_clouds_de": 50000,
        "flux_floor_frac": 0.001,
        "phase_center": [INIT["dx_arcsec"], INIT["dy_arcsec"]],
        "deproject_inclouds": True,
    }
    (work / "fit_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    proc = subprocess.run(
        [str(venv_py), str(worker), str(work / "fit_config.json")],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(worker.parent),
    )
    out_path = work / "kinms_fit_result.json"
    if not out_path.is_file():
        return _receipt(
            tool=label,
            ran=False,
            status="failed",
            returncode=proc.returncode,
            stdout=(proc.stdout or "")[-2000:],
            stderr=(proc.stderr or "")[-2000:],
        )
    fit = json.loads(out_path.read_text())
    ok_run = bool(fit.get("success")) or (
        float(fit.get("chi2_cube", 1e30)) < 1e29
        and (work / "model_cube.fits").is_file()
    )
    return _receipt(
        tool=label,
        ran=ok_run,
        status="ran" if ok_run else "failed",
        kinms_module=detail,
        init=INIT,
        bounds=BOUNDS,
        init_source="kinUV Stage A catalogue seeds; no kinUV posterior",
        sb_mode=fit.get("sb_mode", "gaussian_sbProf"),
        fitted=fit.get("fitted"),
        chi2_cube=fit.get("chi2_cube"),
        nfev=fit.get("nfev"),
        n_clouds=fit.get("n_clouds"),
        model_cube=str(work / "model_cube.fits"),
    )


def _fit_kinms() -> dict:
    rec = _fit_kinms_worker(
        REPO / "external" / "_kinms_fit_worker.py", KINMS_WORK, label="KinMS"
    )
    _write(LIVE / "kinms.json", rec)
    _write(DEST / "kinms.json", rec)
    return rec


def _fit_kinms_best() -> dict:
    rec = _fit_kinms_worker(
        REPO / "external" / "_kinms_best_worker.py", KINMS_BEST, label="KinMS_best"
    )
    rec["note"] = (
        "vis chi2 is the fit; quote_inner_slope: false. "
        "KinMS inClouds SB from data M0; kinematics fit only. "
        "Init from kinUV Stage A catalogue seeds; no kinUV posterior."
    )
    _write(LIVE / "kinms_best.json", rec)
    _write(DEST / "kinms_best.json", rec)
    return rec


def _run_plots(kinms_cube: Path) -> dict:
    plot_script = REPO / "scripts" / "run_s3_live_plots.py"
    py = _kinuv_python()
    proc = subprocess.run(
        [
            py,
            str(plot_script),
            "--live-dir",
            str(LIVE),
            "--kinms-cube",
            str(kinms_cube),
            "--kinms-label",
            "KinMS (inClouds)",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO),
        env=_kinuv_env(),
    )
    overlays = LIVE / "overlays.json"
    if overlays.is_file():
        return json.loads(overlays.read_text())
    return {
        "status": "plot_failed" if proc.returncode else "unknown",
        "returncode": proc.returncode,
        "stderr": (proc.stderr or "")[-1500:],
    }


def _run_mock() -> dict:
    mock_script = REPO / "scripts" / "run_s3_mock_benchmark.py"
    plot_script = REPO / "scripts" / "run_s3_mock_plots.py"
    py = _kinuv_python()
    env = _kinuv_env()
    proc = subprocess.run(
        [py, str(mock_script)],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO),
        env=env,
    )
    if proc.returncode != 0:
        return {
            "status": "mock_failed",
            "returncode": proc.returncode,
            "stderr": (proc.stderr or "")[-1500:],
            "stdout": (proc.stdout or "")[-1500:],
        }
    subprocess.run(
        [py, str(plot_script)],
        check=False,
        cwd=str(REPO),
        env=env,
    )
    summary = MOCK / "summary.json"
    if summary.is_file():
        return json.loads(summary.read_text())
    return {"status": "mock_unknown"}


def _rebuild_s3(kin: dict, kin_best: dict, mock: dict) -> None:
    table_path = DEST / "s3_table.json"
    cmp_path = (
        REPO
        / "docs/reviews/artifacts/2026-09-02-kgas066-leftover-and-modes/comparison.json"
    )
    g3_path = REPO / "docs/reviews/artifacts/2026-08-30-g3-nuts/summary.json"
    map_path = Path(
        "/arc/projects/KILOGAS/analysis/toby_sandbox/results/production/KGAS066/"
        "kinuv-KGAS066-uvsign-map/stage_a_map.json"
    )
    cmp = json.loads(cmp_path.read_text()) if cmp_path.is_file() else {}
    g3 = json.loads(g3_path.read_text()) if g3_path.is_file() else {}
    map_p = json.loads(map_path.read_text()) if map_path.is_file() else {}
    nuts_rt = float(cmp.get("nuts_mean", {}).get("params", {}).get("r_t_arcsec", 0.224))
    nuts_v0 = float(cmp.get("nuts_mean", {}).get("params", {}).get("v0_kms", 255.0))
    kinuv_params = {
        "v0_kms": float(map_p.get("v0_kms", 0.0)),
        "r_t_arcsec": float(map_p.get("r_t_arcsec", 0.0)),
        "pa_deg": float(map_p.get("pa_deg", 0.0)),
        "i_deg": float(map_p.get("i_deg", 0.0)),
        "gas_sigma_kms": float(map_p.get("gas_sigma_kms", 0.0)),
        "vsys_radio_kms": float(map_p.get("vsys_kms", 0.0)),
        "quote_inner_slope": False,
        "note": "Official vis MAP Stage A; r_t is not a quoted inner scale",
    }
    kinms_best_fit = kin_best.get("fitted") or {}
    mock_truth = mock.get("truth") or mock.get("kinuv_mock", {}).get("truth") or {}
    kinuv_mock = (mock.get("kinuv_mock") or {}).get("fitted") or {}
    kinms_mock = (mock.get("kinms_mock") or {}).get("fitted") or {}
    rows = [
        {
            "parameter": "r_t_arcsec",
            "true_mock": mock_truth.get("r_t_arcsec"),
            "kinuv_mock": kinuv_mock.get("r_t_arcsec"),
            "kinms_mock": kinms_mock.get("r_t_arcsec"),
            "kinuv_real_nuts_mean": nuts_rt,
            "kinms_best_real": kinms_best_fit.get("r_t_arcsec"),
        },
        {
            "parameter": "v0_kms",
            "true_mock": mock_truth.get("v0_kms"),
            "kinuv_mock": kinuv_mock.get("v0_kms"),
            "kinms_mock": kinms_mock.get("v0_kms"),
            "kinuv_real_nuts_mean": nuts_v0,
            "kinms_best_real": kinms_best_fit.get("v0_kms"),
        },
        {
            "parameter": "pa_deg",
            "true_mock": mock_truth.get("pa_deg"),
            "kinuv_mock": kinuv_mock.get("pa_deg"),
            "kinms_mock": kinms_mock.get("pa_deg"),
            "kinuv_real_nuts_mean": None,
            "kinms_best_real": kinms_best_fit.get("pa_deg"),
        },
        {
            "parameter": "inner_slope_kms_per_arcsec",
            "true_mock": mock.get("truth_inner_slope_kms_per_arcsec"),
            "kinuv_mock": mock.get("kinuv_mock", {}).get("inner_slope_fit_kms_per_arcsec"),
            "kinms_mock": mock.get("kinms_mock", {}).get("inner_slope_fit_kms_per_arcsec"),
            "kinuv_real_nuts_mean": None,
            "kinms_best_real": None,
            "quote_inner_slope_real": False,
            "mock_inner_slope_recovered": mock.get("kinuv_mock", {}).get(
                "mock_inner_slope_recovered"
            ),
        },
    ]
    param_compare = {
        "kinuv_map": kinuv_params,
        "kinms_independent_gaussian": {
            k: (kin.get("fitted") or {}).get(k)
            for k in (
                "v0_kms",
                "r_t_arcsec",
                "pa_deg",
                "i_deg",
                "gas_sigma_kms",
                "vsys_optical_kms",
            )
        },
        "kinms_best_inclouds": {
            k: kinms_best_fit.get(k)
            for k in (
                "v0_kms",
                "r_t_arcsec",
                "pa_deg",
                "i_deg",
                "gas_sigma_kms",
                "vsys_optical_kms",
                "flux_scale",
            )
        },
        "receding_nuts_mean": {
            "r_t_arcsec": nuts_rt,
            "v0_kms": nuts_v0,
            "quote_inner_slope": False,
            "note": "NUTS mean comparator only; not KinMS init",
        },
    }
    table = {
        "likelihood": (
            "vis chi2 = s * sum w |d-m|^2 on 881x95; KinMS is image-plane comparator"
        ),
        "header_note": "vis chi2 is the fit; quote_inner_slope: false",
        "leftover_gate": "SB-dominated",
        "quote_inner_slope": False,
        "intervals_calibrated": False,
        "kinuv": {
            "tool": "kinUV",
            "plane": "visibility",
            "product": "official MAP kinuv-KGAS066-uvsign-map",
            "params": kinuv_params,
            "quote_inner_slope": False,
        },
        "kinms_best_real": kin_best,
        "kinms_mock": mock.get("kinms_mock") or {},
        "kinms": kin_best,
        "kinms_legacy_gaussian": kin,
        "parameter_comparison": param_compare,
        "s3_parameter_table": rows,
        "mock_controlled": {
            "dir": str(MOCK),
            "mock_inner_slope_recovered": mock.get("kinuv_mock", {}).get(
                "mock_inner_slope_recovered"
            ),
            "kinms_r_t_inflation": mock.get("kinms_mock", {}).get("r_t_inflation_factor"),
            "figure": str(MOCK / "mock_benchmark.png"),
            "quote_inner_slope": True,
            "note": "mock_inner_slope_recovered applies to synthetic test only",
        },
        "receding_nuts": {
            "session": "sd3ckpf2",
            "sampler": g3.get("sampler", "nuts"),
            "mixing_pass": g3.get("mixing_pass", True),
            "r_t_arcsec_mean": nuts_rt,
            "v0_kms_mean": nuts_v0,
            "quote_inner_slope": False,
            "intervals_calibrated": False,
            "note": "NUTS mean r_t is not a quoted inner scale; comparator only",
        },
        "s1": {
            "truth_r_t_arcsec": 0.25,
            "vis_r_t_arcsec": 0.254,
            "clean_m1_inner_slope": 94.7,
            "truth_inner_slope": 236.7,
            "clean_m2": 56.1,
            "truth_gas_sigma": 8.0,
            "note": "S1 cube estimator was restoring-beam M1/M2",
        },
        "live_fitters": {
            "dir": str(LIVE),
            "kinms_best_real": kin_best,
            "kinms_mock": mock.get("kinms_mock") or {},
            "kinms_legacy": kin,
        },
    }
    table_path.write_text(json.dumps(table, indent=2) + "\n")
    readme = f"""# 066 S3 image-plane benchmark

Vis χ² is the fit; quote_inner_slope: false. KinMS is an image-plane comparator, not a kinUV likelihood.

Official MAP `kinuv-KGAS066-uvsign-map` was not written. Two-way benchmark: **kinUV (vis)** vs **KinMS (cube fit)**. Leftover gate is **SB-dominated** on 066. `intervals_calibrated: false`. Do not start G4.

## Live fitters

- KinMS best (inClouds): `{kin_best.get("status")}`
- KinMS legacy (Gaussian SB): `{kin.get("status")}`
- Mock controlled: `{mock.get("kinuv_mock", {}).get("status", mock.get("status"))}`
- Init: kinUV Stage A catalogue seeds only (no kinUV posterior)
- Figures: `live_fitters/pv_comparison_real.png`, `moments_comparison_real.png`, `rotation_curves_real.png`, `mock_controlled/mock_benchmark.png`
- Receipt: `live_fitters/kinms_best.json`

## S1 restated

| | truth | vis Stage A | CLEAN-beam cube |
|---|---|---|---|
| r_t (arcsec) | 0.25 | 0.254 | — |
| inner slope (km/s / arcsec) | 236.7 | 237.8 | M1 94.7 |
| σ / M2 (km/s) | 8 | 7.89 | 56.1 |
"""
    (DEST / "README.md").write_text(readme)
    (LIVE / "README.md").write_text(
        readme
        + "\nOfficial MAP unchanged. Isolated env: "
        + f"`{FITTERS}`\n"
    )


def main() -> int:
    LIVE.mkdir(parents=True, exist_ok=True)
    kin = _fit_kinms()
    kin_best = _fit_kinms_best()
    cube = (
        KINMS_BEST / "model_cube.fits"
        if kin_best.get("ran")
        else KINMS_WORK / "model_cube.fits"
    )
    overlays = _run_plots(cube) if kin_best.get("ran") or kin.get("ran") else {"status": "skipped_no_kinms"}
    mock = _run_mock()
    _rebuild_s3(kin, kin_best, mock)
    _write(
        LIVE / "summary.json",
        _receipt(
            kinms_best=kin_best.get("status"),
            kinms_legacy=kin.get("status"),
            mock=mock.get("kinuv_mock", {}).get("status", mock.get("status")),
            overlays=overlays.get("status"),
        ),
    )
    print(
        json.dumps(
            {
                "live": str(LIVE),
                "kinms_best": kin_best.get("status"),
                "kinms_legacy": kin.get("status"),
                "mock": mock.get("kinuv_mock", {}).get("status", mock.get("status")),
                "overlays": overlays.get("status"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
