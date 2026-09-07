#!/usr/bin/env python3
"""Paired Python-native S4 truth recovery on canonical target samplings.

The analytic sky cube and Fourier visibilities share one arctan disk truth.
No CASA task, Measurement Set imaging, or deconvolution is used.  kinUV fits
the noisy visibilities; frozen stock KinMS fits the noisy restored cube under
its standard image-plane workflow.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from astropy.io import fits

_repo = Path(__file__).resolve().parents[1]
import importlib.util

_scratch_spec = importlib.util.spec_from_file_location(
    "_kinuv_scratch", _repo / "src/kinuv/scratch.py"
)
_scratch_module = importlib.util.module_from_spec(_scratch_spec)
_scratch_spec.loader.exec_module(_scratch_module)
_scratch_module.apply_scratch_env()

from kinuv.constants import freq_to_velocity_kms
from kinuv.diagnostics.imaging import match_model_to_imaging
from kinuv.diagnostics.s1 import add_xx_noise, sky_cube_fits
from kinuv.forward.model import sky_cube
from kinuv.forward.sb import galaxy_r_phi
from kinuv.infer.map import image_grid_for_vis, predict_binned
from kinuv.infer.s2 import fit_s2_start, propagate_equal_weight_ar1
from kinuv.io.vis import load_target_vis, radio_to_optical_kms
from kinuv.response.spectral import hann_native
from kinuv.validation.s4 import (
    channel_noise_from_integrated_error,
    topo_radio_to_lsrk_radio,
)


REPO = Path(__file__).resolve().parents[1]
KINMS_PYTHON = Path("/scratch/kinuv-thbrown/s1-kinms/bin/python")
KINMS_WORKER = REPO / "external/_kinms_best_worker.py"
SEEDS = (7001, 7002, 7003)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _git_state() -> dict:
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=REPO, text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=REPO, text=True
            ).strip()
        ),
    }


def _covariance(metrics: dict, target_id: str, n_bin: int):
    row = next(item for item in metrics["targets"] if item["target_id"] == target_id)
    source = next(iter(row["covariance"]["parameters"]["C1"].values()))
    return propagate_equal_weight_ar1(source["scale"], source["rho"], n_bin)


def _truth(config: dict) -> dict:
    seed = config["stage_a"]["parameter_seed"]
    inclination = float(config["geometry"]["inclination_deg"])
    return {
        "flux": float(seed["flux"]),
        "pa_deg": float(config["geometry"]["pa_seed_deg"]),
        "vsys_kms": float(seed["vsys_kms"]),
        "gas_sigma_kms": float(seed["gas_sigma_kms"]),
        "dx_arcsec": 0.0,
        "dy_arcsec": 0.0,
        "v0_kms": float(seed["v0_kms"]),
        "u_kms": float(seed["v0_kms"]) * np.sin(np.radians(inclination)),
        "r_t_arcsec": 0.30 if config["target_id"] == "KGAS066" else 0.55,
        "inclination_deg": inclination,
    }


def matched_disk_template(grid, truth: dict, bmaj_arcsec: float) -> np.ndarray:
    """Projected exponential disk shared exactly by cube and visibility truth."""

    radius, _ = galaxy_r_phi(
        grid,
        np.radians(float(truth["pa_deg"])),
        np.radians(float(truth["inclination_deg"])),
    )
    image = np.exp(-radius / (1.35 * float(bmaj_arcsec)))
    image *= 0.5 * (
        1.0 - np.tanh((radius - 4.5 * float(bmaj_arcsec)) / (0.3 * float(bmaj_arcsec)))
    )
    return image / (float(np.sum(image)) * float(grid.cell_arcsec) ** 2)


def projected_profile(parameters: dict, radius: np.ndarray) -> np.ndarray:
    inclination = float(parameters.get("inclination_deg", parameters.get("i_deg")))
    v0 = float(parameters.get("v0_kms", parameters.get("v0_kms_diagnostic")))
    rt = float(parameters["r_t_arcsec"])
    return v0 * np.sin(np.radians(inclination)) * (2.0 / np.pi) * np.arctan(radius / rt)


def profile_rmse(truth: dict, fitted: dict, radius: np.ndarray, weight: np.ndarray) -> float:
    residual = projected_profile(fitted, radius) - projected_profile(truth, radius)
    return float(np.sqrt(np.sum(weight * residual**2) / np.sum(weight)))


def _kinms_fit(config: dict, cube: Path, mask: Path, truth_cube: Path, work: Path) -> dict:
    cube = cube.resolve()
    mask = mask.resolve()
    truth_cube = truth_cube.resolve()
    work = work.resolve()
    retained = work / "kinms_fit_result.json"
    if retained.is_file():
        result = json.loads(retained.read_text(encoding="utf-8"))
        if result.get("success"):
            return result
    correction = float(config["spectral_frame"]["frequency_correction_equivalent_kms"])
    truth = config["_truth"]
    vsys_lsrk = float(topo_radio_to_lsrk_radio(truth["vsys_kms"], correction))
    vsys_optical = float(radio_to_optical_kms(vsys_lsrk))
    bounds = {
        "v0_kms": [0.45 * truth["v0_kms"], 1.65 * truth["v0_kms"]],
        "r_t_arcsec": [0.05, 2.0],
        "pa_deg": [truth["pa_deg"] - 35.0, truth["pa_deg"] + 35.0],
        "i_deg": [max(10.0, truth["inclination_deg"] - 20.0), min(80.0, truth["inclination_deg"] + 20.0)],
        "vsys_optical_kms": [vsys_optical - 100.0, vsys_optical + 100.0],
        "gas_sigma_kms": [2.0, 40.0],
    }
    worker_config = {
        "cube": str(cube),
        "mask": str(mask),
        "sb_cube": str(truth_cube),
        "work": str(work),
        "init": {
            "v0_kms": truth["v0_kms"] * 0.9,
            "r_t_arcsec": truth["r_t_arcsec"] * 1.3,
            "pa_deg": truth["pa_deg"] + 5.0,
            "i_deg": truth["inclination_deg"] + 4.0,
            "vsys_optical_kms": vsys_optical + 0.5,
            "gas_sigma_kms": truth["gas_sigma_kms"] * 1.1,
            "dx_arcsec": 0.0,
            "dy_arcsec": 0.0,
        },
        "bounds": bounds,
        "n_clouds": 100000,
        "n_clouds_de": 50000,
        "phase_center": [0.0, 0.0],
        "sb_pa_deg": truth["pa_deg"],
        "sb_inc_deg": truth["inclination_deg"],
    }
    work.mkdir(parents=True, exist_ok=True)
    config_path = work / "fit_config.json"
    _write_json(config_path, worker_config)
    process = subprocess.run(
        [str(KINMS_PYTHON), str(KINMS_WORKER), str(config_path)],
        cwd=REPO / "external",
        capture_output=True,
        text=True,
        check=False,
    )
    (work / "worker.log").write_text(
        (process.stdout or "") + (process.stderr or ""), encoding="ascii", errors="replace"
    )
    result_path = work / "kinms_fit_result.json"
    if not result_path.is_file():
        raise RuntimeError(f"KinMS worker failed with return code {process.returncode}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not result.get("success"):
        raise RuntimeError(f"KinMS fit failed: {result.get('message')}")
    return result


def run_target(config_path: Path, covariance_metrics: dict, output: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    target_id = config["target_id"]
    target_dir = output / target_id
    target_dir.mkdir(parents=True, exist_ok=True)
    phase = config.get("phase_center_deg")
    phase_rad = None if phase is None else np.radians(np.asarray(phase, dtype=float))
    data, load = load_target_vis(
        config["visibility_npz"],
        cube_path=config["fit_window_cube"],
        phase_dir_rad=phase_rad,
    )
    grid = image_grid_for_vis(data)
    bmaj = float(config["diagnostic_beam"]["bmaj_arcsec"])
    truth = _truth(config)
    template = matched_disk_template(grid, truth, bmaj)
    covariance = _covariance(covariance_metrics, target_id, data.n_bin)
    model_vis = np.asarray(
        predict_binned(
            data,
            truth,
            template,
            grid,
            i_rad=np.radians(truth["inclination_deg"]),
        )
    )

    with fits.open(config["diagnostic_cube"], memmap=False) as hdul:
        diagnostic_header = hdul[0].header.copy()
    official_mask = np.asarray(fits.getdata(config["diagnostic_mask"]), dtype=float) > 0.5
    error_m0 = np.squeeze(np.asarray(fits.getdata(config["diagnostic_ico_error"]), dtype=float))
    dv_cube = abs(float(diagnostic_header["CDELT3"]))
    if str(diagnostic_header.get("CUNIT3", "km/s")).lower().replace(" ", "") in {"m/s", "ms-1"}:
        dv_cube /= 1000.0
    sigma_cube, _ = channel_noise_from_integrated_error(error_m0, official_mask, dv_cube)

    cube_yxv = np.asarray(
        sky_cube(
            template,
            grid,
            data.freqs_native,
            flux=truth["flux"],
            pa_rad=np.radians(truth["pa_deg"]),
            vsys_kms=truth["vsys_kms"],
            dx_arcsec=truth["dx_arcsec"],
            dy_arcsec=truth["dy_arcsec"],
            gas_sigma_kms=truth["gas_sigma_kms"],
            v0_kms=truth["v0_kms"],
            r_t_arcsec=truth["r_t_arcsec"],
            i_rad=np.radians(truth["inclination_deg"]),
        )
    )
    cube_yxv = np.asarray(hann_native(cube_yxv, axis=2))
    guard = int(data.n_guard)
    cube_yxv = cube_yxv[:, :, guard:-guard]
    correction = float(config["spectral_frame"]["frequency_correction_equivalent_kms"])
    velocity = topo_radio_to_lsrk_radio(
        freq_to_velocity_kms(data.freqs_native[guard:-guard]), correction
    )
    fits_cube, model_header = sky_cube_fits(cube_yxv, grid, velocity, diagnostic_header)
    truth_cube, _, _ = match_model_to_imaging(
        fits_cube,
        model_header,
        diagnostic_header,
        undo_pb=True,
        nu_hz=float(np.median(data.freqs_native)),
    )
    finite = np.isfinite(truth_cube)
    synthetic_mask = finite & (truth_cube >= 0.005 * float(np.nanmax(truth_cube)))
    mask_path = target_dir / "truth_mask.fits"
    truth_path = target_dir / "truth_cube_k.fits"
    fits.PrimaryHDU(synthetic_mask.astype(np.uint8), diagnostic_header).writeto(mask_path, overwrite=True)
    truth_header = diagnostic_header.copy()
    truth_header["BUNIT"] = "K"
    truth_header["ORIGIN"] = "Python-native analytic kinUV/KinMS S4 truth"
    fits.PrimaryHDU(truth_cube.astype(np.float32), truth_header).writeto(truth_path, overwrite=True)

    radius = np.linspace(0.25 * bmaj, 3.5 * bmaj, 24)
    radial_weight = radius * np.exp(-radius / (1.35 * bmaj))
    realizations = []
    for index, seed in enumerate(SEEDS):
        realization_dir = target_dir / f"seed-{seed}"
        realization_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        noisy_vis = add_xx_noise(model_vis, data.weights, data.s, rng)
        mock_data = replace(data, vis=np.asarray(noisy_vis, dtype=np.complex128))
        perturb = (-1.0, 1.0, -0.5)[index]
        start = {
            "start_id": index,
            "flux": truth["flux"] * (1.0 + 0.08 * perturb),
            "pa_deg": truth["pa_deg"] + 7.0 * perturb,
            "vsys_kms": truth["vsys_kms"] + 0.5 * data.dv_kms * perturb,
            "gas_sigma_kms": truth["gas_sigma_kms"] * (1.0 + 0.1 * perturb),
            "dx_arcsec": 0.05 * perturb,
            "dy_arcsec": -0.04 * perturb,
            "u_kms": truth["u_kms"] * (1.0 + 0.12 * perturb),
            "inclination_deg": float(np.clip(truth["inclination_deg"] + 6.0 * perturb, 12.0, 78.0)),
            "r_t_arcsec": truth["r_t_arcsec"] * (1.0 + 0.25 * perturb),
        }
        kinuv_fit = fit_s2_start(
            mock_data,
            template,
            grid,
            start,
            covariance,
            pa_seed_deg=truth["pa_deg"],
            vsys_seed_kms=truth["vsys_kms"],
            bmaj_arcsec=bmaj,
            fixed_turnover_over_bmaj=None,
            maxiter=140,
        ).to_dict()
        cube_noise = rng.normal(0.0, sigma_cube, truth_cube.shape)
        noisy_cube = np.where(synthetic_mask, truth_cube + cube_noise, 0.0)
        cube_path = realization_dir / "mock_cube_k.fits"
        fits.PrimaryHDU(noisy_cube.astype(np.float32), truth_header).writeto(cube_path, overwrite=True)
        kinms_config = dict(config)
        kinms_config["_truth"] = truth
        kinms = _kinms_fit(
            kinms_config,
            cube_path,
            mask_path,
            truth_path,
            realization_dir / "kinms",
        )
        kinuv_parameters = dict(kinuv_fit["parameters"])
        kinuv_rmse = profile_rmse(truth, kinuv_parameters, radius, radial_weight)
        kinms_rmse = profile_rmse(truth, kinms["fitted"], radius, radial_weight)
        row = {
            "seed": seed,
            "kinuv": {
                "parameters": kinuv_parameters,
                "projected_velocity_rmse_kms": kinuv_rmse,
                "optimizer_success": kinuv_fit["success"],
                "projected_gradient_inf": kinuv_fit["projected_gradient_inf"],
            },
            "kinms": {
                "parameters": kinms["fitted"],
                "projected_velocity_rmse_kms": kinms_rmse,
                "optimizer_success": kinms["success"],
                "nfev": kinms["nfev"],
            },
            "ratio_kinuv_over_kinms": kinuv_rmse / kinms_rmse,
        }
        realizations.append(row)
        _write_json(realization_dir / "recovery.json", row)
        print(json.dumps({"target": target_id, "seed": seed, "rmse_ratio": row["ratio_kinuv_over_kinms"]}, sort_keys=True), flush=True)

    kinuv_all = np.asarray([row["kinuv"]["projected_velocity_rmse_kms"] for row in realizations])
    kinms_all = np.asarray([row["kinms"]["projected_velocity_rmse_kms"] for row in realizations])
    aggregate_ratio = float(np.sqrt(np.mean(kinuv_all**2)) / np.sqrt(np.mean(kinms_all**2)))
    record = {
        "schema_version": "kinuv-s4-python-truth-v1",
        "target_id": target_id,
        "truth": truth,
        "generation": {
            "cube": "analytic sky cube, Hann response, registered restoring beam, Python Gaussian noise",
            "visibilities": "native Python visibility forward operator, Hann plus software bin, complex Gaussian noise",
            "casa_used": False,
            "matched_axisymmetric_emissivity": True,
            "seeds": list(SEEDS),
            "cube_noise_sigma_k": sigma_cube,
            "profile_radius_arcsec": radius.tolist(),
        },
        "load": load,
        "realizations": realizations,
        "aggregate": {
            "kinuv_rms_rmse_kms": float(np.sqrt(np.mean(kinuv_all**2))),
            "kinms_rms_rmse_kms": float(np.sqrt(np.mean(kinms_all**2))),
            "ratio_kinuv_over_kinms": aggregate_ratio,
        },
        "gate": {
            "required_ratio_max": 0.90,
            "passed": aggregate_ratio <= 0.90,
        },
    }
    _write_json(target_dir / "summary.json", record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-config", action="append", type=Path, required=True)
    parser.add_argument("--covariance-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    state = _git_state()
    if state["branch"] != "dev" or state["dirty"]:
        raise RuntimeError("S4 synthetic recovery requires a clean exact commit on dev")
    if not KINMS_PYTHON.is_file() or not KINMS_WORKER.is_file():
        raise FileNotFoundError("frozen stock KinMS worker environment is unavailable")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    covariance = json.loads(args.covariance_metrics.read_text(encoding="utf-8"))
    targets = [run_target(path, covariance, args.output) for path in args.target_config]
    accepted = all(row["gate"]["passed"] for row in targets)
    summary = {
        "schema_version": "kinuv-s4-python-truth-summary-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": state,
        "targets": [
            {
                "target_id": row["target_id"],
                "aggregate": row["aggregate"],
                "gate": row["gate"],
            }
            for row in targets
        ],
        "accepted": accepted,
        "casa_used": False,
    }
    _write_json(args.output / "summary.json", summary)
    manifest = {
        "schema_version": "kinuv-s4-python-truth-manifest-v1",
        "code_commit": state["commit"],
        "files": {},
    }
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            manifest["files"][str(path.relative_to(args.output))] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    _write_json(args.output / "MANIFEST.json", manifest)
    print(json.dumps({"stage": "S4-synthetic", "accepted": accepted}, sort_keys=True), flush=True)
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
