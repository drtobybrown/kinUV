#!/usr/bin/env python3
"""Matched synthetic recovery for the common unified kinematic chart.

The same analytic sky, target sampling, realization identifier, and registered
S4 operators generate a visibility data set for kinUV and a restored cube for
the frozen stock KinMS comparator.  This is an isolated validation harness;
it does not alter production products or use an image likelihood for kinUV.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from astropy.io import fits
from scipy.optimize import minimize

from kinuv.constants import freq_to_velocity_kms
from kinuv.diagnostics.imaging import match_model_to_imaging
from kinuv.diagnostics.s1 import add_xx_noise, sky_cube_fits
from kinuv.forward.model import sky_cube
from kinuv.forward.sb import galaxy_r_phi
from kinuv.infer.map import image_grid_for_vis, predict_binned
from kinuv.infer.s2 import fit_s2_start, propagate_equal_weight_ar1
from kinuv.infer.unified import (
    UnifiedChartSpec,
    build_support_from_template,
    build_unified_log_density,
    decode_unified_chart,
    initial_unified_chart,
    unified_profile_callables,
)
from kinuv.io.vis import load_target_vis, radio_to_optical_kms
from kinuv.profiles.unified import projected_velocity, velocity_dispersion
from kinuv.response.spectral import hann_native
from kinuv.validation.s4 import (
    channel_noise_from_integrated_error,
    projected_arctan_speed,
    topo_radio_to_lsrk_radio,
)


CODE_ROOT = Path(os.environ.get("KINUV_CODE_ROOT", Path(__file__).resolve().parents[2]))
PROJECT = Path(os.environ.get("KINUV_WORKSPACE", CODE_ROOT.parent))
KINMS_WORKER = CODE_ROOT / "external/_kinms_best_worker.py"
DEFAULT_KINMS_PYTHON = Path("/scratch/kinuv-thbrown/s1-kinms/bin/python")
DEFAULT_COVARIANCE = (
    PROJECT / "results/validation/crossdomain-recovery-s2-20260907-r1/metrics.json"
)
SCENARIOS = ("smooth_monotonic", "nonmonotonic_bump", "varying_dispersion")
DEFAULT_SEEDS = (7401,)
R50_FLATNESS_MAX = 0.10
R50_ERROR_MAX_BMAJ = 0.25


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _git_state() -> dict:
    archived_commit = os.environ.get("KINUV_CODE_COMMIT")
    if archived_commit:
        return {
            "commit": archived_commit,
            "branch": "archive-snapshot",
            "production_source_dirty": False,
        }
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, text=True
        ).strip(),
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=CODE_ROOT, text=True
        ).strip(),
        "production_source_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--", "src", "external"],
                cwd=CODE_ROOT,
                text=True,
            ).strip()
        ),
    }


def matched_disk_template(grid, *, pa_rad: float, i_rad: float, bmaj: float):
    """The positive projected exponential disk used by both data products."""
    radius, _ = galaxy_r_phi(grid, pa_rad, i_rad)
    image = np.exp(-radius / (1.35 * bmaj))
    image *= 0.5 * (1.0 - np.tanh((radius - 4.5 * bmaj) / (0.3 * bmaj)))
    return image / (float(np.sum(image)) * float(grid.cell_arcsec) ** 2)


def _covariance(metrics: dict, target_id: str, n_bin: int):
    row = next(item for item in metrics["targets"] if item["target_id"] == target_id)
    source = next(iter(row["covariance"]["parameters"]["C1"].values()))
    return propagate_equal_weight_ar1(source["scale"], source["rho"], n_bin)


def _base_truth(config: dict, bmaj: float) -> dict:
    seed = config["stage_a"]["parameter_seed"]
    inclination = float(config["geometry"]["inclination_deg"])
    return {
        "flux": float(seed["flux"]),
        "pa_deg": float(config["geometry"]["pa_seed_deg"]),
        "vsys_kms": float(seed["vsys_kms"]),
        "sigma0_kms": float(seed["gas_sigma_kms"]),
        "dx_arcsec": 0.0,
        "dy_arcsec": 0.0,
        "v0_kms": float(seed["v0_kms"]),
        "u_inf_kms": float(seed["v0_kms"]) * np.sin(np.radians(inclination)),
        "r_t_arcsec": 0.30 if config["target_id"] == "KGAS066" else 0.55,
        "inclination_deg": inclination,
        "bmaj_arcsec": float(bmaj),
    }


def _scenario_profiles(name: str, truth: dict, support):
    """Return smooth projected-speed and dispersion truth callables."""
    u_inf = float(truth["u_inf_kms"])
    rt = float(truth["r_t_arcsec"])
    bmaj = float(truth["bmaj_arcsec"])
    sigma0 = float(truth["sigma0_kms"])

    def base_u(radius):
        radius = np.asarray(radius)
        return u_inf * (2.0 / np.pi) * np.arctan(radius / rt)

    if name == "smooth_monotonic":
        projected = base_u
        dispersion = lambda radius: np.zeros_like(np.asarray(radius)) + sigma0
        description = "analytic arctan projected speed; constant dispersion"
    elif name == "nonmonotonic_bump":
        def projected(radius):
            radius = np.asarray(radius)
            bump = 0.32 * np.exp(-0.5 * ((radius - 1.20 * bmaj) / (0.28 * bmaj)) ** 2)
            return base_u(radius) * (1.0 + bump)

        dispersion = lambda radius: np.zeros_like(np.asarray(radius)) + sigma0
        description = "analytic arctan carrier times a smooth broad positive bump"
    elif name == "varying_dispersion":
        projected = base_u

        def dispersion(radius):
            return np.asarray(
                velocity_dispersion(
                    radius, np.log(sigma0), np.array([2.5, 2.0]), support
                )
            )

        description = "analytic arctan projected speed; common two-mode log-sigma truth"
    else:
        raise ValueError(f"unknown scenario {name}")
    return projected, dispersion, description


def _truth_turnover_r50(name: str, projected_truth, truth: dict, support) -> dict:
    """Use the known asymptote; arctan controls map exactly to ``r_t``."""
    if name in {"smooth_monotonic", "varying_dispersion"}:
        return {
            "status": "RESOLVED",
            "value_arcsec": float(truth["r_t_arcsec"]),
            "outer_level_kms": float(truth["u_inf_kms"]),
            "definition": "first rising half of known analytic asymptote; exactly arctan r_t",
        }
    radius = np.linspace(0.0, support.emission_r95_arcsec, 4096)
    speed = projected_truth(radius)
    level = 0.5 * float(truth["u_inf_kms"])
    crossings = []
    for index in range(len(radius) - 1):
        if speed[index] <= level < speed[index + 1]:
            fraction = (level - speed[index]) / (speed[index + 1] - speed[index])
            crossings.append(float(radius[index] + fraction * (radius[index + 1] - radius[index])))
    if len(crossings) != 1:
        return {
            "status": "UNRESOLVED_NONUNIQUE" if crossings else "UNRESOLVED_NO_RISING_CROSSING",
            "value_arcsec": None,
            "candidate_crossings_arcsec": crossings,
            "outer_level_kms": float(truth["u_inf_kms"]),
        }
    return {
        "status": "RESOLVED",
        "value_arcsec": crossings[0],
        "outer_level_kms": float(truth["u_inf_kms"]),
        "definition": "first rising half of known analytic asymptote",
    }


def _turnover_r50(radius, speed, *, bmaj: float, r80: float, r95: float) -> dict:
    """First rising half-outer crossing, only when measured support is flat."""
    radius = np.asarray(radius, dtype=np.float64)
    speed = np.asarray(speed, dtype=np.float64)
    supported = np.isfinite(radius) & np.isfinite(speed) & (radius <= r95)
    outer = supported & (radius >= r80)
    if np.count_nonzero(outer) < 3:
        return {"status": "UNRESOLVED_OUTER_SUPPORT", "value_arcsec": None}
    outer_level = float(np.median(speed[outer]))
    if not np.isfinite(outer_level) or outer_level <= 0.0:
        return {"status": "UNRESOLVED_OUTER_LEVEL", "value_arcsec": None}
    outer_values = speed[outer]
    fractional_range = float(np.ptp(outer_values) / outer_level)
    if fractional_range > R50_FLATNESS_MAX:
        return {
            "status": "UNRESOLVED_PLATEAU",
            "value_arcsec": None,
            "outer_level_kms": outer_level,
            "outer_fractional_range": fractional_range,
        }
    level = 0.5 * outer_level
    rr = radius[supported]
    uu = speed[supported]
    crossings = []
    for index in range(len(rr) - 1):
        if uu[index] <= level < uu[index + 1]:
            fraction = (level - uu[index]) / (uu[index + 1] - uu[index])
            crossings.append(float(rr[index] + fraction * (rr[index + 1] - rr[index])))
    if not crossings:
        return {
            "status": "UNRESOLVED_NO_RISING_CROSSING",
            "value_arcsec": None,
            "outer_level_kms": outer_level,
            "outer_fractional_range": fractional_range,
        }
    if len(crossings) > 1:
        return {
            "status": "UNRESOLVED_NONUNIQUE",
            "value_arcsec": None,
            "candidate_crossings_arcsec": crossings,
            "outer_level_kms": outer_level,
            "outer_fractional_range": fractional_range,
        }
    return {
        "status": "RESOLVED",
        "value_arcsec": crossings[0],
        "outer_level_kms": outer_level,
        "outer_fractional_range": fractional_range,
        "definition": "first rising half of data-supported R80--R95 outer level",
        "emission_flux_quantile_note": "R80 and R95 refer to emission-flux quantiles",
    }


def _profile_metrics(
    radius,
    truth_u,
    truth_sigma,
    fit_u,
    fit_sigma,
    support,
    *,
    truth_r50=None,
    fit_r50=None,
    truth_central_slope=None,
    fit_central_slope=None,
) -> dict:
    radius = np.asarray(radius, dtype=np.float64)
    truth_u = np.asarray(truth_u, dtype=np.float64)
    fit_u = np.asarray(fit_u, dtype=np.float64)
    truth_sigma = np.asarray(truth_sigma, dtype=np.float64)
    fit_sigma = np.asarray(fit_sigma, dtype=np.float64)
    inner = radius <= support.bmaj_arcsec
    full = radius <= support.emission_r95_arcsec
    h = max(1.0e-5 * support.bmaj_arcsec, 1.0e-7)
    truth_slope = float(
        np.interp(h, radius, truth_u) / h
        if truth_central_slope is None
        else truth_central_slope
    )
    fit_slope = float(
        np.interp(h, radius, fit_u) / h
        if fit_central_slope is None
        else fit_central_slope
    )
    if truth_r50 is None:
        truth_r50 = _turnover_r50(
            radius,
            truth_u,
            bmaj=support.bmaj_arcsec,
            r80=support.emission_r80_arcsec,
            r95=support.emission_r95_arcsec,
        )
    if fit_r50 is None:
        fit_r50 = _turnover_r50(
            radius,
            fit_u,
            bmaj=support.bmaj_arcsec,
            r80=support.emission_r80_arcsec,
            r95=support.emission_r95_arcsec,
        )
    if truth_r50["value_arcsec"] is None or fit_r50["value_arcsec"] is None:
        recovery = {
            "status": "UNRESOLVED",
            "absolute_error_arcsec": None,
            "within_0p25_bmaj": None,
        }
    else:
        error = abs(fit_r50["value_arcsec"] - truth_r50["value_arcsec"])
        recovery = {
            "status": "RECOVERED" if error <= R50_ERROR_MAX_BMAJ * support.bmaj_arcsec else "RESOLVED_INCORRECT",
            "absolute_error_arcsec": float(error),
            "within_0p25_bmaj": bool(error <= R50_ERROR_MAX_BMAJ * support.bmaj_arcsec),
        }
    return {
        "inner_bmaj_u_rmse_kms": float(np.sqrt(np.mean((fit_u[inner] - truth_u[inner]) ** 2))),
        "full_support_u_rmse_kms": float(np.sqrt(np.mean((fit_u[full] - truth_u[full]) ** 2))),
        "sigma_rmse_kms": float(np.sqrt(np.mean((fit_sigma[full] - truth_sigma[full]) ** 2))),
        "central_du_dR_kms_per_arcsec": fit_slope,
        "central_du_dR_truth_kms_per_arcsec": truth_slope,
        "central_du_dR_absolute_error_kms_per_arcsec": abs(fit_slope - truth_slope),
        "truth_turnover_R50": truth_r50,
        "fit_turnover_R50": fit_r50,
        "turnover_recovery": recovery,
    }


def _fit_unified(mock_data, template, grid, covariance, spec, inclination_deg, maxiter):
    density = build_unified_log_density(mock_data, template, grid, covariance, spec)
    initial = np.asarray(
        initial_unified_chart(spec, inclination_deg=inclination_deg), dtype=np.float64
    )
    starts = [initial.copy(), initial.copy()]
    starts[0][6] += np.log(0.90)
    starts[0][1:6] += np.array([0.08, 0.10, 0.04, -0.04, 0.10])
    starts[1][6] += np.log(1.10)
    starts[1][7:11] += np.array([0.20, -0.20, 0.10, 0.0])
    starts[1][12:14] += np.array([0.30, -0.30])
    objective = jax.jit(jax.value_and_grad(lambda z: -density.log_posterior(z)))
    n_complex = int(mock_data.vis.size)
    attempts = []
    for start_id, start in enumerate(starts, 1):
        compile_started = perf_counter()
        _, gradient = objective(jnp.asarray(start))
        jax.block_until_ready(gradient)
        compile_s = perf_counter() - compile_started

        def normalized(z):
            value, grad = objective(jnp.asarray(z))
            return float(value) / n_complex, np.asarray(grad, dtype=np.float64) / n_complex

        optimize_started = perf_counter()
        fit = minimize(
            normalized,
            start,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": maxiter, "maxls": 40, "ftol": 1.0e-11, "gtol": 1.0e-3},
        )
        optimize_s = perf_counter() - optimize_started
        z = np.asarray(fit.x, dtype=np.float64)
        attempts.append(
            {
                "start_id": start_id,
                "z": z,
                "negative_log_posterior": float(-density.log_posterior(jnp.asarray(z))),
                "visibility_chi2": float(-2.0 * density.log_likelihood(jnp.asarray(z))),
                "success": bool(fit.success),
                "message": str(fit.message),
                "nit": int(fit.nit),
                "nfev": int(fit.nfev),
                "compile_s": compile_s,
                "optimize_s": optimize_s,
            }
        )
    best = min(attempts, key=lambda row: row["negative_log_posterior"])
    physical = decode_unified_chart(best["z"], spec)
    return best, attempts, physical


def _fit_legacy_smooth(mock_data, template, grid, covariance, truth, maxiter):
    start = {
        "start_id": 0,
        "flux": truth["flux"] * 0.92,
        "pa_deg": truth["pa_deg"] - 5.0,
        "vsys_kms": truth["vsys_kms"] - 0.5 * mock_data.dv_kms,
        "gas_sigma_kms": truth["sigma0_kms"] * 1.1,
        "dx_arcsec": 0.05,
        "dy_arcsec": -0.04,
        "u_kms": truth["u_inf_kms"] * 0.9,
        "inclination_deg": float(np.clip(truth["inclination_deg"] - 4.0, 12.0, 78.0)),
        "r_t_arcsec": truth["r_t_arcsec"] * 1.25,
    }
    return fit_s2_start(
        mock_data,
        template,
        grid,
        start,
        covariance,
        pa_seed_deg=truth["pa_deg"],
        vsys_seed_kms=truth["vsys_kms"],
        bmaj_arcsec=truth["bmaj_arcsec"],
        fixed_turnover_over_bmaj=None,
        maxiter=maxiter,
    ).to_dict()


def _best_arctan_approximation(projected_truth, truth: dict, support) -> dict:
    radius = np.linspace(0.01 * support.bmaj_arcsec, support.emission_r95_arcsec, 256)
    target = projected_truth(radius)

    def objective(values):
        model = values[0] * (2.0 / np.pi) * np.arctan(radius / values[1])
        return float(np.mean((model - target) ** 2))

    fit = minimize(
        objective,
        [truth["u_inf_kms"], truth["r_t_arcsec"]],
        method="L-BFGS-B",
        bounds=[(0.3 * truth["u_inf_kms"], 2.0 * truth["u_inf_kms"]), (0.03, 3.0 * support.bmaj_arcsec)],
    )
    return {"u_inf_kms": float(fit.x[0]), "r_t_arcsec": float(fit.x[1])}


def _kinms_fit(config, cube, mask, truth_cube, work, truth, projected_truth, support, kinms_python, seed):
    correction = float(config["spectral_frame"]["frequency_correction_equivalent_kms"])
    vsys_lsrk = float(topo_radio_to_lsrk_radio(truth["vsys_kms"], correction))
    vsys_optical = float(radio_to_optical_kms(vsys_lsrk))
    approximation = _best_arctan_approximation(projected_truth, truth, support)
    v0 = approximation["u_inf_kms"] / np.sin(np.radians(truth["inclination_deg"]))
    rt = approximation["r_t_arcsec"]
    worker_config = {
        "cube": str(cube.resolve()),
        "mask": str(mask.resolve()),
        "sb_cube": str(truth_cube.resolve()),
        "work": str(work.resolve()),
        "init": {
            "v0_kms": 0.9 * v0,
            "r_t_arcsec": 1.2 * rt,
            "pa_deg": truth["pa_deg"] + 5.0,
            "i_deg": truth["inclination_deg"] + 4.0,
            "vsys_optical_kms": vsys_optical + 0.5,
            "gas_sigma_kms": 1.1 * truth["sigma0_kms"],
            "dx_arcsec": 0.0,
            "dy_arcsec": 0.0,
        },
        "bounds": {
            "v0_kms": [0.4 * v0, 1.8 * v0],
            "r_t_arcsec": [0.03, 3.0 * support.bmaj_arcsec],
            "pa_deg": [truth["pa_deg"] - 35.0, truth["pa_deg"] + 35.0],
            "i_deg": [max(10.0, truth["inclination_deg"] - 20.0), min(80.0, truth["inclination_deg"] + 20.0)],
            "vsys_optical_kms": [vsys_optical - 100.0, vsys_optical + 100.0],
            "gas_sigma_kms": [2.0, 40.0],
        },
        "n_clouds": 100000,
        "n_clouds_de": 50000,
        "phase_center": [0.0, 0.0],
        "sb_pa_deg": truth["pa_deg"],
        "sb_inc_deg": truth["inclination_deg"],
        "provenance": {
            "paired_realization_seed": int(seed),
            "cube_sha256": _sha256(cube),
            "mask_sha256": _sha256(mask),
            "truth_cube_sha256": _sha256(truth_cube),
            "worker_sha256": _sha256(KINMS_WORKER),
            "initial_arctan_approximation": approximation,
        },
    }
    work.mkdir(parents=True, exist_ok=True)
    config_path = work / "fit_config.json"
    _write_json(config_path, worker_config)
    process = subprocess.run(
        [str(kinms_python), str(KINMS_WORKER), str(config_path)],
        cwd=KINMS_WORKER.parent,
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


def _truth_cube_and_mask(config, data, grid, template, truth, velocity_profile, dispersion_profile, target_dir):
    with fits.open(config["diagnostic_cube"], memmap=False) as hdul:
        header = hdul[0].header.copy()
    official_mask = np.asarray(fits.getdata(config["diagnostic_mask"]), dtype=float) > 0.5
    error_m0 = np.squeeze(np.asarray(fits.getdata(config["diagnostic_ico_error"]), dtype=float))
    dv_cube = abs(float(header["CDELT3"]))
    if str(header.get("CUNIT3", "km/s")).lower().replace(" ", "") in {"m/s", "ms-1"}:
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
            dx_arcsec=0.0,
            dy_arcsec=0.0,
            gas_sigma_kms=truth["sigma0_kms"],
            i_rad=np.radians(truth["inclination_deg"]),
            velocity_profile=velocity_profile,
            dispersion_profile=dispersion_profile,
        )
    )
    cube_yxv = np.asarray(hann_native(cube_yxv, axis=2))
    guard = int(data.n_guard)
    cube_yxv = cube_yxv[:, :, guard:-guard]
    correction = float(config["spectral_frame"]["frequency_correction_equivalent_kms"])
    velocity = topo_radio_to_lsrk_radio(
        freq_to_velocity_kms(data.freqs_native[guard:-guard]), correction
    )
    fits_cube, model_header = sky_cube_fits(cube_yxv, grid, velocity, header)
    truth_cube, _, _ = match_model_to_imaging(
        fits_cube,
        model_header,
        header,
        undo_pb=True,
        nu_hz=float(np.median(data.freqs_native)),
    )
    finite = np.isfinite(truth_cube)
    mask = finite & (truth_cube >= 0.005 * float(np.nanmax(truth_cube)))
    truth_header = header.copy()
    truth_header["BUNIT"] = "K"
    truth_header["ORIGIN"] = "Phase-4 matched analytic kinUV/KinMS truth"
    truth_path = target_dir / "truth_cube_k.fits"
    mask_path = target_dir / "truth_mask.fits"
    fits.PrimaryHDU(truth_cube.astype(np.float32), truth_header).writeto(truth_path, overwrite=True)
    fits.PrimaryHDU(mask.astype(np.uint8), header).writeto(mask_path, overwrite=True)
    return truth_cube, truth_header, mask, truth_path, mask_path, sigma_cube


def run_target(args) -> dict:
    config_path = Path(args.target_config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    target_id = config["target_id"]
    target_dir = args.output / target_id
    target_dir.mkdir(parents=True, exist_ok=args.resume)
    phase = config.get("phase_center_deg")
    phase_rad = None if phase is None else np.radians(np.asarray(phase, dtype=float))
    data, load = load_target_vis(
        config["visibility_npz"], cube_path=config["fit_window_cube"], phase_dir_rad=phase_rad
    )
    grid = image_grid_for_vis(data)
    bmaj = float(config["diagnostic_beam"]["bmaj_arcsec"])
    truth = _base_truth(config, bmaj)
    i_rad = np.radians(truth["inclination_deg"])
    pa_rad = np.radians(truth["pa_deg"])
    template = matched_disk_template(grid, pa_rad=pa_rad, i_rad=i_rad, bmaj=bmaj)
    support = build_support_from_template(template, grid, pa_rad=pa_rad, i_rad=i_rad, bmaj_arcsec=bmaj)
    covariance_metrics = json.loads(args.covariance_metrics.read_text(encoding="utf-8"))
    covariance = _covariance(covariance_metrics, target_id, data.n_bin)
    radius = np.linspace(0.0, support.emission_r95_arcsec, 320)
    records = []

    for scenario_index, scenario in enumerate(SCENARIOS):
        projected_truth, sigma_truth, description = _scenario_profiles(scenario, truth, support)
        velocity_truth = lambda r, p=projected_truth: p(r) / np.sin(i_rad)
        scenario_dir = target_dir / scenario
        scenario_dir.mkdir(parents=True, exist_ok=args.resume)
        truth_cube, truth_header, mask, truth_path, mask_path, sigma_cube = _truth_cube_and_mask(
            config, data, grid, template, truth, velocity_truth, sigma_truth, scenario_dir
        )
        model_vis = np.asarray(
            predict_binned(
                data,
                {
                    "flux": truth["flux"], "pa_deg": truth["pa_deg"],
                    "vsys_kms": truth["vsys_kms"], "gas_sigma_kms": truth["sigma0_kms"],
                    "dx_arcsec": 0.0, "dy_arcsec": 0.0,
                    "v0_kms": truth["v0_kms"], "r_t_arcsec": truth["r_t_arcsec"],
                },
                template,
                grid,
                i_rad=i_rad,
                velocity_profile=velocity_truth,
                dispersion_profile=sigma_truth,
            )
        )
        truth_u_grid = projected_truth(radius)
        truth_sigma_grid = sigma_truth(radius)
        truth_r50 = _truth_turnover_r50(scenario, projected_truth, truth, support)
        derivative_step = max(1.0e-6 * bmaj, 1.0e-8)
        truth_central_slope = float(projected_truth(derivative_step) / derivative_step)
        scenario_records = []
        for seed in args.seeds:
            realization_dir = scenario_dir / f"seed-{seed}"
            realization_dir.mkdir(parents=True, exist_ok=args.resume)
            seed_sequence = np.random.SeedSequence([int(seed), int(scenario_index)])
            vis_seed, cube_seed = seed_sequence.spawn(2)
            noisy_vis = add_xx_noise(
                model_vis, data.weights, data.s, np.random.default_rng(vis_seed)
            )
            mock_data = replace(data, vis=np.asarray(noisy_vis, dtype=np.complex128))
            u_reference = float(projected_truth(support.reference_radius_arcsec))
            sigma_reference = float(sigma_truth(support.reference_radius_arcsec))
            spec = UnifiedChartSpec(
                support=support,
                pa_reference_rad=pa_rad,
                vsys_reference_kms=truth["vsys_kms"],
                dv_kms=float(data.dv_kms),
                flux_reference=truth["flux"],
                u_reference_kms=u_reference,
                sigma_reference_kms=sigma_reference,
            )
            best, attempts, physical = _fit_unified(
                mock_data, template, grid, covariance, spec, truth["inclination_deg"], args.maxiter
            )
            velocity_fit, sigma_fit_callable = unified_profile_callables(best["z"], spec)
            fit_u = np.asarray(velocity_fit(radius)) * np.sin(float(physical["i_rad"]))
            fit_sigma = np.asarray(sigma_fit_callable(radius))
            unified_central_slope = float(
                projected_velocity(
                    derivative_step,
                    best["z"][6],
                    best["z"][7:11],
                    support,
                )
                / derivative_step
            )
            unified_metrics = _profile_metrics(
                radius,
                truth_u_grid,
                truth_sigma_grid,
                fit_u,
                fit_sigma,
                support,
                truth_r50=truth_r50,
                truth_central_slope=truth_central_slope,
                fit_central_slope=unified_central_slope,
            )

            cube_noise = np.random.default_rng(cube_seed).normal(0.0, sigma_cube, truth_cube.shape)
            noisy_cube = np.where(mask, truth_cube + cube_noise, 0.0)
            cube_path = realization_dir / "mock_cube_k.fits"
            fits.PrimaryHDU(noisy_cube.astype(np.float32), truth_header).writeto(cube_path, overwrite=True)
            kinms = _kinms_fit(
                config,
                cube_path,
                mask_path,
                truth_path,
                realization_dir / "kinms",
                truth,
                projected_truth,
                support,
                args.kinms_python,
                seed,
            )
            fitted = kinms["fitted"]
            kinms_u = projected_arctan_speed(
                {
                    "v0_kms": fitted["v0_kms"],
                    "inclination_deg": fitted["i_deg"],
                    "r_t_arcsec": fitted["r_t_arcsec"],
                },
                radius,
            )
            kinms_sigma = np.zeros_like(radius) + float(fitted["gas_sigma_kms"])
            kinms_r50 = {
                "status": "RESOLVED",
                "value_arcsec": float(fitted["r_t_arcsec"]),
                "definition": "stock arctan parameter; exactly first half-asymptote crossing",
            }
            kinms_central_slope = float(
                fitted["v0_kms"]
                * np.sin(np.radians(fitted["i_deg"]))
                * (2.0 / np.pi)
                / fitted["r_t_arcsec"]
            )
            kinms_metrics = _profile_metrics(
                radius,
                truth_u_grid,
                truth_sigma_grid,
                kinms_u,
                kinms_sigma,
                support,
                truth_r50=truth_r50,
                fit_r50=kinms_r50,
                truth_central_slope=truth_central_slope,
                fit_central_slope=kinms_central_slope,
            )

            legacy = None
            if scenario == "smooth_monotonic":
                legacy_fit = _fit_legacy_smooth(
                    mock_data, template, grid, covariance, truth, args.maxiter
                )
                legacy_parameters = legacy_fit["parameters"]
                legacy_u = projected_arctan_speed(legacy_parameters, radius)
                legacy_sigma = np.zeros_like(radius) + float(legacy_parameters["gas_sigma_kms"])
                legacy = {
                    "parameters": legacy_parameters,
                    "optimizer_success": bool(legacy_fit["success"]),
                    "metrics": _profile_metrics(
                        radius,
                        truth_u_grid,
                        truth_sigma_grid,
                        legacy_u,
                        legacy_sigma,
                        support,
                        truth_r50=truth_r50,
                        fit_r50={
                            "status": "RESOLVED",
                            "value_arcsec": float(legacy_parameters["r_t_arcsec"]),
                            "definition": "legacy kinUV arctan parameter",
                        },
                        truth_central_slope=truth_central_slope,
                        fit_central_slope=float(
                            legacy_parameters["v0_kms"]
                            * np.sin(np.radians(legacy_parameters["inclination_deg"]))
                            * (2.0 / np.pi)
                            / legacy_parameters["r_t_arcsec"]
                        ),
                    ),
                }

            row = {
                "schema_version": "kinuv-unified-phase4-realization-v1",
                "target_id": target_id,
                "scenario": scenario,
                "scenario_description": description,
                "seed": int(seed),
                "paired_noise_contract": {
                    "realization_id": f"{target_id}:{scenario}:{seed}",
                    "root_entropy": [int(seed), int(scenario_index)],
                    "visibility_child_spawn_key": list(vis_seed.spawn_key),
                    "cube_child_spawn_key": list(cube_seed.spawn_key),
                    "note": "paired deterministic child streams; arrays differ because visibility and cube domains differ",
                },
                "truth": {**truth, "turnover_R50_arctan_arcsec": truth["r_t_arcsec"]},
                "support": support.metadata(),
                "profile_grid": {
                    "radius_arcsec": radius.tolist(),
                    "truth_u_projected_kms": np.asarray(truth_u_grid).tolist(),
                    "truth_sigma_kms": np.asarray(truth_sigma_grid).tolist(),
                },
                "kinuv_unified": {
                    "optimum_z": best["z"].tolist(),
                    "physical": {
                        "inclination_deg": float(np.degrees(physical["i_rad"])),
                        "pa_deg": float(np.degrees(physical["pa_rad"])),
                        "u_reference_kms": float(physical["u_reference_kms"]),
                        "sigma0_kms": float(physical["sigma0_kms"]),
                    },
                    "selected_start": {key: value for key, value in best.items() if key != "z"},
                    "attempts": [{key: value for key, value in item.items() if key != "z"} for item in attempts],
                    "metrics": unified_metrics,
                },
                "kinms_stock": {
                    "family": "stock arctan rotation plus constant gasSigma",
                    "parameters": fitted,
                    "optimizer_success": bool(kinms["success"]),
                    "nfev": int(kinms["nfev"]),
                    "metrics": kinms_metrics,
                },
                "legacy_kinuv_arctan_smooth_control": legacy,
                "ratios": {
                    "inner_u_rmse_kinuv_over_kinms": unified_metrics["inner_bmaj_u_rmse_kms"] / kinms_metrics["inner_bmaj_u_rmse_kms"],
                    "full_u_rmse_kinuv_over_kinms": unified_metrics["full_support_u_rmse_kms"] / kinms_metrics["full_support_u_rmse_kms"],
                    "sigma_rmse_kinuv_over_kinms": unified_metrics["sigma_rmse_kms"] / max(kinms_metrics["sigma_rmse_kms"], 1.0e-12),
                    "smooth_inner_u_rmse_unified_over_legacy": None if legacy is None else unified_metrics["inner_bmaj_u_rmse_kms"] / legacy["metrics"]["inner_bmaj_u_rmse_kms"],
                },
            }
            scenario_records.append(row)
            records.append(row)
            _write_json(realization_dir / "recovery.json", row)
            print(json.dumps({"target": target_id, "scenario": scenario, "seed": seed, "inner_ratio": row["ratios"]["inner_u_rmse_kinuv_over_kinms"]}, sort_keys=True), flush=True)
        _write_json(scenario_dir / "summary.json", {"scenario": scenario, "realizations": scenario_records})

    unified_inner = np.array([row["kinuv_unified"]["metrics"]["inner_bmaj_u_rmse_kms"] for row in records])
    kinms_inner = np.array([row["kinms_stock"]["metrics"]["inner_bmaj_u_rmse_kms"] for row in records])
    inner_ratio = float(np.sqrt(np.mean(unified_inner ** 2)) / np.sqrt(np.mean(kinms_inner ** 2)))
    smooth_rows = [row for row in records if row["scenario"] == "smooth_monotonic"]
    unified_smooth = np.array([row["kinuv_unified"]["metrics"]["inner_bmaj_u_rmse_kms"] for row in smooth_rows])
    legacy_smooth = np.array([row["legacy_kinuv_arctan_smooth_control"]["metrics"]["inner_bmaj_u_rmse_kms"] for row in smooth_rows])
    smooth_ratio = float(np.sqrt(np.mean(unified_smooth ** 2)) / np.sqrt(np.mean(legacy_smooth ** 2)))
    turnover_statuses = [row["kinuv_unified"]["metrics"]["turnover_recovery"]["status"] for row in records]
    gates = {
        "kinuv_inner_u_rmse_at_least_10pct_lower_than_kinms": inner_ratio <= 0.90,
        "no_smooth_disk_regression_against_same_data_legacy_kinuv": smooth_ratio <= 1.00,
        "no_false_resolved_turnover": "RESOLVED_INCORRECT" not in turnover_statuses,
    }
    record = {
        "schema_version": "kinuv-unified-phase4-target-v1",
        "created_utc": _now(),
        "target_id": target_id,
        "git": _git_state(),
        "load": load,
        "scenario_names": list(SCENARIOS),
        "seeds": list(args.seeds),
        "operators": {
            "kinuv": "native predict_binned visibility operator with frozen C1 covariance",
            "kinms": "stock KinMS restored-cube worker; arctan rotation and constant gasSigma",
            "truth_cube": "sky_cube, Hann response, registered restoring beam, Gaussian channel noise",
            "truth_visibility": "predict_binned, complex Gaussian XX noise",
            "image_likelihood_used_for_kinuv": False,
            "dark_matter_model_used": False,
        },
        "inputs": {
            "target_config": {"path": str(config_path), "sha256": _sha256(config_path)},
            "covariance_metrics": {"path": str(args.covariance_metrics.resolve()), "sha256": _sha256(args.covariance_metrics)},
            "runner": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__))},
            "kinms_worker": {"path": str(KINMS_WORKER.resolve()), "sha256": _sha256(KINMS_WORKER)},
            "kinms_python": str(args.kinms_python.resolve()),
        },
        "realizations": records,
        "aggregate": {
            "inner_u_rmse_kinuv_rms_kms": float(np.sqrt(np.mean(unified_inner ** 2))),
            "inner_u_rmse_kinms_rms_kms": float(np.sqrt(np.mean(kinms_inner ** 2))),
            "inner_u_rmse_ratio_kinuv_over_kinms": inner_ratio,
            "smooth_control_inner_u_rmse_ratio_unified_over_legacy_kinuv": smooth_ratio,
        },
        "gates": gates,
        "gate_pass": all(gates.values()),
    }
    _write_json(target_dir / "summary.json", record)
    return record


def validate_plan(args) -> int:
    config = json.loads(args.target_config.read_text(encoding="utf-8"))
    phase = config.get("phase_center_deg")
    phase_rad = None if phase is None else np.radians(np.asarray(phase, dtype=float))
    data, _ = load_target_vis(config["visibility_npz"], cube_path=config["fit_window_cube"], phase_dir_rad=phase_rad)
    grid = image_grid_for_vis(data)
    bmaj = float(config["diagnostic_beam"]["bmaj_arcsec"])
    truth = _base_truth(config, bmaj)
    pa = np.radians(truth["pa_deg"])
    inc = np.radians(truth["inclination_deg"])
    template = matched_disk_template(grid, pa_rad=pa, i_rad=inc, bmaj=bmaj)
    support = build_support_from_template(template, grid, pa_rad=pa, i_rad=inc, bmaj_arcsec=bmaj)
    radius = np.linspace(0.0, support.emission_r95_arcsec, 320)
    checks = {}
    for scenario in SCENARIOS:
        u, sigma, description = _scenario_profiles(scenario, truth, support)
        uu = u(radius)
        ss = sigma(radius)
        checks[scenario] = {
            "description": description,
            "finite": bool(np.all(np.isfinite(uu)) and np.all(np.isfinite(ss))),
            "u_origin_exact_zero": bool(uu[0] == 0.0),
            "sigma_positive": bool(np.all(ss > 0.0)),
            "nonmonotonic": bool(np.any(np.diff(uu) < 0.0)),
            "turnover_R50": _truth_turnover_r50(scenario, u, truth, support),
        }
    expected = checks["smooth_monotonic"]["nonmonotonic"] is False and checks["nonmonotonic_bump"]["nonmonotonic"] is True
    output = {"schema_version": "kinuv-unified-phase4-plan-v1", "target_id": config["target_id"], "support": support.metadata(), "checks": checks, "passed": bool(expected and all(row["finite"] and row["u_origin_exact_zero"] and row["sigma_positive"] for row in checks.values()))}
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if output["passed"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--covariance-metrics", type=Path, default=DEFAULT_COVARIANCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--kinms-python", type=Path, default=DEFAULT_KINMS_PYTHON)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-plan", action="store_true")
    args = parser.parse_args()
    args.target_config = args.target_config.resolve()
    args.covariance_metrics = args.covariance_metrics.resolve()
    if args.validate_plan:
        return validate_plan(args)
    if args.output is None:
        parser.error("--output is required unless --validate-plan is used")
    args.output = args.output.resolve()
    if not args.kinms_python.is_file():
        raise FileNotFoundError(f"frozen KinMS Python unavailable: {args.kinms_python}")
    if not KINMS_WORKER.is_file():
        raise FileNotFoundError(KINMS_WORKER)
    state = _git_state()
    if state["production_source_dirty"]:
        raise RuntimeError("production src/external tree is dirty")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    result = run_target(args)
    print(json.dumps({"target": result["target_id"], "gate_pass": result["gate_pass"]}, sort_keys=True))
    return 0 if result["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
