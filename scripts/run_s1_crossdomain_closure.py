#!/usr/bin/env python3
"""Run the frozen S1 intrinsic-KinMS and shared-operator closure dossier.

This is a deterministic validation run, not an optimizer or sampler campaign.
KinMS executes only in an isolated subprocess and returns a beam-free native
cube. kinUV then owns PB attenuation, visibility sampling, and Hann/bin.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
from types import SimpleNamespace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, sleep

import numpy as np

from kinuv.constants import ARCSEC_TO_RAD
from kinuv.diagnostics.comparator import (
    load_intrinsic_kinms_cube,
    sample_intrinsic_kinms,
    sha256_file,
)
from kinuv.forward.sb import galaxy_r_phi, load_sb_template
from kinuv.infer.map import image_grid_for_vis
from kinuv.io.vis import load_target_vis
from kinuv.likelihood.chi2 import chi2
from kinuv.profiles.rotation import arctan_vc
from kinuv.transforms.grid import ImageGrid
from kinuv.transforms.nufft import nufft_backend

REPO = Path(__file__).resolve().parents[1]
WORKER = REPO / "external" / "_kinms_intrinsic_worker.py"
S0_METRICS = Path(
    "/arc/projects/KILOGAS/analysis/toby_sandbox/results/validation/"
    "crossdomain-recovery-s0-20260906/metrics.json"
)
DEFAULT_OUTPUT = Path(
    "/arc/projects/KILOGAS/analysis/toby_sandbox/results/validation/"
    "crossdomain-recovery-s1-20260906-r2"
)
SCHEMA = "kinuv-crossdomain-s1-v1"
TARGETS = ("KGAS066", "KGAS007")
COMMON_SEED = 66
INDEPENDENT_SEED = 1066
EXPECTED_S0_COMMIT = "fb4a14543d579168c9224c8ebca6a7591147f4db"
SOURCE_SNAPSHOT_FILES = (
    "external/_kinms_intrinsic_worker.py",
    "external/requirements-kinms-s1.txt",
    "scripts/run_s1_crossdomain_closure.py",
    "src/kinuv/diagnostics/comparator.py",
    "src/kinuv/forward/model.py",
    "src/kinuv/forward/operators.py",
    "tests/test_s1_comparator.py",
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_state() -> dict:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True
    ).splitlines()
    return {
        "commit": commit,
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=REPO, text=True
        ).strip(),
        "tracked_dirty": any(not line.startswith("??") for line in porcelain),
        "untracked_paths": [line for line in porcelain if line.startswith("??")],
    }


def _require_clean_commit() -> dict:
    state = _git_state()
    if state["branch"] != "dev":
        raise RuntimeError(f"S1 must run on branch dev, got {state['branch']!r}")
    if state["tracked_dirty"] or state["untracked_paths"]:
        raise RuntimeError("S1 dossier requires a clean exact commit")
    return state


def _lock_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _isolated_subprocess_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    return environment


def _validate_kinms_environment(kinms_python: Path) -> dict:
    lock_path = REPO / "external" / "requirements-kinms-s1.txt"
    expected = _lock_lines(lock_path)
    actual = subprocess.check_output(
        [str(kinms_python), "-m", "pip", "freeze"],
        text=True,
        env=_isolated_subprocess_environment(),
    ).splitlines()
    if actual != expected:
        raise RuntimeError("isolated KinMS environment differs from checked-in lock")
    probe = subprocess.check_output(
        [
            str(kinms_python),
            "-c",
            (
                "import hashlib,importlib.metadata,json,kinms,pathlib;"
                "p=pathlib.Path(kinms.__file__).resolve();"
                "print(json.dumps({'version':importlib.metadata.version('kinms'),"
                "'module':str(p),'module_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))"
            ),
        ],
        text=True,
        env=_isolated_subprocess_environment(),
    )
    identity = json.loads(probe)
    if identity["version"] != "3.0.13":
        raise RuntimeError(f"unexpected KinMS version {identity['version']!r}")
    return {
        "lock_path": str(lock_path),
        "lock_sha256": sha256_file(lock_path),
        "freeze": actual,
        **identity,
    }


def _validate_s0_inputs() -> tuple[dict, dict]:
    root = S0_METRICS.parent
    manifest_path = root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "kinuv-s0-artifact-manifest-v1":
        raise RuntimeError("unexpected S0 manifest schema")
    if manifest.get("code_commit") != EXPECTED_S0_COMMIT:
        raise RuntimeError("S0 manifest does not name the accepted implementation")
    for relative, entry in manifest.get("files", {}).items():
        path = root / relative
        if path.stat().st_size != int(entry["bytes"]):
            raise RuntimeError(f"S0 artifact size mismatch: {path}")
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"S0 artifact checksum mismatch: {path}")
    metrics = json.loads(S0_METRICS.read_text())
    if metrics.get("schema_version") != "kinuv-s0-accounting-v1":
        raise RuntimeError("unexpected S0 metrics schema")
    if metrics.get("code", {}).get("commit") != EXPECTED_S0_COMMIT:
        raise RuntimeError("S0 metrics do not name the accepted implementation")
    checked_inputs = {}
    for target in TARGETS:
        target_record = metrics["targets"][target]
        config_record = target_record["config"]
        config_path = REPO / "configs" / "targets" / f"{target}.json"
        if config_record["schema_version"] != "kinuv-production-target-v2":
            raise RuntimeError(f"unexpected {target} target schema")
        if sha256_file(config_path) != config_record["sha256"]:
            raise RuntimeError(f"{target} config differs from accepted S0 input")
        checked_inputs[target] = {"config": config_record, "files": {}}
        for name, entry in target_record["inputs"].items():
            path = Path(entry["path"])
            actual = sha256_file(path)
            if actual != entry["sha256"]:
                raise RuntimeError(f"{target} S0 input changed: {name}")
            checked_inputs[target]["files"][name] = entry
    return metrics, {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "metrics": str(S0_METRICS),
        "metrics_sha256": sha256_file(S0_METRICS),
        "code_commit": EXPECTED_S0_COMMIT,
        "checked_inputs": checked_inputs,
    }


def _environment(kinms_python: Path, kinms_identity: dict) -> dict:
    versions = {}
    for name in ("kinuv", "numpy", "scipy", "jax", "jax-finufft", "astropy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    cpu_model = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    thread_names = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "XLA_FLAGS",
        "JAX_PLATFORM_NAME",
        "JAX_ENABLE_X64",
    )
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count(),
        "thread_environment": {name: os.environ.get(name) for name in thread_names},
        "nufft_backend": nufft_backend(),
        "packages": versions,
        "kinms_python": str(kinms_python),
        "kinms_environment": kinms_identity,
    }


def _target_data(config: dict):
    phase = config.get("phase_center_deg")
    phase_rad = None if phase is None else np.radians(np.asarray(phase, dtype=float))
    return load_target_vis(
        config["visibility_npz"],
        cube_path=config["fit_window_cube"],
        phase_dir_rad=phase_rad,
    )


def _radial_surface_brightness(template, grid, pa_rad, inclination_rad):
    radius, _ = galaxy_r_phi(grid, pa_rad, inclination_rad)
    image = np.asarray(template, dtype=np.float64)
    rmax = float(np.max(radius))
    edges = np.linspace(0.0, rmax, 97)
    centres = 0.5 * (edges[:-1] + edges[1:])
    profile = np.full(centres.size, np.nan)
    for index in range(centres.size):
        selected = (radius >= edges[index]) & (radius < edges[index + 1])
        if np.any(selected):
            values = image[selected]
            values = values[np.isfinite(values) & (values >= 0.0)]
            if values.size:
                profile[index] = float(np.mean(values))
    valid = np.isfinite(profile) & (profile > 0.0)
    if valid.sum() < 2:
        raise ValueError("surface-brightness profile has insufficient support")
    profile = np.interp(
        centres, centres[valid], profile[valid], left=profile[valid][0], right=0.0
    )
    profile = np.maximum(profile, np.max(profile) * 1.0e-12)
    return centres, profile


def _run_worker(kinms_python: Path, config: dict, path: Path) -> dict:
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    log_path = path.with_suffix(".log")
    command = [str(kinms_python), str(WORKER), str(path)]
    start = perf_counter()
    process = subprocess.Popen(
        command,
        cwd=REPO,
        env=_isolated_subprocess_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    peak_rss_kib = 0
    status_path = Path(f"/proc/{process.pid}/status")
    while process.poll() is None:
        try:
            status = status_path.read_text()
            match = re.search(r"^VmHWM:\s*(\d+)\s+kB$", status, re.MULTILINE)
            if match is None:
                match = re.search(r"^VmRSS:\s*(\d+)\s+kB$", status, re.MULTILINE)
            if match is not None:
                peak_rss_kib = max(peak_rss_kib, int(match.group(1)))
        except FileNotFoundError:
            pass
        sleep(0.05)
    stdout, _ = process.communicate()
    elapsed = perf_counter() - start
    log_path.write_text(stdout)
    if process.returncode != 0:
        raise RuntimeError(f"KinMS worker failed ({process.returncode}); see {log_path}")
    if peak_rss_kib <= 0:
        raise RuntimeError("worker peak RSS could not be measured from /proc")
    return {
        "command": command,
        "elapsed_s": elapsed,
        "worker_peak_rss_kib": peak_rss_kib,
        "worker_peak_rss_method": "/proc/<pid>/status VmHWM sampled every 0.05 s",
        "config_sha256": sha256_file(path),
        "log": str(log_path),
        "log_sha256": sha256_file(log_path),
    }


def _thermal_component_rms(delta, data) -> float:
    weights = np.asarray(data.weights, dtype=np.float64)
    good = weights > 0.0
    sigma = np.zeros_like(weights)
    sigma[good] = 1.0 / np.sqrt(float(data.s) * weights[good])
    z_real = np.asarray(delta).real[good] / sigma[good]
    z_imag = np.asarray(delta).imag[good] / sigma[good]
    return float(np.sqrt(np.mean(np.concatenate([z_real, z_imag]) ** 2)))


def _van_der_corput(count: int) -> list[float]:
    values = []
    for integer in range(count):
        value = integer
        denominator = 1.0
        result = 0.0
        while value:
            value, remainder = divmod(value, 2)
            denominator *= 2.0
            result += remainder / denominator
        values.append(result)
    return values


def _render_config(output, grid, data, params, sb_radius, sb_profile, render_settings):
    velocity_radius = np.linspace(0.0, max(15.0, float(sb_radius[-1])), 257)
    return {
        "schema_version": "kinms-continuum-cube-v2",
        "output_npz": str(output),
        "artifact_id": f"{output.parent.name}/{output.name}",
        "grid": {
            "nx": int(grid.nx),
            "ny": int(grid.ny),
            "cell_arcsec": float(grid.cell_arcsec),
        },
        "velocity_centers_kms": np.asarray(data.vel_native, dtype=float).tolist(),
        "seed": int(render_settings["seed"]),
        "radial_samples": int(render_settings["radial_samples"]),
        "azimuth_samples": int(render_settings["azimuth_samples"]),
        "spectral_oversample": int(render_settings.get("spectral_oversample", 1)),
        "azimuth_phase_fractions": [
            float(value) for value in render_settings["azimuth_phase_fractions"]
        ],
        "surface_brightness": {
            "radius_arcsec": np.asarray(sb_radius, dtype=float).tolist(),
            "profile": np.asarray(sb_profile, dtype=float).tolist(),
        },
        "parameters": {
            "velocity_radius_arcsec": velocity_radius.tolist(),
            "circular_speed_kms": np.asarray(
                arctan_vc(velocity_radius, params["v0_kms"], params["r_t_arcsec"]),
                dtype=float,
            ).tolist(),
            "inclination_deg": float(params["inclination_deg"]),
            "pa_deg": float(params["pa_deg"]),
            "gas_sigma_kms": float(params["gas_sigma_kms"]),
            "flux_jy_kms": float(params["flux"]),
            "dx_arcsec": float(params["dx_arcsec"]),
            "dy_arcsec": float(params["dy_arcsec"]),
            "vsys_kms": float(params["vsys_kms"]),
        },
    }


def _cube_geometry(cube, grid, velocity_kms, vsys_kms) -> dict:
    cube = np.asarray(cube, dtype=np.float64)
    velocity = np.asarray(velocity_kms, dtype=np.float64)
    moment0 = cube.sum(axis=2)
    total = float(moment0.sum())
    east, north = grid.pixel_lm_rad()
    east = east / ARCSEC_TO_RAD
    north = north / ARCSEC_TO_RAD
    centroid_east = float(np.sum(east * moment0) / total)
    centroid_north = float(np.sum(north * moment0) / total)
    signed = np.sum(cube * (velocity[None, None, :] - float(vsys_kms)), axis=2)
    dipole_east = float(np.sum(east * signed))
    dipole_north = float(np.sum(north * signed))
    pa = float(np.degrees(np.arctan2(dipole_east, dipole_north)) % 360.0)
    return {
        "flux_weighted_centroid_east_arcsec": centroid_east,
        "flux_weighted_centroid_north_arcsec": centroid_north,
        "signed_velocity_dipole_east": dipole_east,
        "signed_velocity_dipole_north": dipole_north,
        "receding_pa_deg_east_of_north": pa,
    }


def _angle_separation_deg(first, second) -> float:
    return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)


def worker_coordinate_closure(root: Path, kinms_python: Path, kinms_identity: dict) -> dict:
    """Signed, asymmetric external-boundary test independent of target clipping."""
    contract_root = root / "worker-coordinate-contract"
    contract_root.mkdir(parents=True)
    grid = ImageGrid(48, 48, 0.25)
    velocity = 800.0 + np.arange(81, dtype=np.float64) * 5.0
    params = {
        "flux": 12.5,
        "pa_deg": 37.0,
        "vsys_kms": 1000.37,
        "gas_sigma_kms": 8.0,
        "dx_arcsec": 0.62,
        "dy_arcsec": -0.37,
        "v0_kms": 135.0,
        "r_t_arcsec": 0.7,
        "inclination_deg": 53.0,
    }
    radius = np.linspace(0.0, 4.0, 129)
    profile = np.exp(-radius / 1.1)
    settings = {
        "radial_samples": 128,
        "azimuth_samples": 512,
        "spectral_oversample": 2,
        "azimuth_phase_fractions": [0.0],
        "seed": COMMON_SEED,
    }
    output = contract_root / "signed-asymmetric.npz"
    config = _render_config(
        output,
        grid,
        SimpleNamespace(vel_native=velocity),
        params,
        radius,
        profile,
        settings,
    )
    run = _run_worker(kinms_python, config, contract_root / "signed-asymmetric.config.json")
    cube, metadata = load_intrinsic_kinms_cube(
        output, grid=grid, velocity_centers_kms=velocity
    )
    if metadata["kinms_version"] != kinms_identity["version"]:
        raise RuntimeError("coordinate test used an unlocked KinMS version")
    if metadata["kinms_module_sha256"] != kinms_identity["module_sha256"]:
        raise RuntimeError("coordinate test used an unlocked KinMS module")
    geometry = _cube_geometry(cube, grid, velocity, params["vsys_kms"])
    geometry["receding_pa_error_deg"] = _angle_separation_deg(
        geometry["receding_pa_deg_east_of_north"], params["pa_deg"]
    )
    geometry["centroid_error_arcsec"] = float(
        np.hypot(
            geometry["flux_weighted_centroid_east_arcsec"] - params["dx_arcsec"],
            geometry["flux_weighted_centroid_north_arcsec"] - params["dy_arcsec"],
        )
    )
    spectral_flux = cube.sum(axis=(0, 1))
    centroid = float(np.sum(velocity * spectral_flux) / np.sum(spectral_flux))
    centroid_error_channel = abs(centroid - params["vsys_kms"]) / 5.0
    gates = {
        "signed_receding_pa": geometry["receding_pa_error_deg"] <= 3.0,
        "asymmetric_east_north_offset": geometry["centroid_error_arcsec"]
        <= grid.cell_arcsec,
        "spectral_centroid": centroid_error_channel <= 0.02,
    }
    return {
        "parameters": params,
        "grid": {"ny": grid.ny, "nx": grid.nx, "cell_arcsec": grid.cell_arcsec},
        "native_channel_width_kms": 5.0,
        "run": run,
        "artifact": "worker-coordinate-contract/signed-asymmetric.npz",
        "artifact_sha256": sha256_file(output),
        "metadata": metadata,
        "geometry": geometry,
        "spectral_centroid_kms": centroid,
        "spectral_centroid_error_native_channel": centroid_error_channel,
        "thresholds": {
            "receding_pa_error_deg_max": 3.0,
            "centroid_error_arcsec_max": grid.cell_arcsec,
            "spectral_centroid_error_native_channel_max": 0.02,
        },
        "gates": gates,
        "pass": all(gates.values()),
    }


def target_closure(
    target: str,
    root: Path,
    kinms_python: Path,
    s0: dict,
    kinms_identity: dict,
) -> dict:
    config_path = REPO / "configs" / "targets" / f"{target}.json"
    config = json.loads(config_path.read_text())
    data, load_metadata = _target_data(config)
    grid = image_grid_for_vis(data)
    template = load_sb_template(grid, Path(config["template_ico"]))
    selected = s0["targets"][target]["selected_rotating_start"]
    params = {
        name: float(selected[name])
        for name in (
            "flux",
            "pa_deg",
            "vsys_kms",
            "gas_sigma_kms",
            "dx_arcsec",
            "dy_arcsec",
            "v0_kms",
            "r_t_arcsec",
        )
    }
    params["inclination_deg"] = float(config["geometry"]["inclination_deg"])
    pa_rad = np.radians(params["pa_deg"])
    inclination_rad = np.radians(params["inclination_deg"])
    sb_radius, sb_profile = _radial_surface_brightness(
        template, grid, pa_rad, inclination_rad
    )

    target_root = root / target
    target_root.mkdir(parents=True, exist_ok=True)
    common_phases = [0.0]
    independent_phases = [float(np.sqrt(2.0) / 7.0)]
    reference = {
        "radial_samples": 128,
        "azimuth_samples": 512,
        "spectral_oversample": 2,
        "spatial_oversample": 2,
        "azimuth_phase_fractions": common_phases,
        "seed": COMMON_SEED,
    }
    variants = {
        "nominal_common": {
            **reference,
            "radial_samples": 64,
            "azimuth_samples": 256,
        },
        "high_common": dict(reference),
        "high_independent": {
            **reference,
            "azimuth_phase_fractions": independent_phases,
            "seed": INDEPENDENT_SEED,
        },
        "spatial_double": {**reference, "spatial_oversample": 4},
        "radial_double": {**reference, "radial_samples": 256},
        "azimuth_double": {**reference, "azimuth_samples": 1024},
        "spectral_double": {**reference, "spectral_oversample": 4},
    }
    rendered = {}
    visibilities = {}
    reference_cube = None
    reference_grid = None
    for label, render_settings in variants.items():
        output = target_root / f"{label}.npz"
        spatial_oversample = int(render_settings.get("spatial_oversample", 1))
        render_grid = ImageGrid(
            nx=int(grid.nx) * spatial_oversample,
            ny=int(grid.ny) * spatial_oversample,
            cell_arcsec=float(grid.cell_arcsec) / spatial_oversample,
        )
        worker_config = _render_config(
            output,
            render_grid,
            data,
            params,
            sb_radius,
            sb_profile,
            render_settings,
        )
        run = _run_worker(
            kinms_python, worker_config, target_root / f"{label}.config.json"
        )
        start = perf_counter()
        vis, cube, metadata = sample_intrinsic_kinms(
            output, data=data, grid=render_grid, eps=1.0e-10
        )
        operator_elapsed = perf_counter() - start
        if metadata["kinms_version"] != kinms_identity["version"]:
            raise RuntimeError("worker KinMS version differs from locked environment")
        if metadata["kinms_module_sha256"] != kinms_identity["module_sha256"]:
            raise RuntimeError("worker KinMS module differs from locked environment")
        visibilities[label] = np.asarray(vis)
        if label == "high_common":
            reference_cube = np.asarray(cube)
            reference_grid = render_grid
        rendered[label] = {
            "worker": run,
            "operator_elapsed_s": operator_elapsed,
            "artifact": f"{target}/{output.name}",
            "artifact_sha256": sha256_file(output),
            "metadata": metadata,
            "render_grid": {
                "ny": render_grid.ny,
                "nx": render_grid.nx,
                "cell_arcsec": render_grid.cell_arcsec,
            },
            "chi2_visibility": chi2(data.vis, vis, data.weights, data.s),
            "flux_relative_error": abs(
                float(metadata["flux_ledger_jy_kms"]["quadrature_input"])
                - params["flux"]
            )
            / params["flux"],
        }
        spectral_flux = cube.sum(axis=(0, 1))
        centroid = float(np.sum(data.vel_native * spectral_flux) / np.sum(spectral_flux))
        rendered[label]["centroid_kms"] = centroid
        rendered[label]["centroid_error_native_channel"] = abs(
            centroid - params["vsys_kms"]
        ) / float(np.median(np.abs(np.diff(data.vel_native))))

    nominal = visibilities["nominal_common"]
    high = visibilities["high_common"]
    independent = visibilities["high_independent"]
    nominal_high_rms = _thermal_component_rms(nominal - high, data)
    independent_high_rms = _thermal_component_rms(independent - high, data)
    nominal_high_chi2 = abs(
        rendered["nominal_common"]["chi2_visibility"]
        - rendered["high_common"]["chi2_visibility"]
    )
    independent_high_chi2 = abs(
        rendered["high_independent"]["chi2_visibility"]
        - rendered["high_common"]["chi2_visibility"]
    )
    convergence_labels = (
        "spatial_double",
        "radial_double",
        "azimuth_double",
        "spectral_double",
    )
    axis_convergence = {
        label: {
            "absolute_chi2_change": abs(
                rendered[label]["chi2_visibility"]
                - rendered["high_common"]["chi2_visibility"]
            ),
            "relative_complex_visibility_l2": float(
                np.linalg.norm(visibilities[label] - high)
                / max(float(np.linalg.norm(high)), np.finfo(float).tiny)
            ),
            "noise_normalized_component_rms": _thermal_component_rms(
                visibilities[label] - high, data
            ),
        }
        for label in convergence_labels
    }
    geometry = _cube_geometry(
        reference_cube, reference_grid, data.vel_native, params["vsys_kms"]
    )
    geometry["receding_pa_error_deg"] = _angle_separation_deg(
        geometry["receding_pa_deg_east_of_north"], params["pa_deg"]
    )
    geometry["centroid_error_arcsec"] = float(
        np.hypot(
            geometry["flux_weighted_centroid_east_arcsec"] - params["dx_arcsec"],
            geometry["flux_weighted_centroid_north_arcsec"] - params["dy_arcsec"],
        )
    )

    gates = {
        "nominal_rendering_noise_rms": nominal_high_rms <= 0.1,
        "independent_high_rendering_noise_rms": independent_high_rms <= 0.1,
        "high_cloud_chi2_convergence": nominal_high_chi2 <= 0.1,
        "independent_high_chi2_convergence": independent_high_chi2 <= 0.1,
        "all_target_sampling_axes_chi2_convergence": max(
            value["absolute_chi2_change"] for value in axis_convergence.values()
        )
        <= 0.1,
        "all_target_sampling_axes_relative_l2": max(
            value["relative_complex_visibility_l2"]
            for value in axis_convergence.values()
        )
        <= 1.0e-4,
        "flux_error": max(v["flux_relative_error"] for v in rendered.values()) <= 0.001,
        "signed_receding_pa_contract": geometry["receding_pa_error_deg"] <= 3.0,
        "asymmetric_centroid_contract": geometry["centroid_error_arcsec"]
        <= float(grid.cell_arcsec),
    }
    return {
        "target_id": target,
        "target_config": str(config_path),
        "target_config_sha256": sha256_file(config_path),
        "load_metadata": load_metadata,
        "grid": {
            "ny": grid.ny,
            "nx": grid.nx,
            "cell_arcsec": grid.cell_arcsec,
            "native_channels": int(data.vel_native.size),
            "fit_rows": int(data.vis.shape[0]),
            "fit_channels": int(data.vis.shape[1]),
        },
        "parameters_source": "S0 selected rotating MAP; no parameter changes",
        "parameters": params,
        "rendered": rendered,
        "nominal_vs_high_noise_normalized_component_rms": nominal_high_rms,
        "independent_high_repeat_noise_normalized_component_rms": independent_high_rms,
        "nominal_vs_high_absolute_chi2_change": nominal_high_chi2,
        "independent_high_absolute_chi2_change": independent_high_chi2,
        "target_path_sampling_convergence": axis_convergence,
        "signed_geometry_contract": geometry,
        "thresholds": {
            "rendering_noise_rms_thermal_sd_max": 0.1,
            "high_cloud_absolute_chi2_change_max": 0.1,
            "independent_high_absolute_chi2_change_max": 0.1,
            "per_axis_doubled_sampling_absolute_chi2_change_max": 0.1,
            "per_axis_relative_complex_visibility_l2_max": 1.0e-4,
            "flux_relative_error_max": 0.001,
            "target_centroid_status": "diagnostic; finite target support can shift the flux centroid",
            "receding_pa_error_deg_max": 3.0,
            "centroid_error_arcsec_max": float(grid.cell_arcsec),
        },
        "gates": gates,
        "pass": all(gates.values()),
    }


def _write_manifest(root: Path, metrics_path: Path, code_commit: str) -> None:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            files[str(path.relative_to(root))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    manifest = {
        "schema_version": "kinuv-crossdomain-s1-manifest-v1",
        "code_commit": code_commit,
        "principal_metrics": str(metrics_path.relative_to(root)),
        "files": files,
    }
    (root / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def _snapshot_sources(root: Path) -> None:
    source_root = root / "source"
    for relative in SOURCE_SNAPSHOT_FILES:
        source = REPO / relative
        destination = source_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _seal_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o550 if path.is_dir() else 0o440)
    root.chmod(0o550)


def _peak_worker_rss(record: dict) -> int:
    target_peak = max(
        int(render["worker"]["worker_peak_rss_kib"])
        for target in record["targets"].values()
        for render in target["rendered"].values()
    )
    return max(
        target_peak,
        int(record["worker_coordinate_closure"]["run"]["worker_peak_rss_kib"]),
    )


def main() -> int:
    pipeline_start = perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--kinms-python",
        type=Path,
        default=Path("/scratch/kinuv-thbrown/s1-kinms/bin/python"),
    )
    args = parser.parse_args()
    publish_root = args.output.resolve()
    publish_root.parent.mkdir(parents=True, exist_ok=True)
    if publish_root.exists():
        raise RuntimeError(f"refusing to overwrite S1 dossier: {publish_root}")
    root = publish_root.parent / f".{publish_root.name}.attempt-{os.getpid()}"
    if root.exists():
        raise RuntimeError(f"attempt directory already exists: {root}")
    git = _require_clean_commit()
    kinms_identity = _validate_kinms_environment(args.kinms_python)
    s0, s0_validation = _validate_s0_inputs()
    root.mkdir()
    record = {
        "schema_version": SCHEMA,
        "created_utc": _utc(),
        "scope": "S1 deterministic closure; no fit, bootstrap, posterior, or sampler",
        "code": git,
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "environment": _environment(args.kinms_python, kinms_identity),
        "optimizer_budget": {
            "applicable": False,
            "reason": "S1 is deterministic renderer/operator closure; no fit or sampler ran",
        },
        "inputs": {
            "s0": s0_validation,
            "worker": {"path": str(WORKER), "sha256": sha256_file(WORKER)},
            "kinms_lock": {
                "path": str(REPO / "external" / "requirements-kinms-s1.txt"),
                "sha256": sha256_file(REPO / "external" / "requirements-kinms-s1.txt"),
            },
        },
        "worker_coordinate_closure": {},
        "targets": {},
    }
    try:
        record["worker_coordinate_closure"] = worker_coordinate_closure(
            root, args.kinms_python, kinms_identity
        )
        for target in TARGETS:
            record["targets"][target] = target_closure(
                target, root, args.kinms_python, s0, kinms_identity
            )
        record["pass"] = bool(
            record["worker_coordinate_closure"]["pass"]
            and all(value["pass"] for value in record["targets"].values())
        )
        _snapshot_sources(root)
        record["runtime"] = {
            "full_pipeline_wall_s": perf_counter() - pipeline_start,
            "parent_peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "maximum_worker_peak_rss_kib": _peak_worker_rss(record),
        }
        record["runtime"]["full_pipeline_peak_rss_kib"] = max(
            record["runtime"]["parent_peak_rss_kib"],
            record["runtime"]["maximum_worker_peak_rss_kib"],
        )
        metrics_path = root / "metrics.json"
        metrics_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        _write_manifest(root, metrics_path, git["commit"])
        _seal_tree(root)
        root.rename(publish_root)
    except BaseException:
        if root.exists():
            shutil.rmtree(root)
        raise
    print(json.dumps({
        "output": str(publish_root),
        "pass": record["pass"],
        "worker_coordinate": {
            "pass": record["worker_coordinate_closure"]["pass"],
            "gates": record["worker_coordinate_closure"]["gates"],
            "receding_pa_error_deg": record["worker_coordinate_closure"][
                "geometry"
            ]["receding_pa_error_deg"],
            "spectral_centroid_error_native_channel": record[
                "worker_coordinate_closure"
            ]["spectral_centroid_error_native_channel"],
        },
        "targets": {
            key: {
                "pass": value["pass"],
                "gates": value["gates"],
                "nominal_vs_high_rms": value["nominal_vs_high_noise_normalized_component_rms"],
                "independent_high_rms": value["independent_high_repeat_noise_normalized_component_rms"],
                "chi2_change": value["nominal_vs_high_absolute_chi2_change"],
                "independent_chi2_change": value[
                    "independent_high_absolute_chi2_change"
                ],
                "maximum_axis_chi2_change": max(
                    axis["absolute_chi2_change"]
                    for axis in value["target_path_sampling_convergence"].values()
                ),
                "receding_pa_error_deg": value["signed_geometry_contract"][
                    "receding_pa_error_deg"
                ],
            }
            for key, value in record["targets"].items()
        },
    }, indent=2))
    return 0 if record["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
