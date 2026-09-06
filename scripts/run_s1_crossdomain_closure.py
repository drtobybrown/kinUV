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
import resource
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from kinuv.constants import ARCSEC_TO_RAD, F_REST_CO21_HZ
from kinuv.diagnostics.comparator import sample_intrinsic_kinms, sha256_file
from kinuv.forward.model import intrinsic_sky_cube
from kinuv.forward.operators import (
    attenuate_intrinsic_cube,
    sample_intrinsic_cube_native,
)
from kinuv.forward.sb import galaxy_r_phi, load_sb_template
from kinuv.infer.map import image_grid_for_vis, predict_binned
from kinuv.io.vis import load_target_vis
from kinuv.likelihood.chi2 import chi2
from kinuv.profiles.rotation import arctan_vc
from kinuv.transforms.dft import dft_numpy
from kinuv.transforms.grid import ImageGrid

REPO = Path(__file__).resolve().parents[1]
WORKER = REPO / "external" / "_kinms_intrinsic_worker.py"
S0_METRICS = Path(
    "/arc/projects/KILOGAS/analysis/toby_sandbox/results/validation/"
    "crossdomain-recovery-s0-20260906/metrics.json"
)
DEFAULT_OUTPUT = Path(
    "/arc/projects/KILOGAS/analysis/toby_sandbox/results/validation/"
    "crossdomain-recovery-s1-20260906"
)
SCHEMA = "kinuv-crossdomain-s1-v1"
TARGETS = ("KGAS066", "KGAS007")
COMMON_SEED = 66
INDEPENDENT_SEED = 1066


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


def _environment(kinms_python: Path) -> dict:
    versions = {}
    for name in ("kinuv", "numpy", "scipy", "jax", "jax-finufft", "astropy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    kinms_freeze = subprocess.check_output(
        [str(kinms_python), "-m", "pip", "freeze"], text=True
    ).splitlines()
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "jax_enable_x64": os.environ.get("JAX_ENABLE_X64"),
        "packages": versions,
        "kinms_python": str(kinms_python),
        "kinms_environment_freeze": kinms_freeze,
    }


def _analytic_cube(grid: ImageGrid, n_chan: int = 7) -> np.ndarray:
    x = grid.l_rad / ARCSEC_TO_RAD
    y = grid.m_rad / ARCSEC_TO_RAD
    east, north = np.meshgrid(x, y, indexing="xy")
    spatial = np.exp(-0.5 * ((east / 0.55) ** 2 + (north / 0.42) ** 2))
    spatial /= spatial.sum()
    channel = np.arange(n_chan, dtype=np.float64)
    spectral = np.exp(-0.5 * ((channel - 3.17) / 1.1) ** 2)
    return spatial[:, :, None] * spectral[None, None, :]


def analytic_closure() -> dict:
    """Independent DFT and quadrature closure under the frozen S1 gates."""
    rng = np.random.default_rng(8661)
    grid = ImageGrid(32, 32, 0.15)
    freqs = F_REST_CO21_HZ + np.arange(-3, 4, dtype=np.float64) * 1.0e6
    cube = _analytic_cube(grid, freqs.size)
    u_m = rng.uniform(-120.0, 120.0, 19)
    v_m = rng.uniform(-120.0, 120.0, 19)
    attenuated = attenuate_intrinsic_cube(cube, grid, freqs)
    east, north = grid.pixel_lm_rad()
    reference = dft_numpy(
        east.ravel(),
        north.ravel(),
        attenuated.reshape(-1, freqs.size),
        u_m,
        v_m,
        freqs,
    )
    actual = sample_intrinsic_cube_native(
        cube, grid, u_m, v_m, freqs, eps=1.0e-12
    )
    residual = actual - reference
    relative_l2 = float(np.linalg.norm(residual) / np.linalg.norm(reference))
    noise_normalized_component_rms = float(
        np.sqrt(np.mean(np.concatenate([residual.real.ravel(), residual.imag.ravel()]) ** 2))
        / 0.01
    )

    zero = sample_intrinsic_cube_native(
        cube, grid, np.array([0.0]), np.array([0.0]), freqs, eps=1.0e-12
    )[0]
    expected_zero = attenuated.sum(axis=(0, 1))
    zero_flux_error = float(
        np.max(np.abs(zero - expected_zero) / np.maximum(np.abs(expected_zero), 1e-30))
    )

    # A well-contained Gaussian line tests that native midpoint sampling does
    # not displace its intensity centroid at a noninteger channel location.
    native_index = np.arange(-100, 101, dtype=np.float64)
    truth_centroid = 0.37
    line = np.exp(-0.5 * ((native_index - truth_centroid) / 10.0) ** 2)
    centroid = float(np.sum(native_index * line) / np.sum(line))
    centroid_error_native_channel = abs(centroid - truth_centroid)

    # Double spatial midpoint sampling at fixed FoV and compare both numerical
    # models to the high-resolution analytic datum with unit component variance.
    coarse = ImageGrid(48, 48, 0.1)
    fine = ImageGrid(96, 96, 0.05)
    one_freq = np.array([F_REST_CO21_HZ])
    coarse_cube = _analytic_cube(coarse, 1)
    fine_cube = _analytic_cube(fine, 1)
    uv_u = rng.uniform(-80.0, 80.0, 31)
    uv_v = rng.uniform(-80.0, 80.0, 31)
    vis_coarse = sample_intrinsic_cube_native(
        coarse_cube, coarse, uv_u, uv_v, one_freq, eps=1.0e-12
    )
    vis_fine = sample_intrinsic_cube_native(
        fine_cube, fine, uv_u, uv_v, one_freq, eps=1.0e-12
    )
    spatial_delta_chi2 = float(np.sum(np.abs(vis_coarse - vis_fine) ** 2))

    # Compare native midpoint integration with two subchannels per native cell.
    centres = np.arange(-40.0, 42.0, 2.0)
    sigma = 12.0
    mu = 0.73
    coarse_flux = np.exp(-0.5 * ((centres - mu) / sigma) ** 2) * 2.0
    sub = centres[:, None] + np.array([-0.5, 0.5])[None, :]
    fine_flux = np.exp(-0.5 * ((sub - mu) / sigma) ** 2).sum(axis=1)
    spectral_delta_chi2 = float(np.sum((coarse_flux - fine_flux) ** 2))
    doubled_sampling_delta_chi2 = max(spatial_delta_chi2, spectral_delta_chi2)

    gates = {
        "relative_visibility_l2": relative_l2 <= 1.0e-6,
        "noise_normalized_component_rms": noise_normalized_component_rms <= 0.001,
        "zero_baseline_flux_error": zero_flux_error <= 0.001,
        "centroid_error_native_channel": centroid_error_native_channel <= 0.02,
        "doubled_sampling_delta_chi2": doubled_sampling_delta_chi2 <= 0.1,
    }
    return {
        "relative_visibility_l2": relative_l2,
        "noise_normalized_component_rms": noise_normalized_component_rms,
        "zero_baseline_flux_error": zero_flux_error,
        "centroid_error_native_channel": centroid_error_native_channel,
        "spatial_doubling_delta_chi2": spatial_delta_chi2,
        "spectral_doubling_delta_chi2": spectral_delta_chi2,
        "doubled_sampling_delta_chi2": doubled_sampling_delta_chi2,
        "thresholds": {
            "relative_visibility_l2_max": 1.0e-6,
            "noise_normalized_component_rms_max": 0.001,
            "zero_baseline_flux_error_max": 0.001,
            "centroid_error_native_channel_max": 0.02,
            "doubled_sampling_delta_chi2_max": 0.1,
        },
        "gates": gates,
        "pass": all(gates.values()),
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
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = perf_counter()
    result = subprocess.run(
        [str(kinms_python), str(WORKER), str(path)],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    log_path.write_text(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"KinMS worker failed ({result.returncode}); see {log_path}")
    return {
        "command": [str(kinms_python), str(WORKER), str(path)],
        "elapsed_s": elapsed,
        "child_max_rss_kib": int(after.ru_maxrss),
        "child_user_cpu_s_delta": float(after.ru_utime - before.ru_utime),
        "child_system_cpu_s_delta": float(after.ru_stime - before.ru_stime),
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
    velocity_radius = np.linspace(0.01, max(15.0, float(sb_radius[-1])), 256)
    return {
        "schema_version": "kinms-intrinsic-cube-v1",
        "output_npz": str(output),
        "grid": {
            "nx": int(grid.nx),
            "ny": int(grid.ny),
            "cell_arcsec": float(grid.cell_arcsec),
        },
        "velocity_centers_kms": np.asarray(data.vel_native, dtype=float).tolist(),
        "render_mode": "deterministic_quadrature",
        "seed": int(render_settings["seed"]),
        "radial_samples": int(render_settings["radial_samples"]),
        "azimuth_samples": int(render_settings["azimuth_samples"]),
        "dispersion_order": int(render_settings["dispersion_order"]),
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


def target_closure(target: str, root: Path, kinms_python: Path, s0: dict) -> dict:
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
    common_phases = _van_der_corput(64)
    independent_phases = [float((value + np.sqrt(2.0) / 7.0) % 1.0) for value in common_phases]
    variants = {
        "nominal_common": {
            "radial_samples": 256,
            "azimuth_samples": 512,
            "dispersion_order": 9,
            "azimuth_phase_fractions": common_phases[:32],
            "seed": COMMON_SEED,
        },
        "high_common": {
            "radial_samples": 256,
            "azimuth_samples": 512,
            "dispersion_order": 9,
            "azimuth_phase_fractions": common_phases,
            "seed": COMMON_SEED,
        },
        "high_independent": {
            "radial_samples": 256,
            "azimuth_samples": 512,
            "dispersion_order": 9,
            "azimuth_phase_fractions": independent_phases,
            "seed": INDEPENDENT_SEED,
        },
    }
    rendered = {}
    visibilities = {}
    cubes = {}
    for label, render_settings in variants.items():
        output = target_root / f"{label}.npz"
        worker_config = _render_config(
            output, grid, data, params, sb_radius, sb_profile, render_settings
        )
        run = _run_worker(
            kinms_python, worker_config, target_root / f"{label}.config.json"
        )
        start = perf_counter()
        vis, cube, metadata = sample_intrinsic_kinms(
            output, data=data, grid=grid, eps=1.0e-10
        )
        operator_elapsed = perf_counter() - start
        visibilities[label] = np.asarray(vis)
        cubes[label] = cube
        rendered[label] = {
            "worker": run,
            "operator_elapsed_s": operator_elapsed,
            "artifact": str(output),
            "artifact_sha256": sha256_file(output),
            "metadata": metadata,
            "chi2_visibility": chi2(data.vis, vis, data.weights, data.s),
            "flux_relative_error": abs(
                float(metadata["integrated_flux_jy_kms_rendered"]) - params["flux"]
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

    # Refactoring the measurement operator must preserve the exact S0 kinUV
    # likelihood. This replay uses the unmodified target parameters/template.
    replay_start = perf_counter()
    kinuv_vis = predict_binned(data, params, template, grid, i_rad=inclination_rad)
    replay_elapsed = perf_counter() - replay_start
    replay_chi2 = chi2(data.vis, kinuv_vis, data.weights, data.s)
    expected_chi2 = float(selected["chi2_map"])
    replay_error = abs(replay_chi2 - expected_chi2)

    gates = {
        "nominal_rendering_noise_rms": nominal_high_rms <= 0.1,
        "independent_high_rendering_noise_rms": independent_high_rms <= 0.1,
        "high_cloud_chi2_convergence": nominal_high_chi2 <= 0.1,
        "flux_error": max(v["flux_relative_error"] for v in rendered.values()) <= 0.001,
        "centroid_error": max(
            v["centroid_error_native_channel"] for v in rendered.values()
        )
        <= 0.02,
        "baseline_replay": replay_error <= 0.1,
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
        "baseline_replay": {
            "expected_chi2": expected_chi2,
            "actual_chi2": replay_chi2,
            "absolute_error": replay_error,
            "elapsed_s": replay_elapsed,
        },
        "thresholds": {
            "rendering_noise_rms_thermal_sd_max": 0.1,
            "high_cloud_absolute_chi2_change_max": 0.1,
            "flux_relative_error_max": 0.001,
            "centroid_error_native_channel_max": 0.02,
            "baseline_replay_absolute_chi2_max": 0.1,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--kinms-python",
        type=Path,
        default=Path("/scratch/kinuv-thbrown/s1-kinms/bin/python"),
    )
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    s0 = json.loads(S0_METRICS.read_text())
    git = _git_state()
    record = {
        "schema_version": SCHEMA,
        "created_utc": _utc(),
        "scope": "S1 deterministic closure; no fit, bootstrap, posterior, or sampler",
        "code": git,
        "environment": _environment(args.kinms_python),
        "inputs": {
            "s0_metrics": {
                "path": str(S0_METRICS),
                "sha256": sha256_file(S0_METRICS),
            },
            "worker": {"path": str(WORKER), "sha256": sha256_file(WORKER)},
            "kinms_lock": {
                "path": str(REPO / "external" / "requirements-kinms-s1.txt"),
                "sha256": sha256_file(REPO / "external" / "requirements-kinms-s1.txt"),
            },
        },
        "analytic_closure": analytic_closure(),
        "targets": {},
    }
    for target in TARGETS:
        record["targets"][target] = target_closure(
            target, root, args.kinms_python, s0
        )
    record["pass"] = bool(
        record["analytic_closure"]["pass"]
        and all(value["pass"] for value in record["targets"].values())
    )
    metrics_path = root / "metrics.json"
    metrics_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    _write_manifest(root, metrics_path, git["commit"])
    print(json.dumps({
        "output": str(root),
        "pass": record["pass"],
        "analytic": record["analytic_closure"],
        "targets": {
            key: {
                "pass": value["pass"],
                "gates": value["gates"],
                "nominal_vs_high_rms": value["nominal_vs_high_noise_normalized_component_rms"],
                "independent_high_rms": value["independent_high_repeat_noise_normalized_component_rms"],
                "chi2_change": value["nominal_vs_high_absolute_chi2_change"],
                "replay_error": value["baseline_replay"]["absolute_error"],
            }
            for key, value in record["targets"].items()
        },
    }, indent=2))
    return 0 if record["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
