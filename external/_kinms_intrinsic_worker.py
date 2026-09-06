#!/usr/bin/env python3
"""Render the beam-free KinMS continuum comparator.

The adapter retains KinMS's thin circular-disk convention while replacing its
nearest-cell population and stochastic dispersion draws with deterministic
cylindrical quadrature, cubic-B-spline deposition, and analytic channel
integrals. It never renormalizes emission after spatial or spectral clipping.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import kinms
import numpy as np
from scipy.special import ndtr
from scipy.sparse import csr_matrix


SCHEMA_VERSION = "kinms-continuum-cube-v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported continuum contract: {config.get('schema_version')!r}")
    return config


def _velocity_contract(values) -> tuple[np.ndarray, float, bool]:
    velocity = np.asarray(values, dtype=np.float64)
    if velocity.ndim != 1 or velocity.size < 3 or not np.all(np.isfinite(velocity)):
        raise ValueError("velocity_centers_kms must be a finite 1-D axis of length >=3")
    differences = np.diff(velocity)
    descending = bool(np.all(differences < 0.0))
    if not (descending or np.all(differences > 0.0)):
        raise ValueError("velocity_centers_kms must be strictly monotonic")
    dv = float(np.median(np.abs(differences)))
    relative_error = float(np.max(np.abs(np.abs(differences) - dv)) / dv)
    if relative_error > 1.0e-6:
        raise ValueError(f"continuum adapter requires uniform velocity cells; error={relative_error}")
    return velocity, dv, descending


def _b3(distance):
    """Cardinal cubic B-spline in pixel coordinates."""
    absolute = np.abs(np.asarray(distance, dtype=np.float64))
    inner = (4.0 - 6.0 * absolute**2 + 3.0 * absolute**3) / 6.0
    outer = (2.0 - absolute) ** 3 / 6.0
    return np.where(absolute < 1.0, inner, np.where(absolute < 2.0, outer, 0.0))


def _stable_normal_interval(z_lo, z_hi):
    """Return Phi(z_hi)-Phi(z_lo) without positive-tail cancellation."""
    direct = ndtr(z_hi) - ndtr(z_lo)
    survival = ndtr(-z_lo) - ndtr(-z_hi)
    return np.maximum(np.where(z_lo > 0.0, survival, direct), 0.0)


def _channel_probabilities(mu, sigma, velocity, dv, subdivisions):
    """Analytic Gaussian probability in every native top-hat velocity cell."""
    mu = np.asarray(mu, dtype=np.float64)[:, None]
    native_lo = np.asarray(velocity, dtype=np.float64) - 0.5 * dv
    sub_dv = dv / subdivisions
    probability = np.zeros((mu.shape[0], velocity.size), dtype=np.float64)
    for sub_index in range(subdivisions):
        lo = native_lo + sub_index * sub_dv
        hi = lo + sub_dv
        probability += _stable_normal_interval(
            (lo[None, :] - mu) / sigma,
            (hi[None, :] - mu) / sigma,
        )
    return probability


def _quadrature_plane(sb_rad, sb_profile, radial_samples, azimuth_samples, phase):
    r_max = float(sb_rad[-1])
    dr = r_max / radial_samples
    radius = (np.arange(radial_samples, dtype=np.float64) + 0.5) * dr
    phi = (np.arange(azimuth_samples, dtype=np.float64) + phase) * (
        2.0 * np.pi / azimuth_samples
    )
    rr, pp = np.meshgrid(radius, phi, indexing="ij")
    brightness = np.interp(rr, sb_rad, sb_profile, left=sb_profile[0], right=0.0)
    raw_weight = brightness * rr * dr * (2.0 * np.pi / azimuth_samples)
    return rr.ravel(), pp.ravel(), raw_weight.ravel()


def _project_clouds(radius, phi, params, velocity_radius, velocity_profile):
    """Apply the pinned KinMS projection and LOS sign convention."""
    x_face = radius * np.cos(phi)
    y_face = radius * np.sin(phi)
    inclination = np.radians(float(params["inclination_deg"]))
    y_inclined = y_face * np.cos(inclination)
    kinms_pa_deg = (360.0 - float(params["pa_deg"])) % 360.0
    kinms_pa = np.radians(kinms_pa_deg)
    east = (
        np.sin(kinms_pa) * x_face
        + np.cos(kinms_pa) * y_inclined
        + float(params["dx_arcsec"])
    )
    north = (
        -np.cos(kinms_pa) * x_face
        + np.sin(kinms_pa) * y_inclined
        + float(params["dy_arcsec"])
    )
    circular_speed = np.interp(radius, velocity_radius, velocity_profile)
    mean_velocity = (
        float(params["vsys_kms"])
        - circular_speed * np.cos(phi) * np.sin(inclination)
    )
    return east, north, mean_velocity, kinms_pa_deg


def _deposit_phase(
    cube,
    east,
    north,
    mean_velocity,
    integrated_flux,
    *,
    cell,
    velocity,
    dv,
    sigma,
    spectral_subdivisions,
    block_size=32768,
):
    """Deposit one quadrature phase and return its physical flux ledger."""
    ny, nx, _ = cube.shape
    cube_rows = cube.reshape(ny * nx, velocity.size)
    spatial_retained = spectral_retained = joint_retained = 0.0
    for start in range(0, east.size, block_size):
        stop = min(start + block_size, east.size)
        flux = integrated_flux[start:stop]
        x_coordinate = east[start:stop] / cell + nx // 2
        y_coordinate = north[start:stop] / cell + ny // 2
        x_base = np.floor(x_coordinate).astype(np.int64)
        y_base = np.floor(y_coordinate).astype(np.int64)
        offsets = np.arange(-1, 3, dtype=np.int64)
        x_index = x_base[:, None] + offsets[None, :]
        y_index = y_base[:, None] + offsets[None, :]
        x_weight = _b3(x_coordinate[:, None] - x_index)
        y_weight = _b3(y_coordinate[:, None] - y_index)
        x_valid = (x_index >= 0) & (x_index < nx)
        y_valid = (y_index >= 0) & (y_index < ny)
        spatial_fraction = np.sum(x_weight * x_valid, axis=1) * np.sum(
            y_weight * y_valid, axis=1
        )
        probability = _channel_probabilities(
            mean_velocity[start:stop], sigma, velocity, dv, spectral_subdivisions
        )
        spectral_fraction = np.sum(probability, axis=1)
        channel_density = flux[:, None] * probability / dv
        spatial_retained += float(np.sum(flux * spatial_fraction))
        spectral_retained += float(np.sum(flux * spectral_fraction))
        joint_retained += float(np.sum(flux * spatial_fraction * spectral_fraction))
        rows = []
        columns = []
        values = []
        local_column = np.arange(stop - start, dtype=np.int64)
        for iy_offset in range(4):
            for ix_offset in range(4):
                valid = y_valid[:, iy_offset] & x_valid[:, ix_offset]
                if not np.any(valid):
                    continue
                rows.append(
                    y_index[valid, iy_offset] * nx + x_index[valid, ix_offset]
                )
                columns.append(local_column[valid])
                values.append(
                    y_weight[valid, iy_offset] * x_weight[valid, ix_offset]
                )
        deposition = csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
            shape=(ny * nx, stop - start),
        )
        cube_rows += deposition @ channel_density
    return spatial_retained, spectral_retained, joint_retained


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: _kinms_intrinsic_worker.py CONFIG.json")
    config_path = Path(sys.argv[1]).resolve()
    config = _load_config(config_path)
    output_path = Path(config["output_npz"]).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    velocity, native_dv, reverse_spectral = _velocity_contract(
        config["velocity_centers_kms"]
    )

    grid = config["grid"]
    params = config["parameters"]
    brightness = config["surface_brightness"]
    nx, ny = int(grid["nx"]), int(grid["ny"])
    cell = float(grid["cell_arcsec"])
    if nx < 2 or ny < 2 or cell <= 0.0:
        raise ValueError("invalid spatial grid")
    sb_rad = np.asarray(brightness["radius_arcsec"], dtype=np.float64)
    sb_profile = np.asarray(brightness["profile"], dtype=np.float64)
    if sb_rad.ndim != 1 or sb_profile.shape != sb_rad.shape or sb_rad.size < 2:
        raise ValueError("surface-brightness radius/profile arrays must match")
    velocity_radius = np.asarray(params["velocity_radius_arcsec"], dtype=np.float64)
    velocity_profile = np.asarray(params["circular_speed_kms"], dtype=np.float64)
    if velocity_radius.ndim != 1 or velocity_profile.shape != velocity_radius.shape:
        raise ValueError("velocity radius/profile arrays must match")
    if velocity_radius[0] != 0.0 or velocity_profile[0] != 0.0:
        raise ValueError("velocity profile must include the origin (0, 0)")

    radial_samples = int(config["radial_samples"])
    azimuth_samples = int(config["azimuth_samples"])
    if radial_samples < 16 or azimuth_samples < 32 or azimuth_samples % 2:
        raise ValueError("quadrature requires >=16 radial and even >=32 azimuth samples")
    phases = [float(value) % 1.0 for value in config.get("azimuth_phase_fractions", [0.0])]
    if not phases:
        raise ValueError("azimuth_phase_fractions cannot be empty")
    spectral_subdivisions = int(config.get("spectral_oversample", 1))
    if spectral_subdivisions < 1:
        raise ValueError("spectral_oversample must be >=1")
    sigma = float(params["gas_sigma_kms"])
    requested_flux = float(params["flux_jy_kms"])
    if sigma <= 0.0 or requested_flux <= 0.0:
        raise ValueError("gas dispersion and integrated flux must be positive")

    cube = np.zeros((ny, nx, velocity.size), dtype=np.float64)
    spatial_retained = spectral_retained = joint_retained = quadrature_input = 0.0
    kinms_pa_deg = None
    for phase in phases:
        radius, phi, raw_weight = _quadrature_plane(
            sb_rad, sb_profile, radial_samples, azimuth_samples, phase
        )
        raw_sum = float(np.sum(raw_weight))
        if not np.isfinite(raw_sum) or raw_sum <= 0.0:
            raise ValueError("surface-brightness quadrature has no positive support")
        integrated_flux = requested_flux * raw_weight / raw_sum / len(phases)
        east, north, mean_velocity, kinms_pa_deg = _project_clouds(
            radius, phi, params, velocity_radius, velocity_profile
        )
        phase_ledger = _deposit_phase(
            cube,
            east,
            north,
            mean_velocity,
            integrated_flux,
            cell=cell,
            velocity=velocity,
            dv=native_dv,
            sigma=sigma,
            spectral_subdivisions=spectral_subdivisions,
        )
        spatial_retained += phase_ledger[0]
        spectral_retained += phase_ledger[1]
        joint_retained += phase_ledger[2]
        quadrature_input += float(np.sum(integrated_flux))

    rendered_flux = float(np.sum(cube, dtype=np.float64) * native_dv)
    if not np.isclose(rendered_flux, joint_retained, rtol=2.0e-11, atol=2.0e-11):
        raise RuntimeError("deposited cube disagrees with the pre-normalization flux ledger")
    spectral_density = np.sum(cube, axis=(0, 1))
    centroid = float(np.sum(velocity * spectral_density) / np.sum(spectral_density))
    kinms_path = Path(kinms.__file__).resolve()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "renderer": "KinMS continuum adapter",
        "kinms_version": importlib.metadata.version("kinms"),
        "kinms_module": str(kinms_path),
        "kinms_module_sha256": _sha256(kinms_path),
        "worker_sha256": _sha256(Path(__file__).resolve()),
        "config_sha256": _sha256(config_path),
        "clean_out": True,
        "restoring_beam_applied": False,
        "primary_beam_applied": False,
        "spectral_response_applied": False,
        "post_crop_renormalization": False,
        "axis_order": "north,east,velocity",
        "cube_units": "Jy",
        "channel_value_semantics": "channel-average flux density per sky pixel",
        "output_grid": {"ny": ny, "nx": nx, "cell_arcsec": cell},
        "velocity_convention": "radio_kms_from_exact_kinuv_frequency_axis",
        "velocity_descending": reverse_spectral,
        "native_channel_width_kms": native_dv,
        "spectral_subdivisions": spectral_subdivisions,
        "spatial_kernel": "separable cardinal cubic B-spline B3",
        "render_mode": "deterministic_continuum_quadrature",
        "quadrature": {
            "radial_samples": radial_samples,
            "azimuth_samples": azimuth_samples,
            "phase_ensemble_size": len(phases),
            "azimuth_phase_fractions": phases,
        },
        "n_clouds": radial_samples * azimuth_samples * len(phases),
        "seed": int(config["seed"]),
        "coordinate_conversion": {
            "kinuv_pa_deg_east_of_north_receding": float(params["pa_deg"]),
            "kinms_posang_deg": kinms_pa_deg,
            "formula": "kinms_posang=(360-kinuv_pa)%360",
            "post_render_spatial_reflection": False,
        },
        "spectral_registration": {
            "method": "analytic Gaussian probability across native velocity edges",
            "post_render_interpolation": False,
            "centroid_kms": centroid,
            "centroid_error_native_channels": abs(centroid - float(params["vsys_kms"])) / native_dv,
        },
        "flux_ledger_jy_kms": {
            "requested_full_support": requested_flux,
            "quadrature_input": quadrature_input,
            "spatially_retained": spatial_retained,
            "spectrally_retained": spectral_retained,
            "jointly_retained": joint_retained,
            "cube_integral": rendered_flux,
            "spatially_excluded": requested_flux - spatial_retained,
            "spectrally_excluded": requested_flux - spectral_retained,
            "jointly_excluded": requested_flux - joint_retained,
        },
    }
    metadata_text = json.dumps(metadata, sort_keys=True)
    np.savez_compressed(
        output_path,
        cube_yxv=np.ascontiguousarray(cube),
        velocity_centers_kms=velocity,
        metadata_json=np.asarray(metadata_text),
    )
    metadata["output_npz"] = str(config.get("artifact_id", output_path))
    metadata["output_npz_sha256"] = _sha256(output_path)
    sidecar = output_path.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
