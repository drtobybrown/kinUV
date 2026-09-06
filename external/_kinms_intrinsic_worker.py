#!/usr/bin/env python3
"""Render an intrinsic KinMS cube in an isolated KinMS environment.

The worker intentionally imports no ``kinuv`` module. It writes a versioned
NPZ contract consumed by kinUV, and ``cleanOut=True`` guarantees that neither a
restoring beam nor primary beam has been applied. Spectral response is also
disabled; kinUV owns that operator after import.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
from kinms import KinMS


SCHEMA_VERSION = "kinms-intrinsic-cube-v1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported intrinsic contract: {config.get('schema_version')!r}")
    return config


def _velocity_contract(values) -> tuple[np.ndarray, float, bool]:
    velocity = np.asarray(values, dtype=np.float64)
    if velocity.ndim != 1 or velocity.size < 3 or not np.all(np.isfinite(velocity)):
        raise ValueError("velocity_centers_kms must be a finite 1-D axis of length >=3")
    differences = np.diff(velocity)
    if not (np.all(differences > 0.0) or np.all(differences < 0.0)):
        raise ValueError("velocity_centers_kms must be strictly monotonic")
    dv = float(np.median(np.abs(differences)))
    relative_error = float(np.max(np.abs(np.abs(differences) - dv)) / dv)
    if relative_error > 1.0e-6:
        raise ValueError(f"KinMS requires a uniform velocity grid; relative error={relative_error}")
    return velocity, dv, bool(differences[0] < 0.0)


def _quadrature_clouds(sb_rad, sb_profile, config):
    """Deterministic cylindrical/Gauss-Hermite quadrature for KinMS clouds."""
    radial_samples = int(config["radial_samples"])
    azimuth_samples = int(config["azimuth_samples"])
    dispersion_order = int(config["dispersion_order"])
    phase = float(config.get("azimuth_phase_fraction", 0.0))
    if radial_samples < 16 or azimuth_samples < 32 or azimuth_samples % 2:
        raise ValueError("quadrature requires >=16 radial and even >=32 azimuth samples")
    if dispersion_order < 3 or dispersion_order % 2 == 0:
        raise ValueError("dispersion_order must be odd and >=3")
    r_max = float(sb_rad[-1])
    dr = r_max / radial_samples
    radius = (np.arange(radial_samples, dtype=np.float64) + 0.5) * dr
    phi = (
        np.arange(azimuth_samples, dtype=np.float64) + phase
    ) * (2.0 * np.pi / azimuth_samples)
    rr, pp = np.meshgrid(radius, phi, indexing="ij")
    base_sb = np.interp(rr, sb_rad, sb_profile, left=sb_profile[0], right=0.0)
    base_weight = base_sb * rr * dr * (2.0 * np.pi / azimuth_samples)
    x = (rr * np.cos(pp)).ravel()
    y = (rr * np.sin(pp)).ravel()
    spatial_weight = base_weight.ravel()
    gh_x, gh_w = np.polynomial.hermite.hermgauss(dispersion_order)
    standard_normal = np.sqrt(2.0) * gh_x
    normal_weight = gh_w / np.sqrt(np.pi)
    n_spatial = x.size
    inclouds = np.zeros((n_spatial * dispersion_order, 3), dtype=np.float64)
    inclouds[:, 0] = np.repeat(x, dispersion_order)
    inclouds[:, 1] = np.repeat(y, dispersion_order)
    flux_clouds = (
        np.repeat(spatial_weight, dispersion_order)
        * np.tile(normal_weight, n_spatial)
    )
    dispersion_draws = np.tile(standard_normal, n_spatial)
    return inclouds, flux_clouds, dispersion_draws, {
        "radial_samples": radial_samples,
        "azimuth_samples": azimuth_samples,
        "dispersion_order": dispersion_order,
        "azimuth_phase_fraction": phase,
    }


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: _kinms_intrinsic_worker.py CONFIG.json")
    config_path = Path(sys.argv[1]).resolve()
    config = _load_config(config_path)
    output_path = Path(config["output_npz"]).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    velocity, dv, reverse_spectral = _velocity_contract(config["velocity_centers_kms"])
    grid = config["grid"]
    params = config["parameters"]
    sb = config["surface_brightness"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    cell = float(grid["cell_arcsec"])
    if nx < 2 or ny < 2 or cell <= 0.0:
        raise ValueError("invalid spatial grid")
    if nx != ny:
        raise ValueError("KinMS intrinsic v1 supports square grids only")

    sb_rad = np.asarray(sb["radius_arcsec"], dtype=np.float64)
    sb_profile = np.asarray(sb["profile"], dtype=np.float64)
    if sb_rad.ndim != 1 or sb_profile.shape != sb_rad.shape or sb_rad.size < 2:
        raise ValueError("surface-brightness radius/profile arrays must match")
    radial = np.asarray(params["velocity_radius_arcsec"], dtype=np.float64)
    velocity_profile = np.asarray(params["circular_speed_kms"], dtype=np.float64)
    if radial.ndim != 1 or velocity_profile.shape != radial.shape or radial.size < 2:
        raise ValueError("velocity radius/profile arrays must match")

    render_mode = str(config.get("render_mode", "stochastic"))
    n_clouds = int(config.get("n_clouds", 0))
    seed = int(config["seed"])
    quadrature_metadata = None
    inclouds = None
    flux_clouds = None
    dispersion_draws = None
    if render_mode == "deterministic_quadrature":
        phase_offsets = config.get(
            "azimuth_phase_fractions", [config.get("azimuth_phase_fraction", 0.0)]
        )
        if not phase_offsets:
            raise ValueError("azimuth_phase_fractions cannot be empty")
        phase_offsets = [float(value) % 1.0 for value in phase_offsets]
        first_config = dict(config)
        first_config["azimuth_phase_fraction"] = phase_offsets[0]
        inclouds, flux_clouds, dispersion_draws, quadrature_metadata = _quadrature_clouds(
            sb_rad, sb_profile, first_config
        )
        clouds_per_phase = int(inclouds.shape[0])
        n_clouds = clouds_per_phase * len(phase_offsets)
        quadrature_metadata["phase_ensemble_size"] = len(phase_offsets)
        quadrature_metadata["azimuth_phase_fractions"] = phase_offsets
    elif render_mode == "stochastic":
        if n_clouds < 1000:
            raise ValueError("n_clouds must be >=1000 for a scientific render")
    else:
        raise ValueError(f"unsupported render_mode {render_mode!r}")
    centre_index = velocity.size // 2
    centre_velocity = float(velocity[centre_index])
    v_offset = float(params["vsys_kms"]) - centre_velocity
    phase_offsets = (
        [None]
        if render_mode == "stochastic"
        else quadrature_metadata["azimuth_phase_fractions"]
    )
    cube_xyv = None
    for phase_index, phase_offset in enumerate(phase_offsets):
        if phase_index > 0:
            phase_config = dict(config)
            phase_config["azimuth_phase_fraction"] = phase_offset
            inclouds, flux_clouds, dispersion_draws, _ = _quadrature_clouds(
                sb_rad, sb_profile, phase_config
            )
        model = KinMS(
            xs=nx * cell,
            ys=ny * cell,
            vs=velocity.size * dv,
            cellSize=cell,
            dv=dv,
            beamSize=[cell, cell, 0.0],
            nSamps=int(inclouds.shape[0]) if inclouds is not None else n_clouds,
            seed=seed,
            fixSeed=True,
            cleanOut=True,
            spectral_resolution=0.0,
            verbose=False,
        )
        if dispersion_draws is not None:
            # KinMS exposes inClouds/flux_clouds publicly but not a deterministic
            # dispersion quadrature argument. Replacing its pre-generated standard
            # normal draws preserves the documented gasSigma calculation while
            # removing Monte Carlo noise from a numerical-closure comparator.
            model.randompick_vdisp = dispersion_draws
        phase_cube = model.model_cube(
            inc=float(params["inclination_deg"]),
            posAng=float(params["pa_deg"]),
            gasSigma=float(params["gas_sigma_kms"]),
            diskThick=0.0,
            inClouds=[] if inclouds is None else inclouds,
            flux_clouds=flux_clouds,
            sbProf=sb_profile,
            sbRad=sb_rad,
            velProf=velocity_profile,
            velRad=radial,
            intFlux=float(params["flux_jy_kms"]),
            phaseCent=[float(params["dx_arcsec"]), float(params["dy_arcsec"])],
            vOffset=v_offset,
            vSys=float(params["vsys_kms"]),
            returnClouds=False,
        )
        if cube_xyv is None:
            cube_xyv = np.asarray(phase_cube, dtype=np.float64)
        else:
            cube_xyv += np.asarray(phase_cube, dtype=np.float64)
    cube_xyv /= float(len(phase_offsets))
    # KinMS returns channel density in (x,y,v); kinUV consumes channel-integrated
    # flux in (north,east,v). Reverse v when the exact frequency-derived velocity
    # axis is descending.
    cube_yxv = np.transpose(np.asarray(cube_xyv, dtype=np.float64), (1, 0, 2)) * dv
    if reverse_spectral:
        cube_yxv = cube_yxv[:, :, ::-1]
    cube_yxv = np.ascontiguousarray(cube_yxv, dtype=np.float64)

    # KinMS deposits clouds at nearest channel centres. Register its finite-grid
    # centroid to the exact frequency-derived systemic velocity with a linear,
    # flux-conserving subchannel shift before kinUV applies the spectral response.
    spectral_flux = cube_yxv.sum(axis=(0, 1))
    centroid_before = float(np.sum(velocity * spectral_flux) / np.sum(spectral_flux))
    shift_channels = (float(params["vsys_kms"]) - centroid_before) / dv
    if abs(shift_channels) >= 1.0:
        raise ValueError(f"unexpected KinMS spectral registration error: {shift_channels} channels")
    if shift_channels != 0.0:
        original = cube_yxv
        shifted = np.zeros_like(original)
        fraction = abs(shift_channels)
        shifted += (1.0 - fraction) * original
        if shift_channels > 0.0:
            shifted[:, :, 1:] += fraction * original[:, :, :-1]
        else:
            shifted[:, :, :-1] += fraction * original[:, :, 1:]
        before_flux = float(original.sum())
        shifted *= before_flux / float(shifted.sum())
        cube_yxv = shifted
    spectral_flux = cube_yxv.sum(axis=(0, 1))
    centroid_after = float(np.sum(velocity * spectral_flux) / np.sum(spectral_flux))

    kinms_path = Path(sys.modules["kinms"].__file__).resolve()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "renderer": "KinMS",
        "kinms_version": importlib.metadata.version("kinms"),
        "kinms_module": str(kinms_path),
        "kinms_module_sha256": _sha256(kinms_path),
        "worker_sha256": _sha256(Path(__file__).resolve()),
        "config_sha256": _sha256(config_path),
        "clean_out": True,
        "restoring_beam_applied": False,
        "primary_beam_applied": False,
        "spectral_response_applied": False,
        "axis_order": "north,east,velocity",
        "cube_units": "Jy_per_native_channel",
        "velocity_convention": "radio_kms_from_exact_kinuv_frequency_axis",
        "velocity_descending": reverse_spectral,
        "native_channel_width_kms": dv,
        "n_clouds": n_clouds,
        "seed": seed,
        "render_mode": render_mode,
        "quadrature": quadrature_metadata,
        "spectral_registration": {
            "centroid_before_kms": centroid_before,
            "centroid_after_kms": centroid_after,
            "linear_shift_native_channels": shift_channels,
            "flux_conserving": True,
        },
        "integrated_flux_jy_kms_requested": float(params["flux_jy_kms"]),
        "integrated_flux_jy_kms_rendered": float(cube_yxv.sum()),
    }
    metadata_text = json.dumps(metadata, sort_keys=True)
    np.savez_compressed(
        output_path,
        cube_yxv=cube_yxv,
        velocity_centers_kms=velocity,
        metadata_json=np.asarray(metadata_text),
    )
    metadata["output_npz"] = str(output_path)
    metadata["output_npz_sha256"] = _sha256(output_path)
    sidecar = output_path.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
