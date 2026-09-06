"""Validated boundary between kinUV and intrinsic external model renderers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from kinuv.forward.operators import sample_intrinsic_cube_binned

INTRINSIC_SCHEMA_VERSION = "kinms-intrinsic-cube-v1"

_SIDECAR_ONLY_FIELDS = frozenset({"output_npz", "output_npz_sha256"})
_CONTRACT_FIELDS = (
    "schema_version",
    "clean_out",
    "restoring_beam_applied",
    "primary_beam_applied",
    "spectral_response_applied",
    "cube_units",
    "axis_order",
    "output_grid",
    "config_sha256",
    "integrated_flux_jy_kms_requested",
    "integrated_flux_jy_kms_rendered",
)


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_intrinsic_kinms_cube(path, *, grid, velocity_centers_kms):
    """Load a beam-free KinMS cube after validating the immutable contract."""
    cube_path = Path(path)
    sidecar_path = cube_path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text())
    if sidecar.get("output_npz_sha256") != sha256_file(cube_path):
        raise ValueError("KinMS intrinsic cube checksum mismatch")

    with np.load(cube_path, allow_pickle=False) as payload:
        required = {"cube_yxv", "velocity_centers_kms", "metadata_json"}
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"KinMS intrinsic cube missing payload fields {missing}")
        cube = np.asarray(payload["cube_yxv"], dtype=np.float64)
        velocity = np.asarray(payload["velocity_centers_kms"], dtype=np.float64)
        embedded = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    embedded_sidecar = {
        key: value for key, value in sidecar.items() if key not in _SIDECAR_ONLY_FIELDS
    }
    if embedded != embedded_sidecar:
        raise ValueError("KinMS sidecar and checksum-bound payload metadata differ")
    if embedded.get("schema_version") != INTRINSIC_SCHEMA_VERSION:
        raise ValueError("unsupported KinMS intrinsic-cube schema")
    missing_contract = [key for key in _CONTRACT_FIELDS if key not in embedded]
    if missing_contract:
        raise ValueError(f"KinMS intrinsic metadata missing fields {missing_contract}")
    for forbidden in (
        "restoring_beam_applied",
        "primary_beam_applied",
        "spectral_response_applied",
    ):
        if embedded[forbidden] is not False:
            raise ValueError(f"intrinsic comparator has {forbidden}=true or missing")
    if embedded["clean_out"] is not True:
        raise ValueError("KinMS comparator was not rendered with cleanOut=True")
    if embedded["cube_units"] != "Jy_per_native_channel":
        raise ValueError("KinMS intrinsic cube has incompatible units")
    if embedded["axis_order"] != "north,east,velocity":
        raise ValueError("KinMS intrinsic cube has incompatible axis order")
    expected_shape = (int(grid.ny), int(grid.nx), velocity.size)
    expected_grid = {
        "ny": int(grid.ny),
        "nx": int(grid.nx),
        "cell_arcsec": float(grid.cell_arcsec),
    }
    if embedded["output_grid"] != expected_grid:
        raise ValueError("KinMS intrinsic cube grid metadata differs from kinUV grid")
    if cube.shape != expected_shape:
        raise ValueError(f"KinMS cube shape {cube.shape} != {expected_shape}")
    wanted_velocity = np.asarray(velocity_centers_kms, dtype=np.float64)
    if velocity.ndim != 1 or not np.all(np.isfinite(velocity)):
        raise ValueError("KinMS cube velocity axis is not finite and one-dimensional")
    if wanted_velocity.ndim != 1 or not np.all(np.isfinite(wanted_velocity)):
        raise ValueError("requested kinUV velocity axis is not finite and one-dimensional")
    if velocity.shape != wanted_velocity.shape or not np.allclose(
        velocity, wanted_velocity, rtol=0.0, atol=1.0e-9
    ):
        raise ValueError("KinMS cube velocity axis differs from kinUV native axis")
    if not np.all(np.isfinite(cube)) or np.any(cube < 0.0):
        raise ValueError("KinMS intrinsic cube contains invalid emission")
    computed_flux = float(np.sum(cube, dtype=np.float64))
    claimed_flux = float(embedded["integrated_flux_jy_kms_rendered"])
    requested_flux = float(embedded["integrated_flux_jy_kms_requested"])
    if not np.isfinite(claimed_flux) or not np.isfinite(requested_flux):
        raise ValueError("KinMS intrinsic metadata contains nonfinite flux")
    flux_atol = max(abs(computed_flux), 1.0) * 1.0e-12
    if not np.isclose(computed_flux, claimed_flux, rtol=1.0e-12, atol=flux_atol):
        raise ValueError("KinMS claimed rendered flux differs from cube-derived flux")
    validated = dict(embedded)
    validated["output_npz"] = sidecar.get("output_npz")
    validated["output_npz_sha256"] = sidecar["output_npz_sha256"]
    validated["integrated_flux_jy_kms_computed"] = computed_flux
    return cube, validated


def sample_intrinsic_kinms(path, *, data, grid, eps: float = 1e-8):
    """Load KinMS emission and apply the exact kinUV measurement operator."""
    cube, metadata = load_intrinsic_kinms_cube(
        path,
        grid=grid,
        velocity_centers_kms=data.vel_native,
    )
    vis = sample_intrinsic_cube_binned(data, cube, grid, eps=eps)
    return vis, cube, metadata
