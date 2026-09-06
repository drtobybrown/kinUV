"""Validated boundary between kinUV and intrinsic external model renderers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from kinuv.forward.operators import sample_intrinsic_cube_binned

INTRINSIC_SCHEMA_VERSION = "kinms-intrinsic-cube-v1"


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
    if sidecar.get("schema_version") != INTRINSIC_SCHEMA_VERSION:
        raise ValueError("unsupported KinMS intrinsic-cube schema")
    if sidecar.get("output_npz_sha256") != sha256_file(cube_path):
        raise ValueError("KinMS intrinsic cube checksum mismatch")
    for forbidden in (
        "restoring_beam_applied",
        "primary_beam_applied",
        "spectral_response_applied",
    ):
        if sidecar.get(forbidden) is not False:
            raise ValueError(f"intrinsic comparator has {forbidden}=true or missing")
    if sidecar.get("clean_out") is not True:
        raise ValueError("KinMS comparator was not rendered with cleanOut=True")
    if sidecar.get("cube_units") != "Jy_per_native_channel":
        raise ValueError("KinMS intrinsic cube has incompatible units")
    if sidecar.get("axis_order") != "north,east,velocity":
        raise ValueError("KinMS intrinsic cube has incompatible axis order")

    with np.load(cube_path, allow_pickle=False) as payload:
        cube = np.asarray(payload["cube_yxv"], dtype=np.float64)
        velocity = np.asarray(payload["velocity_centers_kms"], dtype=np.float64)
        embedded = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    if embedded.get("config_sha256") != sidecar.get("config_sha256"):
        raise ValueError("KinMS sidecar and payload provenance differ")
    expected_shape = (int(grid.ny), int(grid.nx), velocity.size)
    if cube.shape != expected_shape:
        raise ValueError(f"KinMS cube shape {cube.shape} != {expected_shape}")
    wanted_velocity = np.asarray(velocity_centers_kms, dtype=np.float64)
    if velocity.shape != wanted_velocity.shape or not np.allclose(
        velocity, wanted_velocity, rtol=0.0, atol=1.0e-9
    ):
        raise ValueError("KinMS cube velocity axis differs from kinUV native axis")
    if not np.all(np.isfinite(cube)) or np.any(cube < 0.0):
        raise ValueError("KinMS intrinsic cube contains invalid emission")
    return cube, sidecar


def sample_intrinsic_kinms(path, *, data, grid, eps: float = 1e-8):
    """Load KinMS emission and apply the exact kinUV measurement operator."""
    cube, metadata = load_intrinsic_kinms_cube(
        path,
        grid=grid,
        velocity_centers_kms=data.vel_native,
    )
    vis = sample_intrinsic_cube_binned(data, cube, grid, eps=eps)
    return vis, cube, metadata
