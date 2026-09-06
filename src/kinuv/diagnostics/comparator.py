"""Validated boundary between kinUV and intrinsic external model renderers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from kinuv.forward.operators import sample_intrinsic_cube_binned

INTRINSIC_SCHEMA_VERSION = "kinms-continuum-cube-v2"

_SIDECAR_ONLY_FIELDS = frozenset({"output_npz", "output_npz_sha256"})
_CONTRACT_FIELDS = (
    "schema_version",
    "clean_out",
    "restoring_beam_applied",
    "primary_beam_applied",
    "spectral_response_applied",
    "post_crop_renormalization",
    "cube_units",
    "channel_value_semantics",
    "axis_order",
    "output_grid",
    "config_sha256",
    "spatial_kernel",
    "flux_ledger_jy_kms",
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
    if embedded["post_crop_renormalization"] is not False:
        raise ValueError("continuum comparator applied forbidden post-crop normalization")
    if embedded["cube_units"] != "Jy":
        raise ValueError("KinMS intrinsic cube has incompatible units")
    if embedded["channel_value_semantics"] != "channel-average flux density per sky pixel":
        raise ValueError("KinMS intrinsic cube does not contain channel flux density")
    if embedded["spatial_kernel"] != "separable cardinal cubic B-spline B3":
        raise ValueError("KinMS intrinsic cube has an unapproved deposition kernel")
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
    dv = float(np.median(np.abs(np.diff(velocity))))
    computed_flux = float(np.sum(cube, dtype=np.float64) * dv)
    ledger = embedded["flux_ledger_jy_kms"]
    required_ledger = {
        "requested_full_support",
        "quadrature_input",
        "spatially_retained",
        "spectrally_retained",
        "jointly_retained",
        "cube_integral",
        "spatially_excluded",
        "spectrally_excluded",
        "jointly_excluded",
    }
    missing_ledger = sorted(required_ledger.difference(ledger))
    if missing_ledger:
        raise ValueError(f"KinMS flux ledger missing fields {missing_ledger}")
    ledger = {key: float(value) for key, value in ledger.items()}
    if not all(np.isfinite(value) for value in ledger.values()):
        raise ValueError("KinMS intrinsic metadata contains nonfinite flux")
    claimed_flux = ledger["cube_integral"]
    requested_flux = ledger["requested_full_support"]
    flux_atol = max(abs(computed_flux), 1.0) * 1.0e-12
    if not np.isclose(computed_flux, claimed_flux, rtol=1.0e-12, atol=flux_atol):
        raise ValueError("KinMS claimed rendered flux differs from cube-derived flux")
    if not np.isclose(claimed_flux, ledger["jointly_retained"], rtol=2.0e-11, atol=flux_atol):
        raise ValueError("KinMS cube and joint-retained flux ledger disagree")
    if not np.isclose(
        ledger["quadrature_input"], requested_flux, rtol=1.0e-12, atol=flux_atol
    ):
        raise ValueError("KinMS quadrature input does not conserve requested flux")
    for retained, excluded in (
        ("spatially_retained", "spatially_excluded"),
        ("spectrally_retained", "spectrally_excluded"),
        ("jointly_retained", "jointly_excluded"),
    ):
        if not np.isclose(
            ledger[retained] + ledger[excluded],
            requested_flux,
            rtol=2.0e-11,
            atol=flux_atol,
        ):
            raise ValueError(f"KinMS flux ledger identity fails for {retained}")
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
    vis = sample_intrinsic_cube_binned(
        data, cube, grid, eps=eps, spatial_assignment="cubic_b_spline"
    )
    return vis, cube, metadata
