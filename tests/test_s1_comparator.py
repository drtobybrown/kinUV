"""S1 intrinsic-comparator contract and shared-operator closure."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kinuv.constants import ARCSEC_TO_RAD, F_REST_CO21_HZ
from kinuv.diagnostics.comparator import (
    INTRINSIC_SCHEMA_VERSION,
    load_intrinsic_kinms_cube,
    sha256_file,
)
from kinuv.forward.operators import (
    attenuate_intrinsic_cube,
    sample_intrinsic_cube_native,
)
from kinuv.transforms.dft import dft_numpy
from kinuv.transforms.grid import ImageGrid


def _fixture_cube(grid, freqs):
    x = grid.l_rad / ARCSEC_TO_RAD
    y = grid.m_rad / ARCSEC_TO_RAD
    east, north = np.meshgrid(x, y, indexing="xy")
    spatial = np.exp(-0.5 * ((east / 0.7) ** 2 + (north / 0.5) ** 2))
    spatial /= spatial.sum()
    spectral = np.exp(-0.5 * ((np.arange(len(freqs)) - 2.0) / 0.9) ** 2)
    return spatial[:, :, None] * spectral[None, None, :]


def test_shared_cube_operator_matches_float64_dft():
    grid = ImageGrid(24, 24, 0.2)
    freqs = F_REST_CO21_HZ + np.arange(-2, 3) * 1.0e6
    cube = _fixture_cube(grid, freqs)
    rng = np.random.default_rng(1066)
    u_m = rng.uniform(-100.0, 100.0, 13)
    v_m = rng.uniform(-100.0, 100.0, 13)

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
    relative_l2 = np.linalg.norm(actual - reference) / np.linalg.norm(reference)
    assert relative_l2 <= 1.0e-6


def test_shared_cube_operator_preserves_zero_baseline_flux():
    grid = ImageGrid(24, 24, 0.2)
    freqs = F_REST_CO21_HZ + np.arange(-2, 3) * 1.0e6
    cube = _fixture_cube(grid, freqs)
    attenuated = attenuate_intrinsic_cube(cube, grid, freqs)
    actual = sample_intrinsic_cube_native(
        cube, grid, np.array([0.0]), np.array([0.0]), freqs, eps=1.0e-12
    )[0]
    expected = attenuated.sum(axis=(0, 1))
    relative = np.max(np.abs(actual - expected) / np.maximum(np.abs(expected), 1e-30))
    assert relative <= 1.0e-3


def _write_contract(
    path: Path,
    grid: ImageGrid,
    velocity,
    *,
    restored=False,
    embedded_updates=None,
    embedded_removals=(),
    sidecar_updates=None,
):
    cube = np.ones((grid.ny, grid.nx, len(velocity)), dtype=np.float64)
    embedded = {
        "schema_version": INTRINSIC_SCHEMA_VERSION,
        "config_sha256": "config-digest",
        "clean_out": True,
        "restoring_beam_applied": restored,
        "primary_beam_applied": False,
        "spectral_response_applied": False,
        "post_crop_renormalization": False,
        "cube_units": "Jy",
        "channel_value_semantics": "channel-average flux density per sky pixel",
        "spatial_kernel": "separable cardinal cubic B-spline B3",
        "axis_order": "north,east,velocity",
        "output_grid": {
            "ny": grid.ny,
            "nx": grid.nx,
            "cell_arcsec": grid.cell_arcsec,
        },
        "flux_ledger_jy_kms": {
            "requested_full_support": float(cube.sum()),
            "quadrature_input": float(cube.sum()),
            "spatially_retained": float(cube.sum()),
            "spectrally_retained": float(cube.sum()),
            "jointly_retained": float(cube.sum()),
            "cube_integral": float(cube.sum()),
            "spatially_excluded": 0.0,
            "spectrally_excluded": 0.0,
            "jointly_excluded": 0.0,
        },
    }
    embedded.update(embedded_updates or {})
    for field in embedded_removals:
        embedded.pop(field)
    np.savez_compressed(
        path,
        cube_yxv=cube,
        velocity_centers_kms=np.asarray(velocity),
        metadata_json=np.asarray(json.dumps(embedded)),
    )
    sidecar = {
        **embedded,
        "output_npz_sha256": sha256_file(path),
    }
    sidecar.update(sidecar_updates or {})
    path.with_suffix(".json").write_text(json.dumps(sidecar))


def test_intrinsic_contract_accepts_only_beam_free_exact_axis(tmp_path):
    grid = ImageGrid(8, 8, 0.25)
    velocity = np.arange(5.0)
    path = tmp_path / "intrinsic.npz"
    _write_contract(path, grid, velocity)
    cube, metadata = load_intrinsic_kinms_cube(
        path, grid=grid, velocity_centers_kms=velocity
    )
    assert cube.shape == (8, 8, 5)
    assert metadata["clean_out"] is True
    with pytest.raises(ValueError, match="velocity axis"):
        load_intrinsic_kinms_cube(
            path, grid=grid, velocity_centers_kms=velocity + 0.1
        )


def test_intrinsic_contract_rejects_restored_cube(tmp_path):
    grid = ImageGrid(8, 8, 0.25)
    velocity = np.arange(5.0)
    path = tmp_path / "restored.npz"
    _write_contract(path, grid, velocity, restored=True)
    with pytest.raises(ValueError, match="restoring_beam_applied"):
        load_intrinsic_kinms_cube(path, grid=grid, velocity_centers_kms=velocity)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("primary_beam_applied", True),
        ("spectral_response_applied", True),
        ("clean_out", False),
        ("post_crop_renormalization", True),
        ("cube_units", "Jy_per_beam"),
        ("axis_order", "east,north,velocity"),
    ],
)
def test_intrinsic_contract_rejects_forbidden_embedded_metadata(
    tmp_path, field, value
):
    grid = ImageGrid(8, 8, 0.25)
    velocity = np.arange(5.0)
    path = tmp_path / f"bad-{field}.npz"
    _write_contract(path, grid, velocity, embedded_updates={field: value})
    with pytest.raises(ValueError):
        load_intrinsic_kinms_cube(path, grid=grid, velocity_centers_kms=velocity)


def test_intrinsic_contract_rejects_sidecar_payload_disagreement(tmp_path):
    grid = ImageGrid(8, 8, 0.25)
    velocity = np.arange(5.0)
    path = tmp_path / "contradictory.npz"
    _write_contract(
        path,
        grid,
        velocity,
        embedded_updates={"primary_beam_applied": True},
        sidecar_updates={"primary_beam_applied": False},
    )
    with pytest.raises(ValueError, match="metadata differ"):
        load_intrinsic_kinms_cube(path, grid=grid, velocity_centers_kms=velocity)


def test_intrinsic_contract_rejects_claimed_flux_disagreement(tmp_path):
    grid = ImageGrid(8, 8, 0.25)
    velocity = np.arange(5.0)
    path = tmp_path / "bad-flux.npz"
    _write_contract(
        path,
        grid,
        velocity,
        embedded_updates={
            "flux_ledger_jy_kms": {
                "requested_full_support": 320.0,
                "quadrature_input": 320.0,
                "spatially_retained": 320.0,
                "spectrally_retained": 320.0,
                "jointly_retained": 999.0,
                "cube_integral": 999.0,
                "spatially_excluded": 0.0,
                "spectrally_excluded": 0.0,
                "jointly_excluded": -679.0,
            }
        },
    )
    with pytest.raises(ValueError, match="cube-derived flux"):
        load_intrinsic_kinms_cube(path, grid=grid, velocity_centers_kms=velocity)


def test_intrinsic_contract_rejects_missing_embedded_metadata(tmp_path):
    grid = ImageGrid(8, 8, 0.25)
    velocity = np.arange(5.0)
    path = tmp_path / "missing-axis-order.npz"
    _write_contract(path, grid, velocity, embedded_removals=("axis_order",))
    with pytest.raises(ValueError, match="missing fields"):
        load_intrinsic_kinms_cube(path, grid=grid, velocity_centers_kms=velocity)


def test_intrinsic_contract_rejects_nonfinite_velocity_axis(tmp_path):
    grid = ImageGrid(8, 8, 0.25)
    velocity = np.arange(5.0)
    velocity[2] = np.nan
    path = tmp_path / "bad-velocity.npz"
    _write_contract(path, grid, velocity)
    with pytest.raises(ValueError, match="velocity axis"):
        load_intrinsic_kinms_cube(path, grid=grid, velocity_centers_kms=velocity)


def test_external_worker_is_intrinsic_and_isolated():
    repo = Path(__file__).resolve().parents[1]
    source = (repo / "external" / "_kinms_intrinsic_worker.py").read_text()
    assert "_stable_normal_interval" in source
    assert "_b3" in source
    assert "post_crop_renormalization\": False" in source
    assert "model_cube(" not in source
    assert "randompick_vdisp" not in source
    assert "(360.0 - float(params[\"pa_deg\"])) % 360.0" in source
    assert "from kinuv" not in source
    assert "import kinuv" not in source
