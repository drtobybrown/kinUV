"""Standalone package and ms2kinuv ingestion-contract checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kinuv.constants import C_LIGHT_M_S
from kinuv.io.vis import load_visibility_table
from kinuv.targets import get_target


def _arrays():
    freqs = np.array([220.0e9, 220.1e9])
    vis = np.ones((3, 2), dtype=np.complex64)
    weights = np.ones((3, 2), dtype=np.float32)
    return freqs, vis, weights


def test_loads_canonical_ms2kinuv_contract(tmp_path):
    path = tmp_path / "canonical.npz"
    freqs, vis, weights = _arrays()
    np.savez(
        path,
        u_m=np.array([1.0, 2.0, 3.0]),
        v_m=np.array([-1.0, -2.0, -3.0]),
        vis=vis,
        weights=weights,
        freqs=freqs,
        schema_version=np.array("ms2kinuv-npz-v1"),
        time=np.array([0.0, 10.0, 20.0]),
        baseline=np.array([1, 1, 2]),
        phase_dir_rad=np.array([2.0, 0.1]),
    )
    table = load_visibility_table(path)
    assert table.schema == "ms2kinuv-npz-v1"
    assert table.uv_ref_hz is None
    np.testing.assert_allclose(table.u_m, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(table.phase_dir_rad, [2.0, 0.1])


def test_loads_existing_historical_wavelength_contract(tmp_path):
    path = tmp_path / "historical.npz"
    freqs, vis, weights = _arrays()
    u_m = np.array([1.0, 2.0, 3.0])
    ref_hz = float(np.mean(freqs))
    np.savez(
        path,
        u=u_m * ref_hz / C_LIGHT_M_S,
        v=-u_m * ref_hz / C_LIGHT_M_S,
        vis=vis,
        weights=weights,
        freqs=freqs,
    )
    table = load_visibility_table(path)
    assert table.schema == "historical-reference-wavelengths"
    assert table.uv_ref_hz == ref_hz
    np.testing.assert_allclose(table.u_m, u_m)
    assert table.time is None


def test_rejects_unknown_ms2kinuv_schema(tmp_path):
    path = tmp_path / "future.npz"
    freqs, vis, weights = _arrays()
    np.savez(
        path,
        schema_version=np.array("ms2kinuv-npz-v999"),
        u_m=np.array([1.0, 2.0, 3.0]),
        v_m=np.array([-1.0, -2.0, -3.0]),
        vis=vis,
        weights=weights,
        freqs=freqs,
    )
    with np.testing.assert_raises_regex(ValueError, "unsupported schema_version"):
        load_visibility_table(path)


def test_kgas007_metadata_is_owned_by_kinuv():
    target = get_target("kgas007")
    assert target.source == "kinuv.targets:KGAS007"
    assert target.archive_id == "KILOGAS007"
    assert target.inference_overrides() == {
        "ra_deg": 146.576065,
        "dec_deg": 2.88434,
        "vsys_optical_kms": 14229.0,
        "pa_deg": 212.6,
        "i_deg": 28.9,
    }


def test_runtime_metadata_has_no_legacy_or_casa_dependency():
    repo = Path(__file__).resolve().parents[1]
    pyproject = (repo / "pyproject.toml").read_text(encoding="utf-8").lower()
    dependency_block = pyproject.split("[project.optional-dependencies]", 1)[0]
    for name in ("uvkin", "uvfit", "ms2kinuv", "casatasks", "casacore", "pyuvdata"):
        assert name not in dependency_block
