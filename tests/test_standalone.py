"""Standalone package and ms2kinuv ingestion-contract checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kinuv.constants import C_LIGHT_M_S
from kinuv.io.vis import load_visibility_table, require_s2_provenance
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


def _v2_payload():
    freqs, vis, weights = _arrays()
    n_row, n_chan = vis.shape
    antenna1 = np.array([0, 0, 1], dtype=np.int64)
    antenna2 = np.array([1, 2, 2], dtype=np.int64)
    row_id = np.arange(n_row, dtype=np.int64)
    channel_width = np.full(n_chan, 1.0e8)
    return {
        "schema_version": np.array("ms2kinuv-npz-v2"),
        "u_m": np.array([1.0, 2.0, 3.0]),
        "v_m": np.array([-1.0, -2.0, -3.0]),
        "uvw_m": np.array([[1.0, -1.0, 0.0], [2.0, -2.0, 0.0], [3.0, -3.0, 0.0]]),
        "vis": vis,
        "weights": weights,
        "freqs": freqs,
        "time": np.array([0.0, 10.0, 20.0]),
        "baseline": (antenna1 << np.int64(32)) | antenna2,
        "phase_dir_rad": np.array([2.0, 0.1]),
        "row_id": row_id,
        "source_row_id": row_id.copy(),
        "aggregation_coefficients": np.ones(n_row),
        "antenna1": antenna1,
        "antenna2": antenna2,
        "scan_number": np.array([1, 1, 2]),
        "observation_id": np.zeros(n_row, dtype=np.int64),
        "array_id": np.zeros(n_row, dtype=np.int64),
        "state_id": np.zeros(n_row, dtype=np.int64),
        "field_id": np.zeros(n_row, dtype=np.int64),
        "data_desc_id": np.zeros(n_row, dtype=np.int64),
        "time_centroid": np.array([0.5, 10.5, 20.5]),
        "interval": np.ones(n_row),
        "flags": np.zeros((n_row, n_chan), dtype=bool),
        "channel_width_hz": channel_width,
        "channel_edges_hz": np.column_stack(
            [freqs - channel_width / 2.0, freqs + channel_width / 2.0]
        ),
        "spectral_window_id": np.array(0),
        "polarization_id": np.array(0),
        "polarization_index": np.array(0),
        "correlation_type": np.array(9),
        "frequency_reference_code": np.array(1),
        "frequency_frame": np.array("LSRK"),
        "visibility_unit": np.array("Jy"),
        "weight_convention": np.array(
            "ms_weight_equals_2_over_complex_noise_variance"
        ),
        "history_json": np.array('{"available": true}'),
        "smoothing_history_json": np.array("{}"),
        "extraction_json": np.array(
            '{"source_row_identity_preserved": true, '
            '"row_averaging": "none", "channel_averaging": "none"}'
        ),
    }


def test_loads_fold_safe_ms2kinuv_v2_contract(tmp_path):
    path = tmp_path / "provenance.npz"
    np.savez(path, **_v2_payload())
    table = load_visibility_table(path)
    require_s2_provenance(table)
    assert table.fold_safe
    np.testing.assert_array_equal(table.scan_number, [1, 1, 2])
    np.testing.assert_array_equal(table.antenna1, [0, 0, 1])
    assert table.frequency_frame == "LSRK"
    assert table.extraction["row_averaging"] == "none"


def test_v2_missing_group_metadata_fails_closed(tmp_path):
    path = tmp_path / "incomplete.npz"
    payload = _v2_payload()
    payload.pop("scan_number")
    np.savez(path, **payload)
    with np.testing.assert_raises_regex(KeyError, "scan_number"):
        load_visibility_table(path)


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
