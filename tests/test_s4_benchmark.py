from __future__ import annotations

import numpy as np
import pytest

from kinuv.validation.s4 import (
    channel_noise_from_integrated_error,
    common_reduced_chi2,
    projected_velocity_rmse,
    topo_radio_to_lsrk_radio,
)


def test_frequency_equivalent_frame_correction_has_radio_sign():
    velocity = np.array([8000.0, 8100.0])
    corrected = topo_radio_to_lsrk_radio(velocity, 11.0)
    assert np.all(corrected < velocity)
    assert np.mean(velocity - corrected) == pytest.approx(10.7, rel=0.02)


def test_channel_noise_inverts_integrated_error_rule():
    mask = np.zeros((4, 2, 2), dtype=bool)
    mask[:, 0, 0] = True
    mask[:2, 1, 1] = True
    error = np.zeros((2, 2))
    error[0, 0] = 3.0 * 5.0 * np.sqrt(4.0)
    error[1, 1] = 3.0 * 5.0 * np.sqrt(2.0)
    sigma, count = channel_noise_from_integrated_error(error, mask, 5.0)
    assert sigma == pytest.approx(3.0)
    assert count == 2


def test_common_reduced_chi2_and_projected_rmse():
    data = np.ones((2, 2, 2))
    model = np.zeros_like(data)
    metric = common_reduced_chi2(data, model, np.ones_like(data, bool), 0.5, 2)
    assert metric["chi2"] == pytest.approx(32.0)
    assert metric["reduced_chi2"] == pytest.approx(32.0 / 6.0)
    profiles = {
        "rotation_radius_data": np.array([1.0, 2.0, 3.0]),
        "rotation_speed_data": np.array([10.0, 20.0, 30.0]),
        "rotation_radius_kinuv": np.array([1.0, 2.0, 3.0]),
        "rotation_speed_kinuv": np.array([11.0, 21.0, 31.0]),
        "rotation_radius_kinms": np.array([1.0, 2.0, 3.0]),
        "rotation_speed_kinms": np.array([12.0, 22.0, 32.0]),
    }
    recovery = projected_velocity_rmse(profiles, 30.0)
    assert recovery["kinuv"]["rmse_kms"] == pytest.approx(0.5)
    assert recovery["kinms"]["rmse_kms"] == pytest.approx(1.0)
    assert recovery["ratio_kinuv_over_kinms"] == pytest.approx(0.5)
    assert recovery["gate_eligible"] is True


def test_two_point_velocity_profile_is_diagnostic_but_not_gate_eligible():
    profiles = {
        "rotation_radius_data": np.array([1.0, 2.0]),
        "rotation_speed_data": np.array([10.0, 20.0]),
        "rotation_radius_kinuv": np.array([1.0, 2.0]),
        "rotation_speed_kinuv": np.array([11.0, 21.0]),
        "rotation_radius_kinms": np.array([1.0, 2.0]),
        "rotation_speed_kinms": np.array([12.0, 22.0]),
    }
    recovery = projected_velocity_rmse(profiles, 30.0)
    assert recovery["gate_eligible"] is False
    assert np.isfinite(recovery["ratio_kinuv_over_kinms"])


def test_s4_runner_uses_same_observed_frequency_to_invert_primary_beam():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "scripts/run_s4_real_benchmark.py"
    ).read_text(encoding="utf-8")
    assert "pb_frequency_hz = float(np.median(data.freqs_native))" in source
    assert "nu_hz=pb_frequency_hz" in source
    assert "hann_native(cube_yxv, axis=2)" in source
    assert '"--replay-all"' in source
    assert '"promotion_eligible": False' in source


def test_replay_records_sparse_profile_instead_of_aborting():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/run_s4_real_benchmark.py"
    spec = importlib.util.spec_from_file_location("s4_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    record = module._diagnostic_projected_velocity_rmse(
        {
            "rotation_radius_data": np.array([1.0]),
            "rotation_speed_data": np.array([10.0]),
        },
        30.0,
    )
    assert record["gate_eligible"] is False
    assert record["ratio_kinuv_over_kinms"] is None
    assert record["data_n_radius"] == 1


def test_grouped_prediction_bootstrap_and_frame_round_trip():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/run_s4_grouped_prediction.py"
    spec = importlib.util.spec_from_file_location("s4_grouped", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [
        {"delta_chi2": value, "n_real_components": 100}
        for value in (40.0, 50.0, 60.0, 45.0, 55.0)
    ]
    result = module._bootstrap_delta(rows, seed=2, draws=2000)
    assert result["delta_chi2_per_real_component"] == pytest.approx(0.5)
    assert result["lower_95_percent"] > 0.0

    correction = 10.9850680313608
    vopt_lsrk = 8300.0
    topo = module._lsrk_optical_to_topo_radio(vopt_lsrk, correction)
    recovered = topo_radio_to_lsrk_radio(topo, correction)
    from kinuv.io.vis import optical_to_radio_kms

    assert recovered == pytest.approx(float(optical_to_radio_kms(vopt_lsrk)))

    source = path.read_text(encoding="utf-8")
    assert '"casa_reimaging_required": False' in source
    assert '"promotion_eligible": audit_pass' in source
    assert "training-only stock-KinMS confirmation" not in source


def test_continuum_worker_skips_fully_cropped_deposition_chunks():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "external/_kinms_intrinsic_worker.py"
    ).read_text(encoding="utf-8")
    assert "if not values:" in source
    assert "there is no sparse deposition contribution" in source
