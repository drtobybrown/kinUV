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
