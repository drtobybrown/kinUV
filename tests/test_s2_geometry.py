from __future__ import annotations

import numpy as np
import pytest

from kinuv.infer.s2 import (
    FitCovariance,
    correlated_chi2,
    deterministic_start_grid,
    pack_s2,
    projected_gradient,
    propagate_equal_weight_ar1,
    s2_bounds,
    unpack_s2,
)
from kinuv.io.vis import VisData
from kinuv.transforms.grid import ImageGrid


def test_ar1_propagation_matches_explicit_block_covariance():
    source_scale = 1.3
    source_rho = 0.3
    output = propagate_equal_weight_ar1(source_scale, source_rho, 4)
    indices = np.arange(8)
    covariance = source_scale * source_rho ** np.abs(
        indices[:, None] - indices[None, :]
    )
    left = np.zeros(8)
    right = np.zeros(8)
    left[:4] = 0.25
    right[4:] = 0.25
    variance = float(left @ covariance @ left)
    adjacent = float(left @ covariance @ right)
    # A binned MS weight is four times one native weight.
    assert output.scale == pytest.approx(4.0 * variance)
    assert output.rho == pytest.approx(adjacent / variance)


def test_correlated_chi2_matches_manual_two_channel_form():
    data = np.array([[1.0 + 2.0j, 3.0 + 5.0j]])
    model = np.array([[0.5 + 1.0j, 2.0 + 4.0j]])
    weight = np.array([[4.0, 9.0]])
    covariance = FitCovariance(2.0, 0.2, 2.0, 0.2, 1)
    residual = np.sqrt(weight) * (data - model)
    manual = (
        abs(residual[0, 0]) ** 2
        + abs(residual[0, 1] - 0.2 * residual[0, 0]) ** 2 / (1.0 - 0.2**2)
    ) / 2.0
    assert correlated_chi2(data, model, weight, covariance) == pytest.approx(manual)


def test_correlated_chi2_restarts_after_flagged_gap():
    data = np.ones((1, 3), dtype=np.complex128)
    weight = np.array([[1.0, 0.0, 1.0]])
    covariance = FitCovariance(1.0, 0.4, 1.0, 0.4, 1)
    assert correlated_chi2(data, np.zeros_like(data), weight, covariance) == 2.0


def test_s2_chart_round_trip_and_projected_velocity():
    parameters = {
        "flux": 4.0,
        "pa_deg": 210.0,
        "vsys_kms": 8100.0,
        "gas_sigma_kms": 11.0,
        "dx_arcsec": 0.1,
        "dy_arcsec": -0.2,
        "u_kms": 175.0,
        "inclination_deg": 30.0,
        "r_t_arcsec": 0.4,
    }
    z = pack_s2(parameters, vsys_seed_kms=8090.0, dv_kms=5.0, bmaj_arcsec=1.0)
    back = unpack_s2(z, vsys_seed_kms=8090.0, dv_kms=5.0, bmaj_arcsec=1.0)
    for key in parameters:
        assert back[key] == pytest.approx(parameters[key])
    assert back["v0_kms_diagnostic"] == pytest.approx(350.0)


def test_registered_start_grid_and_bounds():
    starts = deterministic_start_grid(205.0, 8100.0, 1.0, 0.4)
    assert len(starts) == 12
    assert len({row["start_id"] for row in starts}) == 12
    assert {row["inclination_deg"] for row in starts} == {30.0, 60.0}
    assert {row["u_kms"] for row in starts} == {75.0, 175.0, 300.0}
    bounds = s2_bounds(
        pa_seed_deg=205.0,
        dv_kms=5.0,
        bmaj_arcsec=1.0,
        turnover_ratio=0.4,
    )
    assert bounds[2] == (-20.0, 20.0)
    assert bounds[8] == (0.4, 0.4)


def test_projected_gradient_removes_outward_bound_component():
    z = np.array([0.0, 1.0])
    gradient = np.array([2.0, -3.0])
    bounds = [(0.0, 4.0), (0.0, 1.0)]
    np.testing.assert_allclose(projected_gradient(z, gradient, bounds), 0.0)


def test_s2_objective_differentiates_center_and_inclination():
    pytest.importorskip("jax")
    from kinuv.infer.s2 import build_s2_value_gradient
    from kinuv.transforms.nufft import BACKEND

    if BACKEND != "jax-finufft":
        pytest.skip("requires jax-finufft")
    n_row, n_native, n_guard, n_bin = 4, 10, 1, 2
    frequencies = 224.0e9 + np.arange(n_native) * 1.0e6
    rng = np.random.default_rng(42)
    data = VisData(
        u_m=np.linspace(10.0, 40.0, n_row),
        v_m=np.linspace(-20.0, 20.0, n_row),
        vis=rng.normal(size=(n_row, 4)) + 1j * rng.normal(size=(n_row, 4)),
        weights=np.ones((n_row, 4)),
        freqs=frequencies[1:-1].reshape(4, 2).mean(axis=1),
        vel=np.linspace(8000.0, 8040.0, 4),
        freqs_native=frequencies,
        vel_native=np.linspace(7990.0, 8050.0, n_native),
        n_bin=n_bin,
        dv_kms=10.0,
        s=1.0,
        phase_dir_rad=np.zeros(2),
        line_free_mask=np.ones(4, dtype=bool),
        n_guard=n_guard,
        weights_native=np.ones((n_row, n_native - 2)),
    )
    grid = ImageGrid(nx=12, ny=12, cell_arcsec=0.4)
    x = (np.arange(12) - 6) * 0.4
    east, north = np.meshgrid(x, x, indexing="xy")
    template = np.exp(-0.5 * (east**2 + north**2))
    template /= template.sum() * grid.cell_arcsec**2
    start = deterministic_start_grid(200.0, 8020.0, 1.0, 0.4)[0]
    start["flux"] = 2.0
    z = pack_s2(start, vsys_seed_kms=8020.0, dv_kms=10.0, bmaj_arcsec=1.0)
    value_gradient = build_s2_value_gradient(
        data,
        template,
        grid,
        FitCovariance(1.0, 0.1, 1.0, 0.1, 2),
        vsys_seed_kms=8020.0,
        bmaj_arcsec=1.0,
    )
    value, gradient = value_gradient(z)
    assert np.isfinite(float(value))
    gradient = np.asarray(gradient)
    assert np.all(np.isfinite(gradient))
    assert gradient[4] != 0.0
    assert gradient[5] != 0.0
    assert gradient[7] != 0.0
