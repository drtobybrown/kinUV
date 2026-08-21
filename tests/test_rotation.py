"""Stage A arctan + Stage B rings (DEC-066-VC, DEC-066-OSCMETRIC)."""

from pathlib import Path

import numpy as np
import pytest

from kinuv.profiles.rotation import (
    BMAJ_ICO_ARCSEC,
    CALIBRATION_RT_ARCSEC,
    CALIBRATION_V0_KM_S,
    DISK_RADIUS_ARCSEC,
    DV_CHAN_NATIVE_KM_S,
    N_RINGS_MAX,
    N_RINGS_MIN,
    R0_MIN_OVER_BMAJ,
    V_K_MAX_KM_S,
    V_K_MIN_KM_S,
    aic_keep_stage_a,
    arctan_vc,
    curvature_penalty,
    k_extra_rings,
    monotonicity_penalty,
    omega_k,
    r0_min_arcsec,
    ring_regulariser,
    ring_vc,
    ring_velocity_bounds,
    rings_from_arctan,
    select_lambda_reg,
    uniform_knot_radii,
)


def _knots(n_rings: int = 7):
    return uniform_knot_radii(n_rings, r_last_arcsec=DISK_RADIUS_ARCSEC)


def test_arctan_formula_and_origin():
    r = np.array([0.0, 3.0, 9.0])
    v0, rt = 200.0, 3.0
    got = arctan_vc(r, v0, rt)
    want = v0 * (2.0 / np.pi) * np.arctan(r / rt)
    assert np.allclose(got, want)
    assert float(arctan_vc(0.0, v0, rt)) == 0.0


def test_arctan_not_flattened_beyond_disk():
    """Stage A is defined for all R; V(15″) still rises toward V_0."""
    v0, rt = 200.0, 3.0
    r = np.array([DISK_RADIUS_ARCSEC, 15.0])
    v = arctan_vc(r, v0, rt)
    assert v[1] > v[0]
    assert v[1] < v0
    flat_control = np.array([v[0], v[0]])
    assert not np.allclose(v, flat_control)


def test_rings_from_arctan_matches_stage_a_at_knots():
    knots = _knots()
    v0, rt = CALIBRATION_V0_KM_S, CALIBRATION_RT_ARCSEC
    v = rings_from_arctan(knots, v0, rt)
    assert np.allclose(v, arctan_vc(knots, v0, rt))
    assert not np.allclose(v, 0.0)


def test_r0_at_least_half_bmaj():
    floor = r0_min_arcsec(BMAJ_ICO_ARCSEC)
    assert floor == pytest.approx(R0_MIN_OVER_BMAJ * BMAJ_ICO_ARCSEC)
    assert floor == pytest.approx(0.65)
    knots = _knots()
    assert knots[0] >= floor
    with pytest.raises(ValueError, match="0.5 BMAJ"):
        uniform_knot_radii(7, r0_arcsec=0.3, bmaj_arcsec=BMAJ_ICO_ARCSEC)
    with pytest.raises(ValueError, match="N_rings"):
        uniform_knot_radii(N_RINGS_MIN - 1)
    with pytest.raises(ValueError, match="N_rings"):
        uniform_knot_radii(N_RINGS_MAX + 1)


def test_default_spacing_over_disk():
    knots = _knots(7)
    assert knots[-1] == pytest.approx(DISK_RADIUS_ARCSEC)
    assert N_RINGS_MIN <= knots.size <= N_RINGS_MAX
    dr = float(np.diff(knots).mean())
    assert 1.0 - 0.05 <= dr <= 1.3 + 0.05


def test_outer_flat_and_v0_on_grid_to_15_arcsec():
    """ADR unit test: grid to 15″, r_last=7.5″; dV/dR=0 for R>r_last; V(0)=0."""
    r_last = DISK_RADIUS_ARCSEC
    knots = uniform_knot_radii(7, r_last_arcsec=r_last)
    v_k = rings_from_arctan(knots, CALIBRATION_V0_KM_S, CALIBRATION_RT_ARCSEC)
    r = np.linspace(0.0, 15.0, 1501)
    v = ring_vc(r, knots, v_k)
    assert float(v[0]) == pytest.approx(0.0)
    assert float(ring_vc(0.0, knots, v_k)) == pytest.approx(0.0)
    outer = r > r_last
    dv_dr = np.diff(v[outer]) / np.diff(r[outer])
    assert np.max(np.abs(dv_dr)) < 1e-12
    assert np.allclose(v[outer], v_k[-1])
    just_out = ring_vc(np.array([r_last + 0.1, 10.0, 15.0]), knots, v_k)
    assert np.allclose(just_out, v_k[-1])


def test_inner_solid_body_vs_flat_control():
    knots = _knots()
    v_k = rings_from_arctan(knots, CALIBRATION_V0_KM_S, CALIBRATION_RT_ARCSEC)
    r0, v0 = knots[0], v_k[0]
    r_in = np.array([0.0, 0.5 * r0, r0])
    solid = ring_vc(r_in, knots, v_k, inner_bc="solid_body")
    flat = ring_vc(r_in, knots, v_k, inner_bc="flat")
    assert solid[0] == pytest.approx(0.0)
    assert solid[1] == pytest.approx(0.5 * v0)
    assert solid[2] == pytest.approx(v0)
    assert np.allclose(flat, v0)
    assert not np.allclose(solid, flat)
    # Continuity at r_0: both BCs meet the first knot.
    assert solid[2] == pytest.approx(flat[2])


def test_lbfgs_bounds_are_constants_not_a_fit():
    bounds = ring_velocity_bounds(7)
    assert bounds.shape == (7, 2)
    assert np.all(bounds[:, 0] == V_K_MIN_KM_S)
    assert np.all(bounds[:, 1] == V_K_MAX_KM_S)
    assert V_K_MIN_KM_S == 0.0
    assert V_K_MAX_KM_S == 400.0


def test_curvature_penalty_is_second_difference_not_gp():
    v = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])  # linear: d²=0
    assert curvature_penalty(v, 3.5) == pytest.approx(0.0)
    bump = v.copy()
    bump[3] += 5.0
    d2 = bump[2:] - 2.0 * bump[1:-1] + bump[:-2]
    assert curvature_penalty(bump, 2.0) == pytest.approx(2.0 * np.sum(d2**2))


def test_monotonicity_allows_outermost_decline():
    rising = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
    assert monotonicity_penalty(rising, 4.0) == pytest.approx(0.0)
    inner_drop = rising.copy()
    inner_drop[2] = 5.0  # V2 < V1, inside last knot
    assert monotonicity_penalty(inner_drop, 1.0) > 0.0
    outer_drop = rising.copy()
    outer_drop[-1] = 0.0  # outermost may decline
    assert monotonicity_penalty(outer_drop, 9.0) == pytest.approx(0.0)
    lam = 2.5
    assert ring_regulariser(rising, lam) == pytest.approx(
        curvature_penalty(rising, lam) + monotonicity_penalty(rising, lam)
    )


def test_omega_k_uses_passed_channel_width():
    v = np.array([0.0, 10.0, 30.0, 40.0, 55.0, 60.0, 58.0])
    d2 = np.abs(v[2:] - 2.0 * v[1:-1] + v[:-2])
    native = omega_k(v)
    assert native.shape == (v.size - 2,)
    assert np.allclose(native, d2 / DV_CHAN_NATIVE_KM_S)
    replica = omega_k(v, dv_chan_kms=5.3)
    bin8 = omega_k(v, dv_chan_kms=10.6)
    assert np.allclose(replica, d2 / 5.3)
    assert np.allclose(bin8, d2 / 10.6)
    assert not np.allclose(native, bin8)
    assert DV_CHAN_NATIVE_KM_S == pytest.approx(1.270)


def test_aic_keeps_stage_a_unless_delta_exceeds_two_k_extra():
    n = 7
    k_extra = k_extra_rings(n)
    assert k_extra == n - 2
    threshold = 2 * k_extra
    aic_a = 100.0
    assert aic_keep_stage_a(aic_a, aic_a - threshold, n) is True
    assert aic_keep_stage_a(aic_a, aic_a - threshold - 1e-9, n) is False
    assert aic_keep_stage_a(aic_a, aic_a - 1.0, n) is True  # small win: keep A


def test_select_lambda_reg_acceptance_on_fake_omega():
    n_mock = 20
    lambdas = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
    n_lam = lambdas.size
    max_omega = np.full((n_lam, n_mock), 0.10)
    v0_a = np.full(n_mock, 200.0)
    v0_b = np.full((n_lam, n_mock), 200.0)
    rt_b = np.full((n_lam, n_mock), 3.0)
    max_omega[0] = 0.6  # fails max Ω < 0.3 in 95% of mocks
    v0_b[-1] = 150.0  # biased low vs Stage A by more than 1σ scatter
    chosen = select_lambda_reg(
        lambdas,
        max_omega,
        v0_b,
        rt_b,
        v0_sigma=8.0,
        rt_sigma=0.5,
        v0_stage_a=v0_a,
    )
    assert chosen == pytest.approx(0.1)


def test_select_lambda_reg_returns_none_if_criteria_conflict():
    n_mock = 20
    lambdas = np.array([1.0, 10.0])
    max_omega = np.full((2, n_mock), 0.05)
    v0_b = np.full((2, n_mock), 200.0)
    rt_b = np.full((2, n_mock), 3.0)
    v0_a = np.full(n_mock, 200.0)
    max_omega[0] = 0.9  # fails Ω
    v0_b[1] = 100.0  # biased low
    assert (
        select_lambda_reg(
            lambdas,
            max_omega,
            v0_b,
            rt_b,
            v0_sigma=8.0,
            rt_sigma=0.5,
            v0_stage_a=v0_a,
        )
        is None
    )


def test_lambda_reg_campaign_delegates_to_calibrator():
    from kinuv.profiles import rotation as rot

    src = Path(rot.__file__).read_text(encoding="utf-8")
    chunk = src.split("def run_lambda_reg_campaign", 1)[1]
    assert "calibrate_lambda_reg" in chunk
    assert "NotImplementedError" not in chunk.split("def ", 1)[0]
