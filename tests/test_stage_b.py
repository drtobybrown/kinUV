"""Stage B ring MAP helpers (066-12). No visibility campaign here."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kinuv.forward.model import VSYS_SEED_KM_S, los_velocity
from kinuv.geometry import inclination_rad, pa_seed_rad, sky_to_galaxy
from kinuv.infer.stage_b import predict_binned, recover_arctan_from_rings
from kinuv.profiles.rotation import (
    CALIBRATION_RT_ARCSEC,
    CALIBRATION_V0_KM_S,
    DISK_RADIUS_ARCSEC,
    ring_vc,
    rings_from_arctan,
    uniform_knot_radii,
)


def test_recover_arctan_from_truth_rings():
    knots = uniform_knot_radii(7, r_last_arcsec=DISK_RADIUS_ARCSEC)
    v_k = rings_from_arctan(knots, CALIBRATION_V0_KM_S, CALIBRATION_RT_ARCSEC)
    v0, rt = recover_arctan_from_rings(knots, v_k)
    assert abs(v0 - CALIBRATION_V0_KM_S) < 15.0
    assert abs(rt - CALIBRATION_RT_ARCSEC) < 0.8


def test_los_velocity_knots_match_ring_vc():
    knots = uniform_knot_radii(7)
    v_k = rings_from_arctan(knots, CALIBRATION_V0_KM_S, CALIBRATION_RT_ARCSEC)
    pa = pa_seed_rad()
    i = inclination_rad()
    x_e, y_n = 1.2, 0.4
    v_ring = los_velocity(
        x_e,
        y_n,
        pa,
        i,
        VSYS_SEED_KM_S,
        r_knots_arcsec=knots,
        v_knots_kms=v_k,
    )
    xg, yg = sky_to_galaxy(x_e, y_n, pa, i)
    r = float(np.hypot(xg, yg))
    vc = float(ring_vc(r, knots, v_k))
    want = VSYS_SEED_KM_S + vc * np.sin(i) * (xg / r)
    assert v_ring == pytest.approx(float(want), rel=1e-10)
    v_a = los_velocity(
        x_e,
        y_n,
        pa,
        i,
        VSYS_SEED_KM_S,
        CALIBRATION_V0_KM_S,
        CALIBRATION_RT_ARCSEC,
    )
    assert v_ring != pytest.approx(v_a)


def test_run_stage_b_map_stores_residual_omega():
    from kinuv.infer import stage_b as sb

    src = Path(sb.__file__).read_text(encoding="utf-8")
    chunk = src.split("def run_stage_b_map", 1)[1]
    assert "om = omega_residual(v_k, v_init, data.dv_kms)" in chunk
    assert "max_omega=float(np.max(om))" in chunk


def test_stage_b_uses_caller_inclination(monkeypatch):
    seen = {}

    def fake_predict(*_args, **kwargs):
        seen["i_rad"] = kwargs["i_rad"]
        return np.zeros((2, 6), dtype=np.complex128)

    def fake_bin(*_args, **_kwargs):
        return np.zeros((2, 1), dtype=np.complex128)

    monkeypatch.setattr("kinuv.infer.stage_b.predict_vis", fake_predict)
    monkeypatch.setattr("kinuv.infer.stage_b.hann_then_bin", fake_bin)
    data = SimpleNamespace(
        n_guard=1,
        u_m=np.zeros(2),
        v_m=np.zeros(2),
        freqs_native=np.arange(6.0),
        vel_native=np.arange(6.0),
        weights_native=np.ones((2, 6)),
        n_bin=4,
        vis=np.zeros((2, 1), dtype=np.complex128),
    )
    nuisance = {
        "flux": 1.0,
        "pa_deg": 10.0,
        "vsys_kms": 1000.0,
        "dx_arcsec": 0.0,
        "dy_arcsec": 0.0,
        "gas_sigma_kms": 8.0,
    }
    predict_binned(data, nuisance, np.ones((2, 2)), object(), i_rad=0.37)
    assert seen["i_rad"] == pytest.approx(0.37)
