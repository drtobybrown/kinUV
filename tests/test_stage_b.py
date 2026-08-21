"""Stage B ring MAP helpers (066-12). No visibility campaign here."""

import numpy as np
import pytest

from kinuv.forward.model import VSYS_SEED_KM_S, los_velocity
from kinuv.geometry import inclination_rad, pa_seed_rad, sky_to_galaxy
from kinuv.infer.stage_b import recover_arctan_from_rings
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
