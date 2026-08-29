"""S1 metric unit tests (no npz). Inner slope formula and Hann+bin operator."""

from __future__ import annotations

import numpy as np
import pytest

from kinuv.diagnostics.s1 import (
    PIPELINE_KERNEL,
    assert_hann_bin_operator,
    inner_slope_arctan,
    r_eval_arcsec,
)
from kinuv.profiles.rotation import BMAJ_ICO_ARCSEC, arctan_vc
from kinuv.response.spectral import hann_then_bin


def test_s1_pipeline_kernel_is_hann_then_bin():
    assert assert_hann_bin_operator() == "hann_then_bin"
    assert PIPELINE_KERNEL == "hann_then_bin"
    assert hann_then_bin.__module__ == "kinuv.response.spectral"


def test_inner_slope_matches_finite_difference():
    v0, rt, r = 250.0, 0.25, r_eval_arcsec()
    analytic = inner_slope_arctan(v0, rt, r)
    h = 1.0e-4
    fd = float((arctan_vc(r + h, v0, rt) - arctan_vc(r - h, v0, rt)) / (2.0 * h))
    assert analytic == pytest.approx(fd, rel=1e-6)
    assert r == pytest.approx(0.25 * BMAJ_ICO_ARCSEC)
    assert inner_slope_arctan(v0, rt, r) > 4.0 * inner_slope_arctan(v0, 3.0, r)
