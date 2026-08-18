"""DEC-066-WEIGHT / VIS / ZEROMODEL: empirical s, diagonal χ², 066 loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kinuv.io.vis import (
    DEFAULT_NPZ,
    N_BIN,
    NATIVE_N_ROW,
    average_time_steps,
    bin_uv_plane,
    load_kgas066,
    optical_to_radio_kms,
)
from kinuv.likelihood.chi2 import chi2, chi2_zero, delta_chi2, empirical_s
from kinuv.response.spectral import s_theory

NPZ = DEFAULT_NPZ


@pytest.fixture(scope="session")
def kgas066():
    if not NPZ.is_file():
        pytest.skip("KILOGAS066.npz not on this machine")
    return load_kgas066(NPZ)


def test_chi2_zero_is_v_equals_zero_and_delta():
    rng = np.random.default_rng(7)
    vis = rng.standard_normal((20, 8)) + 1j * rng.standard_normal((20, 8))
    w = np.full(vis.shape, 2.5, dtype=np.float64)
    s = 0.8
    z = chi2_zero(vis, w, s)
    assert z == pytest.approx(chi2(vis, np.zeros_like(vis), w, s))
    assert chi2(vis, vis, w, s) == pytest.approx(0.0)
    model = 0.4 * vis
    c = chi2(vis, model, w, s)
    assert delta_chi2(c, z) == pytest.approx(z - c)
    assert delta_chi2(c, z) > 0.0


def test_chi2_accumulates_float64():
    vis = np.array([[1.0 + 1e-8j]], dtype=np.complex128)
    model = np.array([[1.0 + 0.0j]], dtype=np.complex128)
    w = np.array([[1.0]], dtype=np.float64)
    got = chi2(vis, model, w, 1.0)
    assert got == pytest.approx(1e-16, rel=1e-12)
    assert isinstance(got, float)


def test_empirical_s_on_synthetic_line_free():
    rng = np.random.default_rng(11)
    n_row, n_chan = 400, 10
    sigma = 0.4
    vis = (rng.standard_normal((n_row, n_chan)) + 1j * rng.standard_normal((n_row, n_chan))) * sigma
    w = np.full((n_row, n_chan), 1.0 / sigma**2)
    vis[:, 3:7] += 5.0
    mask = np.ones(n_chan, dtype=bool)
    mask[3:7] = False
    s = empirical_s(vis, w, mask)
    assert 0.3 < s < 1.5
    stat = np.mean((s * w[:, mask] * (np.abs(vis[:, mask]) ** 2)))
    assert stat == pytest.approx(2.0, rel=0.08)
    assert s != pytest.approx(0.5, rel=0.01)
    assert s != pytest.approx(s_theory(8), rel=0.01)


def test_time_average_and_uv_bin_weighted():
    u_m = np.array([10.0, 10.0])
    v_m = np.array([20.0, 20.0])
    vis = np.array([[1.0 + 0j, 3.0 + 0j], [3.0 + 0j, 1.0 + 0j]], dtype=np.complex128)
    w = np.ones((2, 2), dtype=np.float64)
    time_s = np.array([0.0, 0.5])
    bl = np.array([1, 1], dtype=np.int64)
    uo, vo, viso, wo = average_time_steps(u_m, v_m, vis, w, time_s, 1.0, bl)
    assert uo.shape[0] == 1
    np.testing.assert_allclose(viso[0], [2.0 + 0j, 2.0 + 0j])
    np.testing.assert_allclose(wo[0], [2.0, 2.0])
    ub, vb, visb, wb = bin_uv_plane(u_m, v_m, vis, w, 50.0)
    assert ub.shape[0] == 1
    np.testing.assert_allclose(visb[0], [2.0 + 0j, 2.0 + 0j])


def test_optical_to_radio_is_not_identity_at_066():
    v_opt = np.array([8034.0, 8536.0])
    v_rad = optical_to_radio_kms(v_opt)
    assert v_rad[0] < v_opt[0] - 150.0
    assert v_rad[1] < v_opt[1] - 150.0


@pytest.mark.skipif(not NPZ.is_file(), reason="KILOGAS066.npz not on this machine")
def test_real_066_line_free_s_applied(kgas066):
    d = kgas066
    assert 0.3 < d.s < 1.5
    assert d.s != pytest.approx(0.5, rel=0.02)
    assert d.s != pytest.approx(12.0 / 29.0, rel=0.02)
    w = d.weights[:, d.line_free_mask]
    v = d.vis[:, d.line_free_mask]
    mag2 = v.real.astype(np.float64) ** 2 + v.imag.astype(np.float64) ** 2
    good = w > 0.0
    mean = float(np.mean((w * mag2)[good], dtype=np.float64))
    assert 0.3 < 2.0 / mean < 1.5
    stat = d.s * mean
    assert stat == pytest.approx(2.0, rel=1e-10)
    z = chi2_zero(d.vis, d.weights, d.s)
    assert z > 0.0
    assert chi2(d.vis, d.vis, d.weights, d.s) == pytest.approx(0.0)


@pytest.mark.skipif(not NPZ.is_file(), reason="KILOGAS066.npz not on this machine")
def test_loader_records_n4_dv_and_collapses_rows(kgas066):
    d = kgas066
    assert d.n_bin == N_BIN == 4
    assert d.dv_kms == pytest.approx(4.0 * 1.270, rel=0.01)
    assert d.dv_kms == pytest.approx(5.08, rel=0.01)
    n_row, n_chan = d.vis.shape
    assert n_row < NATIVE_N_ROW // 5
    assert n_chan < 1920 // 4
    assert d.freqs_native.size == d.vel_native.size
    n_trim = d.freqs_native.size - 2 * d.n_guard
    assert n_trim // d.n_bin == n_chan
    assert d.line_free_mask.shape == (n_chan,)
    assert np.any(d.line_free_mask)
    assert d.phase_dir_rad.shape == (2,)
    # YAML obs_freq_range [224.148, 224.506] GHz clips; cube trim must be wider
    # on the approaching (high-freq) side than 224.506 GHz.
    assert d.freqs.max() > 224.506e9
    test_loader_records_n4_dv_and_collapses_rows.record = (
        n_row,
        n_chan,
        d.dv_kms,
        d.n_bin,
        d.s,
    )
