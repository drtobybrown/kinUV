"""DEC-066-PB primary-beam unit tests. Stationary-PB gate included."""

from __future__ import annotations

import numpy as np
import pytest

from kinuv.constants import ARCSEC_TO_RAD
from kinuv.response.primary_beam import (
    D_ANT_M,
    attenuate,
    fwhm_pb_arcsec,
    primary_beam,
    translate_then_attenuate,
)
from kinuv.template.fourier_shift import fourier_shift
from kinuv.template.resample import sky_axes
from kinuv.template.wiener import flux_weighted_centroid
from kinuv.transforms import dft_numpy

NU = 224.5e9
CELL = 0.4
NPIX = 151


def _axes(n=NPIX, cell=CELL):
    return sky_axes(n, cell), sky_axes(n, cell)


def _disk(radius, n=NPIX, cell=CELL):
    x, y = _axes(n, cell)
    X, Y = np.meshgrid(x, y)
    return (np.hypot(X, Y) <= radius).astype(np.float64)


def _gaussian(sx, sy, n=NPIX, cell=CELL):
    x, y = _axes(n, cell)
    X, Y = np.meshgrid(x, y)
    return np.exp(-0.5 * ((X / sx) ** 2 + (Y / sy) ** 2))


def _dft(image, u_m, v_m, nu_hz, cell=CELL):
    n = image.shape[0]
    x, y = _axes(n, cell)
    X, Y = np.meshgrid(x, y)
    l = (X * ARCSEC_TO_RAD).ravel()
    m = (Y * ARCSEC_TO_RAD).ravel()
    d_omega = (cell * ARCSEC_TO_RAD) ** 2
    s = (image.ravel() * d_omega)[:, None]
    return dft_numpy(l, m, s, np.asarray(u_m), np.asarray(v_m), np.asarray(nu_hz))


def test_fwhm_25p9_not_56p6_over_ghz():
    fwhm = fwhm_pb_arcsec(NU, D_ANT_M)
    assert fwhm == pytest.approx(25.9, rel=0.01)
    wrong = 56.6 / (NU / 1e9)
    assert abs(fwhm - wrong) > 20.0


def test_a_from_phase_centre_not_dxdy():
    x, y = _axes()
    a = primary_beam(x, y, NU, x_phase_arcsec=0.0, y_phase_arcsec=0.0)
    X, Y = np.meshgrid(x, y)
    r = np.hypot(X, Y)
    at_edge = a.ravel()[np.argmin(np.abs(r.ravel() - 7.5))]
    want = np.exp(-4.0 * np.log(2.0) * 7.5**2 / fwhm_pb_arcsec(NU) ** 2)
    assert at_edge == pytest.approx(want, rel=0.02)
    # peak stays on the phase centre even if a galaxy offset is in play
    iy, ix = np.unravel_index(np.argmax(a), a.shape)
    assert abs(x[ix]) < CELL and abs(y[iy]) < CELL


def test_uniform_disk_short_baseline_suppression():
    disk = _disk(15.0)
    x, y = _axes()
    att = attenuate(disk, x, y, NU)
    u_m = np.array([0.0, 12.0, 18.0, 25.0])
    v_m = np.zeros_like(u_m)
    freqs = np.array([NU])
    v0 = _dft(disk, u_m, v_m, freqs)
    vp = _dft(att, u_m, v_m, freqs)
    ratio = np.abs(vp) / np.abs(v0)
    # r=15″ disk; edge A(7.5″)≈0.79. Short baselines show ~20%+ suppression.
    assert np.all(ratio < 0.85)
    assert ratio[0, 0] < 0.80
    assert ratio[0, 0] > 0.55


def test_stationary_pb_gate_shift_then_a():
    img = _gaussian(3.0, 2.5)
    dx, dy = 1.0, 1.0
    sky, att = translate_then_attenuate(img, dx, dy, CELL, NU)
    cx, cy = flux_weighted_centroid(att, CELL)
    assert cx > 0.4 and cy > 0.4
    x, y = _axes()
    a = primary_beam(x, y, NU)
    implied = np.divide(att, sky, out=np.zeros_like(att), where=np.abs(sky) > 1e-8 * sky.max())
    core = np.abs(sky) > 1e-3 * sky.max()
    assert np.max(np.abs(implied[core] - a[core])) < 0.02
    iy, ix = np.unravel_index(np.argmax(a), a.shape)
    assert abs(x[ix]) < CELL and abs(y[iy]) < CELL


def test_ramp_after_a_control_fails_stationary_pb():
    """A then Fourier shift (≡ visibility ramp after A) drags the PB."""
    img = _gaussian(3.0, 2.5)
    dx, dy = 1.0, 1.0
    x, y = _axes()
    a = primary_beam(x, y, NU)
    wrong = fourier_shift(img * a, dx, dy, CELL)
    sky = fourier_shift(img, dx, dy, CELL)
    implied = np.divide(
        wrong, sky, out=np.zeros_like(wrong), where=np.abs(sky) > 1e-8 * sky.max()
    )
    iy, ix = np.unravel_index(np.argmax(implied), implied.shape)
    # envelope followed the galaxy — fails "PB stays on phase centre"
    assert abs(x[ix] - dx) < 0.4
    assert abs(y[iy] - dy) < 0.4
    _, correct = translate_then_attenuate(img, dx, dy, CELL, NU)
    iy_c, ix_c = np.unravel_index(np.argmax(correct / np.maximum(sky, 1e-30)), sky.shape)
    assert abs(x[ix_c]) < CELL and abs(y[iy_c]) < CELL
    assert np.hypot(x[ix] - x[ix_c], y[iy] - y[iy_c]) > 0.5
