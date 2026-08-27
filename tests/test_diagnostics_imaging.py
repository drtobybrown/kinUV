"""Image-plane diagnostics: moments, spectral rebin, radio↔optical."""

from __future__ import annotations

import numpy as np

from kinuv.diagnostics.imaging import (
    jy_per_pixel_to_k,
    masked_moments,
    offset_world,
    rebin_spectrum,
    restoring_beam_kernel,
)
from kinuv.io.vis import optical_to_radio_kms, radio_to_optical_kms
from kinuv.template.wiener import k_to_jy_per_beam


def test_radio_optical_roundtrip():
    v = np.array([8000.0, 8299.563, 8500.0])
    assert np.allclose(optical_to_radio_kms(radio_to_optical_kms(v)), v)
    assert np.allclose(radio_to_optical_kms(optical_to_radio_kms(v)), v)


def test_rebin_spectrum_preserves_constant():
    cube = np.full((11, 3, 3), 2.5)
    v_in = np.linspace(8000.0, 8100.0, 11)
    v_out = np.array([8040.0, 8060.0, 8080.0])
    out = rebin_spectrum(cube, v_in, v_out, 20.0)
    assert out.shape == (3, 3, 3)
    assert np.allclose(out, 2.5)


def test_rebin_spectrum_nearest_if_no_overlap():
    cube = np.arange(5, dtype=np.float64)[:, None, None] * np.ones((5, 2, 2))
    v_in = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    out = rebin_spectrum(cube, v_in, np.array([100.0]), 1.0)
    assert np.allclose(out, 4.0)


def test_masked_moments_single_channel():
    cube = np.zeros((3, 4, 4))
    cube[1, 1:3, 1:3] = 2.0
    mask = cube > 0
    vel = np.array([10.0, 20.0, 30.0])
    m0, m1, m2 = masked_moments(cube, vel, mask, dv_kms=5.0)
    assert m0[1, 1] == 10.0
    assert m1[1, 1] == 20.0
    assert m2[1, 1] == 0.0
    assert np.isnan(m0[0, 0])


def test_jy_per_pixel_to_k_inverts_k_to_jy_per_beam():
    bmaj, bmin, cell = 1.0, 0.8, 0.25
    nu = 224.3e9
    omega_beam = np.pi * bmaj * bmin / (4.0 * np.log(2.0))
    jy_pix = 0.01
    t = jy_per_pixel_to_k(jy_pix, cell, nu, bmaj, bmin)
    jy_beam = jy_pix * (omega_beam / cell**2)
    assert np.isclose(k_to_jy_per_beam(t, nu, bmaj, bmin), jy_beam)


def test_restoring_beam_kernel_unit_sum():
    k = restoring_beam_kernel(0.3, 1.04, 0.95, -44.8, -8.3e-5)
    assert k.ndim == 2
    assert np.isclose(k.sum(), 1.0)


def test_offset_world_east_decreases_ra():
    ra, dec = offset_world(345.0, 13.0, 1.0, 0.0)
    assert ra < 345.0
    ra2, dec2 = offset_world(345.0, 13.0, 0.0, 2.0)
    assert dec2 > 13.0
    assert np.isclose(ra2, 345.0)


def _toy_header():
    from astropy.io import fits

    h = fits.Header()
    h["NAXIS"] = 3
    h["NAXIS1"] = 21
    h["NAXIS2"] = 21
    h["NAXIS3"] = 3
    h["CRPIX1"] = 11.0
    h["CRPIX2"] = 11.0
    h["CRPIX3"] = 1.0
    h["CRVAL1"] = 345.0
    h["CRVAL2"] = 13.0
    h["CRVAL3"] = 8200.0
    h["CDELT1"] = -8.333333333334e-5
    h["CDELT2"] = 8.333333333334e-5
    h["CDELT3"] = 10.0
    h["CTYPE1"] = "RA---SIN"
    h["CTYPE2"] = "DEC--SIN"
    h["CTYPE3"] = "VOPT"
    h["CUNIT1"] = "deg"
    h["CUNIT2"] = "deg"
    h["CUNIT3"] = "km s-1"
    return h


def test_pv_positive_offset_is_receding_north():
    from kinuv.diagnostics.imaging import pv_diagram

    hdr = _toy_header()
    cube = np.zeros((3, 21, 21))
    # 0.3" pixels; +4" north of centre is +13 rows in y (0-index 10+13=23 out of range)
    # +3" = 10 pixels → y=20 too far. +2.1" = 7 pixels → y=17
    cube[1, 10 + 7, 10] = 5.0
    pv, off = pv_diagram(cube, hdr, 345.0, 13.0, pa_deg=0.0, length_arcsec=6.0, width_arcsec=0.4)
    i_peak = int(np.nanargmax(pv[1]))
    assert off[i_peak] > 0.5

