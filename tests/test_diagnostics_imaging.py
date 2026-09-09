"""Image-plane diagnostics: moments, spectral rebin, radio↔optical."""

from __future__ import annotations

import numpy as np
import pytest

from kinuv.diagnostics.imaging import (
    flux_weighted_velocity,
    jy_per_pixel_to_k,
    masked_moments,
    offset_world,
    radio_header_velocity_kms,
    rebin_spectrum,
    restoring_beam_kernel,
    spectral_axis_kms,
    spectral_wcs_report,
)
from kinuv.io.vis import optical_to_radio_kms, radio_to_optical_kms
from kinuv.template.wiener import k_to_jy_per_beam
from kinuv.validation.s4 import topo_radio_to_lsrk_radio


def test_radio_optical_roundtrip():
    v = np.array([8000.0, 8299.563, 8500.0])
    assert np.allclose(optical_to_radio_kms(radio_to_optical_kms(v)), v)
    assert np.allclose(radio_to_optical_kms(optical_to_radio_kms(v)), v)


def test_native_radio_pvd_curve_transforms_each_endpoint_to_optical_lsrk():
    from kinuv.diagnostics.delivery import (
        _native_radio_curve_on_optical_lsrk_offsets,
    )

    offset = np.array([-4.0, 0.0, 4.0])
    radius = np.array([0.0, 4.0])
    projected = np.array([0.0, 180.60325616735634])
    native_vsys = 8099.2759442714005
    correction = 10.9850680313608
    actual = _native_radio_curve_on_optical_lsrk_offsets(
        offset, radius, projected, native_vsys, correction
    )
    expected = radio_to_optical_kms(
        topo_radio_to_lsrk_radio(
            native_vsys + np.array([-projected[-1], 0.0, projected[-1]]),
            correction,
        )
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-10)

    reporting_vsys = radio_to_optical_kms(
        topo_radio_to_lsrk_radio(native_vsys, correction)
    )
    assert actual[1] == pytest.approx(reporting_vsys)
    assert actual[2] - reporting_vsys != pytest.approx(projected[-1])
    assert reporting_vsys - actual[0] != pytest.approx(projected[-1])
    assert actual[2] - reporting_vsys != pytest.approx(reporting_vsys - actual[0])


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


def test_offset_world_east_increases_ra():
    ra, dec = offset_world(345.0, 13.0, 1.0, 0.0)
    assert ra > 345.0
    ra2, dec2 = offset_world(345.0, 13.0, 0.0, 2.0)
    assert dec2 > 13.0
    assert np.isclose(ra2, 345.0)


@pytest.mark.parametrize("pa_deg", [35.0, 90.0, 199.73, 289.73])
def test_pv_slit_follows_celestial_position_angle(pa_deg):
    """A linear east/north field has its physical derivative along the slit."""
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS
    import astropy.units as u
    from kinuv.diagnostics.imaging import pv_diagram

    header = _toy_header()
    yy, xx = np.mgrid[:21, :21]
    sky = WCS(header).celestial.pixel_to_world(xx, yy)
    origin = SkyCoord(345.0 * u.deg, 13.0 * u.deg)
    east, north = origin.spherical_offsets_to(sky)
    field = 2.0 * east.arcsec + 3.0 * north.arcsec
    cube = np.broadcast_to(field, (3, 21, 21)).copy()
    pv, offsets = pv_diagram(cube, header, 345.0, 13.0, pa_deg, 4.0, 0.3)
    expected = 2.0 * np.sin(np.deg2rad(pa_deg)) + 3.0 * np.cos(np.deg2rad(pa_deg))
    np.testing.assert_allclose(np.polyfit(offsets, pv[0], 1)[0], expected, atol=1e-5)


def test_spectral_axis_kms_divides_si_metres():
    from astropy.io import fits

    h = fits.Header()
    h["NAXIS3"] = 3
    h["CRPIX3"] = 1.0
    h["CRVAL3"] = 7_820_168.28
    h["CDELT3"] = 1269.925445
    h["CUNIT3"] = "m/s"
    v = spectral_axis_kms(h)
    np.testing.assert_allclose(v[0], 7820.16828)
    np.testing.assert_allclose(np.diff(v), 1.269925445)


def test_radio_header_to_optical_roundtrip_at_066():
    from astropy.io import fits

    h = fits.Header()
    h["NAXIS3"] = 2
    h["CRPIX3"] = 1.0
    h["CRVAL3"] = 8098.773150512066
    h["CDELT3"] = 5.08
    h["CUNIT3"] = "km/s"
    h["CTYPE3"] = "VRAD"
    v_opt = radio_header_velocity_kms(h)
    np.testing.assert_allclose(optical_to_radio_kms(v_opt[0]), 8098.773150512066)


def test_rebin_delta_lands_on_matching_optical_channel():
    """A native channel whose optical velocity matches cube chan 0 stays in chan 0."""
    v_model = np.array([8029.62, 8036.32, 8044.35, 8051.05])
    v_data = np.array([8044.32, 8054.77, 8065.22])
    cube = np.zeros((4, 2, 2))
    cube[2] = 1.0
    out = rebin_spectrum(cube, v_model, v_data, 10.45)
    spec = out[:, 0, 0]
    assert int(np.argmax(spec)) == 0
    assert spec[0] > spec[1]


def test_flux_weighted_velocity_centroid():
    spec = np.array([0.0, 1.0, 1.0, 0.0])
    vel = np.array([10.0, 20.0, 30.0, 40.0])
    assert flux_weighted_velocity(spec, vel) == pytest.approx(25.0)
    assert np.isnan(flux_weighted_velocity(np.zeros(3), np.arange(3.0)))


def test_spectral_wcs_report_reads_restfrq_and_specsys():
    from astropy.io import fits

    h = _toy_header()
    h["RESTFRQ"] = 230.538e9
    h["SPECSYS"] = "LSRK"
    rec = spectral_wcs_report(h, label="toy")
    assert rec["label"] == "toy"
    assert rec["ctype3"] == "VOPT"
    assert rec["specsys"] == "LSRK"
    assert rec["restfrq_hz"] == pytest.approx(230.538e9)
    assert rec["vel_chan0_kms"] == pytest.approx(8200.0)
    assert rec["header_axis"] == "optical"
    assert rec["vel_optical_chan0_kms"] == pytest.approx(8200.0)


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
