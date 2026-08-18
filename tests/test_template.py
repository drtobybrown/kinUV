"""DEC-066-SB Wiener unit tests. Synthetic tests do not need the Ico FITS."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kinuv.constants import ARCSEC_TO_RAD, C_LIGHT_M_S, F_REST_CO21_HZ
from kinuv.template.fftpad import default_pad_n
from kinuv.template.resample import resample_flux_conserving, sky_axes
from kinuv.template.wiener import (
    SignedFluxError,
    assert_signed_flux_gate,
    clip_if_centroid_stable,
    convolve_restoring_beam,
    ico_to_template,
    k_to_jy_per_beam,
    restoring_beam_ft,
    wiener_deconvolve,
)
from kinuv.transforms import dft_numpy

BMAJ, BMIN, BPA = 1.30, 1.18, -18.3
CELL = 0.4
NPIX = 135
NU = 224.3e9
ICO_FITS = Path(
    "/Users/thbrown/kilogas/analysis/kinms_test/kgas066/KGAS66_Ico_K_kms-1.fits"
)


def _grid(n=NPIX, cell=CELL):
    x = sky_axes(n, cell)
    y = sky_axes(n, cell)
    return np.meshgrid(x, y)


def _gaussian(sx, sy, amp=1.0, x0=0.0, y0=0.0, n=NPIX, cell=CELL):
    X, Y = _grid(n, cell)
    return amp * np.exp(-0.5 * (((X - x0) / sx) ** 2 + ((Y - y0) / sy) ** 2))


def _exponential(re, n=NPIX, cell=CELL):
    X, Y = _grid(n, cell)
    return np.exp(-np.hypot(X, Y) / re)


def _mask_like_ico(n=NPIX, cell=CELL, n_pix=1709):
    X, Y = _grid(n, cell)
    r0 = cell * np.sqrt(n_pix / np.pi)
    return np.hypot(X, Y) < r0


def _fit_re(image, mask, r_min=1.5, r_max=8.0, cell=CELL):
    X, Y = _grid(image.shape[0], cell)
    r = np.hypot(X, Y)
    sel = mask & (image > 0) & (r >= r_min) & (r <= r_max)
    a = np.vstack([np.ones(int(sel.sum())), -r[sel]]).T
    coeff, *_ = np.linalg.lstsq(a, np.log(image[sel]), rcond=None)
    return 1.0 / coeff[1]


def _dft(image, u_m, v_m, nu_hz, cell=CELL):
    n = image.shape[0]
    x = sky_axes(n, cell)
    y = sky_axes(n, cell)
    X, Y = np.meshgrid(x, y)
    l = (X * ARCSEC_TO_RAD).ravel()
    m = (Y * ARCSEC_TO_RAD).ravel()
    d_omega = (cell * ARCSEC_TO_RAD) ** 2
    s = np.repeat((image.ravel() * d_omega)[:, None], np.size(nu_hz), axis=1)
    return dft_numpy(l, m, s, np.asarray(u_m), np.asarray(v_m), np.asarray(nu_hz))


def _inside_taper(u_m, v_m, nu_hz, taper=0.05):
    scale = float(nu_hz) / C_LIGHT_M_S
    u_cyc = np.asarray(u_m) * scale * ARCSEC_TO_RAD
    v_cyc = np.asarray(v_m) * scale * ARCSEC_TO_RAD
    beam = restoring_beam_ft(u_cyc, v_cyc, BMAJ, BMIN, BPA)
    return np.abs(beam) >= taper, beam


def test_k_to_jy_recomputed_not_hardcoded():
    bmaj, bmin = 1.2948374156468399, 1.17954483683112
    got = k_to_jy_per_beam(1.0, NU, bmaj, bmin)
    rest = k_to_jy_per_beam(1.0, F_REST_CO21_HZ, bmaj, bmin)
    assert abs(got - 0.063) > 1e-5
    assert abs(1.0 - (NU / F_REST_CO21_HZ) ** 2) > 0.05
    assert got / rest == pytest.approx((NU / F_REST_CO21_HZ) ** 2, rel=1e-12)


def test_gaussian_fake_long_baseline_matches_unconvolved_not_restored():
    truth = _gaussian(2.0, 1.5, amp=5.0)
    restored = convolve_restoring_beam(truth, CELL, BMAJ, BMIN, BPA)
    wiener = wiener_deconvolve(restored, CELL, BMAJ, BMIN, BPA, k_wiener=1e-16)
    rng = np.random.default_rng(66)
    b = 10 ** rng.uniform(np.log10(80.0), np.log10(220.0), 60)
    th = rng.uniform(0.0, 2.0 * np.pi, 60)
    u_m, v_m = b * np.cos(th), b * np.sin(th)
    inside, _ = _inside_taper(u_m, v_m, NU)
    assert inside.sum() > 10
    freqs = np.array([NU])
    v_true = _dft(truth, u_m[inside], v_m[inside], freqs)
    v_rest = _dft(restored, u_m[inside], v_m[inside], freqs)
    v_w = _dft(wiener, u_m[inside], v_m[inside], freqs)
    err_true = np.abs(v_w - v_true) / np.maximum(np.abs(v_true), 1e-12)
    err_rest = np.abs(v_w - v_rest) / np.maximum(np.abs(v_true), 1e-12)
    assert err_true.max() < 0.05
    assert err_true.mean() < 0.3 * err_rest.mean()
    assert err_rest.mean() > 0.05


def test_exponential_disk_scale_length_unbiased_no_ringing():
    re = 7.4
    truth = _exponential(re)
    restored = convolve_restoring_beam(truth, CELL, BMAJ, BMIN, BPA)
    wiener = wiener_deconvolve(restored, CELL, BMAJ, BMIN, BPA, k_wiener=1e-16)
    mask = _mask_like_ico()
    re_hat = _fit_re(wiener, mask)
    assert abs(re_hat - re) / re < 0.05
    X, Y = _grid()
    r = np.hypot(X, Y)
    annulus = mask & (r > 7.5) & (r < 9.2)
    model = np.exp(-r / re_hat) * (wiener[mask & (r > 1.5)].max())
    # match amplitude at r~4"
    core = mask & (r > 3.5) & (r < 4.5)
    model *= np.median(wiener[core]) / np.median(model[core])
    resid = wiener[annulus] - model[annulus]
    assert np.std(resid) < 0.12 * np.median(np.abs(model[annulus]))
    # even/odd pixel difference would spike if the taper/mask rings
    even = resid[::2]
    odd = resid[1::2]
    n = min(even.size, odd.size)
    assert abs(even[:n].mean() - odd[:n].mean()) < 0.05 * np.median(np.abs(model[annulus]))


def test_correlated_dirty_beam_noise_unbiased_scale_length():
    re = 7.4
    truth = _exponential(re)
    restored = convolve_restoring_beam(truth, CELL, BMAJ, BMIN, BPA)
    rng = np.random.default_rng(20260818)
    n = restored.shape[0]
    uy = np.fft.fftfreq(n, d=CELL)[:, None]
    ux = np.fft.fftfreq(n, d=CELL)[None, :]
    q = np.hypot(ux, uy)
    cov = (q < 0.35) & (rng.random((n, n)) < 0.35)
    cov[0, 0] = True
    noise_ft = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) * cov
    residual = np.fft.ifft2(noise_ft).real
    residual *= 0.03 * restored.max() / np.std(residual)
    dirty = restored + residual
    sig = float(np.std(np.concatenate([
        dirty[:12, :12].ravel(),
        dirty[:12, -12:].ravel(),
        dirty[-12:, :12].ravel(),
        dirty[-12:, -12:].ravel(),
    ])))
    k = (sig / dirty.max()) ** 2
    wiener = wiener_deconvolve(dirty, CELL, BMAJ, BMIN, BPA, k_wiener=k)
    re_hat = _fit_re(np.maximum(wiener, 0.0), _mask_like_ico())
    assert abs(re_hat - re) / re < 0.12


def test_pad_gate_wraparound_drops_below_taper_floor():
    truth = _gaussian(1.6, 1.6, amp=4.0, y0=-18.0)
    restored = convolve_restoring_beam(truth, CELL, BMAJ, BMIN, BPA, pad_n=512)
    unpadded = wiener_deconvolve(
        restored, CELL, BMAJ, BMIN, BPA, k_wiener=1e-16, pad_n=NPIX
    )
    padded = wiener_deconvolve(
        restored, CELL, BMAJ, BMIN, BPA, k_wiener=1e-16, pad_n=default_pad_n(NPIX)
    )
    assert default_pad_n(NPIX) >= 2 * NPIX
    edge_u = np.max(np.abs(unpadded[:5, :]))
    edge_p = np.max(np.abs(padded[:5, :]))
    peak = padded.max()
    assert edge_u > edge_p
    assert edge_p / peak < 0.05


def test_flux_conservation_resample():
    img = _gaussian(2.5, 2.0, amp=3.0, n=81, cell=0.4)
    cell_out = 0.1
    out = resample_flux_conserving(img, 0.4, cell_out)
    flux_in = img.sum() * 0.4**2
    flux_out = out.sum() * cell_out**2
    assert abs(flux_out / flux_in - 1.0) < 1e-4


def test_centroid_gate_and_asymmetric_mask_skips_clip():
    truth = _gaussian(2.0, 1.5, amp=5.0)
    restored = convolve_restoring_beam(truth, CELL, BMAJ, BMIN, BPA)
    signed = wiener_deconvolve(restored, CELL, BMAJ, BMIN, BPA, k_wiener=1e-16)
    clipped, did = clip_if_centroid_stable(signed, CELL)
    assert did is True
    from kinuv.template.wiener import flux_weighted_centroid

    cx_s, cy_s = flux_weighted_centroid(signed, CELL)
    cx_c, cy_c = flux_weighted_centroid(clipped, CELL)
    assert np.hypot(cx_c - cx_s, cy_c - cy_s) < 0.01

    X, Y = _grid()
    # strongly asymmetric mask covering a one-sided negative lobe
    control = signed - 8.0 * np.exp(
        -0.5 * (((X + 8.0) / 2.0) ** 2 + ((Y - 6.0) / 2.0) ** 2)
    )
    mask = X > -4.0
    gated, did_ctrl = clip_if_centroid_stable(control, CELL, mask=mask)
    assert did_ctrl is False
    assert np.any(gated < 0.0)


def test_signed_flux_dominance_abort():
    mask = _mask_like_ico()
    pos = _gaussian(3.0, 3.0, amp=1.0)
    neg = -1.2 * pos
    with pytest.raises(SignedFluxError):
        assert_signed_flux_gate(neg, mask)
    mixed = pos.copy()
    mixed[:, : NPIX // 2] = -pos[:, : NPIX // 2]
    with pytest.raises(SignedFluxError):
        assert_signed_flux_gate(mixed, mask)
    ico_to_template(
        np.maximum(pos, 0.0),
        CELL,
        NU,
        BMAJ,
        BMIN,
        BPA,
        mask=mask,
        units="Jy",
        k_wiener=1e-16,
    )


@pytest.mark.skipif(not ICO_FITS.is_file(), reason="Ico FITS not on this machine")
def test_live_ico_optional():
    from astropy.io import fits

    with fits.open(ICO_FITS) as hdul:
        h = hdul[0].header
        data = np.array(hdul[0].data, dtype=np.float64)
    assert data.shape == (135, 135)
    assert int(np.isfinite(data).sum()) == 1709
    bmaj = float(h["BMAJ"]) * 3600.0
    bmin = float(h["BMIN"]) * 3600.0
    bpa = float(h["BPA"])
    cell = abs(float(h["CDELT2"])) * 3600.0
    tmpl = ico_to_template(
        data,
        cell,
        NU,
        bmaj,
        bmin,
        bpa,
        sigma_empty=0.02 * np.nanmax(data),
        model_cell_arcsec=0.1,
    )
    d_omega = tmpl.cell_arcsec**2
    assert abs(tmpl.sb.sum() * d_omega - 1.0) < 1e-8
    factor = k_to_jy_per_beam(1.0, NU, bmaj, bmin)
    assert abs(factor - 0.063) > 1e-5
