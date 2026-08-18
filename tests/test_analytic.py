"""Closed-form visibility oracles. Transform error must be < 1e-7 (066-1)."""

import numpy as np
from scipy.special import j0

from kinuv.constants import ARCSEC_TO_RAD
from kinuv.transforms import dft_numpy


def analytic_gaussian_vis(u_lam, v_lam, sx_arcsec, sy_arcsec, flux):
    sx = sx_arcsec * ARCSEC_TO_RAD
    sy = sy_arcsec * ARCSEC_TO_RAD
    return flux * np.exp(-2.0 * np.pi**2 * ((sx * u_lam) ** 2 + (sy * v_lam) ** 2))


def gaussian_nodes(sx_arcsec, sy_arcsec, flux, n=201):
    t, w = np.polynomial.hermite.hermgauss(n)
    x = np.sqrt(2.0) * sx_arcsec * t
    y = np.sqrt(2.0) * sy_arcsec * t
    wx = w / np.sqrt(np.pi)
    wy = w / np.sqrt(np.pi)
    X, Y = np.meshgrid(x, y, indexing="ij")
    W = np.outer(wx, wy) * flux
    return X.ravel() * ARCSEC_TO_RAD, Y.ravel() * ARCSEC_TO_RAD, W.ravel()


def test_gaussian_visibility_matches_closed_form(uv_sampling, freqs):
    u_m, v_m = uv_sampling
    sx, sy, flux = 1.2, 0.8, 5.0
    l, m, w = gaussian_nodes(sx, sy, flux)
    strengths = np.repeat(w[:, None], len(freqs), axis=1)
    got = dft_numpy(l, m, strengths, u_m, v_m, freqs)
    from kinuv.constants import C_LIGHT_M_S

    scale = np.asarray(freqs) / C_LIGHT_M_S
    u_lam = np.asarray(u_m)[:, None] * scale
    v_lam = np.asarray(v_m)[:, None] * scale
    want = analytic_gaussian_vis(u_lam, v_lam, sx, sy, flux)
    err = np.abs(got - want).max() / flux
    assert err < 1e-7, f"Gaussian DFT error {err:.3e}"


def test_thin_ring_matches_j0(uv_sampling, freqs):
    """Face-on thin ring: V = F J0(2π q R)."""
    u_m, v_m = uv_sampling
    r_arcsec, flux, n_phi = 4.0, 3.0, 4096
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    l = r_arcsec * ARCSEC_TO_RAD * np.cos(phi)
    m = r_arcsec * ARCSEC_TO_RAD * np.sin(phi)
    strengths = np.full((n_phi, len(freqs)), flux / n_phi)
    got = dft_numpy(l, m, strengths, u_m, v_m, freqs)
    from kinuv.constants import C_LIGHT_M_S

    scale = np.asarray(freqs) / C_LIGHT_M_S
    q = np.hypot(np.asarray(u_m)[:, None] * scale, np.asarray(v_m)[:, None] * scale)
    want = flux * j0(2.0 * np.pi * q * r_arcsec * ARCSEC_TO_RAD)
    err = np.abs(got - want).max() / flux
    assert err < 1e-7, f"thin-ring J0 error {err:.3e}"


def test_zero_spacing_equals_total_flux(freqs):
    sx, sy, flux = 1.5, 1.0, 7.3
    l, m, w = gaussian_nodes(sx, sy, flux)
    strengths = np.repeat(w[:, None], 1, axis=1)
    got = dft_numpy(l, m, strengths, np.zeros(1), np.zeros(1), freqs[:1])
    assert abs(got[0, 0].real - flux) < 1e-12 * flux
    assert abs(got[0, 0].imag) < 1e-12 * flux


def double_horn(v_kms, vmax):
    """Normalised 1/(π sqrt(vmax² − v²)) for |v|<vmax."""
    v = np.asarray(v_kms, dtype=np.float64)
    out = np.zeros_like(v)
    inside = np.abs(v) < vmax
    out[inside] = 1.0 / (np.pi * np.sqrt(vmax**2 - v[inside] ** 2))
    return out


def test_thin_ring_double_horn_at_phase_centre():
    """At (u,v)=0 a rotating ring's spectrum is the double-horn (DEC-066-OSCMETRIC)."""
    vmax, flux = 120.0, 2.5
    v = np.linspace(-200.0, 200.0, 401)
    n_phi = 2000
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    v_los = vmax * np.cos(phi)
    dv = v[1] - v[0]
    hist = np.zeros_like(v)
    for vl in v_los:
        hist += np.exp(-0.5 * ((v - vl) / (0.4 * dv)) ** 2)
    hist *= flux / hist.sum() / dv
    want = flux * double_horn(v, vmax)
    # histogram of finite samples approximates the horn away from the edges
    mid = np.abs(v) < 0.8 * vmax
    rel = np.abs(hist[mid] - want[mid]).max() / (want[mid].max())
    assert rel < 0.05
    assert abs(double_horn(np.array([0.0]), vmax)[0] - 1.0 / (np.pi * vmax)) < 1e-12
    assert np.all(double_horn(np.array([-vmax - 1.0, vmax + 1.0]), vmax) == 0.0)
