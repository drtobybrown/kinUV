"""Opt-in m=2 SB helpers. No FITS. Do not change load_sb_template."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from kinuv.forward.sb import (
    apply_m2,
    axisymmetrise_template,
    galaxy_r_phi,
    image_grid_xy_arcsec,
    load_sb_template,
)
from kinuv.geometry import inclination_rad, sky_to_galaxy
from kinuv.transforms.grid import ImageGrid

PA = np.radians(199.73)
I_RAD = inclination_rad()


def _grid(n=64, cell=0.15):
    return ImageGrid(nx=n, ny=n, cell_arcsec=cell)


def test_load_sb_template_signature_unchanged():
    sig = inspect.signature(load_sb_template)
    assert list(sig.parameters) == ["grid", "ico_path"]
    assert sig.parameters["ico_path"].default is None


def test_phi_atan2_yg_xg_receding_is_zero():
    grid = _grid()
    radius, phi = galaxy_r_phi(grid, PA, I_RAD)
    x, y = image_grid_xy_arcsec(grid)
    xe, yn = np.meshgrid(x, y, indexing="xy")
    xg, yg = sky_to_galaxy(xe, yn, PA, I_RAD)
    rec = np.argmin((xg - 1.0) ** 2 + yg**2)
    assert phi.ravel()[rec] == pytest.approx(0.0, abs=0.15)
    plus_y = np.argmin(xg**2 + (yg - 1.0) ** 2)
    assert phi.ravel()[plus_y] == pytest.approx(np.pi / 2, abs=0.15)
    assert np.allclose(phi, np.arctan2(yg, xg))


def test_sky_atan2_cannot_absorb_galaxy_phi():
    grid = _grid()
    x, y = image_grid_xy_arcsec(grid)
    xe, yn = np.meshgrid(x, y, indexing="xy")
    i0 = np.exp(-np.hypot(xe, yn) / 3.0)
    i0 = i0 / (i0.sum() * grid.cell_arcsec**2)
    gal, _ = apply_m2(i0, grid, PA, I_RAD, 0.5, 0.0)
    sky_phi = np.arctan2(yn, xe)
    radius, _ = galaxy_r_phi(grid, PA, I_RAD)
    a2 = 0.5 * np.exp(-0.5 * ((radius - 2.5) / 1.5) ** 2)
    best = None
    best_phi = 0.0
    for phi2 in np.linspace(0.0, np.pi, 72, endpoint=False):
        raw = i0 * (1.0 + a2 * np.cos(2.0 * (sky_phi - phi2)))
        raw = np.maximum(raw, 0.0)
        cand = raw / (raw.sum() * grid.cell_arcsec**2)
        err = float(np.max(np.abs(gal - cand)))
        if best is None or err < best:
            best = err
            best_phi = phi2
    assert best / np.max(i0) > 0.02, (best, best_phi)


def test_a0_equals_i0():
    grid = _grid()
    x, y = image_grid_xy_arcsec(grid)
    xe, yn = np.meshgrid(x, y, indexing="xy")
    i0 = np.exp(-np.hypot(xe, yn) / 3.0)
    i0 = i0 / (i0.sum() * grid.cell_arcsec**2)
    out, _ = apply_m2(i0, grid, PA, I_RAD, 0.0, 0.3)
    rel = np.max(np.abs(out - i0)) / np.max(i0)
    assert rel < 1e-6


def test_phi2_period_pi():
    grid = _grid()
    x, y = image_grid_xy_arcsec(grid)
    xe, yn = np.meshgrid(x, y, indexing="xy")
    i0 = np.exp(-np.hypot(xe, yn) / 3.0)
    i0 = i0 / (i0.sum() * grid.cell_arcsec**2)
    a, _ = apply_m2(i0, grid, PA, I_RAD, 0.4, 0.2)
    b, _ = apply_m2(i0, grid, PA, I_RAD, 0.4, 0.2 + np.pi)
    assert np.max(np.abs(a - b)) < 1e-12


def test_positive_for_a_le_0p8():
    grid = _grid()
    x, y = image_grid_xy_arcsec(grid)
    xe, yn = np.meshgrid(x, y, indexing="xy")
    i0 = np.exp(-np.hypot(xe, yn) / 3.0)
    i0 = i0 / (i0.sum() * grid.cell_arcsec**2)
    out, info = apply_m2(i0, grid, PA, I_RAD, 0.8, 0.1)
    assert np.all(out >= 0.0)
    assert info["centroid_shift_arcsec"] < 0.01


def test_zero_frame_does_not_dilute_i0():
    grid = _grid(n=80, cell=0.15)
    x, y = image_grid_xy_arcsec(grid)
    xe, yn = np.meshgrid(x, y, indexing="xy")
    gauss = np.exp(-0.5 * (xe**2 + yn**2) / (1.2**2))
    framed = gauss.copy()
    framed[np.hypot(xe, yn) > 4.0] = 0.0
    naned = gauss.copy()
    naned[np.hypot(xe, yn) > 4.0] = np.nan
    _, info_g = axisymmetrise_template(gauss, grid, PA, I_RAD)
    _, info_f = axisymmetrise_template(framed, grid, PA, I_RAD)
    _, info_n = axisymmetrise_template(naned, grid, PA, I_RAD)
    inner = info_g["r_centre_arcsec"] < 3.0
    g = info_g["I0_R"][inner]
    f = info_f["I0_R"][inner]
    n = info_n["I0_R"][inner]
    ok = np.isfinite(g) & np.isfinite(f) & np.isfinite(n)
    assert np.max(np.abs(f[ok] - g[ok])) / np.max(g[ok]) < 1e-6
    assert np.max(np.abs(n[ok] - g[ok])) / np.max(g[ok]) < 1e-6
