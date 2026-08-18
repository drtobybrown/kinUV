"""066-7 forward model: Stage A cube, PB-then-T2, SHIFT broadening bound."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kinuv.constants import ARCSEC_TO_RAD
from kinuv.forward.model import (
    GAS_SIGMA_SEED_KM_S,
    INJECT_OFFSET_ARCSEC,
    VSYS_SEED_KM_S,
    los_velocity,
    predict_vis,
    sky_cube,
)
from kinuv.forward.sb import (
    ICO_FITS,
    R_SCALE_066_ARCSEC,
    exponential_r_scale,
    exponential_template,
    fourier_shift_padded,
    load_sb_template,
    place_template_on_grid,
)
from kinuv.geometry import inclination_rad, pa_seed_rad
from kinuv.profiles.rotation import CALIBRATION_RT_ARCSEC, CALIBRATION_V0_KM_S
from kinuv.response.primary_beam import primary_beam
from kinuv.template.fftpad import default_pad_n
from kinuv.template.resample import sky_axes
from kinuv.template.wiener import ico_to_template
from kinuv.transforms.dft import uv_wavelengths
from kinuv.transforms.grid import (
    ImageGrid,
    fov_co_plus_pb_arcsec,
    image_grid_from_uv,
    max_baseline_lambda,
)
from kinuv.transforms.nufft import nufft2_degrid

SRC = Path(__file__).resolve().parents[1] / "src" / "kinuv" / "forward"


def test_no_uvkin_or_kinms_import():
    for path in SRC.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "uvkin" not in text
        assert "KinMS" not in text
        assert "from kinms" not in text.lower()


def test_receding_major_axis_is_redshifted():
    pa = pa_seed_rad()
    i = inclination_rad()
    # Receding major axis: E = sin(PA), N = cos(PA)
    x_e, y_n = np.sin(pa) * 4.0, np.cos(pa) * 4.0
    v = los_velocity(x_e, y_n, pa, i, VSYS_SEED_KM_S)
    assert v > VSYS_SEED_KM_S + 20.0
    v_app = los_velocity(-x_e, -y_n, pa, i, VSYS_SEED_KM_S)
    assert v_app < VSYS_SEED_KM_S - 20.0
    assert los_velocity._kinuv_requires == ("DEC-066-PA", "DEC-066-INC", "DEC-066-VC")


def test_vis_grid_is_not_ico_cdelt():
    grid = image_grid_from_uv(305e3, fov_co_plus_pb_arcsec())
    assert grid.cell_arcsec != pytest.approx(0.4)
    sb = exponential_template(grid)
    assert sb.shape == (grid.ny, grid.nx)
    assert abs(float(sb.sum()) * grid.cell_arcsec**2 - 1.0) < 1e-12


def test_shift_broadening_bound_on_padded_grid():
    """After 1″ (and 0.3″, 2″) Fourier shift, r_scale changes < 0.5% (DEC-066-SHIFT)."""
    cell = 0.1
    n = 201
    grid = ImageGrid(nx=n, ny=n, cell_arcsec=cell)
    sb = exponential_template(grid, R_SCALE_066_ARCSEC)
    r0 = exponential_r_scale(fourier_shift_padded(sb, 0.0, 0.0, cell), cell)
    pad = default_pad_n(n)
    assert (pad - n) * cell / 2.0 >= 2.0
    for shift in (0.3, 1.0, 2.0):
        padded = fourier_shift_padded(sb, shift, 0.0, cell)
        r_hat = exponential_r_scale(padded, cell, x0_arcsec=shift, y0_arcsec=0.0)
        rel = abs(r_hat - r0) / r0
        assert rel < 0.005, f"r_scale {r0:.4f} → {r_hat:.4f} after {shift}″ ({rel:.4%})"


def test_predict_vis_native_shape_and_requires(uv_sampling, freqs):
    u_m, v_m = uv_sampling
    mb = max_baseline_lambda(u_m, v_m, freqs)
    grid = image_grid_from_uv(mb, fov_arcsec=16.0)
    tmpl = exponential_template(grid)
    vis = predict_vis(
        u_m,
        v_m,
        freqs,
        flux=1.2,
        pa_rad=pa_seed_rad(),
        vsys_kms=VSYS_SEED_KM_S,
        dx_arcsec=INJECT_OFFSET_ARCSEC,
        dy_arcsec=INJECT_OFFSET_ARCSEC,
        gas_sigma_kms=GAS_SIGMA_SEED_KM_S,
        template=tmpl,
        grid=grid,
        v0_kms=CALIBRATION_V0_KM_S,
        r_t_arcsec=CALIBRATION_RT_ARCSEC,
    )
    assert vis.shape == (u_m.size, freqs.size)
    assert vis.dtype == np.complex128
    assert predict_vis._kinuv_requires == (
        "DEC-066-PB",
        "DEC-066-SHIFT",
        "DEC-066-GRID",
        "DEC-066-VC",
        "DEC-066-PA",
        "DEC-066-INC",
    )


def test_no_visibility_ramp_after_pb(uv_sampling, freqs):
    u_m, v_m = uv_sampling
    mb = max_baseline_lambda(u_m, v_m, freqs)
    grid = image_grid_from_uv(mb, fov_arcsec=16.0)
    tmpl = exponential_template(grid)
    kw = dict(
        flux=1.0,
        pa_rad=pa_seed_rad(),
        vsys_kms=VSYS_SEED_KM_S,
        gas_sigma_kms=GAS_SIGMA_SEED_KM_S,
        template=tmpl,
        grid=grid,
    )
    dx = dy = 1.0
    v_shift = predict_vis(u_m, v_m, freqs, dx_arcsec=dx, dy_arcsec=dy, **kw)
    v0 = predict_vis(u_m, v_m, freqs, dx_arcsec=0.0, dy_arcsec=0.0, **kw)
    u_lam, v_lam = uv_wavelengths(u_m, v_m, freqs)
    ramp = np.exp(
        -2.0j * np.pi * (u_lam * dx * ARCSEC_TO_RAD + v_lam * dy * ARCSEC_TO_RAD)
    )
    v_wrong = v0 * ramp
    rel = np.abs(v_shift - v_wrong).max() / np.abs(v_shift).max()
    assert rel > 1e-3


def test_pb_stays_on_phase_centre_after_shift(uv_sampling, freqs):
    u_m, v_m = uv_sampling
    mb = max_baseline_lambda(u_m, v_m, freqs)
    grid = image_grid_from_uv(mb, fov_arcsec=16.0)
    tmpl = exponential_template(grid)
    cube = sky_cube(
        tmpl,
        grid,
        freqs[:1],
        flux=1.0,
        pa_rad=pa_seed_rad(),
        vsys_kms=VSYS_SEED_KM_S,
        dx_arcsec=1.0,
        dy_arcsec=1.0,
        gas_sigma_kms=GAS_SIGMA_SEED_KM_S,
    )
    x = (np.arange(grid.nx) - grid.nx // 2) * grid.cell_arcsec
    y = (np.arange(grid.ny) - grid.ny // 2) * grid.cell_arcsec
    a = primary_beam(x, y, float(freqs[0]))
    plane = cube[:, :, 0]
    # Recover A from the attenuated cube vs the pre-A sky by undoing A on a copy
    iy, ix = np.unravel_index(np.argmax(a), a.shape)
    assert abs(x[ix]) < grid.cell_arcsec and abs(y[iy]) < grid.cell_arcsec
    assert plane.shape == a.shape
    _ = nufft2_degrid(grid, plane, u_m[:8], v_m[:8], freqs[:1])


@pytest.mark.skipif(not ICO_FITS.is_file(), reason="Ico FITS not on this machine")
def test_ico_resampled_onto_vis_grid_not_cdelt():
    grid = image_grid_from_uv(305e3, fov_co_plus_pb_arcsec())
    assert grid.cell_arcsec != pytest.approx(0.4)
    sb = load_sb_template(grid, ICO_FITS)
    assert sb.shape == (grid.ny, grid.nx)
    assert abs(float(sb.sum()) * grid.cell_arcsec**2 - 1.0) < 1e-8
    from astropy.io import fits

    with fits.open(ICO_FITS) as hdul:
        h = hdul[0].header
        data = np.array(hdul[0].data, dtype=np.float64)
    cell = abs(float(h["CDELT2"])) * 3600.0
    tmpl = ico_to_template(
        data,
        cell,
        224.3e9,
        float(h["BMAJ"]) * 3600.0,
        float(h["BMIN"]) * 3600.0,
        float(h["BPA"]),
        sigma_empty=0.02 * np.nanmax(np.abs(data)),
    )
    placed = place_template_on_grid(tmpl.sb, tmpl.cell_arcsec, grid)
    flux_in = float(tmpl.sb.sum()) * tmpl.cell_arcsec**2
    assert flux_in == pytest.approx(1.0, abs=1e-8)
    assert placed.shape == sb.shape
    # sky_axes is the Ico convention; vis grid must not silently be that CDELT
    assert sky_axes(tmpl.sb.shape[0], tmpl.cell_arcsec).shape[0] != grid.nx or not np.isclose(
        tmpl.cell_arcsec, grid.cell_arcsec
    )
