"""SB template on the vis ImageGrid (DEC-066-SB, DEC-066-GRID, DEC-066-SHIFT)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kinuv.decisions import requires
from kinuv.template.fftpad import default_pad_n, embed_centered
from kinuv.template.fourier_shift import fourier_shift
from kinuv.template.resample import resample_flux_conserving, sky_axes
from kinuv.template.wiener import ico_to_template
from kinuv.transforms.grid import ImageGrid

R_SCALE_066_ARCSEC = 3.0
ICO_FITS = Path(
    "/Users/thbrown/kilogas/analysis/kinms_test/kgas066/KGAS66_Ico_K_kms-1.fits"
)
NU_OBS_ICO_HZ = 224.3e9


def image_grid_xy_arcsec(grid: ImageGrid):
    """East / North pixel centres [arcsec] in FINUFFT mode order (DEC-066-GRID)."""
    x = (np.arange(grid.nx, dtype=np.float64) - grid.nx // 2) * grid.cell_arcsec
    y = (np.arange(grid.ny, dtype=np.float64) - grid.ny // 2) * grid.cell_arcsec
    return x, y


def _overlap_axes(x_in, cell_in, x_out, cell_out) -> np.ndarray:
    """Same area-overlap kernel as ``kinuv.template.resample._overlap_1d``."""
    left_in = x_in - 0.5 * cell_in
    right_in = x_in + 0.5 * cell_in
    left_out = x_out[:, None] - 0.5 * cell_out
    right_out = x_out[:, None] + 0.5 * cell_out
    return np.maximum(
        0.0,
        np.minimum(right_in[None, :], right_out) - np.maximum(left_in[None, :], left_out),
    )


@requires("DEC-066-SB", "DEC-066-GRID")
def place_template_on_grid(sb, cell_arcsec, grid: ImageGrid) -> np.ndarray:
    """Flux-conserving remap of a centred SB stamp onto ``ImageGrid``.

    Uses :func:`resample_flux_conserving` when the cell changes, then the same
    area-overlap kernel onto FINUFFT pixel centres (not Ico 0.4″ CDELT).
    Output is unit ``∫ I dΩ`` on the vis grid.
    """
    img = np.asarray(sb, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("sb must be 2-D")
    cell_in = float(cell_arcsec)
    cell_out = float(grid.cell_arcsec)
    if not np.isclose(cell_in, cell_out, rtol=0.0, atol=1e-15):
        img = resample_flux_conserving(img, cell_in, cell_out)
        cell_in = cell_out
    ny, nx = img.shape
    x_in = sky_axes(nx, cell_in)
    y_in = sky_axes(ny, cell_in)
    x_out, y_out = image_grid_xy_arcsec(grid)
    flux = _overlap_axes(y_in, cell_in, y_out, cell_out) @ img @ _overlap_axes(
        x_in, cell_in, x_out, cell_out
    ).T
    out = flux / (cell_out * cell_out)
    tot = float(out.sum() * cell_out * cell_out)
    if abs(tot) < 1e-30:
        raise ValueError("template integral vanishes on ImageGrid")
    return out / tot


@requires("DEC-066-GRID")
def exponential_template(grid: ImageGrid, r_scale_arcsec: float = R_SCALE_066_ARCSEC):
    """Unit-integral exponential disk on ImageGrid axes. Not the 0.4″ Ico cell."""
    x, y = image_grid_xy_arcsec(grid)
    xg, yg = np.meshgrid(x, y, indexing="xy")
    sb = np.exp(-np.hypot(xg, yg) / float(r_scale_arcsec))
    d_omega = grid.cell_arcsec**2
    return sb / (float(sb.sum()) * d_omega)


@requires("DEC-066-SB", "DEC-066-GRID")
def load_sb_template(grid: ImageGrid, ico_path: Path | None = None) -> np.ndarray:
    """``ico_to_template`` when the FITS is present; exponential otherwise."""
    path = ICO_FITS if ico_path is None else Path(ico_path)
    if path.is_file():
        from astropy.io import fits

        with fits.open(path) as hdul:
            h = hdul[0].header
            data = np.array(hdul[0].data, dtype=np.float64)
        bmaj = float(h["BMAJ"]) * 3600.0
        bmin = float(h["BMIN"]) * 3600.0
        bpa = float(h["BPA"])
        cell = abs(float(h["CDELT2"])) * 3600.0
        tmpl = ico_to_template(
            data,
            cell,
            NU_OBS_ICO_HZ,
            bmaj,
            bmin,
            bpa,
            sigma_empty=0.02 * np.nanmax(np.abs(data)),
        )
        return place_template_on_grid(tmpl.sb, tmpl.cell_arcsec, grid)
    return exponential_template(grid)


@requires("DEC-066-SHIFT")
def fourier_shift_padded(image, dx_arcsec, dy_arcsec, cell_arcsec, pad_n=None):
    """Fourier shift on the Wiener pad; **no crop**. For the SHIFT broadening bound."""
    img = np.asarray(image, dtype=np.float64)
    ny, nx = img.shape
    pad = int(pad_n) if pad_n is not None else default_pad_n(max(ny, nx))
    padded, _, _ = embed_centered(img, pad, pad)
    # Reuse the production interpolator on the padded canvas (crop is identity).
    return fourier_shift(padded, dx_arcsec, dy_arcsec, cell_arcsec, pad_n=pad)


@requires("DEC-066-SHIFT")
def exponential_r_scale(
    image,
    cell_arcsec,
    x0_arcsec: float = 0.0,
    y0_arcsec: float = 0.0,
    r_min_arcsec: float = 1.5,
    r_max_arcsec: float = 8.0,
) -> float:
    """Azimuthally averaged exponential scale length about ``(x0, y0)`` [arcsec]."""
    img = np.asarray(image, dtype=np.float64)
    ny, nx = img.shape
    x = (np.arange(nx, dtype=np.float64) - nx // 2) * float(cell_arcsec) - float(
        x0_arcsec
    )
    y = (np.arange(ny, dtype=np.float64) - ny // 2) * float(cell_arcsec) - float(
        y0_arcsec
    )
    X, Y = np.meshgrid(x, y, indexing="xy")
    r = np.hypot(X, Y)
    sel = (img > 0.0) & np.isfinite(img) & (r >= r_min_arcsec) & (r <= r_max_arcsec)
    if int(sel.sum()) < 8:
        raise ValueError("too few pixels to fit r_scale")
    design = np.vstack([np.ones(int(sel.sum())), -r[sel]]).T
    coeff, *_ = np.linalg.lstsq(design, np.log(img[sel]), rcond=None)
    slope = float(coeff[1])
    if slope <= 0.0:
        raise ValueError(f"non-positive inverse scale {slope}")
    return 1.0 / slope
