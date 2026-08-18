"""Flux-conserving resample of a regular sky grid (DEC-066-SB)."""

from __future__ import annotations

import numpy as np

from kinuv.decisions import requires


def _centers(n: int, cell: float) -> np.ndarray:
    return (np.arange(n, dtype=np.float64) - (n - 1) * 0.5) * cell


def _overlap_1d(n_in: int, cell_in: float, n_out: int, cell_out: float) -> np.ndarray:
    """Overlap lengths [arcsec] between output pixels (rows) and input pixels (cols)."""
    x_in = _centers(n_in, cell_in)
    x_out = _centers(n_out, cell_out)
    left_in = x_in - 0.5 * cell_in
    right_in = x_in + 0.5 * cell_in
    left_out = x_out[:, None] - 0.5 * cell_out
    right_out = x_out[:, None] + 0.5 * cell_out
    return np.maximum(
        0.0,
        np.minimum(right_in[None, :], right_out) - np.maximum(left_in[None, :], left_out),
    )


@requires("DEC-066-SB")
def resample_flux_conserving(image, cell_in_arcsec, cell_out_arcsec):
    """Area-overlap remap conserving ``Σ I ΔΩ``. Not bilinear-without-ΔΩ.

    Input and output grids share a common centre. Output shape is chosen so the
    field of view matches the input to the nearest output pixel.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("image must be 2-D")
    cell_in = float(cell_in_arcsec)
    cell_out = float(cell_out_arcsec)
    if cell_in <= 0.0 or cell_out <= 0.0:
        raise ValueError("cell sizes must be positive")
    if np.isclose(cell_in, cell_out, rtol=0.0, atol=1e-15):
        return img.copy()
    ny, nx = img.shape
    ny_out = max(1, int(round(ny * cell_in / cell_out)))
    nx_out = max(1, int(round(nx * cell_in / cell_out)))
    wy = _overlap_1d(ny, cell_in, ny_out, cell_out)
    wx = _overlap_1d(nx, cell_in, nx_out, cell_out)
    # I is SB; overlap lengths give flux = I * Δx_ov * Δy_ov.
    flux = wy @ img @ wx.T
    return flux / (cell_out * cell_out)


def sky_axes(n: int, cell_arcsec: float) -> np.ndarray:
    """Pixel-centre coordinates [arcsec] for a centred square-cell axis."""
    return _centers(int(n), float(cell_arcsec))
