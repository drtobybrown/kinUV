"""Sky grid for FINUFFT type-2. Cell comes from uv coverage, never CDELT.

DEC-066-GRID: ``1 / (2 · cell_rad) > max_baseline_λ`` with margin. The 0.4″
cell was uvkin silently replacing YAML 0.1″ with an imaging-cube header.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kinuv.constants import ARCSEC_TO_RAD, C_LIGHT_M_S
from kinuv.decisions import requires

#: ADR figure for KGAS066 (~305 kλ). Prefer :func:`max_baseline_lambda` on data.
KGAS066_MAX_BASELINE_LAMBDA = 305e3

#: Extra factor when *choosing* a cell so the assert is not sitting on the edge.
CHOOSE_MARGIN = 1.2

#: CO extent and PB FWHM (DEC-066-PB); FoV is this scale, not 256²@0.1″.
CO_EXTENT_ARCSEC = 15.0
PB_FWHM_ARCSEC = 25.9


def max_baseline_lambda(u_m, v_m, freqs_hz) -> float:
    """Largest |b|/λ in the supplied visibilities."""
    u_m = np.asarray(u_m, dtype=np.float64)
    v_m = np.asarray(v_m, dtype=np.float64)
    nu = np.asarray(freqs_hz, dtype=np.float64)
    return float(np.hypot(u_m, v_m).max() * nu.max() / C_LIGHT_M_S)


def nyquist_u_max_lambda(cell_arcsec: float) -> float:
    """Grid band limit ``1 / (2 · cell_rad)`` in wavelengths."""
    cell_rad = float(cell_arcsec) * ARCSEC_TO_RAD
    if cell_rad <= 0.0:
        raise ValueError(f"cell_arcsec must be positive, got {cell_arcsec}")
    return 1.0 / (2.0 * cell_rad)


def nyquist_assert(cell_arcsec, max_baseline_lambda, *, margin: float = 1.0) -> None:
    """Fail if the cell cannot represent ``max_baseline_lambda``.

    ADR inequality is ``u_max > max_baseline_λ``. ``margin`` > 1 tightens it.
    """
    u_max = nyquist_u_max_lambda(cell_arcsec)
    limit = float(margin) * float(max_baseline_lambda)
    if not (u_max > limit):
        raise ValueError(
            f"cell {float(cell_arcsec):.4g}\" is not Nyquist for "
            f"{float(max_baseline_lambda):.6g} λ (u_max={u_max:.6g} λ, "
            f"margin={margin:g}). 0.4\" is the uvkin imaging-header override "
            f"bug; choose the cell from uv coverage, not FITS CDELT."
        )


def cell_arcsec_from_max_baseline(
    max_baseline_lambda, *, margin: float = CHOOSE_MARGIN
) -> float:
    """Largest cell that Nyquist-samples the data (no header CDELT)."""
    mb = float(max_baseline_lambda)
    if mb <= 0.0:
        raise ValueError(f"max_baseline_lambda must be positive, got {mb}")
    cell_rad = 1.0 / (2.0 * float(margin) * mb)
    return cell_rad / ARCSEC_TO_RAD


def fov_co_plus_pb_arcsec(
    co_extent_arcsec: float = CO_EXTENT_ARCSEC,
    pb_fwhm_arcsec: float = PB_FWHM_ARCSEC,
) -> float:
    """Box covering the CO disk and the PB, not a fixed 256-pixel canvas."""
    return float(max(co_extent_arcsec, pb_fwhm_arcsec))


@dataclass(frozen=True)
class ImageGrid:
    """Uniform sky grid, phase centre at the array centre pixel."""

    nx: int
    ny: int
    cell_arcsec: float

    def __post_init__(self):
        if self.nx < 2 or self.ny < 2:
            raise ValueError(f"grid too small: nx={self.nx} ny={self.ny}")
        if self.cell_arcsec <= 0.0:
            raise ValueError(f"cell_arcsec must be positive, got {self.cell_arcsec}")

    @property
    def cell_rad(self) -> float:
        return self.cell_arcsec * ARCSEC_TO_RAD

    @property
    def l_rad(self) -> np.ndarray:
        """East–west coordinates, ``k = i - nx//2`` (FINUFFT mode order)."""
        return (np.arange(self.nx, dtype=np.float64) - self.nx // 2) * self.cell_rad

    @property
    def m_rad(self) -> np.ndarray:
        """North–south coordinates, ``k = j - ny//2``."""
        return (np.arange(self.ny, dtype=np.float64) - self.ny // 2) * self.cell_rad

    def pixel_lm_rad(self):
        """``(L, M)`` each shape ``(ny, nx)``, ``indexing='xy'``."""
        return np.meshgrid(self.l_rad, self.m_rad, indexing="xy")

    @property
    def fov_arcsec(self) -> float:
        return self.nx * self.cell_arcsec


@requires("DEC-066-GRID")
def image_grid_from_uv(
    max_baseline_lambda,
    fov_arcsec,
    *,
    margin: float = CHOOSE_MARGIN,
    cell_arcsec=None,
) -> ImageGrid:
    """Size ``n`` from FoV / cell. Cell from uv data unless given explicitly."""
    if cell_arcsec is None:
        cell_arcsec = cell_arcsec_from_max_baseline(
            max_baseline_lambda, margin=margin
        )
    nyquist_assert(cell_arcsec, max_baseline_lambda, margin=1.0)
    fov = float(fov_arcsec)
    if fov <= 0.0:
        raise ValueError(f"fov_arcsec must be positive, got {fov}")
    n = int(np.ceil(fov / float(cell_arcsec)))
    n = max(n, 2)
    if n % 2:
        n += 1
    return ImageGrid(nx=n, ny=n, cell_arcsec=float(cell_arcsec))
