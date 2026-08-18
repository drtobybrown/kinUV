"""FINUFFT type-2 degrid: uniform sky → visibilities (DEC-066-GRID).

Production path. The 066-1 DFT remains the correctness reference; this module
does not call ``dft_numpy``.

Backend on this Mac: **jax-finufft** (pip wheel, one ``libomp``). If that
import fails, **python-finufft** / conda-forge ``finufft`` is the CPU fallback
behind the same ``nufft2_degrid`` interface. Type-3 is not implemented.
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np

from kinuv.decisions import requires
from kinuv.transforms.dft import uv_wavelengths
from kinuv.transforms.grid import ImageGrid, nyquist_assert

BACKEND: str | None
_t2: Callable | None
_IMPORT_ERROR: BaseException | None = None

try:
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    import jax

    jax.config.update("jax_enable_x64", True)
    from jax_finufft import nufft2 as _jax_nufft2

    def _t2_jax(source, x, y, eps):
        import jax.numpy as jnp

        vis = _jax_nufft2(
            jnp.asarray(source),
            jnp.asarray(x),
            jnp.asarray(y),
            iflag=-1,
            eps=float(eps),
        )
        return np.asarray(vis)

    _t2 = _t2_jax
    BACKEND = "jax-finufft"
except Exception as exc:  # jax-finufft missing or OpenMP clash → CPU FINUFFT
    _IMPORT_ERROR = exc
    try:
        import finufft as _finufft

        def _t2_cpu(source, x, y, eps):
            # python-finufft n_trans shares (x, y); channels have distinct λ.
            source = np.ascontiguousarray(source)
            x = np.ascontiguousarray(x)
            y = np.ascontiguousarray(y)
            if source.ndim == 2:
                return _finufft.nufft2d2(x, y, source, isign=-1, eps=float(eps))
            out = np.empty((source.shape[0], x.shape[-1]), dtype=np.complex128)
            for c in range(source.shape[0]):
                out[c] = _finufft.nufft2d2(
                    np.ascontiguousarray(x[c]),
                    np.ascontiguousarray(y[c]),
                    source[c],
                    isign=-1,
                    eps=float(eps),
                )
            return out

        _t2 = _t2_cpu
        BACKEND = "finufft"
        _IMPORT_ERROR = None
    except Exception as exc2:
        _t2 = None
        BACKEND = None
        _IMPORT_ERROR = exc2


def nufft_backend() -> str:
    """``'jax-finufft'`` or ``'finufft'``. Raises if neither is importable."""
    if BACKEND is None or _t2 is None:
        raise ImportError(
            "FINUFFT is required for type-2 degridding. Install kinuv[nufft] "
            "(jax-finufft) or conda-forge python-finufft. DFT is not a substitute. "
            f"Last import error: {_IMPORT_ERROR!r}"
        ) from _IMPORT_ERROR
    return BACKEND


@requires("DEC-066-GRID")
def nufft2_degrid(grid: ImageGrid, image, u_m, v_m, freqs_hz, *, eps: float = 1e-8):
    """Type-2 NUFFT: ``V[k,c] = sum_{x,y} I[y,x,c] exp(-2πi (u_λ l + v_λ m))``.

    ``image`` is ``(ny, nx)`` (broadcast over channels) or ``(ny, nx, n_chan)``.
    Pixel ``[j, i]`` sits at ``(l, m) = ((i-nx//2) cell, (j-ny//2) cell)``.
    Returns ``(n_row, n_chan)`` complex128, matching :func:`dft_numpy`.
    """
    impl = _t2
    nufft_backend()
    freqs = np.asarray(freqs_hz, dtype=np.float64)
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 2:
        if image.shape != (grid.ny, grid.nx):
            raise ValueError(
                f"image shape {image.shape} != grid (ny, nx)=({grid.ny}, {grid.nx})"
            )
        image = np.repeat(image[:, :, None], freqs.size, axis=2)
    elif image.ndim == 3:
        want = (grid.ny, grid.nx, freqs.size)
        if image.shape != want:
            raise ValueError(f"image shape {image.shape} != {want}")
    else:
        raise ValueError(f"image ndim must be 2 or 3, got {image.ndim}")

    u_lam, v_lam = uv_wavelengths(u_m, v_m, freqs)
    nyquist_assert(grid.cell_arcsec, float(np.hypot(u_lam, v_lam).max()))

    source = np.transpose(image, (2, 1, 0)).astype(np.complex128)
    scale = 2.0 * np.pi * grid.cell_rad
    x = np.ascontiguousarray((scale * u_lam).T)
    y = np.ascontiguousarray((scale * v_lam).T)
    vis = impl(source, x, y, eps)
    return np.ascontiguousarray(np.asarray(vis, dtype=np.complex128).T)


@requires("DEC-066-GRID")
def nufft3_degrid(*_args, **_kwargs):
    """Type-3 is not the 066-5 production path."""
    raise NotImplementedError("type-3 is not the 066-5 production path")
