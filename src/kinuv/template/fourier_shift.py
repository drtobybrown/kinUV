"""Fourier image shift for the DEC-066-PB stationary-PB gate.

Scope exception: DEC-066-SHIFT interpolator for that test only. Not MAP, not a
fitter, not 066-8 ownership.
"""

from __future__ import annotations

import numpy as np

from kinuv.decisions import requires
from kinuv.template.fftpad import crop_centered, default_pad_n, embed_centered
from kinuv.xp import is_jax


@requires("DEC-066-PB")
def fourier_shift(image, dx_arcsec, dy_arcsec, cell_arcsec, pad_n=None):
    """``I_out(x, y) = I_in(x − dx, y − dy)`` via a padded Fourier phase ramp.

    Content moves to ``(+dx, +dy)``. Pad is the Wiener pad (≥2× NAXIS, 512
    default), then crop back. Do not use a visibility ramp after PB.
    """
    if is_jax(image):
        import jax.numpy as jnp

        img = jnp.asarray(image)
        if img.ndim != 2:
            raise ValueError("image must be 2-D")
        ny, nx = int(img.shape[0]), int(img.shape[1])
        naxis = max(ny, nx)
        pad = int(pad_n) if pad_n is not None else default_pad_n(naxis)
        padded = jnp.zeros((pad, pad), dtype=img.dtype)
        y0 = pad // 2 - ny // 2
        x0 = pad // 2 - nx // 2
        padded = padded.at[y0 : y0 + ny, x0 : x0 + nx].set(img)
        ft = jnp.fft.fft2(jnp.fft.ifftshift(padded))
        cell = float(cell_arcsec)
        uy = jnp.fft.fftfreq(pad, d=cell)[:, None]
        ux = jnp.fft.fftfreq(pad, d=cell)[None, :]
        phase = jnp.exp(
            -2.0j * jnp.pi * (ux * float(dx_arcsec) + uy * float(dy_arcsec))
        )
        shifted = jnp.fft.fftshift(jnp.fft.ifft2(ft * phase)).real
        return shifted[y0 : y0 + ny, x0 : x0 + nx]

    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("image must be 2-D")
    ny, nx = img.shape
    naxis = max(ny, nx)
    pad = int(pad_n) if pad_n is not None else default_pad_n(naxis)
    padded, y0, x0 = embed_centered(img, pad, pad)
    ft = np.fft.fft2(np.fft.ifftshift(padded))
    cell = float(cell_arcsec)
    uy = np.fft.fftfreq(pad, d=cell)[:, None]
    ux = np.fft.fftfreq(pad, d=cell)[None, :]
    phase = np.exp(-2.0j * np.pi * (ux * float(dx_arcsec) + uy * float(dy_arcsec)))
    shifted = np.fft.fftshift(np.fft.ifft2(ft * phase)).real
    return crop_centered(shifted, ny, nx, y0, x0)
