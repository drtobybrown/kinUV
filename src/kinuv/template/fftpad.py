"""Centered FFT pad/crop. Pad is ≥2× NAXIS; 512 is the 066-safe default.

Do not tie this pad to ImageGrid.nx (DEC-066-GRID).
"""

from __future__ import annotations

import numpy as np


def default_pad_n(naxis: int) -> int:
    """At least 2× NAXIS; 512 is the safe 066 default."""
    return max(2 * int(naxis), 512)


def embed_centered(image, pad_y: int, pad_x: int):
    """Place ``image`` so its centre pixel lands on ``(pad_y//2, pad_x//2)``."""
    img = np.asarray(image, dtype=np.float64)
    ny, nx = img.shape
    if pad_y < ny or pad_x < nx:
        raise ValueError("pad smaller than image")
    out = np.zeros((int(pad_y), int(pad_x)), dtype=np.float64)
    y0 = pad_y // 2 - ny // 2
    x0 = pad_x // 2 - nx // 2
    out[y0 : y0 + ny, x0 : x0 + nx] = img
    return out, y0, x0


def crop_centered(padded, ny: int, nx: int, y0: int, x0: int) -> np.ndarray:
    return np.asarray(padded, dtype=np.float64)[y0 : y0 + ny, x0 : x0 + nx].copy()
