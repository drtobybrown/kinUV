"""Exact direct Fourier transform: the 066-1 correctness reference.

Evaluates the coplanar sum in float64. No grid, no interpolation, no band
limit. Production NUFFT (DEC-066-GRID) must match this to 1e-7 on a Gaussian.
"""

from __future__ import annotations

import numpy as np

from kinuv.constants import C_LIGHT_M_S
from kinuv.decisions import requires


def uv_wavelengths(u_m, v_m, freqs_hz):
    """Metre baselines → wavelengths per channel, shape ``(n_row, n_chan)``."""
    u_m = np.asarray(u_m, dtype=np.float64)
    v_m = np.asarray(v_m, dtype=np.float64)
    scale = np.asarray(freqs_hz, dtype=np.float64) / C_LIGHT_M_S
    return u_m[:, None] * scale[None, :], v_m[:, None] * scale[None, :]


@requires("DEC-066-GRID")
def dft_numpy(l_rad, m_rad, strengths, u_m, v_m, freqs_hz):
    """``V[k,c] = sum_n S[n,c] exp(-2πi (u_λ l + v_λ m))``."""
    l_rad = np.asarray(l_rad, dtype=np.float64)
    m_rad = np.asarray(m_rad, dtype=np.float64)
    strengths = np.asarray(strengths, dtype=np.float64)
    u_lam, v_lam = uv_wavelengths(u_m, v_m, freqs_hz)
    n_row, n_chan = u_lam.shape
    out = np.zeros((n_row, n_chan), dtype=np.complex128)
    for c in range(n_chan):
        phase = -2.0 * np.pi * (
            np.outer(u_lam[:, c], l_rad) + np.outer(v_lam[:, c], m_rad)
        )
        out[:, c] = np.exp(1j * phase) @ strengths[:, c]
    return out
