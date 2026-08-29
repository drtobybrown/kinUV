"""Exact direct Fourier transform: the 066-1 correctness reference.

Evaluates the coplanar sum in float64. No grid, no interpolation, no band
limit. Production NUFFT (DEC-066-GRID) must match this to 1e-7 on a Gaussian.
"""

from __future__ import annotations

import numpy as np

from kinuv.constants import C_LIGHT_M_S
from kinuv.decisions import requires

# CASA/ALMA visibilities (066 npz UVW as stored) match
# V = ∫ I exp(+2πi (u l + v m)) with l east, m north — the CASA imager
# convention, not a conjugated export. Textbook −2πi is recovered by
# (u, v) → (−u, −v). FT of the WCS-true CLEAN cube onto the npz confirms
# this sign; the opposite kernel is χ²-indistinguishable from V=0.
NPZ_UV_SIGN = -1.0


def uv_wavelengths(u_m, v_m, freqs_hz):
    """Metre baselines → wavelengths per channel, shape ``(n_row, n_chan)``."""
    u_m = np.asarray(u_m, dtype=np.float64)
    v_m = np.asarray(v_m, dtype=np.float64)
    scale = np.asarray(freqs_hz, dtype=np.float64) / C_LIGHT_M_S
    return u_m[:, None] * scale[None, :], v_m[:, None] * scale[None, :]


def vis_uv_wavelengths(u_m, v_m, freqs_hz):
    """``uv_wavelengths`` times :data:`NPZ_UV_SIGN` for the −2πi kernel."""
    u_lam, v_lam = uv_wavelengths(u_m, v_m, freqs_hz)
    s = float(NPZ_UV_SIGN)
    return s * u_lam, s * v_lam


@requires("DEC-066-GRID")
def dft_numpy(l_rad, m_rad, strengths, u_m, v_m, freqs_hz):
    """``V[k,c] = sum_n S[n,c] exp(-2πi (u_λ l + v_λ m))`` with npz (u, v) sign."""
    l_rad = np.asarray(l_rad, dtype=np.float64)
    m_rad = np.asarray(m_rad, dtype=np.float64)
    strengths = np.asarray(strengths, dtype=np.float64)
    u_lam, v_lam = vis_uv_wavelengths(u_m, v_m, freqs_hz)
    n_row, n_chan = u_lam.shape
    out = np.zeros((n_row, n_chan), dtype=np.complex128)
    for c in range(n_chan):
        phase = -2.0 * np.pi * (
            np.outer(u_lam[:, c], l_rad) + np.outer(v_lam[:, c], m_rad)
        )
        out[:, c] = np.exp(1j * phase) @ strengths[:, c]
    return out
