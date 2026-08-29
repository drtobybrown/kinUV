"""Standard diagnostic figures. Cosmetics via ``style``; not a likelihood.

Plot labels are ASCII (``chi2``, ``gas_sigma``, ``r_t``) so titles do not
depend on a Unicode renderer. Do not copy rcParams. Do not use viridis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kinuv.diagnostics.style import COLOUR, apply_style, intensity_cmap, save_fig


def binned_mean(x, y, n_bin: int = 16):
    """Mean ``y`` in percentile-clipped bins of ``x``."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lo, hi = np.percentile(x, [2, 98])
    edges = np.linspace(lo, hi, n_bin + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    means = np.full(n_bin, np.nan)
    for i in range(n_bin):
        sel = (x >= edges[i]) & (x < edges[i + 1])
        if np.any(sel):
            means[i] = float(np.mean(y[sel]))
    return centres, means


def plot_leftover_chi2(baseline_m, per_row, vel_kms, per_chan, path) -> Path:
    """Residual chi2 vs uv-distance and velocity (SB leftover vs missing flux)."""
    apply_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
    bc, bm = binned_mean(baseline_m, per_row)
    axes[0].plot(
        baseline_m, per_row, ".", color=COLOUR["zero"], ms=2, alpha=0.4, rasterized=True
    )
    axes[0].plot(bc, bm, "-", color=COLOUR["model"], lw=1.6)
    axes[0].set_xlabel("baseline (m)")
    axes[0].set_ylabel("chi2 per row")
    axes[0].set_title("leftover vs uv-distance")
    axes[1].plot(vel_kms, per_chan, "-", color=COLOUR["model"], lw=1.4)
    axes[1].set_xlabel("radio velocity (km/s)")
    axes[1].set_ylabel("chi2 per channel")
    axes[1].set_title("leftover vs velocity")
    fig.tight_layout()
    return save_fig(fig, path)


def _contour(ax, x, y, z, xlab, ylab, title):
    im = ax.pcolormesh(x, y, z, shading="auto", cmap=intensity_cmap())
    ax.contour(x, y, z, colors="white", linewidths=0.6, alpha=0.7)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    return im


def plot_chi2_slices(
    pa,
    sigma,
    r_t,
    i_deg,
    z_pa_sigma,
    z_sigma_i,
    z_pa_rt,
    mark: dict,
    path,
) -> Path:
    """2-D chi2 slices: PA-gas_sigma, gas_sigma-i (scan), PA-r_t."""
    apply_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4))
    _contour(axes[0], pa, sigma, z_pa_sigma, "PA (deg)", "gas_sigma (km/s)", "PA-gas_sigma")
    axes[0].plot(mark["pa_deg"], mark["gas_sigma_kms"], "o", color=COLOUR["data"], ms=4)
    _contour(
        axes[1],
        sigma,
        i_deg,
        z_sigma_i,
        "gas_sigma (km/s)",
        "i (deg)",
        "gas_sigma-i (i unfrozen scan)",
    )
    axes[1].plot(mark["gas_sigma_kms"], mark["i_deg"], "o", color=COLOUR["data"], ms=4)
    _contour(axes[2], pa, r_t, z_pa_rt, "PA (deg)", "r_t (arcsec)", "PA-r_t")
    axes[2].plot(mark["pa_deg"], mark["r_t_arcsec"], "o", color=COLOUR["data"], ms=4)
    fig.tight_layout()
    return save_fig(fig, path)
