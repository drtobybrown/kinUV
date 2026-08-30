"""Standard diagnostic figures. Cosmetics via ``style``; not a likelihood.

Plot labels are ASCII (``chi2``, ``gas_sigma``, ``r_t``) so titles do not
depend on a Unicode renderer. Do not copy rcParams. Do not use viridis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kinuv.diagnostics.style import COLOUR, apply_style, intensity_cmap, save_fig

STAGE_A_NAMES = (
    "flux",
    "pa_deg",
    "vsys_kms",
    "gas_sigma_kms",
    "dx_arcsec",
    "dy_arcsec",
    "v0_kms",
    "r_t_arcsec",
)


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


def _as_nuts_draws(rec):
    """Require provenance ``sampler == 'nuts'`` and an 8-column draw array."""
    if isinstance(rec, (str, Path)):
        raise ValueError("plot_posterior_corner takes a draws record, not a path")
    if isinstance(rec, dict):
        sampler = rec.get("sampler")
        draws = rec.get("draws")
        intervals = rec.get("intervals")
    else:
        sampler = getattr(rec, "sampler", None)
        draws = getattr(rec, "draws", None)
        intervals = getattr(rec, "intervals", None)
    if sampler != "nuts":
        raise ValueError(
            f"plot_posterior_corner requires sampler == 'nuts'; got {sampler!r}"
        )
    if draws is None:
        raise ValueError(
            "draws array required; p16/p84 interval tables are not a NUTS posterior"
        )
    arr = np.asarray(draws, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.ndim != 2 or arr.shape[1] != 8:
        raise ValueError("draws must be (n_draw, 8) or (n_chain, n_draw, 8)")
    if intervals is not None and np.size(draws) <= 8:
        raise ValueError("interval tables with no chain draws are refused")
    return arr


def plot_posterior_corner(rec, path, *, title=None) -> Path:
    """Stage A corner from NUTS draws only. Not laplace_mh. ASCII names."""
    draws = _as_nuts_draws(rec)
    apply_style()
    import matplotlib.pyplot as plt

    n = draws.shape[1]
    fig, axes = plt.subplots(n, n, figsize=(9.6, 9.6))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                ax.hist(
                    draws[:, i],
                    bins=24,
                    color=COLOUR["model"],
                    histtype="stepfilled",
                    alpha=0.35,
                    density=True,
                )
                q16, q50, q84 = np.quantile(draws[:, i], [0.16, 0.50, 0.84])
                ax.axvline(q16, color=COLOUR["vsys"], lw=0.8, ls="--")
                ax.axvline(q50, color=COLOUR["data"], lw=0.9)
                ax.axvline(q84, color=COLOUR["vsys"], lw=0.8, ls="--")
            else:
                ax.plot(
                    draws[:, j],
                    draws[:, i],
                    ".",
                    color=COLOUR["zero"],
                    ms=1.5,
                    alpha=0.35,
                    rasterized=True,
                )
            if i == n - 1:
                ax.set_xlabel(STAGE_A_NAMES[j], fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(STAGE_A_NAMES[i], fontsize=8)
            else:
                ax.set_yticklabels([])
    fig.suptitle(
        title or "synthetic nuts fixture; not 066; not laplace_mh",
        fontsize=11,
    )
    fig.tight_layout()
    return save_fig(fig, path)
