"""Publication figure style for kinUV image-plane diagnostics.

Import this module. Do not copy rcParams into scripts. Cosmetics only:
matching physics stays in ``kinuv.diagnostics.imaging``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
from matplotlib.ticker import MultipleLocator

# --- colour tokens (hex + role). Never use the default C0/C1 cycle. ---
COLOUR = {
    "data": "#1A1A1A",  # observed spectra / traces
    "model": "#2A6F97",  # Stage B / model traces (desaturated blue, not C0)
    "vsys": "#737373",  # systemic-velocity guide
    "zero": "#C8C8C8",  # zero-flux / zero-offset line
    "mask": "#FFFFFF",  # blanked / masked pixels (not a fake zero)
    "beam_face": "#E6E6E6",
    "beam_edge": "#1A1A1A",
    "text": "#1A1A1A",
    "muted": "#555555",
}

# Sequential intensity (M0, M2, PV). Dark-low so white mask stays distinct.
# matplotlib has no mako; this is a colourblind-safe teal–sand ramp.
INTENSITY_HEX = (
    "#08111A",
    "#0E2A3A",
    "#124354",
    "#155E68",
    "#1B7870",
    "#3A9274",
    "#6AAB78",
    "#A4C488",
    "#D9DCA4",
    "#F3F1D3",
)

CMAP_VELOCITY = "coolwarm"  # M1; light mid-tone, centred on 0 after v−vsys
CMAP_RESIDUAL = "RdBu_r"  # data−model; not the velocity map

DPI = 200
CROP_ARCSEC = 12.0
TITLE_SIZE = 11
TICK_SIZE = 8
LABEL_SIZE = 9

__all__ = [
    "CMAP_RESIDUAL",
    "CMAP_VELOCITY",
    "COLOUR",
    "CROP_ARCSEC",
    "DPI",
    "INTENSITY_HEX",
    "apply_style",
    "beam_ellipse",
    "cbar",
    "data_model_residual_grid",
    "format_sky_ax",
    "imshow_masked",
    "intensity_cmap",
    "panel_letter",
    "residual_cmap",
    "save_fig",
    "sequential_clim",
    "sky_extent_arcsec",
    "symmetric_clim",
    "velocity_cmap",
    "vsys_line",
]


def _register_cmaps() -> None:
    if "kinuv_intensity" in mpl.colormaps:
        return
    cmap = LinearSegmentedColormap.from_list(
        "kinuv_intensity", INTENSITY_HEX, N=256
    )
    mpl.colormaps.register(cmap.with_extremes(bad=COLOUR["mask"]))


def apply_style() -> None:
    """Set rcParams. Call once per plotting process before creating figures."""
    _register_cmaps()
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
            "font.family": "DejaVu Sans",
            "font.size": LABEL_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "axes.titlepad": 4.0,
            "axes.linewidth": 0.7,
            "axes.grid": False,
            "axes.unicode_minus": True,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "image.interpolation": "nearest",
            "image.origin": "lower",
            "image.cmap": "kinuv_intensity",
            "axes.prop_cycle": mpl.cycler(color=[COLOUR["model"], COLOUR["data"]]),
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": DPI,
            "figure.dpi": 100,
        }
    )


def _cmap_copy(name: str):
    _register_cmaps()
    return mpl.colormaps[name].with_extremes(bad=COLOUR["mask"])


def intensity_cmap():
    """Sequential map for M0 / M2 / PV brightness. Prefers mako if registered."""
    _register_cmaps()
    name = "mako" if "mako" in mpl.colormaps else "kinuv_intensity"
    return _cmap_copy(name)


def velocity_cmap():
    return _cmap_copy(CMAP_VELOCITY)


def residual_cmap():
    return _cmap_copy(CMAP_RESIDUAL)


def sky_extent_arcsec(header) -> tuple[float, float, float, float]:
    """Imshow extent (x0, x1, y0, y1) in arcsec; east-positive.

    Radio FITS has ``CDELT1 < 0``, so ``x0 > x1`` and east is already left.
    """
    nx, ny = int(header["NAXIS1"]), int(header["NAXIS2"])
    dx = float(header["CDELT1"]) * 3600.0
    dy = float(header["CDELT2"]) * 3600.0
    x0 = (0.5 - float(header["CRPIX1"])) * dx
    x1 = (nx + 0.5 - float(header["CRPIX1"])) * dx
    y0 = (0.5 - float(header["CRPIX2"])) * dy
    y1 = (ny + 0.5 - float(header["CRPIX2"])) * dy
    return x0, x1, y0, y1


def format_sky_ax(
    ax,
    crop: float = CROP_ARCSEC,
    centre=(0.0, 0.0),
    *,
    xlabel: bool = False,
    ylabel: bool = False,
):
    """East left, north up, cropped to the galaxy. Ticks in arcsec."""
    cx, cy = float(centre[0]), float(centre[1])
    crop = float(crop)
    ax.set_xlim(cx + crop, cx - crop)
    ax.set_ylim(cy - crop, cy + crop)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    if xlabel:
        ax.set_xlabel("East offset (arcsec)")
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
    if ylabel:
        ax.set_ylabel("North offset (arcsec)")
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)


def imshow_masked(ax, img, extent, vmin, vmax, cmap, *, aspect="equal"):
    """Sky / PV image with NaNs as white, not a mapped zero."""
    data = np.ma.masked_invalid(np.asarray(img, dtype=float))
    if isinstance(cmap, str):
        cmap = _cmap_copy(cmap)
    return ax.imshow(
        data,
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="nearest",
        aspect=aspect,
        rasterized=True,
    )


def sequential_clim(*arrays, p: float = 99.0) -> tuple[float, float]:
    """vmin=0, vmax=percentile of finite values (data and model together)."""
    vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    vmax = float(np.percentile(vals, p))
    if vmax <= 0.0:
        vmax = 1.0
    return 0.0, vmax


def symmetric_clim(*arrays, p: float = 95.0) -> tuple[float, float]:
    """Diverging limits, percentile-clipped, symmetric about 0."""
    vals = np.concatenate(
        [np.abs(np.asarray(a, dtype=float).ravel()) for a in arrays]
    )
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -1.0, 1.0
    span = float(np.percentile(vals, p))
    if span <= 0.0:
        span = 1.0
    return -span, span


def cbar(fig, mappable, label, *, ax=None, cax=None, **kwargs):
    """One colourbar; ``label`` must include units."""
    if cax is not None:
        cb = fig.colorbar(mappable, cax=cax, **kwargs)
    else:
        kwargs.setdefault("fraction", 0.046)
        kwargs.setdefault("pad", 0.02)
        cb = fig.colorbar(mappable, ax=ax, **kwargs)
    cb.set_label(label, fontsize=LABEL_SIZE, labelpad=3)
    cb.ax.tick_params(labelsize=TICK_SIZE, width=0.6, length=3)
    if cb.outline is not None:
        cb.outline.set_linewidth(0.6)
    return cb


def data_model_residual_grid(
    fig,
    nrows: int,
    *,
    left=0.10,
    right=0.90,
    top=0.90,
    bottom=0.08,
    hspace=0.16,
    wspace=0.12,
):
    """Equal panels: Data | Model | pair-cbar | Residual | residual-cbar."""
    from matplotlib.gridspec import GridSpec

    gs = GridSpec(
        nrows,
        5,
        figure=fig,
        width_ratios=[1.0, 1.0, 0.055, 1.0, 0.055],
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        wspace=wspace,
        hspace=hspace,
    )
    axes, cax_pair, cax_res = [], [], []
    for i in range(nrows):
        ax0 = fig.add_subplot(gs[i, 0])
        share = dict(sharex=ax0, sharey=ax0)
        axes.append(
            [ax0, fig.add_subplot(gs[i, 1], **share), fig.add_subplot(gs[i, 3], **share)]
        )
        cax_pair.append(fig.add_subplot(gs[i, 2]))
        cax_res.append(fig.add_subplot(gs[i, 4]))
        for ax in axes[i][1:]:
            ax.tick_params(labelleft=False)
        if i < nrows - 1:
            for ax in axes[i]:
                ax.tick_params(labelbottom=False)
    return axes, cax_pair, cax_res


def beam_ellipse(ax, bmaj_arcsec, bmin_arcsec, bpa_deg, xy):
    """Restoring beam in the east/north plane. ``bpa_deg`` is east of north.

    Place on the data column (M0, or every data-column panel). ``xy`` is the
    ellipse centre in east/north arcsec (lower-left of the cropped map).
    """
    ell = Ellipse(
        xy,
        width=float(bmaj_arcsec),
        height=float(bmin_arcsec),
        angle=90.0 - float(bpa_deg),
        facecolor=COLOUR["beam_face"],
        edgecolor=COLOUR["beam_edge"],
        lw=0.7,
        zorder=5,
    )
    ax.add_patch(ell)
    return ell


def panel_letter(ax, letter, *, x=0.06, y=0.94):
    ax.text(
        x,
        y,
        f"({letter})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=COLOUR["text"],
        zorder=10,
        bbox={
            "boxstyle": "round,pad=0.12",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
        },
    )


def vsys_line(ax, vsys, *, orientation="v"):
    """Dashed grey systemic-velocity line. Not a legend entry."""
    kw = dict(color=COLOUR["vsys"], ls="--", lw=0.8, zorder=4)
    if orientation == "v":
        ax.axvline(float(vsys), **kw)
    else:
        ax.axhline(float(vsys), **kw)


def save_fig(fig, path, *, dpi: int | None = None) -> Path:
    """Write a white-background PNG at publication dpi and close the figure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=DPI if dpi is None else int(dpi),
        facecolor="white",
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)
    return path
