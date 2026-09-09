"""Publication figure style for kinUV image-plane diagnostics.

Import this module. Do not copy rcParams into scripts. Cosmetics only:
matching physics stays in ``kinuv.diagnostics.imaging``.
"""

from __future__ import annotations

from pathlib import Path
import warnings

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

# Fallback sequential intensity palette. Production plots use matplotlib's
# perceptually uniform ``magma`` map; the fallback remains registered for old
# artifacts that explicitly request ``kinuv_intensity``.
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

# Geometry and base rcParams follow drtobybrown/apj-formatter (MIT), vendored
# here to keep kinUV standalone. Production typography uses a larger 12-point
# base so labels and 10-point ticks remain legible in dense science panels.
SINGLE_COLUMN_WIDTH = 3.5
DOUBLE_COLUMN_WIDTH = 7.1
MAX_PAGE_HEIGHT = 9.0
DEFAULT_ASPECT = 0.60
DPI = 300
CROP_ARCSEC = 12.0
TITLE_SIZE = 12
TICK_SIZE = 10
LABEL_SIZE = 12
LEGEND_SIZE = 11

__all__ = [
    "CMAP_RESIDUAL",
    "CMAP_VELOCITY",
    "COLOUR",
    "CROP_ARCSEC",
    "DEFAULT_ASPECT",
    "DPI",
    "DOUBLE_COLUMN_WIDTH",
    "INTENSITY_HEX",
    "MAX_PAGE_HEIGHT",
    "SINGLE_COLUMN_WIDTH",
    "apj_dimensions",
    "apj_rcparams",
    "apply_style",
    "beam_ellipse",
    "cbar",
    "data_model_residual_grid",
    "format_sky_ax",
    "imshow_masked",
    "intensity_cmap",
    "panel_letter",
    "publication_figure",
    "publication_subplots",
    "residual_cmap",
    "save_fig",
    "save_publication",
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


def apj_dimensions(
    columns: int = 1,
    *,
    aspect_ratio: float = DEFAULT_ASPECT,
    width_ratio: float = 1.0,
) -> tuple[float, float]:
    """Return the ApJ single- or double-column figure dimensions in inches."""

    if columns == 1:
        width = SINGLE_COLUMN_WIDTH * float(width_ratio)
    elif columns == 2:
        width = DOUBLE_COLUMN_WIDTH * float(width_ratio)
    else:
        raise ValueError("columns must be 1 or 2")
    if aspect_ratio <= 0.0 or width_ratio <= 0.0:
        raise ValueError("aspect_ratio and width_ratio must be positive")
    height = width * float(aspect_ratio)
    if height > MAX_PAGE_HEIGHT:
        warnings.warn(
            f"figure height {height:.2f} inches exceeds the ApJ page limit "
            f"of {MAX_PAGE_HEIGHT:.1f} inches",
            UserWarning,
            stacklevel=2,
        )
    return width, height


def apj_rcparams(
    columns: int = 2,
    *,
    aspect_ratio: float = DEFAULT_ASPECT,
    width_ratio: float = 1.0,
    fontsize_pt: float = LABEL_SIZE,
) -> dict:
    """Return the standalone ApJ rcParams contract used by kinUV."""

    width, height = apj_dimensions(
        columns, aspect_ratio=aspect_ratio, width_ratio=width_ratio
    )
    base = float(fontsize_pt)
    if base <= 0.0:
        raise ValueError("fontsize_pt must be positive")
    return {
        "figure.figsize": (width, height),
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "none",
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "TeX Gyre Termes",
            "DejaVu Serif",
            "STIXGeneral",
            "serif",
        ],
        "font.size": base,
        "mathtext.fontset": "stix",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "text.usetex": False,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "axes.titlepad": 4.0,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.unicode_minus": True,
        "lines.linewidth": 1.0,
        "lines.markersize": 4.0,
        "lines.markeredgewidth": 0.5,
        "xtick.labelsize": max(base - 2.0, 6.0),
        "ytick.labelsize": max(base - 2.0, 6.0),
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "legend.frameon": True,
        "legend.framealpha": 0.85,
        "legend.fancybox": False,
        "legend.edgecolor": "0.8",
        "legend.borderpad": 0.4,
        "legend.fontsize": max(base - 1.0, 6.0),
        "image.interpolation": "nearest",
        "image.origin": "lower",
        "image.cmap": "magma",
        "axes.prop_cycle": mpl.cycler(color=[COLOUR["model"], COLOUR["data"]]),
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def apply_style(
    *,
    columns: int = 2,
    aspect_ratio: float = DEFAULT_ASPECT,
    width_ratio: float = 1.0,
    fontsize_pt: float = LABEL_SIZE,
) -> tuple[float, float]:
    """Apply the vendored ApJ style and return its figure dimensions."""

    _register_cmaps()
    rc = apj_rcparams(
        columns,
        aspect_ratio=aspect_ratio,
        width_ratio=width_ratio,
        fontsize_pt=fontsize_pt,
    )
    mpl.rcParams.update(rc)
    return tuple(rc["figure.figsize"])


def publication_figure(*, columns: int = 1, aspect_ratio: float = DEFAULT_ASPECT, **kwargs):
    """Create an ApJ-sized figure using only the in-repo style contract."""

    size = apply_style(columns=columns, aspect_ratio=aspect_ratio)
    return plt.figure(figsize=size, **kwargs)


def publication_subplots(
    nrows: int = 1,
    ncols: int = 1,
    *,
    columns: int = 1,
    aspect_ratio: float = DEFAULT_ASPECT,
    **kwargs,
):
    """Create ApJ-sized subplots using only the in-repo style contract."""

    size = apply_style(columns=columns, aspect_ratio=aspect_ratio)
    return plt.subplots(nrows, ncols, figsize=size, **kwargs)


def _cmap_copy(name: str):
    _register_cmaps()
    return mpl.colormaps[name].with_extremes(bad=COLOUR["mask"])


def intensity_cmap():
    """Perceptually uniform high-dynamic-range map for M0, M2, and PV."""
    _register_cmaps()
    return _cmap_copy("magma")


def velocity_cmap():
    return _cmap_copy(CMAP_VELOCITY)


def residual_cmap():
    """Zero-white diverging map for signed data-minus-model residuals."""

    base = _cmap_copy(CMAP_RESIDUAL)
    colours = base(np.linspace(0.0, 1.0, 257))
    colours[128] = (1.0, 1.0, 1.0, 1.0)
    return mpl.colors.ListedColormap(colours, name="kinuv_residual").with_extremes(
        bad=COLOUR["mask"]
    )


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


def panel_letter(ax, letter, *, x=0.06, y=0.94, fontsize=TITLE_SIZE):
    ax.text(
        x,
        y,
        f"({letter})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
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


def save_publication(fig, stem, *, dpi: int = DPI) -> dict[str, Path]:
    """Write one figure as vector PDF and high-resolution PNG, then close it."""

    stem = Path(stem)
    if stem.suffix:
        raise ValueError("publication figure stem must not include a suffix")
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {"pdf": stem.with_suffix(".pdf"), "png": stem.with_suffix(".png")}
    for path in outputs.values():
        fig.savefig(
            path,
            dpi=int(dpi),
            facecolor="white",
            edgecolor="none",
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(fig)
    return outputs
