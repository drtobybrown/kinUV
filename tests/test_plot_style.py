"""Plot style helpers. No /arc FITS required."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from kinuv.diagnostics.figures import plot_chi2_slices, plot_leftover_chi2
from kinuv.diagnostics.style import (
    apj_dimensions,
    apply_style,
    format_sky_ax,
    save_fig,
    save_publication,
)


def test_apply_style_callable():
    import matplotlib as mpl

    apply_style()
    assert float(mpl.rcParams["axes.labelsize"]) >= 12.0
    assert float(mpl.rcParams["xtick.labelsize"]) >= 10.0
    assert float(mpl.rcParams["ytick.labelsize"]) >= 10.0
    assert mpl.rcParams["mathtext.fontset"] == "stix"
    assert mpl.rcParams["pdf.fonttype"] == 42


def test_apj_dimensions_match_journal_columns():
    assert apj_dimensions(1) == (3.5, 2.1)
    assert apj_dimensions(2) == (7.1, 4.26)


def test_save_fig_writes(tmp_path):
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0])
    out = tmp_path / "style_probe.png"
    save_fig(fig, out)
    assert out.is_file()
    assert out.stat().st_size > 0


def test_save_publication_writes_vector_and_raster(tmp_path):
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0])
    outputs = save_publication(fig, tmp_path / "publication")
    assert outputs["pdf"].is_file() and outputs["pdf"].stat().st_size > 0
    assert outputs["png"].is_file() and outputs["png"].stat().st_size > 0


def test_format_sky_ax_east_left():
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots()
    format_sky_ax(ax, crop=12.0)
    x0, x1 = ax.get_xlim()
    assert x0 > x1
    plt.close(fig)


def test_leftover_and_slice_figures_write(tmp_path):
    import numpy as np

    b = np.linspace(10.0, 200.0, 40)
    row = 1.0 + 0.01 * b
    vel = np.linspace(7900.0, 8300.0, 20)
    chan = np.ones(20)
    p1 = plot_leftover_chi2(b, row, vel, chan, tmp_path / "leftover.png")
    assert p1.is_file() and p1.stat().st_size > 0

    pa = np.linspace(198.0, 202.0, 5)
    sig = np.linspace(6.0, 10.0, 4)
    rt = np.linspace(0.2, 0.4, 4)
    i_deg = np.linspace(40.0, 48.0, 4)
    z = np.outer(np.ones(4), (pa - 200.0) ** 2) + np.outer((sig - 8.0) ** 2, np.ones(5))
    z2 = np.outer(np.ones(4), (sig - 8.0) ** 2) + np.outer((i_deg - 44.0) ** 2, np.ones(4))
    z3 = np.outer(np.ones(4), (pa - 200.0) ** 2) + np.outer((rt - 0.3) ** 2, np.ones(5))
    p2 = plot_chi2_slices(
        pa,
        sig,
        rt,
        i_deg,
        z,
        z2,
        z3,
        {"pa_deg": 200.0, "gas_sigma_kms": 8.0, "r_t_arcsec": 0.3, "i_deg": 44.0},
        tmp_path / "slices.png",
    )
    assert p2.is_file() and p2.stat().st_size > 0
