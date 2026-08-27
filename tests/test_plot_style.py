"""Plot style helpers. No /arc FITS required."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from kinuv.diagnostics.style import apply_style, format_sky_ax, save_fig


def test_apply_style_callable():
    apply_style()


def test_save_fig_writes(tmp_path):
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0])
    out = tmp_path / "style_probe.png"
    save_fig(fig, out)
    assert out.is_file()
    assert out.stat().st_size > 0


def test_format_sky_ax_east_left():
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots()
    format_sky_ax(ax, crop=12.0)
    x0, x1 = ax.get_xlim()
    assert x0 > x1
    plt.close(fig)
