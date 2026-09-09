"""Final, stage-labelled production diagnostics.

These functions consume frozen cubes and parameter records.  They never fit or
rescore a model.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import numpy as np

from kinuv.diagnostics.imaging import offset_world, pv_diagram, spectral_axis_kms
from kinuv.diagnostics.kinms_benchmark import fits_sky_offsets_arcsec
from kinuv.diagnostics.style import (
    COLOUR,
    apply_style,
    beam_ellipse,
    cbar,
    format_sky_ax,
    imshow_masked,
    intensity_cmap,
    panel_letter,
    residual_cmap,
    save_publication,
    sequential_clim,
    sky_extent_arcsec,
    symmetric_clim,
    velocity_cmap,
)

KINMS_COLOUR = "#D55E00"
POSTERIOR_COLOUR = "#56B4E9"


def _save(figure, output: Path, name: str) -> list[Path]:
    products = save_publication(figure, output / name)
    return [products["pdf"], products["png"]]


def _integrated_spectrum(cube, support, header):
    velocity = spectral_axis_kms(header)
    rest_hz = float(header["RESTFRQ"])
    frequency_hz = rest_hz / (1.0 + float(np.nanmedian(velocity)) / 299792.458)
    beam_area_deg2 = (
        np.pi * float(header["BMAJ"]) * float(header["BMIN"]) / (4.0 * np.log(2.0))
    )
    pixel_area_deg2 = abs(float(header["CDELT1"]) * float(header["CDELT2"]))
    jy_per_beam_per_k = (
        2.0
        * 1.380649e-23
        * frequency_hz**2
        / 299792458.0**2
        * beam_area_deg2
        * (np.pi / 180.0) ** 2
        * 1.0e26
    )
    kelvin_sum = np.nansum(np.where(support[None, :, :], cube, np.nan), axis=(1, 2))
    return kelvin_sum * jy_per_beam_per_k * pixel_area_deg2 / beam_area_deg2


def _crop(moment0, header, centre, beam):
    east, north = fits_sky_offsets_arcsec(header, moment0.shape)
    finite = np.isfinite(moment0)
    support = finite & (moment0 > 0.05 * np.nanmax(moment0))
    if not np.any(support):
        support = finite
    radius = max(
        float(np.max(np.abs(east[support] - centre[0]))),
        float(np.max(np.abs(north[support] - centre[1]))),
    )
    extent = sky_extent_arcsec(header)
    full = 0.48 * min(abs(extent[1] - extent[0]), abs(extent[3] - extent[2]))
    return min(full, max(3.0 * beam, 1.15 * radius + beam))


def render_moments(target, moments, header, geometry, stage, output):
    """Render M0/M1/M2 as Data | kinUV | KinMS | two residuals."""

    apply_style(columns=2, aspect_ratio=7.0 / 11.5, width_ratio=11.5 / 7.1)
    figure = plt.figure(figsize=(11.5, 7.0))
    grid = GridSpec(
        3, 5, figure=figure,
        left=0.06, right=0.80, bottom=0.13, top=0.91, wspace=0.06, hspace=0.08,
    )
    extent = sky_extent_arcsec(header)
    centre = (float(geometry["dx_arcsec"]), float(geometry["dy_arcsec"]))
    beam = (float(header["BMAJ"]) * 3600.0, float(header["BMIN"]) * 3600.0, float(header["BPA"]))
    crop = _crop(moments["data_moment0"], header, centre, beam[0])
    vsys = float(geometry["vsys_kms"])
    rows = (
        ("moment0", "Moment 0", r"$I_{\rm CO}\ (\mathrm{K\ km\ s^{-1}})$", False),
        ("moment1", "Moment 1", r"$v-v_{\rm sys}\ (\mathrm{km\ s^{-1}})$", True),
        ("moment2", "Moment 2", r"$\sigma_v\ (\mathrm{km\ s^{-1}})$", False),
    )
    titles = ("Data", f"kinUV {stage}", "KinMS", f"Data - kinUV {stage}", "Data - KinMS")
    letters = iter("abcdefghijklmno")
    for row, (key, row_name, unit, velocity) in enumerate(rows):
        images = [
            moments[f"data_{key}"], moments[f"kinuv_{key}"], moments[f"kinms_{key}"],
            moments[f"data_minus_kinuv_{key}"], moments[f"data_minus_kinms_{key}"],
        ]
        if velocity:
            images[:3] = [image - vsys for image in images[:3]]
            common_limits, common_cmap = symmetric_clim(*images[:3], p=99), velocity_cmap()
        else:
            common_limits, common_cmap = sequential_clim(*images[:3], p=99), intensity_cmap()
        residual_limits = symmetric_clim(*images[3:], p=97.5)
        common_mask = np.isfinite(images[0])
        images = [np.where(common_mask, image, np.nan) for image in images]
        common_artist = residual_artist = None
        axes = []
        for column, image in enumerate(images):
            axis = figure.add_subplot(grid[row, column])
            axes.append(axis)
            limits = common_limits if column < 3 else residual_limits
            cmap = common_cmap if column < 3 else residual_cmap()
            artist = imshow_masked(axis, image, extent, *limits, cmap)
            common_artist = artist if column < 3 else common_artist
            residual_artist = artist if column >= 3 else residual_artist
            format_sky_ax(axis, crop, centre)
            axis.tick_params(labelbottom=row == 2, labelleft=column == 0)
            if row == 0:
                axis.set_title(titles[column], fontsize=11)
            panel_letter(axis, next(letters), fontsize=11)
        axes[0].text(0.96, 0.94, row_name, transform=axes[0].transAxes, ha="right", va="top", fontsize=10,
                     bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})
        bounds = axes[-1].get_position()
        common_cax = figure.add_axes((0.835, bounds.y0, 0.016, bounds.height))
        residual_cax = figure.add_axes((0.925, bounds.y0, 0.016, bounds.height))
        common_bar = cbar(figure, common_artist, unit, cax=common_cax, orientation="vertical")
        residual_bar = cbar(figure, residual_artist, r"$\Delta$ " + unit, cax=residual_cax, orientation="vertical")
        common_bar.locator = MaxNLocator(nbins=5)
        residual_bar.locator = MaxNLocator(nbins=3, symmetric=True)
        common_bar.update_ticks(); residual_bar.update_ticks()
        common_bar.ax.yaxis.set_ticks_position("left")
        common_bar.ax.yaxis.set_label_position("left")
        residual_bar.ax.yaxis.set_ticks_position("right")
        residual_bar.ax.yaxis.set_label_position("right")
        if row == 0:
            beam_ellipse(axes[0], *beam, (centre[0] + crop - 1.4, centre[1] - crop + 1.4))
    figure.suptitle(f"{target}: matched moments - kinUV {stage}", y=0.975)
    figure.supxlabel(r"East offset (arcsec)", y=0.055)
    figure.supylabel(r"North offset (arcsec)", x=0.012)
    figure.text(
        0.5, 0.012,
        r"east left, north up  $\cdot$  optical LSRK  $\cdot$  identical clim per row  $\cdot$  restoring beam on M0 data",
        ha="center", va="bottom", fontsize=10, color=COLOUR["muted"],
    )
    return _save(figure, output, "moments_kinuv_vs_kinms")


def _curve_on_offsets(offset, radius, speed, vsys):
    return float(vsys) + np.sign(offset) * np.interp(np.abs(offset), radius, speed)


def render_pvd(target, cubes, mask, header, geometry, stage, profiles, output):
    """Render the matched major-axis PVD and intrinsic rotation comparison."""

    ra, dec = offset_world(float(header["CRVAL1"]), float(header["CRVAL2"]),
                           float(geometry["dx_arcsec"]), float(geometry["dy_arcsec"]))
    beam = float(header["BMAJ"]) * 3600.0
    width = float(header["BMIN"]) * 3600.0
    crop = _crop(profiles["moment0"], header, (geometry["dx_arcsec"], geometry["dy_arcsec"]), beam)
    angle = float(geometry["pa_deg"]) % 360.0
    pvs, offset = [], None
    for cube in cubes:
        pv, current = pv_diagram(cube, header, ra, dec, angle, 2 * crop, width)
        if offset is not None and not np.allclose(offset, current):
            raise ValueError("matched PVD offsets differ")
        pvs.append(pv); offset = current
    pv_support, support_offset = pv_diagram(mask.astype(float), header, ra, dec, angle, 2 * crop, width)
    if not np.allclose(offset, support_offset):
        raise ValueError("PVD signal-mask offsets differ")
    signal = np.isfinite(pv_support) & (pv_support > 0.05)
    pvs = [np.where(signal, pv, np.nan) for pv in pvs]
    velocity = spectral_axis_kms(header)
    residuals = [pvs[0] - pvs[1], pvs[0] - pvs[2]]
    common_limits = sequential_clim(*pvs, p=99.2)
    residual_limits = symmetric_clim(*residuals, p=97.5)
    apply_style(columns=2, aspect_ratio=6.8 / 10.2, width_ratio=10.2 / 7.1)
    figure = plt.figure(figsize=(10.2, 6.8))
    grid = GridSpec(3, 5, figure=figure, height_ratios=(1.28, 0.07, 1.0), left=0.065, right=0.98,
                    bottom=0.10, top=0.91, wspace=0.06, hspace=0.31)
    titles = ("Data", f"kinUV {stage}", "KinMS", f"Data - kinUV {stage}", "Data - KinMS")
    images = pvs + residuals
    extent = (float(offset[0]), float(offset[-1]), float(velocity[0]), float(velocity[-1]))
    for column, image in enumerate(images):
        axis = figure.add_subplot(grid[0, column])
        artist = imshow_masked(axis, image, extent, *(common_limits if column < 3 else residual_limits),
                               intensity_cmap() if column < 3 else residual_cmap(), aspect="auto")
        if column < 3: common_artist = artist
        else: residual_artist = artist
        axis.axvline(0.0, color="white", lw=0.8, alpha=0.9)
        axis.axhline(float(geometry["vsys_kms"]), color="white", lw=0.8, ls=":", alpha=0.9)
        axis.axvspan(-beam / 2, beam / 2, color="white", alpha=0.10)
        if column in (0, 1, 3):
            curve = _curve_on_offsets(offset, profiles["radius"], profiles["kinuv_projected"], geometry["vsys_kms"])
            if profiles.get("projected_lo") is not None:
                lo = _curve_on_offsets(offset, profiles["radius"], profiles["projected_lo"], geometry["vsys_kms"])
                hi = _curve_on_offsets(offset, profiles["radius"], profiles["projected_hi"], geometry["vsys_kms"])
                axis.fill_between(offset, np.minimum(lo, hi), np.maximum(lo, hi), color=POSTERIOR_COLOUR, alpha=0.25)
            axis.plot(offset, curve, color="#00E5FF", lw=1.6, label=f"kinUV {stage}")
        if column in (0, 2, 4):
            curve = _curve_on_offsets(offset, profiles["radius"], profiles["kinms_projected"], profiles["kinms_vsys"])
            axis.plot(offset, curve, color="#FFD166", lw=1.5, ls="--", label="KinMS")
        axis.set_title(titles[column], fontsize=11)
        if column == 2:
            axis.set_xlabel(r"Major-axis offset (arcsec; receding $+$)")
        if column == 0:
            axis.set_ylabel(r"$v_{\rm opt,LSRK}\ (\mathrm{km\ s^{-1}})$")
            axis.legend(loc="upper right", fontsize=8)
            axis.text(0.04, 0.04, f"Major axis\nPA = {angle:.1f} deg", transform=axis.transAxes,
                      color="white", fontsize=9, bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.65})
        else:
            axis.tick_params(labelleft=False)
        panel_letter(axis, chr(ord("a") + column), fontsize=11)
    cbar(figure, common_artist, r"$T_{\rm B}\ (\mathrm{K})$", cax=figure.add_subplot(grid[1, :3]), orientation="horizontal")
    bar = cbar(figure, residual_artist, r"$\Delta T_{\rm B}\ (\mathrm{K})$", cax=figure.add_subplot(grid[1, 3:]), orientation="horizontal")
    bar.locator = MaxNLocator(nbins=3, symmetric=True); bar.update_ticks()

    axis = figure.add_subplot(grid[2, :])
    radius = profiles["radius"]
    axis.axvspan(0, beam, color="0.92", label=r"$R\leq1\,\mathrm{BMAJ}$")
    axis.plot(radius, profiles["kinuv_intrinsic"], color=COLOUR["model"], lw=2.1, label=f"kinUV {stage}")
    if profiles.get("intrinsic_lo") is not None:
        axis.fill_between(radius, profiles["intrinsic_lo"], profiles["intrinsic_hi"],
                          color=POSTERIOR_COLOUR, alpha=0.28, label="NUTS 16th-84th percentile")
    axis.plot(radius, profiles["kinms_intrinsic"], color=KINMS_COLOUR, lw=1.7, ls="--", label="KinMS")
    if profiles.get("turnover") is not None:
        axis.axvline(profiles["turnover"], color=COLOUR["model"], ls=":", lw=1.1)
    axis.set(xlabel=r"Galactocentric radius $R\ (\mathrm{arcsec})$",
             ylabel=r"$V_{\rm c}\ (\mathrm{km\ s^{-1}})$", xlim=(0, float(radius[-1])), ylim=(0, None))
    axis.xaxis.set_minor_locator(AutoMinorLocator()); axis.yaxis.set_minor_locator(AutoMinorLocator())
    axis.legend(fontsize=9, ncol=3, loc="lower right")
    metrics = profiles["profile_metrics"]
    badge = "\n".join((
        metrics["turnover"], metrics["velocity"], metrics["inner_gradient"], metrics["smearing"],
    ))
    axis.text(0.080, 0.96, badge, transform=axis.transAxes, ha="left", va="top", fontsize=10,
              bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "0.45", "alpha": 0.94})
    panel_letter(axis, "f", fontsize=11)
    figure.suptitle(f"{target}: major-axis PVD and intrinsic rotation - kinUV {stage}", y=0.975)
    return _save(figure, output, "pvd_kinuv_vs_kinms")


def render_spectra(target, cubes, mask, header, geometry, stage, output):
    """Render full-disk and one-BMAJ spectra with residual tracks."""

    spatial = np.any(mask, axis=0)
    east, north = fits_sky_offsets_arcsec(header, spatial.shape)
    centre = (float(geometry["dx_arcsec"]), float(geometry["dy_arcsec"]))
    beam = float(header["BMAJ"]) * 3600.0
    central = spatial & (np.hypot(east - centre[0], north - centre[1]) <= beam)
    velocity = spectral_axis_kms(header)
    apply_style(columns=2, aspect_ratio=5.7 / 7.1)
    figure, axes = plt.subplots(2, 2, figsize=(7.1, 5.7), sharex="col",
                               gridspec_kw={"height_ratios": (2.1, 1.0), "hspace": 0.07, "wspace": 0.25})
    for column, (title, support) in enumerate((("Integrated disk", spatial), (r"Central $R\leq1\,\mathrm{BMAJ}$", central))):
        data, kinuv, kinms = [_integrated_spectrum(cube, support, header) for cube in cubes]
        top, bottom = axes[0, column], axes[1, column]
        top.step(velocity, data, where="mid", color=COLOUR["data"], label="Data")
        top.step(velocity, kinuv, where="mid", color=COLOUR["model"], label=f"kinUV {stage}")
        top.step(velocity, kinms, where="mid", color=KINMS_COLOUR, label="KinMS")
        bottom.step(velocity, data - kinuv, where="mid", color=COLOUR["model"], label=f"Data - kinUV {stage}")
        bottom.step(velocity, data - kinms, where="mid", color=KINMS_COLOUR, label="Data - KinMS")
        bottom.axhline(0, color=COLOUR["zero"], lw=1)
        for axis in (top, bottom):
            axis.axvline(float(geometry["vsys_kms"]), color=COLOUR["vsys"], lw=0.8, ls=":")
            axis.xaxis.set_minor_locator(AutoMinorLocator()); axis.yaxis.set_minor_locator(AutoMinorLocator())
        top.set_title(title); top.tick_params(labelbottom=False)
        bottom.set_xlabel(r"$v_{\rm opt,LSRK}\ (\mathrm{km\ s^{-1}})$")
        if column == 0:
            top.set_ylabel(r"$S_\nu\ (\mathrm{Jy})$"); bottom.set_ylabel(r"Residual (Jy)")
        top.legend(fontsize=8.5, loc="upper right")
        bottom.legend(fontsize=7.5, loc="best")
        panel_letter(top, "a" if column == 0 else "b"); panel_letter(bottom, "c" if column == 0 else "d")
    figure.suptitle(f"{target}: aperture spectra - kinUV {stage}", y=0.98)
    figure.subplots_adjust(left=0.11, right=0.98, bottom=0.13, top=0.89)
    return _save(figure, output, "spectra_kinuv_vs_kinms")


def render_radial_profiles(target, profiles, stage, output, *, benchmark):
    """Render intrinsic rotation and dispersion in a vertical two-panel layout."""

    apply_style(columns=2, aspect_ratio=6.2 / 7.1)
    figure, axes = plt.subplots(2, 1, figsize=(7.1, 6.2), sharex=True,
                               gridspec_kw={"hspace": 0.08, "height_ratios": (1.65, 1.0)})
    radius = profiles["radius"]
    for axis in axes:
        axis.axvspan(0, profiles["beam"], color="0.92", label=r"$R\leq1\,\mathrm{BMAJ}$")
    axes[0].plot(radius, profiles["kinuv_intrinsic"], color=COLOUR["model"], lw=2.1, label=f"kinUV {stage}")
    if profiles.get("intrinsic_lo") is not None:
        axes[0].fill_between(radius, profiles["intrinsic_lo"], profiles["intrinsic_hi"],
                             color=POSTERIOR_COLOUR, alpha=0.28, label="NUTS 16th-84th percentile")
    if benchmark:
        axes[0].plot(radius, profiles["kinms_intrinsic"], color=KINMS_COLOUR, lw=1.7, ls="--", label="KinMS intrinsic")
    if profiles.get("turnover") is not None:
        axes[0].axvline(profiles["turnover"], color=COLOUR["model"], ls=":", lw=1.1, label=r"kinUV $R_{\rm turn}$")
    axes[0].set_ylabel(r"$V_{\rm c}\ (\mathrm{km\ s^{-1}})$")
    axes[0].legend(fontsize=9, ncol=2, loc="lower right")
    axes[1].plot(radius, profiles["sigma_map"], color=COLOUR["model"], lw=2.1, label=f"kinUV {stage}")
    if profiles.get("sigma_lo") is not None:
        axes[1].fill_between(radius, profiles["sigma_lo"], profiles["sigma_hi"],
                             color=POSTERIOR_COLOUR, alpha=0.28, label="NUTS 16th-84th percentile")
    if benchmark:
        axes[1].plot(radius, np.full_like(radius, profiles["kinms_sigma"]), color=KINMS_COLOUR, lw=1.7, ls="--", label="KinMS")
    axes[1].set(xlabel=r"Galactocentric radius $R\ (\mathrm{arcsec})$",
                ylabel=r"$\sigma(R)\ (\mathrm{km\ s^{-1}})$", xlim=(0, float(radius[-1])), ylim=(0, None))
    axes[1].legend(fontsize=9, ncol=2, loc="best")
    for axis in axes:
        axis.xaxis.set_minor_locator(AutoMinorLocator()); axis.yaxis.set_minor_locator(AutoMinorLocator())
    panel_letter(axes[0], "a"); panel_letter(axes[1], "b")
    kind = "kinUV versus KinMS" if benchmark else "kinUV"
    figure.suptitle(f"{target}: {kind} radial profiles - {stage}", y=0.98)
    figure.subplots_adjust(left=0.13, right=0.98, bottom=0.12, top=0.91)
    return _save(figure, output, "rotation_curve_kinuv_vs_kinms" if benchmark else "rotation_curve")


def render_synthetic(target, synthetic, subbeam, output):
    """Render registered mock recovery with explicit truth turnover annotation."""

    from kinuv.validation.s4 import projected_arctan_speed

    beam = float(subbeam["bmaj_arcsec"]); truth = synthetic["truth"]
    radius = np.linspace(0, 3.5 * beam, 240)
    truth_profile = projected_arctan_speed(truth, radius)
    kinuv = np.asarray([projected_arctan_speed(row["kinuv"]["parameters"], radius) for row in synthetic["realizations"]])
    kinms = np.asarray([projected_arctan_speed(row["kinms"]["parameters"], radius) for row in synthetic["realizations"]])
    apply_style(columns=2, aspect_ratio=7.2 / 7.1)
    figure = plt.figure(figsize=(7.1, 7.2)); grid = GridSpec(2, 2, figure=figure, height_ratios=(1.35, 1), hspace=0.38, wspace=0.32)
    axis = figure.add_subplot(grid[0, :]); x = radius / beam; rt = float(truth["r_t_arcsec"]); rt_b = rt / beam
    axis.axvspan(0, 1, color="0.93", label=r"Inner $1\,\mathrm{BMAJ}$")
    axis.fill_between(x, np.min(kinuv, axis=0), np.max(kinuv, axis=0), color=COLOUR["model"], alpha=0.20)
    axis.fill_between(x, np.min(kinms, axis=0), np.max(kinms, axis=0), color=KINMS_COLOUR, alpha=0.18)
    axis.plot(x, truth_profile, color="black", lw=2.2, label="Truth")
    axis.plot(x, np.median(kinuv, axis=0), color=COLOUR["model"], lw=1.8, label="kinUV median")
    axis.plot(x, np.median(kinms, axis=0), color=KINMS_COLOUR, lw=1.8, label="KinMS median")
    axis.axvline(rt_b, color="black", ls=":", lw=1.2)
    axis.annotate(fr"Truth $R_{{\rm turn}}={rt:.3f}''={rt_b:.3f}\,\mathrm{{BMAJ}}$", xy=(rt_b, 0.96), xycoords=("data", "axes fraction"),
                  xytext=(8, -3), textcoords="offset points", ha="left", va="top", fontsize=10)
    axis.axvline(1, color="0.45", ls="--", lw=1)
    axis.set(xlabel=r"$R/\mathrm{BMAJ}$", ylabel=r"$u(R)=V_{\rm c}\sin i\ (\mathrm{km\ s^{-1}})$", xlim=(0, 3.5), title="Registered matched-family recovery")
    axis.legend(fontsize=9, loc="lower right"); panel_letter(axis, "a")
    aggregate = subbeam["aggregate"]
    values = ((aggregate["kinuv_mean_absolute_turnover_error_arcsec"] / beam, aggregate["kinms_mean_absolute_turnover_error_arcsec"] / beam,
               r"Mean $|\Delta R_{\rm turn}|/\mathrm{BMAJ}$", aggregate["absolute_turnover_error_ratio_kinuv_over_kinms"]),
              (aggregate["kinuv_inner_rms_rmse_kms"], aggregate["kinms_inner_rms_rmse_kms"],
               r"Inner $u(R)$ RMSE $(\mathrm{km\ s^{-1}})$", aggregate["inner_rmse_ratio_kinuv_over_kinms"]))
    for subaxis, vals, letter in zip((figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1])), values, "bc"):
        a, b, label, ratio = vals
        subaxis.bar((0, 1), (a, b), color=(COLOUR["model"], KINMS_COLOUR), width=0.68)
        subaxis.set_xticks((0, 1), ("kinUV", "KinMS")); subaxis.set_ylabel(label); subaxis.set_ylim(bottom=0)
        subaxis.text(0.5, 0.96, f"ratio = {ratio:.3f}", transform=subaxis.transAxes, ha="center", va="top", fontsize=11)
        panel_letter(subaxis, letter)
    figure.suptitle(f"{target}: synthetic sub-beam kinematic recovery", y=0.99)
    return _save(figure, output, "synthetic_benchmark")
