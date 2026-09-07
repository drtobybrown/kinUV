#!/usr/bin/env python3
"""Build the canonical production hierarchy and ApJ diagnostic suite.

The script consumes an accepted model bundle and accepted S4 synthetic records.
It performs no fit and changes no scientific parameter. It writes the complete
``best_model/``, ``plots/``, and ``benchmarks/`` target hierarchy with hashed
provenance and no legacy figures.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter

from kinuv.diagnostics.imaging import offset_world, pv_diagram, spectral_axis_kms
from kinuv.diagnostics.kinms_benchmark import (
    fits_sky_offsets_arcsec,
    major_axis_rotation_profile,
)
from kinuv.diagnostics.style import (
    COLOUR,
    CROP_ARCSEC,
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
    vsys_line,
)
from kinuv.validation.s4 import projected_arctan_speed


REPO = Path(__file__).resolve().parents[1]
KINMS_COLOUR = "#D55E00"
FIDUCIAL_COLOUR = "#6A3D9A"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def git_state() -> dict:
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=REPO, text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=REPO, text=True
            ).strip()
        ),
    }


def accepted_run(production_root: Path, target_id: str) -> Path:
    standardized = production_root / target_id / "best_model" / "summary.json"
    if standardized.is_file():
        summary = load_json(standardized)
        if summary.get("status") == "accepted" and summary.get("target_id") == target_id:
            return standardized.parent
    matches = []
    for summary_path in sorted((production_root / target_id).glob("kinuv-*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") == "accepted" and summary.get("target_id") == target_id:
            matches.append(summary_path.parent)
    if len(matches) != 1:
        raise ValueError(f"expected one accepted run for {target_id}; found {matches}")
    return matches[0]


def source_paths(run: Path, config_path: Path, synthetic_root: Path, subbeam_root: Path) -> dict:
    """Resolve equivalent inputs from a legacy bundle or standardized target."""

    target_id = load_json(config_path)["target_id"]
    if run.name == "best_model":
        target_root = run.parent
        return {
            "config": run / "config.json",
            "run_summary": run / "summary.json",
            "stage_a": run / "stage_a_map.json",
            "stage_b": run / "stage_b_map.json",
            "plot_summary": run / "plot_summary.json",
            "data_cube": Path(load_json(config_path)["diagnostic_cube"]),
            "mask_cube": Path(load_json(config_path)["diagnostic_mask"]),
            "model_cube": run / "model_on_10kms.fits",
            "model_native": run / "model_native.fits",
            "rotation_curve": run / "rotation_curve.npz",
            "posterior_samples": run / "posterior/posterior_samples.json",
            "posterior_summary": run / "posterior/summary.json",
            "benchmark_record": target_root / "benchmarks/benchmark.json",
            "moments": target_root / "benchmarks/moments.npz",
            "kinms_cube": target_root / "benchmarks/kinms_model_k.fits",
            "kinms_fit": target_root / "benchmarks/kinms_fit_result.json",
            "synthetic": synthetic_root / target_id / "summary.json",
            "subbeam": subbeam_root / target_id / "summary.json",
        }
    config = load_json(config_path)
    return {
        "config": run / "config.json",
        "run_summary": run / "summary.json",
        "stage_a": run / "stage_a_map.json",
        "stage_b": run / "stage_b_map.json",
        "plot_summary": run / "plots/summary.json",
        "data_cube": Path(config["diagnostic_cube"]),
        "mask_cube": Path(config["diagnostic_mask"]),
        "model_cube": run / "plots/model_on_10kms.fits",
        "model_native": run / "plots/model_native.fits",
        "rotation_curve": run / "plots/rotation_curve.npz",
        "posterior_samples": run / "posterior/posterior_samples.json",
        "posterior_summary": run / "posterior/summary.json",
        "benchmark_record": run / "benchmark/benchmark.json",
        "moments": run / "benchmark/moments.npz",
        "kinms_cube": run / "benchmark/kinms_model_k.fits",
        "kinms_fit": run / "benchmark/kinms_fit_result.json",
        "synthetic": synthetic_root / target_id / "summary.json",
        "subbeam": subbeam_root / target_id / "summary.json",
    }


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_cube(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float64).squeeze(), hdul[0].header.copy()


def save_pair(fig, output_dir: Path, name: str) -> list[Path]:
    products = save_publication(fig, output_dir / name)
    return [products["pdf"], products["png"]]


def source_crop_arcsec(
    moment0: np.ndarray,
    header,
    centre: tuple[float, float],
    beam_major_arcsec: float,
) -> float:
    """Choose a square field that contains the detected source plus one beam."""

    image = np.asarray(moment0, dtype=np.float64)
    finite = np.isfinite(image)
    if not np.any(finite):
        raise ValueError("moment-0 map has no finite source support")
    threshold = 0.05 * float(np.nanmax(image))
    support = finite & (image > threshold)
    if not np.any(support):
        support = finite
    east, north = fits_sky_offsets_arcsec(header, image.shape)
    source_radius = max(
        float(np.max(np.abs(east[support] - centre[0]))),
        float(np.max(np.abs(north[support] - centre[1]))),
    )
    full_extent = sky_extent_arcsec(header)
    half_width = 0.48 * min(
        abs(full_extent[1] - full_extent[0]),
        abs(full_extent[3] - full_extent[2]),
    )
    return min(CROP_ARCSEC, half_width, max(3.0 * beam_major_arcsec, 1.15 * source_radius + beam_major_arcsec))


def moment_figure(
    target_id: str,
    moments: dict[str, np.ndarray],
    header,
    geometry: dict,
    output_dir: Path,
) -> list[Path]:
    figure_size = apply_style(columns=2, aspect_ratio=9.0 / 7.1)
    figure = plt.figure(figsize=figure_size)
    grid = GridSpec(
        6,
        3,
        figure=figure,
        height_ratios=(1.0, 0.11, 1.0, 0.11, 1.0, 0.11),
        left=0.11,
        right=0.98,
        bottom=0.13,
        top=0.92,
        wspace=0.08,
        hspace=0.28,
    )
    extent = sky_extent_arcsec(header)
    centre = (float(geometry["dx_arcsec"]), float(geometry["dy_arcsec"]))
    beam = (
        float(header["BMAJ"]) * 3600.0,
        float(header["BMIN"]) * 3600.0,
        float(header["BPA"]),
    )
    crop = source_crop_arcsec(moments["data_moment0"], header, centre, beam[0])
    vsys = float(geometry["vsys_kms"])
    rows = (
        ("moment0", r"Moment 0", r"$I_{\rm CO}\ (\mathrm{K\ km\ s^{-1}})$", False),
        ("moment1", r"Moment 1", r"$v-v_{\rm sys}\ (\mathrm{km\ s^{-1}})$", True),
        ("moment2", r"Moment 2", r"$\sigma_v\ (\mathrm{km\ s^{-1}})$", False),
    )
    letters = iter("abcdefghi")
    for row, (key, row_label, unit, velocity) in enumerate(rows):
        data = moments[f"data_{key}"]
        model = moments[f"kinuv_{key}"]
        residual = moments[f"data_minus_kinuv_{key}"]
        if velocity:
            data = data - vsys
            model = model - vsys
            low, high = symmetric_clim(data, model, p=99.0)
            cmap = velocity_cmap()
        else:
            low, high = sequential_clim(data, model, p=99.0)
            cmap = intensity_cmap()
        residual_low, residual_high = symmetric_clim(residual, p=97.5)
        axes = []
        pair_image = None
        residual_image = None
        for column, (image, vmin, vmax, colourmap) in enumerate(
            (
                (data, low, high, cmap),
                (model, low, high, cmap),
                (residual, residual_low, residual_high, residual_cmap()),
            )
        ):
            axis = figure.add_subplot(grid[2 * row, column])
            axes.append(axis)
            artist = imshow_masked(axis, image, extent, vmin, vmax, colourmap)
            if column < 2:
                pair_image = artist
            else:
                residual_image = artist
            format_sky_ax(
                axis,
                crop,
                centre,
                xlabel=False,
                ylabel=False,
            )
            axis.tick_params(labelbottom=row == 2, labelleft=column == 0)
            if row == 0:
                axis.set_title(("Data", "kinUV model", "Data - kinUV")[column])
            panel_letter(axis, next(letters))
        axes[0].text(
            0.04,
            0.05,
            row_label,
            transform=axes[0].transAxes,
            va="bottom",
            ha="left",
            fontsize=11,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80},
        )
        pair_bar = cbar(
            figure,
            pair_image,
            unit,
            cax=figure.add_subplot(grid[2 * row + 1, 0:2]),
            orientation="horizontal",
        )
        residual_bar = cbar(
            figure,
            residual_image,
            unit,
            cax=figure.add_subplot(grid[2 * row + 1, 2]),
            orientation="horizontal",
        )
        if row < 2:
            pair_bar.ax.xaxis.set_label_position("top")
            residual_bar.ax.xaxis.set_label_position("top")
        if row == 0:
            beam_ellipse(
                axes[0],
                *beam,
                (centre[0] + crop - 1.4, centre[1] - crop + 1.4),
            )
    figure.suptitle(f"{target_id}: visibility-fit moment diagnostics", y=0.975)
    figure.supxlabel(r"East offset (arcsec)", y=0.012)
    figure.supylabel(r"North offset (arcsec)", x=0.015)
    return save_pair(figure, output_dir, "moments_comparison")


def pv_figure(
    target_id: str,
    data_cube: np.ndarray,
    model_cube: np.ndarray,
    mask_cube: np.ndarray,
    header,
    geometry: dict,
    output_dir: Path,
) -> list[Path]:
    figure_size = apply_style(columns=2, aspect_ratio=6.8 / 7.1)
    spatial_support = np.any(mask_cube, axis=0)
    data = np.where(spatial_support[None, :, :], data_cube, np.nan)
    model = np.where(spatial_support[None, :, :], model_cube, np.nan)
    ra, dec = offset_world(
        float(header["CRVAL1"]),
        float(header["CRVAL2"]),
        float(geometry["dx_arcsec"]),
        float(geometry["dy_arcsec"]),
    )
    pa = float(geometry["pa_deg"])
    width = float(header["BMIN"]) * 3600.0
    rows = []
    for name, angle in (("Major axis", pa), ("Minor axis", pa + 90.0)):
        data_pv, offset = pv_diagram(data, header, ra, dec, angle, 16.0, width)
        model_pv, model_offset = pv_diagram(model, header, ra, dec, angle, 16.0, width)
        if not np.allclose(offset, model_offset):
            raise ValueError("data and model PVD offsets differ")
        rows.append((name, angle % 360.0, offset, data_pv, model_pv))
    intensity_limits = sequential_clim(
        *(item for row in rows for item in (row[3], row[4])), p=99.2
    )
    residual_limits = symmetric_clim(*(row[3] - row[4] for row in rows), p=97.5)
    velocity = spectral_axis_kms(header)
    vsys = float(geometry["vsys_kms"])
    figure = plt.figure(figsize=figure_size)
    grid = GridSpec(
        3,
        3,
        figure=figure,
        height_ratios=(1.0, 1.0, 0.065),
        left=0.09,
        right=0.98,
        bottom=0.13,
        top=0.90,
        wspace=0.08,
        hspace=0.10,
    )
    letters = iter("abcdef")
    for row_index, (name, angle, offset, data_pv, model_pv) in enumerate(rows):
        extent = (float(offset[0]), float(offset[-1]), float(velocity[0]), float(velocity[-1]))
        residual = data_pv - model_pv
        pair_image = None
        residual_image = None
        for column, (image, limits, colourmap) in enumerate(
            (
                (data_pv, intensity_limits, intensity_cmap()),
                (model_pv, intensity_limits, intensity_cmap()),
                (residual, residual_limits, residual_cmap()),
            )
        ):
            axis = figure.add_subplot(grid[row_index, column])
            artist = imshow_masked(
                axis, image, extent, limits[0], limits[1], colourmap, aspect="auto"
            )
            if column < 2:
                pair_image = artist
            else:
                residual_image = artist
            if row_index == 0:
                axis.set_title(("Data", "kinUV model", "Data - kinUV")[column])
                axis.tick_params(labelbottom=False)
            if column == 0:
                axis.set_ylabel(r"$v_{\rm opt,LSRK}\ (\mathrm{km\ s^{-1}})$")
                axis.text(
                    0.03,
                    0.04,
                    f"{name}\nPA = {angle:.1f} deg",
                    transform=axis.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=11,
                    color="white",
                    bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.65},
                )
            else:
                axis.tick_params(labelleft=False)
            vsys_line(axis, vsys, orientation="h")
            panel_letter(axis, next(letters))
    cbar(
        figure,
        pair_image,
        r"$T_{\rm B}\ (\mathrm{K})$",
        cax=figure.add_subplot(grid[2, 0:2]),
        orientation="horizontal",
    )
    cbar(
        figure,
        residual_image,
        r"$\Delta T_{\rm B}\ (\mathrm{K})$",
        cax=figure.add_subplot(grid[2, 2]),
        orientation="horizontal",
    )
    figure.suptitle(f"{target_id}: major- and minor-axis position-velocity diagrams", y=0.975)
    figure.supxlabel(r"Offset (arcsec; receding $+$)", y=0.015)
    return save_pair(figure, output_dir, "pv_diagrams")


def arctan_curve(radius: np.ndarray, speed: float, turnover: float) -> np.ndarray:
    return float(speed) * (2.0 / np.pi) * np.arctan(radius / float(turnover))


def rotation_figure(
    target_id: str,
    run_summary: dict,
    stage_a: dict,
    stage_b: dict,
    rotation: dict[str, np.ndarray],
    centroid_radius_arcsec: np.ndarray,
    centroid_speed_kms: np.ndarray,
    kinms_fit: dict,
    beam_arcsec: float,
    output_dir: Path,
) -> list[Path]:
    figure_size = apply_style(columns=2, aspect_ratio=5.4 / 7.1)
    radius = rotation["radius_arcsec"]
    selected = run_summary["selected_model"]
    selected_curve = rotation["stage_b_kms"] if selected == "stage_b" else rotation["stage_a_kms"]
    figure, axis = plt.subplots(figsize=figure_size)
    figure.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.30)
    axis.axvspan(0.0, beam_arcsec, color="0.90", zorder=0, label=r"Inner $1\times\mathrm{BMAJ}$")
    axis.plot(
        radius,
        selected_curve,
        color=COLOUR["model"],
        linewidth=2.2,
        label=f"kinUV selected ({selected.replace('_', ' ').title()})",
        zorder=4,
    )
    if selected != "stage_a":
        axis.plot(
            radius,
            rotation["stage_a_kms"],
            color=FIDUCIAL_COLOUR,
            linestyle="--",
            linewidth=1.5,
            label="Stage A arctan baseline",
        )
    if kinms_fit:
        axis.plot(
            radius,
            arctan_curve(radius, kinms_fit["v0_kms"], kinms_fit["r_t_arcsec"]),
            color=KINMS_COLOUR,
            linestyle=":",
            linewidth=1.7,
            label="KinMS arctan fit",
        )
    axis.scatter(
        centroid_radius_arcsec,
        centroid_speed_kms,
        facecolor="white",
        edgecolor=COLOUR["data"],
        linewidth=0.9,
        s=28,
        label="Restored-cube centroids (WCS-recomputed)",
        zorder=5,
    )
    turnover = (
        float(stage_b.get("r_t_recovered", stage_a["r_t_arcsec"]))
        if selected == "stage_b"
        else float(stage_a["r_t_arcsec"])
    )
    turnover_label = (
        r"$R_{\rm turn}$ (arctan-equivalent)"
        if selected == "stage_b"
        else r"$R_{\rm turn}$"
    )
    axis.axvline(turnover, color=COLOUR["model"], linestyle="--", linewidth=1.1)
    axis.text(
        turnover,
        0.98,
        turnover_label,
        color=COLOUR["model"],
        rotation=90,
        ha="right",
        va="top",
        transform=axis.get_xaxis_transform(),
        fontsize=11,
    )
    axis.axvline(beam_arcsec, color="0.45", linestyle="-.", linewidth=1.0)
    axis.set(
        xlabel=r"Galactocentric radius $R\ (\mathrm{arcsec})$",
        ylabel=r"$V_{\rm c}\ (\mathrm{km\ s^{-1}})$",
        xlim=(0.0, float(np.nanmax(radius))),
        ylim=(0.0, None),
        title=f"{target_id}: conditional MAP rotation curve",
    )
    axis.xaxis.set_minor_locator(AutoMinorLocator())
    axis.yaxis.set_minor_locator(AutoMinorLocator())
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        fontsize=10,
    )
    return save_pair(figure, output_dir, "rotation_curve")


def synthetic_figure(
    target_id: str,
    synthetic: dict,
    subbeam: dict,
    output_dir: Path,
) -> list[Path]:
    figure_size = apply_style(columns=2, aspect_ratio=7.2 / 7.1)
    truth = synthetic["truth"]
    beam = float(subbeam["bmaj_arcsec"])
    radius = np.linspace(0.0, 3.5 * beam, 240)
    truth_profile = projected_arctan_speed(truth, radius)
    kinuv_profiles = np.asarray(
        [projected_arctan_speed(row["kinuv"]["parameters"], radius) for row in synthetic["realizations"]]
    )
    kinms_profiles = np.asarray(
        [projected_arctan_speed(row["kinms"]["parameters"], radius) for row in synthetic["realizations"]]
    )
    figure = plt.figure(figsize=figure_size)
    grid = GridSpec(2, 2, figure=figure, height_ratios=(1.35, 1.0), hspace=0.38, wspace=0.32)
    profile_axis = figure.add_subplot(grid[0, :])
    profile_axis.axvspan(0.0, 1.0, color="0.93", zorder=0, label=r"Inner $1\times\mathrm{BMAJ}$")
    profile_axis.fill_between(
        radius / beam,
        np.min(kinuv_profiles, axis=0),
        np.max(kinuv_profiles, axis=0),
        color=COLOUR["model"],
        alpha=0.20,
    )
    profile_axis.fill_between(
        radius / beam,
        np.min(kinms_profiles, axis=0),
        np.max(kinms_profiles, axis=0),
        color=KINMS_COLOUR,
        alpha=0.18,
    )
    profile_axis.plot(radius / beam, truth_profile, color="black", linewidth=2.2, label="Truth")
    profile_axis.plot(
        radius / beam,
        np.median(kinuv_profiles, axis=0),
        color=COLOUR["model"],
        linewidth=1.8,
        label="kinUV median",
    )
    profile_axis.plot(
        radius / beam,
        np.median(kinms_profiles, axis=0),
        color=KINMS_COLOUR,
        linewidth=1.8,
        label="KinMS median",
    )
    profile_axis.axvline(float(truth["r_t_arcsec"]) / beam, color="black", linestyle=":", linewidth=1.0)
    profile_axis.axvline(1.0, color="0.45", linestyle="--", linewidth=1.0)
    profile_axis.set(
        xlabel=r"$R/\mathrm{BMAJ}$",
        ylabel=r"$u(R)=V_{\rm c}\sin i\ (\mathrm{km\ s^{-1}})$",
        xlim=(0.0, 3.5),
        title="Matched-family recovery",
    )
    profile_axis.legend(fontsize=9, loc="lower right")
    panel_letter(profile_axis, "a")

    aggregate = subbeam["aggregate"]
    axes = [figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1])]
    values = (
        (
            aggregate["kinuv_mean_absolute_turnover_error_arcsec"] / beam,
            aggregate["kinms_mean_absolute_turnover_error_arcsec"] / beam,
            r"Mean $|\Delta R_{\rm turn}|/\mathrm{BMAJ}$",
            aggregate["absolute_turnover_error_ratio_kinuv_over_kinms"],
        ),
        (
            aggregate["kinuv_inner_rms_rmse_kms"],
            aggregate["kinms_inner_rms_rmse_kms"],
            r"Inner $u(R)$ RMSE $(\mathrm{km\ s^{-1}})$",
            aggregate["inner_rmse_ratio_kinuv_over_kinms"],
        ),
    )
    for axis, (kinuv_value, kinms_value, ylabel, metric_ratio), letter in zip(axes, values, "bc"):
        axis.bar((0, 1), (kinuv_value, kinms_value), color=(COLOUR["model"], KINMS_COLOUR), width=0.68)
        axis.set_xticks((0, 1), ("kinUV", "KinMS"), rotation=25)
        axis.set_ylabel(ylabel)
        axis.set_ylim(bottom=0.0)
        axis.text(
            0.5,
            0.96,
            f"ratio = {metric_ratio:.3f}",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=11,
        )
        panel_letter(axis, letter)
    figure.suptitle(f"{target_id}: synthetic sub-beam kinematic recovery", y=0.99)
    return save_pair(figure, output_dir, "synthetic_benchmark")


def integrated_flux_spectrum(cube: np.ndarray, spatial_support: np.ndarray, header) -> np.ndarray:
    """Return an aperture-integrated spectrum in Jy from a Kelvin cube."""

    velocity = spectral_axis_kms(header)
    rest_hz = float(header["RESTFRQ"])
    c_ms = 299792458.0
    k_b = 1.380649e-23
    frequency_hz = rest_hz / (1.0 + float(np.nanmedian(velocity)) / 299792.458)
    beam_area_deg2 = (
        np.pi * float(header["BMAJ"]) * float(header["BMIN"]) / (4.0 * np.log(2.0))
    )
    pixel_area_deg2 = abs(float(header["CDELT1"]) * float(header["CDELT2"]))
    jy_per_beam_per_k = (
        2.0
        * k_b
        * frequency_hz**2
        / c_ms**2
        * beam_area_deg2
        * (np.pi / 180.0) ** 2
        * 1.0e26
    )
    kelvin_sum = np.nansum(
        np.where(spatial_support[None, :, :], np.asarray(cube), np.nan), axis=(1, 2)
    )
    return kelvin_sum * jy_per_beam_per_k * pixel_area_deg2 / beam_area_deg2


def spectral_figure(
    target_id: str,
    data_cube: np.ndarray,
    model_cube: np.ndarray,
    mask_cube: np.ndarray,
    header,
    geometry: dict,
    output_dir: Path,
) -> list[Path]:
    """Plot the matched channel-by-channel integrated flux profile."""

    apply_style(columns=2, aspect_ratio=5.2 / 7.1)
    support = np.any(mask_cube, axis=0)
    velocity = spectral_axis_kms(header)
    data = integrated_flux_spectrum(data_cube, support, header)
    model = integrated_flux_spectrum(model_cube, support, header)
    residual = data - model
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(7.1, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": (2.2, 1.0), "hspace": 0.06},
    )
    axes[0].step(velocity, data, where="mid", color=COLOUR["data"], label="Data")
    axes[0].step(velocity, model, where="mid", color=COLOUR["model"], label="kinUV")
    axes[0].fill_between(velocity, data, model, step="mid", color=COLOUR["model"], alpha=0.12)
    axes[0].set_ylabel(r"Integrated flux density $S_\nu\ (\mathrm{Jy})$")
    axes[0].legend(loc="upper right")
    axes[0].tick_params(labelbottom=False)
    axes[1].step(velocity, residual, where="mid", color="#9C2F2F")
    axes[1].axhline(0.0, color=COLOUR["zero"], linewidth=1.0)
    axes[1].set(
        xlabel=r"$v_{\rm opt,LSRK}\ (\mathrm{km\ s^{-1}})$",
        ylabel=r"Data $-$ kinUV $\ (\mathrm{Jy})$",
    )
    for axis in axes:
        vsys_line(axis, float(geometry["vsys_kms"]), orientation="v")
        axis.xaxis.set_minor_locator(AutoMinorLocator())
        axis.yaxis.set_minor_locator(AutoMinorLocator())
    panel_letter(axes[0], "a")
    panel_letter(axes[1], "b")
    figure.suptitle(f"{target_id}: aperture-integrated spectral profile", y=0.98)
    figure.subplots_adjust(left=0.14, right=0.98, bottom=0.16, top=0.89)
    return save_pair(figure, output_dir, "spectral_profiles")


def _density_levels(histogram: np.ndarray) -> list[float]:
    flat = np.sort(np.asarray(histogram, dtype=np.float64).ravel())[::-1]
    cumulative = np.cumsum(flat)
    if cumulative[-1] <= 0.0:
        return []
    cumulative /= cumulative[-1]
    thresholds = []
    for probability in (0.95, 0.68):
        index = min(int(np.searchsorted(cumulative, probability)), flat.size - 1)
        thresholds.append(float(flat[index]))
    return sorted(set(thresholds))


def posterior_corner_figure(
    target_id: str,
    posterior: dict,
    inclination_deg: float,
    output_dir: Path,
) -> list[Path]:
    """Render primary posterior covariance, showing inclination as fixed."""

    names = list(posterior["param_names"])
    draws = np.asarray(posterior["draws"], dtype=np.float64).reshape(-1, len(names))
    vsys_draws = draws[:, names.index("vsys_kms")]
    vsys_reference = float(np.median(vsys_draws))
    columns = {
        "v0_kms": draws[:, names.index("v0_kms")],
        "r_t_arcsec": draws[:, names.index("r_t_arcsec")],
        "pa_deg": draws[:, names.index("pa_deg")],
        "delta_vsys_kms": vsys_draws - vsys_reference,
        "gas_sigma_kms": draws[:, names.index("gas_sigma_kms")],
    }
    keys = tuple(columns)
    labels = (
        r"$V_{\rm flat}\ (\mathrm{km\ s^{-1}})$",
        r"$R_{\rm turn}\ (\mathrm{arcsec})$",
        r"$\mathrm{PA}\ (\mathrm{deg})$",
        fr"$V_{{\rm sys,radio}}-{vsys_reference:.1f}$" + "\n" + r"$(\mathrm{km\ s^{-1}})$",
        r"$\sigma_v\ (\mathrm{km\ s^{-1}})$",
    )
    limits = {}
    for key, values in columns.items():
        lo, hi = np.quantile(values, (0.002, 0.998))
        pad = 0.08 * (hi - lo)
        limits[key] = (float(lo - pad), float(hi + pad))

    size = len(keys)
    apply_style(columns=2, aspect_ratio=7.4 / 7.1)
    figure, axes = plt.subplots(size, size, figsize=(7.1, 7.4), squeeze=False)
    for row, y_key in enumerate(keys):
        for column, x_key in enumerate(keys):
            axis = axes[row, column]
            if column > row:
                axis.set_visible(False)
                continue
            x = columns[x_key]
            if row == column:
                axis.hist(
                    x,
                    bins=32,
                    density=True,
                    histtype="stepfilled",
                    color=COLOUR["model"],
                    alpha=0.40,
                )
                for quantile in np.quantile(x, (0.16, 0.50, 0.84)):
                    axis.axvline(quantile, color=COLOUR["data"], linewidth=0.8)
                axis.set_yticks([])
            else:
                histogram, x_edges, y_edges = np.histogram2d(
                    x, columns[y_key], bins=34, range=(limits[x_key], limits[y_key])
                )
                smooth = gaussian_filter(histogram.T, sigma=1.0)
                levels = _density_levels(smooth)
                if levels:
                    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
                    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
                    axis.contourf(
                        x_centres,
                        y_centres,
                        smooth,
                        levels=levels + [float(np.max(smooth)) + np.finfo(float).eps],
                        colors=("#B8D8E8", COLOUR["model"]),
                        alpha=0.85,
                    )
            axis.set_xlim(*limits[x_key])
            if row != column:
                axis.set_ylim(*limits[y_key])
            axis.tick_params(labelsize=10)
            axis.xaxis.set_major_locator(MaxNLocator(3))
            axis.yaxis.set_major_locator(MaxNLocator(3))
            if row < size - 1:
                axis.tick_params(labelbottom=False)
            else:
                axis.set_xlabel(labels[column], fontsize=12)
                axis.tick_params(axis="x", labelrotation=35)
            if column > 0:
                axis.tick_params(labelleft=False)
            elif row > 0:
                axis.set_ylabel(labels[row], fontsize=12)
    figure.suptitle(
        f"{target_id}: conditional posterior covariance; "
        fr"$i={inclination_deg:.3f}^\circ$ fixed; intervals uncalibrated",
        y=0.985,
    )
    figure.subplots_adjust(
        left=0.18,
        right=0.98,
        bottom=0.18,
        top=0.92,
        wspace=0.10,
        hspace=0.10,
    )
    return save_pair(figure, output_dir, "posterior_corner")


def benchmark_moment_figure(
    target_id: str,
    moments: dict[str, np.ndarray],
    header,
    geometry: dict,
    output_dir: Path,
) -> list[Path]:
    """Compare matched data, kinUV, and KinMS moments on one scale."""

    apply_style(columns=2, aspect_ratio=9.0 / 7.1)
    figure = plt.figure(figsize=(7.1, 9.0))
    grid = GridSpec(
        6,
        5,
        figure=figure,
        height_ratios=(1.0, 0.10, 1.0, 0.10, 1.0, 0.10),
        left=0.08,
        right=0.99,
        bottom=0.12,
        top=0.93,
        wspace=0.06,
        hspace=0.25,
    )
    extent = sky_extent_arcsec(header)
    centre = (float(geometry["dx_arcsec"]), float(geometry["dy_arcsec"]))
    beam = (
        float(header["BMAJ"]) * 3600.0,
        float(header["BMIN"]) * 3600.0,
        float(header["BPA"]),
    )
    crop = source_crop_arcsec(moments["data_moment0"], header, centre, beam[0])
    vsys = float(geometry["vsys_kms"])
    rows = (
        ("moment0", "Moment 0", r"$I_{\rm CO}\ (\mathrm{K\ km\ s^{-1}})$", False),
        ("moment1", "Moment 1", r"$v-v_{\rm sys}\ (\mathrm{km\ s^{-1}})$", True),
        ("moment2", "Moment 2", r"$\sigma_v\ (\mathrm{km\ s^{-1}})$", False),
    )
    titles = ("Data", "kinUV", "KinMS", "Data - kinUV", "Data - KinMS")
    letters = iter("abcdefghijklmno")
    for row, (key, row_name, unit, velocity) in enumerate(rows):
        images = [
            moments[f"data_{key}"],
            moments[f"kinuv_{key}"],
            moments[f"kinms_{key}"],
            moments[f"data_minus_kinuv_{key}"],
            moments[f"data_minus_kinms_{key}"],
        ]
        if velocity:
            images[:3] = [image - vsys for image in images[:3]]
            common_limits = symmetric_clim(*images[:3], p=99.0)
            common_cmap = velocity_cmap()
        else:
            common_limits = sequential_clim(*images[:3], p=99.0)
            common_cmap = intensity_cmap()
        residual_limits = symmetric_clim(*images[3:], p=97.5)
        common_artist = residual_artist = None
        row_axes = []
        for column, image in enumerate(images):
            axis = figure.add_subplot(grid[2 * row, column])
            row_axes.append(axis)
            limits = common_limits if column < 3 else residual_limits
            cmap = common_cmap if column < 3 else residual_cmap()
            artist = imshow_masked(axis, image, extent, *limits, cmap)
            common_artist = artist if column < 3 else common_artist
            residual_artist = artist if column >= 3 else residual_artist
            format_sky_ax(axis, crop, centre, xlabel=False, ylabel=False)
            axis.tick_params(labelbottom=row == 2, labelleft=column == 0)
            if row == 0:
                axis.set_title(titles[column], fontsize=11)
            panel_letter(axis, next(letters), fontsize=10)
        row_axes[0].text(
            0.04,
            0.05,
            row_name,
            transform=row_axes[0].transAxes,
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
        cbar(
            figure,
            common_artist,
            unit,
            cax=figure.add_subplot(grid[2 * row + 1, :3]),
            orientation="horizontal",
        )
        cbar(
            figure,
            residual_artist,
            unit,
            cax=figure.add_subplot(grid[2 * row + 1, 3:]),
            orientation="horizontal",
        )
        if row == 0:
            beam_ellipse(
                row_axes[0],
                *beam,
                (centre[0] + crop - 1.4, centre[1] - crop + 1.4),
            )
    figure.suptitle(f"{target_id}: matched moment comparison", y=0.98)
    figure.supxlabel(r"East offset (arcsec)", y=0.012)
    figure.supylabel(r"North offset (arcsec)", x=0.012)
    return save_pair(figure, output_dir, "moments_kinuv_vs_kinms")


def _matched_pv_rows(cubes, mask_cube, header, geometry, length_arcsec):
    support = np.any(mask_cube, axis=0)
    ra, dec = offset_world(
        float(header["CRVAL1"]),
        float(header["CRVAL2"]),
        float(geometry["dx_arcsec"]),
        float(geometry["dy_arcsec"]),
    )
    width = float(header["BMIN"]) * 3600.0
    rows = []
    for name, angle in (
        ("Major axis", float(geometry["pa_deg"])),
        ("Minor axis", float(geometry["pa_deg"]) + 90.0),
    ):
        profiles = []
        offset = None
        for cube in cubes:
            profile, current_offset = pv_diagram(
                np.where(support[None, :, :], cube, np.nan),
                header,
                ra,
                dec,
                angle,
                length_arcsec,
                width,
            )
            if offset is not None and not np.allclose(offset, current_offset):
                raise ValueError("matched PVD offsets differ")
            offset = current_offset
            profiles.append(profile)
        rows.append((name, angle % 360.0, offset, profiles))
    return rows


def benchmark_pv_figure(
    target_id,
    data_cube,
    kinuv_cube,
    kinms_cube,
    mask_cube,
    header,
    geometry,
    crop,
    output_dir,
) -> list[Path]:
    """Compare matched kinUV and KinMS major/minor PVDs."""

    rows = _matched_pv_rows(
        (data_cube, kinuv_cube, kinms_cube), mask_cube, header, geometry, 2.0 * crop
    )
    velocity = spectral_axis_kms(header)
    common_limits = sequential_clim(
        *(profile for row in rows for profile in row[3]), p=99.2
    )
    residual_limits = symmetric_clim(
        *(row[3][0] - model for row in rows for model in row[3][1:]), p=97.5
    )
    apply_style(columns=2, aspect_ratio=6.8 / 7.1)
    figure = plt.figure(figsize=(7.1, 6.8))
    grid = GridSpec(
        3,
        5,
        figure=figure,
        height_ratios=(1.0, 1.0, 0.07),
        left=0.075,
        right=0.99,
        bottom=0.14,
        top=0.90,
        wspace=0.05,
        hspace=0.10,
    )
    titles = ("Data", "kinUV", "KinMS", "Data - kinUV", "Data - KinMS")
    letters = iter("abcdefghij")
    for row_index, (name, angle, offset, profiles) in enumerate(rows):
        images = profiles + [profiles[0] - profiles[1], profiles[0] - profiles[2]]
        extent = (float(offset[0]), float(offset[-1]), float(velocity[0]), float(velocity[-1]))
        common_artist = residual_artist = None
        for column, image in enumerate(images):
            axis = figure.add_subplot(grid[row_index, column])
            limits = common_limits if column < 3 else residual_limits
            cmap = intensity_cmap() if column < 3 else residual_cmap()
            artist = imshow_masked(axis, image, extent, *limits, cmap, aspect="auto")
            common_artist = artist if column < 3 else common_artist
            residual_artist = artist if column >= 3 else residual_artist
            if row_index == 0:
                axis.set_title(titles[column], fontsize=11)
                axis.tick_params(labelbottom=False)
            if column == 0:
                axis.set_ylabel(r"$v_{\rm opt,LSRK}\ (\mathrm{km\ s^{-1}})$")
                axis.text(
                    0.04,
                    0.04,
                    f"{name}\nPA = {angle:.1f} deg",
                    transform=axis.transAxes,
                    color="white",
                    fontsize=10,
                    bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.65},
                )
            else:
                axis.tick_params(labelleft=False)
            vsys_line(axis, float(geometry["vsys_kms"]), orientation="h")
            panel_letter(axis, next(letters), fontsize=10)
    cbar(
        figure,
        common_artist,
        r"$T_{\rm B}\ (\mathrm{K})$",
        cax=figure.add_subplot(grid[2, :3]),
        orientation="horizontal",
    )
    residual_bar = cbar(
        figure,
        residual_artist,
        r"$\Delta T_{\rm B}\ (\mathrm{K})$",
        cax=figure.add_subplot(grid[2, 3:]),
        orientation="horizontal",
    )
    # The residual colourbar spans only two of five columns.  Keep its tick
    # labels legible at ApJ two-column width, including low-amplitude targets.
    residual_bar.locator = MaxNLocator(nbins=3, symmetric=True)
    residual_bar.update_ticks()
    figure.suptitle(f"{target_id}: matched PVD comparison", y=0.975)
    figure.supxlabel(r"Offset (arcsec; receding $+$)", y=0.018)
    return save_pair(figure, output_dir, "pvd_kinuv_vs_kinms")


def benchmark_spectral_figure(
    target_id,
    data_cube,
    kinuv_cube,
    kinms_cube,
    mask_cube,
    header,
    geometry,
    output_dir,
) -> list[Path]:
    support = np.any(mask_cube, axis=0)
    velocity = spectral_axis_kms(header)
    data = integrated_flux_spectrum(data_cube, support, header)
    kinuv = integrated_flux_spectrum(kinuv_cube, support, header)
    kinms = integrated_flux_spectrum(kinms_cube, support, header)
    apply_style(columns=2, aspect_ratio=5.2 / 7.1)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(7.1, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": (2.2, 1.0), "hspace": 0.06},
    )
    axes[0].step(velocity, data, where="mid", color=COLOUR["data"], label="Data")
    axes[0].step(velocity, kinuv, where="mid", color=COLOUR["model"], label="kinUV")
    axes[0].step(velocity, kinms, where="mid", color=KINMS_COLOUR, label="KinMS")
    axes[0].set_ylabel(r"Integrated flux density $S_\nu\ (\mathrm{Jy})$")
    axes[0].legend(loc="upper right", ncol=3)
    axes[0].tick_params(labelbottom=False)
    axes[1].step(velocity, data - kinuv, where="mid", color=COLOUR["model"], label="Data - kinUV")
    axes[1].step(velocity, data - kinms, where="mid", color=KINMS_COLOUR, label="Data - KinMS")
    axes[1].axhline(0.0, color=COLOUR["zero"], linewidth=1.0)
    axes[1].set(
        xlabel=r"$v_{\rm opt,LSRK}\ (\mathrm{km\ s^{-1}})$",
        ylabel=r"Residual $\ (\mathrm{Jy})$",
    )
    axes[1].legend(loc="lower right", ncol=2, fontsize=10)
    for axis in axes:
        vsys_line(axis, float(geometry["vsys_kms"]), orientation="v")
        axis.xaxis.set_minor_locator(AutoMinorLocator())
        axis.yaxis.set_minor_locator(AutoMinorLocator())
    panel_letter(axes[0], "a")
    panel_letter(axes[1], "b")
    figure.suptitle(f"{target_id}: matched integrated-spectrum comparison", y=0.98)
    figure.subplots_adjust(left=0.14, right=0.98, bottom=0.16, top=0.89)
    return save_pair(figure, output_dir, "spectra_kinuv_vs_kinms")


def benchmark_rotation_figure(
    target_id,
    run_summary,
    stage_a,
    stage_b,
    rotation,
    moments,
    header,
    geometry,
    kinms_fit,
    beam_arcsec,
    output_dir,
) -> list[Path]:
    radius = rotation["radius_arcsec"]
    selected = run_summary["selected_model"]
    kinuv_curve = rotation["stage_b_kms"] if selected == "stage_b" else rotation["stage_a_kms"]
    data_r, data_v = major_axis_rotation_profile(
        moments["data_moment0"], moments["data_moment1"], header, **geometry,
        radius_max_arcsec=7.5, support_moment0=moments["data_moment0"]
    )
    kinuv_r, kinuv_v = major_axis_rotation_profile(
        moments["kinuv_moment0"], moments["kinuv_moment1"], header, **geometry,
        radius_max_arcsec=7.5, support_moment0=moments["data_moment0"]
    )
    kinms_r, kinms_v = major_axis_rotation_profile(
        moments["kinms_moment0"], moments["kinms_moment1"], header, **geometry,
        radius_max_arcsec=7.5, support_moment0=moments["data_moment0"]
    )
    apply_style(columns=2, aspect_ratio=5.5 / 7.1)
    figure, axis = plt.subplots(figsize=(7.1, 5.5))
    axis.axvspan(0.0, beam_arcsec, color="0.92", label=r"Inner $1\times\mathrm{BMAJ}$")
    axis.plot(radius, kinuv_curve, color=COLOUR["model"], linewidth=2.2, label="kinUV intrinsic")
    axis.plot(
        radius,
        arctan_curve(radius, kinms_fit["v0_kms"], kinms_fit["r_t_arcsec"]),
        color=KINMS_COLOUR,
        linewidth=2.0,
        linestyle="--",
        label="KinMS intrinsic",
    )
    axis.scatter(data_r, data_v, facecolor="white", edgecolor=COLOUR["data"], s=28, label="Data cube centroids")
    axis.plot(kinuv_r, kinuv_v, color=COLOUR["model"], marker="o", markersize=3.5, linewidth=1.0, alpha=0.75, label="kinUV cube centroids")
    axis.plot(kinms_r, kinms_v, color=KINMS_COLOUR, marker="s", markersize=3.2, linewidth=1.0, alpha=0.75, label="KinMS cube centroids")
    axis.axvline(float(stage_a["r_t_arcsec"]), color=FIDUCIAL_COLOUR, linestyle=":", linewidth=1.1, label=r"Stage A $R_{\rm turn}$")
    axis.axvline(float(kinms_fit["r_t_arcsec"]), color=KINMS_COLOUR, linestyle=":", linewidth=1.1, label=r"KinMS $R_{\rm turn}$")
    axis.set(
        xlabel=r"Galactocentric radius $R\ (\mathrm{arcsec})$",
        ylabel=r"$V_{\rm c}\ (\mathrm{km\ s^{-1}})$",
        xlim=(0.0, float(np.nanmax(radius))),
        ylim=(0.0, None),
        title=f"{target_id}: rotation-profile comparison",
    )
    axis.xaxis.set_minor_locator(AutoMinorLocator())
    axis.yaxis.set_minor_locator(AutoMinorLocator())
    figure.legend(loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=2, fontsize=9.5)
    figure.subplots_adjust(left=0.13, right=0.98, top=0.90, bottom=0.31)
    return save_pair(figure, output_dir, "rotation_curve_kinuv_vs_kinms")


def target_products(
    config_path: Path,
    source_root: Path,
    output_root: Path,
    synthetic_root: Path,
    subbeam_root: Path,
    state: dict,
) -> dict:
    config = load_json(config_path)
    target_id = config["target_id"]
    run = accepted_run(source_root, target_id)
    target_root = output_root / target_id
    if target_root.exists():
        raise FileExistsError(f"refusing to overwrite production target: {target_root}")
    best_model = target_root / "best_model"
    plots_dir = target_root / "plots"
    benchmarks_dir = target_root / "benchmarks"
    for directory in (best_model, plots_dir, benchmarks_dir, best_model / "posterior"):
        directory.mkdir(parents=True, exist_ok=True)

    paths = source_paths(run, config_path, synthetic_root, subbeam_root)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing production inputs for {target_id}: {missing}")

    best_model_copies = {
        "config.json": paths["config"],
        "summary.json": paths["run_summary"],
        "stage_a_map.json": paths["stage_a"],
        "stage_b_map.json": paths["stage_b"],
        "plot_summary.json": paths["plot_summary"],
        "model_native.fits": paths["model_native"],
        "model_on_10kms.fits": paths["model_cube"],
        "rotation_curve.npz": paths["rotation_curve"],
        "posterior/posterior_samples.json": paths["posterior_samples"],
        "posterior/summary.json": paths["posterior_summary"],
    }
    source_posterior = paths["posterior_samples"].parent
    for name in ("config.yaml", "validation.json", "METRICS.md"):
        candidate = source_posterior / name
        if candidate.is_file():
            best_model_copies[f"posterior/{name}"] = candidate
    for name in ("environment.json", "METRICS.md", "run.log"):
        candidate = run / name
        if candidate.is_file():
            best_model_copies[name] = candidate
    for destination, source in best_model_copies.items():
        output_path = best_model / destination
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, output_path)

    benchmark_copies = {
        "benchmark.json": paths["benchmark_record"],
        "moments.npz": paths["moments"],
        "kinms_model_k.fits": paths["kinms_cube"],
        "kinms_fit_result.json": paths["kinms_fit"],
    }
    for destination, source in benchmark_copies.items():
        shutil.copy2(source, benchmarks_dir / destination)

    run_summary = load_json(paths["run_summary"])
    stage_a = load_json(paths["stage_a"])
    stage_b = load_json(paths["stage_b"])
    plot_summary = load_json(paths["plot_summary"])
    kinms_document = load_json(paths["kinms_fit"])
    kinms_fit = kinms_document.get("fitted", kinms_document)
    synthetic = load_json(paths["synthetic"])
    subbeam = load_json(paths["subbeam"])
    data_cube, header = load_cube(paths["data_cube"])
    model_cube, model_header = load_cube(paths["model_cube"])
    kinms_cube, kinms_header = load_cube(paths["kinms_cube"])
    mask_cube, _ = load_cube(paths["mask_cube"])
    if not (data_cube.shape == model_cube.shape == kinms_cube.shape == mask_cube.shape):
        raise ValueError(
            f"cube mismatch for {target_id}: data={data_cube.shape}, "
            f"kinUV={model_cube.shape}, KinMS={kinms_cube.shape}, mask={mask_cube.shape}"
        )
    for label, model_path, current_header in (
        ("kinUV", paths["model_cube"], model_header),
        ("KinMS", paths["kinms_cube"], kinms_header),
    ):
        if current_header.get("BUNIT", "").strip().lower() not in {"k", "kelvin"}:
            raise ValueError(f"{label} production model cube must be in kelvin: {model_path}")
    geometry = {
        "pa_deg": stage_a["pa_deg"],
        "inclination_deg": stage_a["inclination_deg_frozen"],
        "vsys_kms": plot_summary["vsys_optical_kms"],
        "dx_arcsec": stage_a["dx_arcsec"],
        "dy_arcsec": stage_a["dy_arcsec"],
    }
    with np.load(paths["moments"]) as npz:
        moments = {name: np.asarray(npz[name]) for name in npz.files}
    with np.load(paths["rotation_curve"]) as npz:
        rotation = {name: np.asarray(npz[name]) for name in npz.files}
    posterior = load_json(paths["posterior_samples"])

    centroid_radius, centroid_speed = major_axis_rotation_profile(
        moments["data_moment0"],
        moments["data_moment1"],
        header,
        pa_deg=geometry["pa_deg"],
        inclination_deg=geometry["inclination_deg"],
        vsys_kms=geometry["vsys_kms"],
        dx_arcsec=geometry["dx_arcsec"],
        dy_arcsec=geometry["dy_arcsec"],
        slit_half_width_arcsec=0.5,
        radius_max_arcsec=7.5,
        n_bin=48,
        support_moment0=moments["data_moment0"],
    )
    centroid_path = plots_dir / "rotation_centroids.npz"
    np.savez(
        centroid_path,
        radius_arcsec=centroid_radius,
        speed_kms=centroid_speed,
        extraction_contract=np.asarray(
            "current WCS tangent-plane offsets; no FITS east-west reflection"
        ),
    )

    plot_products = []
    plot_products.extend(moment_figure(target_id, moments, header, geometry, plots_dir))
    plot_products.extend(
        pv_figure(
            target_id, data_cube, model_cube, mask_cube > 0.5, header, geometry, plots_dir
        )
    )
    plot_products.extend(
        rotation_figure(
            target_id,
            run_summary,
            stage_a,
            stage_b,
            rotation,
            centroid_radius,
            centroid_speed,
            kinms_fit,
            float(config["diagnostic_beam"]["bmaj_arcsec"]),
            plots_dir,
        )
    )
    plot_products.extend(
        spectral_figure(
            target_id, data_cube, model_cube, mask_cube > 0.5, header, geometry, plots_dir
        )
    )
    plot_products.extend(
        posterior_corner_figure(
            target_id, posterior, float(geometry["inclination_deg"]), plots_dir
        )
    )
    plot_products.append(centroid_path)

    centre = (float(geometry["dx_arcsec"]), float(geometry["dy_arcsec"]))
    crop = source_crop_arcsec(
        moments["data_moment0"],
        header,
        centre,
        float(header["BMAJ"]) * 3600.0,
    )
    benchmark_products = []
    benchmark_products.extend(
        benchmark_moment_figure(target_id, moments, header, geometry, benchmarks_dir)
    )
    benchmark_products.extend(
        benchmark_pv_figure(
            target_id,
            data_cube,
            model_cube,
            kinms_cube,
            mask_cube > 0.5,
            header,
            geometry,
            crop,
            benchmarks_dir,
        )
    )
    benchmark_products.extend(
        benchmark_spectral_figure(
            target_id,
            data_cube,
            model_cube,
            kinms_cube,
            mask_cube > 0.5,
            header,
            geometry,
            benchmarks_dir,
        )
    )
    benchmark_products.extend(
        benchmark_rotation_figure(
            target_id,
            run_summary,
            stage_a,
            stage_b,
            rotation,
            moments,
            header,
            geometry,
            kinms_fit,
            float(config["diagnostic_beam"]["bmaj_arcsec"]),
            benchmarks_dir,
        )
    )
    benchmark_products.extend(
        synthetic_figure(target_id, synthetic, subbeam, benchmarks_dir)
    )

    if run_summary["selected_model"] == "stage_b":
        selection_limit = (
            "KGAS066 Stage B is an immutable historical baseline whose former "
            "oscillation gate was invalidated. Its displayed turnover marker is "
            "an arctan-equivalent diagnostic compression of the selected ring "
            "curve, not a directly fitted ring parameter."
        )
    else:
        selection_limit = (
            "KGAS007 Stage B was rejected because a ring reached 0 km/s; the "
            "accepted Stage A arctan curve is displayed."
        )

    science_note = "\n".join(
        [
            f"# {target_id} production science figures",
            "",
            "These figures use the accepted production model and retained S4 synthetic fits. No fit was run during plotting.",
            "",
            "- `best_model/`: selected fit parameters, model cubes, retained posterior checkpoint, and model manifest.",
            "- `plots/`: moments, major/minor PVDs, conditional-MAP rotation curve, integrated spectrum, and primary-parameter covariance.",
            "- `benchmarks/`: matched kinUV/KinMS moments, PVDs, spectra, rotation profiles, and synthetic sub-beam recovery.",
            "",
            "Image-plane products are diagnostics; the accepted scientific objective remains visibility-domain chi-square. Restored-cube centroid points are beam-correlated and are not independent measurements. Synthetic evidence is limited to the registered thin axisymmetric arctan family.",
            "",
            "The plotted real-galaxy curve is a conditional MAP diagnostic; its posterior intervals are uncalibrated. No real-data inner-slope or sub-beam turnover claim is promoted. A formal rotation claim still requires a fitted non-rotating emitting-disk null and complete-refit bootstrap, which remain pending.",
            "",
            selection_limit,
            "",
        ]
    )
    note_path = target_root / "README.md"
    note_path.write_text(science_note, encoding="utf-8")
    best_manifest = {
        "schema_version": "kinuv-best-model-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_id": target_id,
        "selected_model": run_summary["selected_model"],
        "source_bundle": str(run.resolve()),
        "git": state,
        "files": {
            path.relative_to(best_model).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(best_model.rglob("*"))
            if path.is_file()
        },
    }
    write_json(best_model / "MANIFEST.json", best_manifest)

    all_files = [
        path
        for path in sorted(target_root.rglob("*"))
        if path.is_file() and path != target_root / "MANIFEST.json"
    ]
    manifest = {
        "schema_version": "kinuv-production-layout-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_id": target_id,
        "source_bundle": str(run.resolve()),
        "layout": ["best_model", "plots", "benchmarks"],
        "git": state,
        "fit_performed": False,
        "sources": {
            name: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for name, path in paths.items()
        },
        "files": {
            path.relative_to(target_root).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in all_files
        },
        "science_content": {
            "moments_0_1_2": True,
            "major_axis_pvd": True,
            "minor_axis_pvd": True,
            "intrinsic_rotation_curve": True,
            "restored_cube_centroids": True,
            "centroids_recomputed_with_current_wcs_contract": True,
            "turnover_radius": True,
            "inner_beam_scale": True,
            "synthetic_kinms_comparison": True,
            "subbeam_recovery": True,
            "integrated_spectral_profile": True,
            "posterior_covariance": True,
            "inclination_reported_as_fixed": True,
            "matched_kinms_moments": True,
            "matched_kinms_pvds": True,
            "matched_kinms_spectra": True,
            "matched_kinms_rotation_profile": True,
            "claim_limits_explicit": True,
        },
        "rendering": {
            "source_adaptive_crop_arcsec": crop,
            "style": "vendored ApJ contract",
            "legacy_figures_present": False,
        },
    }
    write_json(target_root / "MANIFEST.json", manifest)
    print(
        f"{target_id}: wrote {len(plot_products)} plot products and "
        f"{len(benchmark_products)} benchmark products to {target_root}"
    )
    return manifest


def recovery_target_products(config_path, source_root, output_root, synthetic_root, subbeam_root, state, recovery_root):
    """Plot one frozen S4-bound checkpoint; never combine old cubes/posteriors."""
    from kinuv.io.vis import radio_to_optical_kms
    from kinuv.validation.s4 import topo_radio_to_lsrk_radio

    config = load_json(config_path)
    target = config["target_id"]
    replay_root = recovery_root / "replay"
    replay_path = replay_root / target / "replay.json"
    replay = load_json(replay_path)
    checkpoint_path = recovery_root / "s3" / target / "ablations.json"
    checkpoint = load_json(checkpoint_path)
    if not checkpoint["accepted"] or not replay["mechanical_replay_complete"]:
        raise ValueError("recovery checkpoint or mechanical replay is not accepted")
    if sha256(checkpoint_path) != replay["inputs"]["s3"]["sha256"]:
        raise ValueError("checkpoint hash differs from cube replay")
    selected = next(item for item in replay["candidates"] if item["selected_in_s3"])
    fit = next(item for item in checkpoint["fits"] if item["candidate"] == selected["candidate"])
    if fit["parameters"] != selected["frozen_parameters"]:
        raise ValueError("cube and fit parameter checkpoints differ")
    bundle = replay_root / selected["products"]
    source_files = {
        "checkpoint": checkpoint_path,
        "replay": replay_path,
        "replay_manifest": replay_root / "MANIFEST.json",
        "model_cube": bundle / "kinuv_model_k.fits",
        "moments": bundle / "benchmark/moments.npz",
        "kinms_cube": bundle / "benchmark/kinms_model_k.fits",
        "kinms_fit": source_root / target / "benchmarks/kinms_fit_result.json",
        "data_cube": Path(replay["inputs"]["data_cube"]["path"]),
        "mask_cube": Path(replay["inputs"]["mask_cube"]["path"]),
        "config": config_path,
        "synthetic": synthetic_root / target / "summary.json",
        "subbeam": subbeam_root / target / "summary.json",
    }
    replay_manifest = load_json(source_files["replay_manifest"])
    for name in ("model_cube", "moments", "kinms_cube", "replay"):
        path = source_files[name]
        expected = replay_manifest["files"][path.relative_to(replay_root).as_posix()]["sha256"]
        if sha256(path) != expected:
            raise ValueError(f"replay source checksum failed: {path}")
    for name, key in (("data_cube", "data_cube"), ("mask_cube", "mask_cube"), ("kinms_fit", "kinms_result")):
        if sha256(source_files[name]) != replay["inputs"][key]["sha256"]:
            raise ValueError(f"replay input checksum failed: {name}")
    # Routing paths changed during production curation. Recover the exact fit
    # configuration from its recorded commit instead of claiming new routing
    # metadata was the configuration used for inference.
    config_revision = replay_manifest["code_commit"]
    frozen_config = subprocess.check_output(
        ["git", "show", f"{config_revision}:configs/targets/{target}.json"], cwd=REPO
    )
    if hashlib.sha256(frozen_config).hexdigest() != replay["inputs"]["config"]["sha256"]:
        raise ValueError("archived fit configuration checksum failed")
    target_root = output_root / target
    if target_root.exists():
        raise ValueError(f"output already exists; archive before replacement: {target_root}")
    best, plots, benchmarks = (target_root / name for name in ("best_model", "plots", "benchmarks"))
    for path in (best, plots, benchmarks):
        path.mkdir(parents=True)
    for name, destination in {
        "checkpoint": best / "checkpoint.json", "replay": best / "replay.json",
        "model_cube": best / "model_on_science_grid.fits",
        "kinms_fit": benchmarks / "kinms_fit_result.json", "kinms_cube": benchmarks / "kinms_model_k.fits",
        "moments": benchmarks / "moments.npz",
    }.items():
        shutil.copy2(source_files[name], destination)
    (best / "config.json").write_bytes(frozen_config)
    p = fit["parameters"]
    geometry = {key: float(p[key]) for key in ("pa_deg", "inclination_deg", "dx_arcsec", "dy_arcsec")}
    geometry["vsys_kms"] = float(radio_to_optical_kms(topo_radio_to_lsrk_radio(
        p["vsys_kms"], replay["frame"]["frequency_equivalent_correction_kms"])))
    write_json(best / "parameters.json", {"candidate": fit["candidate"], "fitted_native_parameters": p, "diagnostic_geometry_lsrk_optical": geometry})
    data, header = load_cube(source_files["data_cube"])
    model, model_header = load_cube(source_files["model_cube"])
    kinms, kinms_header = load_cube(source_files["kinms_cube"])
    mask, _ = load_cube(source_files["mask_cube"])
    if not (data.shape == model.shape == kinms.shape == mask.shape):
        raise ValueError("recovery cubes and mask have different shapes")
    for label, hdr in (("kinUV", model_header), ("KinMS", kinms_header)):
        if hdr.get("BUNIT", "").strip().lower() not in {"k", "kelvin"}:
            raise ValueError(f"{label} cube is not kelvin")
        for key in ("CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CDELT1", "CDELT2", "CRVAL3", "CRPIX3", "CDELT3"):
            if not np.isclose(hdr[key], header[key], rtol=0, atol=1e-9):
                raise ValueError(f"{label} cube WCS mismatch: {key}")
    with np.load(source_files["moments"]) as npz:
        moments = {key: np.asarray(npz[key]) for key in npz.files}
    beam = float(header["BMAJ"]) * 3600.0
    kinms_doc = load_json(source_files["kinms_fit"])
    kinms_fit = kinms_doc.get("fitted", kinms_doc)
    moment_figure(target, moments, header, geometry, plots)
    pv_figure(target, data, model, mask > 0.5, header, geometry, plots)
    spectral_figure(target, data, model, mask > 0.5, header, geometry, plots)
    benchmark_moment_figure(target, moments, header, geometry, benchmarks)
    crop = source_crop_arcsec(moments["data_moment0"], header, (p["dx_arcsec"], p["dy_arcsec"]), beam)
    benchmark_pv_figure(target, data, model, kinms, mask > 0.5, header, geometry, crop, benchmarks)
    benchmark_spectral_figure(target, data, model, kinms, mask > 0.5, header, geometry, benchmarks)
    synthetic_figure(target, load_json(source_files["synthetic"]), load_json(source_files["subbeam"]), benchmarks)
    # Show only the selected checkpoint's intrinsic profile, with no borrowed posterior.
    radii = np.asarray(checkpoint["velocity_support"]["knot_radii_arcsec"])
    radius = np.linspace(0.0, radii[-1], 240)
    if fit["candidate"] == "supported_rings":
        speeds = np.asarray(p["u_knots_kms"])
        u_profile = np.interp(radius, radii, speeds)
        u_profile = np.where(radius < radii[0], speeds[0] * radius / radii[0], u_profile)
        turnover = None
    else:
        turnover = p["turnover_over_bmaj"] * checkpoint["bmaj_arcsec"]
        u_profile = arctan_curve(radius, p["arctan_u_kms"], turnover)
    vc = u_profile / np.sin(np.deg2rad(p["inclination_deg"]))
    np.savez(best / "rotation_curve.npz", radius_arcsec=radius, projected_speed_kms=u_profile, intrinsic_speed_kms=vc)
    apply_style(columns=2, aspect_ratio=5.0 / 7.1)
    fig, ax = plt.subplots(figsize=(7.1, 5.0))
    ax.axvspan(0, beam, color="0.92", label="Inner beam scale")
    ax.plot(radius, vc, color=COLOUR["model"], label=f"kinUV {fit['candidate'].replace('_', ' ')}")
    ax.plot(radius, arctan_curve(radius, kinms_fit["v0_kms"], kinms_fit["r_t_arcsec"]), color=KINMS_COLOUR, ls="--", label="KinMS intrinsic")
    if turnover is not None:
        ax.axvline(turnover, color=COLOUR["model"], ls=":", label=r"kinUV $R_{\rm turn}$")
    ax.set(xlabel=r"Radius $R\ (\mathrm{arcsec})$", ylabel=r"$V_{\rm c}\ (\mathrm{km\ s^{-1}})$", title=f"{target}: selected conditional MAP profiles", xlim=(0, radius[-1]), ylim=(0, None))
    ax.legend(fontsize=11, loc="lower right")
    fig.tight_layout()
    save_pair(fig, plots, "rotation_curve")
    for suffix in ("pdf", "png"):
        shutil.copy2(plots / f"rotation_curve.{suffix}", benchmarks / f"rotation_curve_kinuv_vs_kinms.{suffix}")
    summary = {"target_id": target, "status": "accepted_checkpoint_diagnostic", "selected_model": fit["candidate"], "source_fit": fit, "no_fit_performed": True, "posterior_available_for_selected_checkpoint": False, "accounting": replay["accounting_repairs"], "geometry": geometry}
    write_json(best / "summary.json", summary)
    note = f"""# {target}: accepted recovery checkpoint diagnostics

The selected `{fit['candidate']}` checkpoint and its exact cube come from the S4-sealed smooth-emissivity replay. No fit, parameter tuning, or new benchmark scoring was performed. All direct and comparator plots use these same frozen parameters and matched cubes. Celestial east increases RA; slit PA is east of north. All three cubes share each selected-fit diagnostic slit and aperture.

Visibility likelihood is primary. Restored science cubes are supporting diagnostics, not ground truth; cleaner image residuals do not establish more accurate intrinsic kinematics. The measured synthetic turnover-error ratio 0.02493 and inner-velocity RMSE ratio 0.01438 for KGAS066 apply to the registered axisymmetric arctan test family, not arbitrary real galaxies.

No matching posterior exists for this selected checkpoint. Its historical, fixed-inclination arctan corner plot is preserved only in the superseded archive. MAP profiles have no calibrated credible intervals. No real inner slope or sub-beam turnover precision claim is promoted; the supported-ring model has no scalar turnover parameter.

Visibility fitting avoids the CLEAN inversion and restoring-beam pixel covariance while retaining phase information. Exported data still have measured spectral correlation (native adjacent C1 rho approximately 0.2977), which the likelihood models. In the marginally resolved, short-baseline regime, phase obeys phi(q,v) approximately -2 pi q dot xbar(v), with q in wavelengths and xbar in radians; differential phase replaces xbar by its difference from a reference channel. Flux-weighted clumps also affect the centroid. This is complementary information, not exact morphology/kinematics independence. See [Lachaume 2003](https://arxiv.org/abs/astro-ph/0304259).

The preserved replay explicitly records the same observed PB frequency in forward/inverse accounting, one native Hann response, guard disposal, and channel-edge overlap onto the official LSRK optical cube. This plotting pass does not reapply spectral or beam convolution.
"""
    (target_root / "README.md").write_text(note, encoding="ascii")
    file_record = lambda path: {"bytes": path.stat().st_size, "sha256": sha256(path)}
    write_json(best / "MANIFEST.json", {"schema_version": "kinuv-recovery-checkpoint-v1", "git": state, "selected_model": fit["candidate"], "files": {p.relative_to(best).as_posix(): file_record(p) for p in sorted(best.rglob("*")) if p.is_file()}})
    manifest = {"schema_version": "kinuv-production-layout-v3", "target_id": target, "git": state, "fit_performed": False, "scoring_performed": False, "selected_model": fit["candidate"], "fit_config_commit": config_revision, "fit_config_sha256": hashlib.sha256(frozen_config).hexdigest(), "geometry": geometry, "accounting": replay["accounting_repairs"], "sources": {k: {"path": str(v.resolve()), **file_record(v)} for k,v in source_files.items()}, "files": {p.relative_to(target_root).as_posix(): file_record(p) for p in sorted(target_root.rglob("*")) if p.is_file()}, "rendering": {"source_adaptive_crop_arcsec": crop, "east_increases_ra": True, "pa_east_of_north": True, "legacy_figures_present": False}, "posterior_available_for_selected_checkpoint": False}
    write_json(target_root / "MANIFEST.json", manifest)
    print(f"{target}: generated checkpoint-consistent diagnostics ({fit['candidate']})")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-root", type=Path, required=True, help="S4-bound root containing s3/ and replay/")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--subbeam-root", type=Path, required=True)
    parser.add_argument("--target-config", action="append", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = git_state()
    if state["branch"] != "dev" or state["dirty"]:
        raise SystemExit("production plotting requires a clean dev checkout")
    manifests = [
        recovery_target_products(
            path,
            args.source_root,
            args.output_root,
            args.synthetic_root,
            args.subbeam_root,
            state,
            args.recovery_root,
        )
        for path in args.target_config
    ]
    print(f"Completed {len(manifests)} production targets at commit {state['commit']}")


if __name__ == "__main__":
    main()
