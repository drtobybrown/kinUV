#!/usr/bin/env python3
"""Generate the canonical ApJ production figure suite for accepted kinUV runs.

The script consumes promoted model products and accepted S4 synthetic records.
It performs no fit, changes no scientific parameter, and writes an additive
``publication`` directory with its own provenance manifest.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from astropy.io import fits

from kinuv.diagnostics.imaging import offset_world, pv_diagram, spectral_axis_kms
from kinuv.diagnostics.kinms_benchmark import major_axis_rotation_profile
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
    matches = []
    for summary_path in sorted((production_root / target_id).glob("kinuv-*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") == "accepted" and summary.get("target_id") == target_id:
            matches.append(summary_path.parent)
    if len(matches) != 1:
        raise ValueError(f"expected one accepted run for {target_id}; found {matches}")
    return matches[0]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_cube(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float64).squeeze(), hdul[0].header.copy()


def save_pair(fig, output_dir: Path, name: str) -> list[Path]:
    products = save_publication(fig, output_dir / name)
    return [products["pdf"], products["png"]]


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
    crop = min(CROP_ARCSEC, 0.48 * abs(extent[1] - extent[0]))
    beam = (
        float(header["BMAJ"]) * 3600.0,
        float(header["BMIN"]) * 3600.0,
        float(header["BPA"]),
    )
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
    axis.text(
        0.98,
        0.04,
        "Beam-correlated cube centroids; intervals uncalibrated",
        transform=axis.transAxes,
        ha="right",
        color=COLOUR["muted"],
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


def target_products(
    config_path: Path,
    production_root: Path,
    synthetic_root: Path,
    subbeam_root: Path,
    state: dict,
) -> dict:
    config = load_json(config_path)
    target_id = config["target_id"]
    run = accepted_run(production_root, target_id)
    output = run / "publication"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing publication directory: {output}")
    output.mkdir()

    paths = {
        "config": config_path,
        "run_summary": run / "summary.json",
        "stage_a": run / "stage_a_map.json",
        "stage_b": run / "stage_b_map.json",
        "plot_summary": run / "plots/summary.json",
        "data_cube": Path(config["diagnostic_cube"]),
        "mask_cube": Path(config["diagnostic_mask"]),
        "model_cube": run / "plots/model_on_10kms.fits",
        "moments": run / "benchmark/moments.npz",
        "rotation_curve": run / "plots/rotation_curve.npz",
        "kinms_fit": run / "benchmark/kinms_fit_result.json",
        "synthetic": synthetic_root / target_id / "summary.json",
        "subbeam": subbeam_root / target_id / "summary.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing publication inputs for {target_id}: {missing}")

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
    mask_cube, _ = load_cube(paths["mask_cube"])
    if data_cube.shape != model_cube.shape or data_cube.shape != mask_cube.shape:
        raise ValueError(
            f"cube mismatch for {target_id}: data={data_cube.shape}, "
            f"model={model_cube.shape}, mask={mask_cube.shape}"
        )
    if model_header.get("BUNIT", "").strip().lower() not in {"k", "kelvin"}:
        raise ValueError(f"production model cube must be in kelvin: {paths['model_cube']}")
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
    centroid_path = output / "rotation_centroids.npz"
    np.savez(
        centroid_path,
        radius_arcsec=centroid_radius,
        speed_kms=centroid_speed,
        extraction_contract=np.asarray(
            "current WCS tangent-plane offsets; no FITS east-west reflection"
        ),
    )

    products = []
    products.extend(moment_figure(target_id, moments, header, geometry, output))
    products.extend(
        pv_figure(target_id, data_cube, model_cube, mask_cube > 0.5, header, geometry, output)
    )
    products.extend(
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
            output,
        )
    )
    products.extend(synthetic_figure(target_id, synthetic, subbeam, output))
    products.append(centroid_path)

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
            "- `moments_comparison`: masked moments 0/1/2 for data, kinUV, and data minus kinUV, with the registered restoring beam.",
            "- `pv_diagrams`: major- and minor-axis data, kinUV, and residual PVDs through the fitted center and PA.",
            "- `rotation_curve`: conditional-MAP kinUV curve, analytic/KinMS context, WCS-recomputed restored-cube centroid diagnostics, turnover radius, and one-beam scale.",
            "- `synthetic_benchmark`: matched-family truth recovery, sub-beam turnover error, and inner-beam projected-velocity accuracy against KinMS.",
            "",
            "Image-plane products are diagnostics; the accepted scientific objective remains visibility-domain chi-square. Restored-cube centroid points are beam-correlated and are not independent measurements. Synthetic evidence is limited to the registered thin axisymmetric arctan family.",
            "",
            "The plotted real-galaxy curve is a conditional MAP diagnostic; its posterior intervals are uncalibrated. No real-data inner-slope or sub-beam turnover claim is promoted. A formal rotation claim still requires a fitted non-rotating emitting-disk null and complete-refit bootstrap, which remain pending.",
            "",
            selection_limit,
            "",
        ]
    )
    note_path = output / "SCIENCE_DELIVERABLES.md"
    note_path.write_text(science_note, encoding="utf-8")
    products.append(note_path)
    manifest = {
        "schema_version": "kinuv-production-publication-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_id": target_id,
        "run": str(run.resolve()),
        "git": state,
        "fit_performed": False,
        "original_promotion_files_modified": False,
        "sources": {
            name: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for name, path in paths.items()
        },
        "products": {
            path.name: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in products
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
            "claim_limits_explicit": True,
        },
    }
    write_json(output / "MANIFEST.json", manifest)
    print(f"{target_id}: wrote {len(products)} publication products to {output}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, required=True)
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
        target_products(
            path,
            args.production_root,
            args.synthetic_root,
            args.subbeam_root,
            state,
        )
        for path in args.target_config
    ]
    print(f"Completed {len(manifests)} production targets at commit {state['commit']}")


if __name__ == "__main__":
    main()
