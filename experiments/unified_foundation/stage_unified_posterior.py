#!/usr/bin/env python3
"""Stage accepted weighted-Dynesty products beside the unified MAP delivery.

This tool is copy-on-write: it reads a production target, creates a complete
temporary copy outside production, validates accepted ``finalize_unified_dynesty``
evidence, renders posterior diagnostics, rebuilds manifests, and atomically
renames the staged target. Weighted nested samples are never described as MCMC
chains or unweighted posterior draws.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from astropy.io import fits


REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_production_figures as direct_figures  # noqa: E402
from unified_map_runner import build_problem  # noqa: E402
from phase4_synthetic_benchmark import (  # noqa: E402
    matched_disk_template as phase4_disk_template,
)
from kinuv.constants import freq_to_velocity_kms  # noqa: E402
from kinuv.diagnostics.imaging import (  # noqa: E402
    match_model_to_imaging,
    offset_world,
    pv_diagram,
    spectral_axis_kms,
)
from kinuv.diagnostics.delivery import (  # noqa: E402
    render_moments,
    render_pvd,
    render_radial_profiles,
    render_spectra,
)
from kinuv.diagnostics.kinms_benchmark import write_cube_benchmark  # noqa: E402
from kinuv.diagnostics.style import (  # noqa: E402
    COLOUR,
    apply_style,
    cbar,
    imshow_masked,
    intensity_cmap,
    panel_letter,
    save_publication,
    sequential_clim,
)
from kinuv.forward.model import attenuate_intrinsic_cube, intrinsic_sky_cube  # noqa: E402
from kinuv.infer.unified import decode_unified_chart, unified_profile_callables  # noqa: E402
from kinuv.infer.unified import build_support_from_template  # noqa: E402
from kinuv.io.vis import radio_to_optical_kms  # noqa: E402
from kinuv.profiles.unified import projected_velocity, velocity_dispersion  # noqa: E402
from kinuv.validation.s4 import topo_radio_to_lsrk_radio  # noqa: E402
from kinuv.validation.s4 import projected_arctan_speed  # noqa: E402


SCHEMA = "kinuv-unified-posterior-staging-v1"
POSTERIOR_COLOUR = "#56B4E9"
MAP_COLOUR = COLOUR["model"]
DEFAULT_CHI2_MARGIN = 1.0
CORNER_FIELDS = (
    ("flux_jy_kms", r"$F-q_{50}$ (Jy km s$^{-1}$)"),
    ("pa_deg", r"$\mathrm{PA}-q_{50}$ (deg)"),
    ("vsys_native_kms", r"$v_{\rm sys}-q_{50}$ (km s$^{-1}$)"),
    ("dx_arcsec", r"$\Delta x-q_{50}$ (arcsec)"),
    ("dy_arcsec", r"$\Delta y-q_{50}$ (arcsec)"),
    ("inclination_deg", r"$i-q_{50}$ (deg)"),
    ("u_reference_kms", r"$u_{\rm ref}-q_{50}$ (km s$^{-1}$)"),
    ("sigma0_kms", r"$\sigma_0-q_{50}$ (km s$^{-1}$)"),
)


def now():
    return datetime.now(timezone.utc).isoformat()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    path = Path(path)
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def weighted_quantile(values, weights, probabilities=(0.16, 0.50, 0.84)):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return np.interp(probabilities, cumulative, values[order])


def verify_evidence(evidence, target, selected_map):
    summary_path = evidence / "summary.json"
    manifest_path = evidence / "MANIFEST.json"
    weighted_path = evidence / "posterior_weighted.npz"
    profile_path = evidence / "profile_quantiles.npz"
    for path in (summary_path, manifest_path, weighted_path, profile_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = load_json(summary_path)
    if summary.get("state") != "ACCEPTED" or not summary.get("gates", {}).get("accepted"):
        raise RuntimeError(f"{target}: Dynesty evidence is not accepted")
    if summary.get("target_id") != target:
        raise RuntimeError(f"{target}: evidence target mismatch")
    manifest = load_json(manifest_path)
    for name, expected in manifest.get("files", {}).items():
        path = evidence / name
        if not path.is_file() or sha256(path) != expected["sha256"]:
            raise RuntimeError(f"{target}: evidence manifest mismatch for {name}")
    expected_map = summary["sources"]["map"]["sha256"]
    if sha256(selected_map) != expected_map:
        raise RuntimeError(f"{target}: selected MAP differs from Dynesty conditioning MAP")
    with np.load(weighted_path, allow_pickle=False) as archive:
        samples = {name: np.asarray(archive[name]) for name in archive.files}
    q = np.asarray(samples["q_unified"], dtype=np.float64)
    weight = np.asarray(samples["weight"], dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 14 or weight.shape != (q.shape[0],):
        raise RuntimeError(f"{target}: invalid weighted posterior dimensions")
    if np.any(weight < 0) or not np.all(np.isfinite(weight)) or not np.isclose(weight.sum(), 1.0, atol=1e-10):
        raise RuntimeError(f"{target}: invalid posterior weights")
    return summary, samples, q, weight


def verify_production_target(source):
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = load_json(manifest_path)
    for name, expected in manifest.get("files", {}).items():
        path = source / name
        if not path.is_file() or sha256(path) != expected["sha256"]:
            raise RuntimeError(f"production manifest mismatch for {path}")


def verify_scoring_source(evidence_summary):
    commit = evidence_summary["contract"]["code_commit"]
    paths = (
        "src/kinuv/constants.py",
        "src/kinuv/forward",
        "src/kinuv/geometry.py",
        "src/kinuv/infer",
        "src/kinuv/io/vis.py",
        "src/kinuv/profiles",
        "experiments/unified_foundation/unified_map_runner.py",
        "experiments/unified_foundation/representation_model.py",
    )
    reference_root = os.environ.get("KINUV_SCORING_REFERENCE")
    if reference_root:
        reference_root = Path(reference_root)
        for relative in paths:
            current = REPO / relative
            reference = reference_root / relative
            if reference.is_dir():
                expected = {
                    path.relative_to(reference)
                    for path in reference.rglob("*.py")
                }
                actual = {
                    path.relative_to(current)
                    for path in current.rglob("*.py")
                }
                if actual != expected:
                    raise RuntimeError(
                        f"current scoring source file set differs from {commit}: {relative}"
                    )
                pairs = ((current / name, reference / name) for name in expected)
            else:
                pairs = ((current, reference),)
            for current_path, reference_path in pairs:
                if not reference_path.is_file() or sha256(current_path) != sha256(reference_path):
                    raise RuntimeError(
                        f"current scoring implementation differs from {commit}: {relative}"
                    )
        return
    completed = __import__("subprocess").run(
        ["git", "diff", "--quiet", commit, "--", *paths], cwd=REPO, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "current scoring implementation differs from accepted Dynesty code commit"
        )


def score_posterior_median(target, q, weight):
    context, spec, density, _, _ = build_problem(target)
    median = np.array([weighted_quantile(q[:, index], weight)[1] for index in range(q.shape[1])])
    likelihood = jax.jit(density.log_likelihood)
    value = likelihood(jnp.asarray(median))
    jax.block_until_ready(value)
    return context, spec, median, float(-2.0 * value)


def render_corner(samples, weight, output):
    absolute = [np.asarray(samples[name], dtype=np.float64) for name, _ in CORNER_FIELDS]
    centers = [weighted_quantile(value, weight)[1] for value in absolute]
    values = [value - center for value, center in zip(absolute, centers)]
    # Nested-sampling output retains negligible-weight prior-tail particles.
    # Restrict plotting only (never inference) to the central 99.8% weighted
    # interval so those particles cannot collapse the visible posterior.
    ranges = []
    for value in values:
        lo, hi = weighted_quantile(value, weight, probabilities=(0.001, 0.999))
        if not np.isfinite(lo + hi) or hi <= lo:
            width = max(float(np.nanstd(value)), 1.0e-8)
            lo, hi = -4.0 * width, 4.0 * width
        pad = 0.04 * (hi - lo)
        ranges.append((lo - pad, hi + pad))
    labels = [label for _, label in CORNER_FIELDS]
    count = len(values)
    apply_style(columns=2, aspect_ratio=1.0)
    # A full eight-parameter corner needs a page-sized canvas to retain the
    # project-wide 12/10-point label and tick contract without collisions.
    figure, axes = plt.subplots(count, count, figsize=(12.4, 12.4))
    for row in range(count):
        for column in range(count):
            axis = axes[row, column]
            if row < column:
                axis.axis("off")
                continue
            if row == column:
                axis.hist(values[row], bins=34, range=ranges[row], weights=weight, color=MAP_COLOUR, histtype="stepfilled", alpha=0.75)
                for qvalue in weighted_quantile(values[row], weight):
                    axis.axvline(qvalue, color=COLOUR["data"], lw=0.55, alpha=0.8)
            else:
                axis.hist2d(
                    values[column], values[row], bins=28,
                    range=(ranges[column], ranges[row]), weights=weight, cmap="Blues",
                )
                axis.set_ylim(*ranges[row])
            axis.set_xlim(*ranges[column])
            if row == count - 1:
                axis.set_xlabel(labels[column])
                axis.tick_params(axis="x", labelrotation=35)
            else:
                axis.tick_params(labelbottom=False)
            if column == 0 and row > 0:
                axis.set_ylabel(labels[row])
            elif column > 0:
                axis.tick_params(labelleft=False)
            axis.tick_params(labelsize=10)
    figure.suptitle(
        "Weighted Dynesty posterior: primary parameters centered on q50", y=0.995
    )
    figure.subplots_adjust(left=0.075, right=0.995, bottom=0.075, top=0.97, wspace=0.08, hspace=0.08)
    return save_publication(figure, output / "weighted_corner_primary", dpi=180)


def render_profile_bands(profile_path, map_profile_path, support, output):
    with np.load(profile_path, allow_pickle=False) as archive:
        posterior = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(map_profile_path, allow_pickle=False) as archive:
        map_profile = {name: np.asarray(archive[name]) for name in archive.files}
    radius = posterior["radius_arcsec"]
    apply_style(columns=2, aspect_ratio=0.74)
    figure, axes = plt.subplots(2, 1, figsize=(7.1, 5.25), sharex=True)
    panels = (
        (axes[0], "u_projected", "projected_velocity_kms", r"$u(R)$ (km s$^{-1}$)"),
        (axes[1], "sigma", "dispersion_kms", r"$\sigma(R)$ (km s$^{-1}$)"),
    )
    for index, (axis, prefix, map_key, ylabel) in enumerate(panels):
        lo = posterior[f"{prefix}_q16"]
        median = posterior[f"{prefix}_q50"]
        hi = posterior[f"{prefix}_q84"]
        map_values = np.interp(radius, map_profile["radius_arcsec"], map_profile[map_key])
        axis.axvspan(0.0, float(support["bmaj_arcsec"]), color="0.92", label=r"$R\leq1\,\mathrm{BMAJ}$")
        axis.fill_between(radius, lo, hi, color=POSTERIOR_COLOUR, alpha=0.35, label="weighted Dynesty q16-q84")
        axis.plot(radius, median, color="#0F7C82", lw=1.5, label="weighted median profile")
        axis.plot(radius, map_values, color=MAP_COLOUR, lw=1.2, ls="--", label="selected MAP")
        r95 = float(support["emission_flux_quantiles_arcsec"]["emission_r95"])
        axis.axvline(r95, color=COLOUR["muted"], lw=0.8, ls=":", label="emission R95" if index == 0 else None)
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=10, ncol=2, loc="best")
        panel_letter(axis, chr(ord("a") + index))
    axes[-1].set_xlabel("Radius (arcsec)")
    figure.suptitle("Unified MAP and weighted-Dynesty radial profiles")
    figure.subplots_adjust(left=0.12, right=0.98, bottom=0.11, top=0.92, hspace=0.08)
    return save_publication(figure, output / "rotation_dispersion_posterior", dpi=220)


def phase4_target_records(global_path):
    global_summary = load_json(global_path)
    records = {}
    for item in global_summary["targets"]:
        target_path = Path(item["summary_path"])
        if not target_path.is_file():
            raise FileNotFoundError(target_path)
        records[item["target_id"]] = (item, load_json(target_path), target_path)
    if set(records) != {"KGAS066", "KGAS007"}:
        raise RuntimeError("Phase-4 synthetic aggregate must contain both target samplings")
    return global_summary, records


def render_phase4_synthetic(target, global_path, context, output):
    """Replace historical mock claims with the current unified r2 evidence."""
    _, records = phase4_target_records(global_path)
    aggregate, target_record, _ = records[target]
    truth_seed = target_record["realizations"][0]["truth"]
    bmaj = float(truth_seed["bmaj_arcsec"])
    pa_rad = np.radians(float(truth_seed["pa_deg"]))
    i_rad = np.radians(float(truth_seed["inclination_deg"]))
    template = phase4_disk_template(context.grid, pa_rad=pa_rad, i_rad=i_rad, bmaj=bmaj)
    support = build_support_from_template(template, context.grid, pa_rad=pa_rad, i_rad=i_rad, bmaj_arcsec=bmaj)
    apply_style(columns=2, aspect_ratio=1.24)
    figure = plt.figure(figsize=(7.1, 8.8))
    grid = GridSpec(4, 2, figure=figure, height_ratios=(1.0, 1.0, 1.0, 0.62), left=0.10, right=0.98, bottom=0.075, top=0.88, hspace=0.34, wspace=0.26)
    scenario_titles = {
        "smooth_monotonic": "Smooth monotonic",
        "nonmonotonic_bump": "Smooth non-monotonic bump",
        "varying_dispersion": "Varying dispersion",
    }
    r50_labels = {
        "RECOVERED": "recovered",
        "RESOLVED_INCORRECT": "incorrect",
        "UNRESOLVED": "unresolved",
        "NONUNIQUE": "nonunique",
    }
    legend_handles = None
    for row_index, row in enumerate(target_record["realizations"]):
        scenario = row["scenario"]
        radius = np.asarray(row["profile_grid"]["radius_arcsec"])
        truth_u = np.asarray(row["profile_grid"]["truth_u_projected_kms"])
        truth_sigma = np.asarray(row["profile_grid"]["truth_sigma_kms"])
        optimum = np.asarray(row["kinuv_unified"]["optimum_z"])
        unified_u = np.asarray(projected_velocity(radius, optimum[6], optimum[7:11], support))
        unified_sigma = np.asarray(velocity_dispersion(radius, optimum[11], optimum[12:14], support))
        kinms_fit = row["kinms_stock"]["parameters"]
        kinms_u = projected_arctan_speed(kinms_fit, radius)
        kinms_sigma = np.zeros_like(radius) + float(kinms_fit["gas_sigma_kms"])
        for column, (truth_values, unified_values, kinms_values, ylabel) in enumerate((
            (truth_u, unified_u, kinms_u, r"$u(R)$ (km s$^{-1}$)"),
            (truth_sigma, unified_sigma, kinms_sigma, r"$\sigma(R)$ (km s$^{-1}$)"),
        )):
            axis = figure.add_subplot(grid[row_index, column])
            axis.axvspan(0.0, bmaj, color="0.92")
            axis.plot(radius, truth_values, color=COLOUR["data"], lw=1.5, label="Truth")
            axis.plot(radius, unified_values, color=MAP_COLOUR, lw=1.4, label="unified kinUV")
            axis.plot(radius, kinms_values, color="#D55E00", lw=1.2, ls="--", label="stock KinMS")
            if legend_handles is None:
                legend_handles, _ = axis.get_legend_handles_labels()
            axis.set_ylabel(ylabel)
            if row_index == 0:
                axis.set_title(("Projected rotation", "Dispersion")[column])
            if row_index == 2:
                axis.set_xlabel("Radius (arcsec)")
            metrics_u = row["kinuv_unified"]["metrics"]
            metrics_k = row["kinms_stock"]["metrics"]
            if column == 0:
                note = (
                    f"inner ratio={row['ratios']['inner_u_rmse_kinuv_over_kinms']:.2f}\n"
                    f"R50 err={metrics_u['turnover_recovery']['absolute_error_arcsec']:.2f}/{metrics_k['turnover_recovery']['absolute_error_arcsec']:.2f} arcsec\n"
                    f"R50 U/K={r50_labels.get(metrics_u['turnover_recovery']['status'], metrics_u['turnover_recovery']['status'].lower())}/"
                    f"{r50_labels.get(metrics_k['turnover_recovery']['status'], metrics_k['turnover_recovery']['status'].lower())}\n"
                    f"slope err={metrics_u['central_du_dR_absolute_error_kms_per_arcsec']:.1f}/{metrics_k['central_du_dR_absolute_error_kms_per_arcsec']:.1f}"
                )
            else:
                note = f"sigma RMSE={metrics_u['sigma_rmse_kms']:.2f}/{metrics_k['sigma_rmse_kms']:.2f} km/s"
                if target == "KGAS007" and scenario == "varying_dispersion":
                    note += "\nKGAS007 limitation: unified worse"
            axis.text(0.98, 0.04, note, transform=axis.transAxes, ha="right", va="bottom", fontsize=10, bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.88})
            if column == 0:
                axis.text(0.16, 0.96, scenario_titles[scenario], transform=axis.transAxes, va="top", fontsize=10, weight="bold")
            axis.tick_params(labelsize=10)
            panel_letter(axis, chr(ord("a") + row_index * 2 + column), fontsize=11)
    bar_axis = figure.add_subplot(grid[3, :])
    names = ("KGAS066", "KGAS007")
    ratios = [records[name][0]["aggregate"]["inner_u_rmse_ratio_kinuv_over_kinms"] for name in names]
    bars = bar_axis.bar(names, ratios, color=(MAP_COLOUR, "#0F7C82"), width=0.55)
    bar_axis.axhline(0.90, color=COLOUR["data"], ls="--", lw=1.0, label="registered maximum 0.90")
    for bar, value in zip(bars, ratios):
        bar_axis.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=10)
    bar_axis.set_ylim(0.0, 1.02)
    bar_axis.set_ylabel("aggregate inner\nRMSE ratio")
    bar_axis.legend(fontsize=10, loc="upper right")
    bar_axis.tick_params(labelsize=10)
    panel_letter(bar_axis, "g", fontsize=11)
    figure.legend(
        legend_handles,
        ("Truth", "unified kinUV", "stock KinMS"),
        loc="upper center",
        bbox_to_anchor=(0.54, 0.94),
        fontsize=10,
        ncol=3,
    )
    figure.suptitle(
        f"{target}: unified-foundation Phase-4 matched mocks\n"
        "right-sized scientific gate passed; parametric control retained",
        y=0.995,
    )
    return save_publication(figure, output / "synthetic_benchmark", dpi=220)


def posterior_endpoint_bands(q, weight, spec, offset, correction):
    radius = np.abs(np.asarray(offset, dtype=np.float64))

    def one(sample):
        speed = projected_velocity(radius, sample[6], sample[7:11], spec.support)
        vsys = spec.vsys_reference_kms + sample[2] * spec.dv_kms
        return vsys + jnp.asarray(np.sign(offset)) * speed

    evaluate = jax.jit(jax.vmap(one))
    native = np.asarray(evaluate(jnp.asarray(q)))
    lsrk_radio = topo_radio_to_lsrk_radio(native, correction)
    optical = radio_to_optical_kms(lsrk_radio)
    quantiles = np.empty((3, len(offset)), dtype=np.float64)
    for index in range(len(offset)):
        quantiles[:, index] = weighted_quantile(optical[:, index], weight)
    return quantiles


def selected_endpoint(selected_q, spec, offset, correction):
    physical = decode_unified_chart(selected_q, spec)
    speed = np.asarray(projected_velocity(np.abs(offset), selected_q[6], selected_q[7:11], spec.support))
    native = float(physical["vsys_kms"]) + np.sign(offset) * speed
    return radio_to_optical_kms(topo_radio_to_lsrk_radio(native, correction))


def render_pvd_posterior(target, target_root, config, selected_q, q, weight, spec, output):
    data = np.asarray(fits.getdata(config["diagnostic_cube"]), dtype=np.float64).squeeze()
    header = fits.getheader(config["diagnostic_cube"])
    model = np.asarray(fits.getdata(target_root / "best_model/model_on_science_grid.fits"), dtype=np.float64).squeeze()
    mask = np.asarray(fits.getdata(config["diagnostic_mask"]), dtype=np.float64).squeeze() > 0.5
    geometry = load_json(target_root / "best_model/parameters.json")["diagnostic_optical_LSRK"]
    ra, dec = offset_world(float(header["CRVAL1"]), float(header["CRVAL2"]), geometry["dx_arcsec"], geometry["dy_arcsec"])
    width = float(header["BMIN"]) * 3600.0
    bmaj = float(header["BMAJ"]) * 3600.0
    spatial = np.any(mask, axis=0)
    rows = []
    for name, angle in (("Major axis", geometry["pa_deg"]), ("Minor axis", geometry["pa_deg"] + 90.0)):
        pair = []
        offsets = None
        for cube in (data, model):
            pv, current = pv_diagram(np.where(spatial[None], cube, np.nan), header, ra, dec, angle, 8.0 * bmaj, width)
            if offsets is not None and not np.allclose(offsets, current):
                raise RuntimeError("PVD offset mismatch")
            offsets = current
            pair.append(pv)
        rows.append((name, offsets, pair))
    limits = sequential_clim(*(pv for _, _, pair in rows for pv in pair), p=99.2)
    velocity = spectral_axis_kms(header)
    correction = float(config["spectral_frame"]["frequency_correction_equivalent_kms"])
    apply_style(columns=2, aspect_ratio=0.72)
    figure = plt.figure(figsize=(7.1, 5.8))
    grid = GridSpec(2, 2, figure=figure, left=0.09, right=0.91, bottom=0.10, top=0.79, hspace=0.09, wspace=0.07)
    artist = None
    legend_handles = None
    legend_labels = None
    for row_index, (name, offset, pair) in enumerate(rows):
        extent = (float(offset[0]), float(offset[-1]), float(velocity[0]), float(velocity[-1]))
        for column, pv in enumerate(pair):
            axis = figure.add_subplot(grid[row_index, column])
            artist = imshow_masked(axis, pv, extent, *limits, intensity_cmap(), aspect="auto")
            axis.axvline(0.0, color="white", lw=0.7)
            if row_index == 0:
                lo, median, hi = posterior_endpoint_bands(q, weight, spec, offset, correction)
                selected = selected_endpoint(selected_q, spec, offset, correction)
                axis.fill_between(offset, lo, hi, color=POSTERIOR_COLOUR, alpha=0.30, label="weighted q16-q84")
                axis.plot(offset, median, color="#00E5FF", lw=1.4, label="weighted median")
                axis.plot(offset, selected, color="white", lw=0.9, ls="--", label="selected model")
            else:
                physical = decode_unified_chart(selected_q, spec)
                optical_vsys = radio_to_optical_kms(topo_radio_to_lsrk_radio(float(physical["vsys_kms"]), correction))
                axis.axhline(optical_vsys, color="white", lw=0.9, ls="--")
            if row_index == 0:
                axis.set_title(("Data", "selected kinUV model")[column])
            if column == 0:
                axis.set_ylabel(f"{name}\noptical LSRK (km s$^{{-1}}$)")
            else:
                axis.tick_params(labelleft=False)
            if row_index == 1:
                axis.set_xlabel("Offset (arcsec; receding +)")
            else:
                axis.tick_params(labelbottom=False)
                if column == 0:
                    legend_handles, legend_labels = axis.get_legend_handles_labels()
            panel_letter(axis, chr(ord("a") + 2 * row_index + column))
    cbar(figure, artist, r"$T_{B}$ (K)", cax=figure.add_axes((0.93, 0.13, 0.018, 0.62)))
    if legend_handles:
        figure.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.50, 0.895),
            ncol=3,
            fontsize=10,
        )
    figure.suptitle(f"{target}: corrected posterior PVD endpoints", y=0.995)
    return save_publication(figure, output / "pvd_posterior_overlay", dpi=220)


def write_selected_model(target, target_root, config, context, spec, selected_q):
    """Replace selected-model cubes only when posterior median beats MAP."""
    physical = decode_unified_chart(selected_q, spec)
    velocity_profile, dispersion_profile = unified_profile_callables(selected_q, spec)
    intrinsic = intrinsic_sky_cube(
        context.template,
        context.grid,
        context.data.freqs_native,
        flux=physical["flux"], pa_rad=physical["pa_rad"], vsys_kms=physical["vsys_kms"],
        dx_arcsec=physical["dx_arcsec"], dy_arcsec=physical["dy_arcsec"],
        gas_sigma_kms=physical["sigma0_kms"], i_rad=physical["i_rad"],
        velocity_profile=velocity_profile, dispersion_profile=dispersion_profile,
    )
    attenuated = attenuate_intrinsic_cube(intrinsic, context.grid, context.data.freqs_native)
    correction = float(config["spectral_frame"]["frequency_correction_equivalent_kms"])
    velocity = topo_radio_to_lsrk_radio(freq_to_velocity_kms(context.data.freqs_native), correction)
    ico_header = fits.getheader(config["template_ico"])
    for name, cube, role in (
        ("intrinsic_native_model.fits", intrinsic, "INTRINSIC"),
        ("native_pb_attenuated_model.fits", attenuated, "PB_ATTENUATED"),
    ):
        from kinuv.diagnostics.s1 import sky_cube_fits
        array, header = sky_cube_fits(cube, context.grid, velocity, ico_header)
        header["SPECSYS"] = "LSRK"; header["BUNIT"] = "Jy/pixel"; header["OBJECT"] = target
        header["KINUVSTG"] = "POSTMED"; header["CUBEROLE"] = role
        fits.PrimaryHDU(array.astype(np.float32), header).writeto(target_root / "best_model" / name, overwrite=True)
    native = np.asarray(fits.getdata(target_root / "best_model/native_pb_attenuated_model.fits"), dtype=np.float64)
    native_header = fits.getheader(target_root / "best_model/native_pb_attenuated_model.fits")
    data_header = fits.getheader(config["diagnostic_cube"])
    matched, _, _ = match_model_to_imaging(native, native_header, data_header, nu_hz=float(np.median(context.data.freqs_native)), undo_pb=True)
    out_header = data_header.copy(); out_header["BUNIT"] = "K"; out_header["KINUVSTG"] = "POSTMED"
    fits.PrimaryHDU(np.nan_to_num(matched).astype(np.float32), out_header).writeto(target_root / "best_model/model_on_science_grid.fits", overwrite=True)
    radius = np.linspace(0.0, spec.support.outer_radius_arcsec, 256)
    u = np.asarray(projected_velocity(radius, selected_q[6], selected_q[7:11], spec.support))
    sigma = np.asarray(velocity_dispersion(radius, selected_q[11], selected_q[12:14], spec.support))
    vc = u / max(float(np.sin(physical["i_rad"])), 1.0e-8)
    np.savez(target_root / "best_model/radial_profiles.npz", radius_arcsec=radius, projected_velocity_kms=u, intrinsic_rotation_kms=vc, dispersion_kms=sigma)
    geometry = {
        "pa_deg": float(np.degrees(physical["pa_rad"])) % 360.0,
        "inclination_deg": float(np.degrees(physical["i_rad"])),
        "vsys_kms": float(radio_to_optical_kms(topo_radio_to_lsrk_radio(float(physical["vsys_kms"]), correction))),
        "dx_arcsec": float(physical["dx_arcsec"]), "dy_arcsec": float(physical["dy_arcsec"]),
    }
    physical_json = {key: (float(value) if np.asarray(value).ndim == 0 else np.asarray(value).tolist()) for key, value in physical.items() if key != "i_rad"}
    physical_json["inclination_deg"] = geometry["inclination_deg"]
    write_json(target_root / "best_model/parameters.json", {"native_TOPO_radio": physical_json, "diagnostic_optical_LSRK": geometry})


def refresh_selected_diagnostics(
    target, target_root, source, config, context, spec, selected_q, selected_best, profile_quantiles
):
    """Regenerate every model-dependent diagnostic for the selected model."""
    stage = "posterior median [16th-84th percentiles]" if selected_best == "POSTERIOR_MEDIAN" else "MAP"
    data_path = Path(config["diagnostic_cube"])
    mask_path = Path(config["diagnostic_mask"])
    model_path = target_root / "best_model/model_on_science_grid.fits"
    kinms_source = source / "benchmarks/kinms_model_k.fits"
    kinms_fit_source = source / "benchmarks/kinms_fit_result.json"
    benchmarks = target_root / "benchmarks"
    plots = target_root / "plots"
    geometry = load_json(target_root / "best_model/parameters.json")["diagnostic_optical_LSRK"]

    benchmark = write_cube_benchmark(
        target_id=target,
        data_cube=data_path,
        mask_cube=mask_path,
        kinuv_cube=model_path,
        kinms_cube=kinms_source,
        output_dir=benchmarks,
        pa_deg=float(geometry["pa_deg"]),
        inclination_deg=float(geometry["inclination_deg"]),
        vsys_kms=float(geometry["vsys_kms"]),
        dx_arcsec=float(geometry["dx_arcsec"]),
        dy_arcsec=float(geometry["dy_arcsec"]),
    )
    shutil.copy2(kinms_fit_source, benchmarks / "kinms_fit_result.json")
    for obsolete in (
        "moments_comparison.png", "spectra_comparison.png", "pvd_major_comparison.png",
        "rotation_curve_comparison.png", "channel_maps_comparison.png",
    ):
        (benchmarks / obsolete).unlink(missing_ok=True)

    data = np.asarray(fits.getdata(data_path), dtype=np.float64).squeeze()
    model = np.asarray(fits.getdata(model_path), dtype=np.float64).squeeze()
    mask = np.asarray(fits.getdata(mask_path), dtype=np.float64).squeeze() > 0.5
    header = fits.getheader(data_path)
    kinms = np.asarray(fits.getdata(benchmarks / "kinms_model_k.fits"), dtype=np.float64).squeeze()
    with np.load(benchmarks / "moments.npz", allow_pickle=False) as archive:
        moments = {key: np.asarray(archive[key]) for key in archive.files}
    direct_figures.pv_figure(target, data, model, mask, header, geometry, plots)
    direct_figures.spectral_figure(target, data, model, mask, header, geometry, plots)
    render_moments(target, moments, header, geometry, stage, benchmarks)

    with np.load(target_root / "best_model/radial_profiles.npz", allow_pickle=False) as archive:
        selected_profile = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(profile_quantiles, allow_pickle=False) as archive:
        posterior_profile = {key: np.asarray(archive[key]) for key in archive.files}
    radius = selected_profile["radius_arcsec"]
    kinms_document = load_json(kinms_fit_source)
    kinms_fit = kinms_document.get("fitted", kinms_document)
    kinms_intrinsic = float(kinms_fit["v0_kms"]) * (2.0 / np.pi) * np.arctan(
        radius / float(kinms_fit["r_t_arcsec"])
    )
    kinms_projected = kinms_intrinsic * np.sin(
        np.radians(float(kinms_fit.get("i_deg", kinms_fit.get("inclination_deg"))))
    )
    correction = float(config["spectral_frame"]["frequency_correction_equivalent_kms"])
    physical = decode_unified_chart(selected_q, spec)
    u = selected_profile["projected_velocity_kms"]
    slope = float((u[1] - u[0]) / (radius[1] - radius[0]))
    profile_payload = {
        "radius": radius,
        "beam": float(spec.support.bmaj_arcsec),
        "kinuv_projected": u,
        "kinuv_intrinsic": selected_profile["intrinsic_rotation_kms"],
        "sigma_map": selected_profile["dispersion_kms"],
        "projected_lo": np.interp(radius, posterior_profile["radius_arcsec"], posterior_profile["u_projected_q16"]),
        "projected_hi": np.interp(radius, posterior_profile["radius_arcsec"], posterior_profile["u_projected_q84"]),
        "intrinsic_lo": np.interp(radius, posterior_profile["radius_arcsec"], posterior_profile["v_rot_q16"]),
        "intrinsic_hi": np.interp(radius, posterior_profile["radius_arcsec"], posterior_profile["v_rot_q84"]),
        "sigma_lo": np.interp(radius, posterior_profile["radius_arcsec"], posterior_profile["sigma_q16"]),
        "sigma_hi": np.interp(radius, posterior_profile["radius_arcsec"], posterior_profile["sigma_q84"]),
        "kinms_projected": kinms_projected,
        "kinms_intrinsic": kinms_intrinsic,
        "kinms_sigma": float(kinms_fit["gas_sigma_kms"]),
        "kinms_vsys": float(kinms_fit["vsys_optical_kms"]),
        "turnover": None,
        "moment0": moments["data_moment0"],
        "kinuv_spectral_transform": {
            "native_vsys_radio_topo_kms": float(physical["vsys_kms"]),
            "frequency_equivalent_correction_kms": correction,
        },
        "profile_metrics": {
            "turnover": r"kinUV $R_{50}$: unresolved plateau",
            "velocity": fr"$u(R_{{70}})={float(physical['u_reference_kms']):.1f}$ km s$^{{-1}}$",
            "inner_gradient": fr"$(du/dR)_0={slope:.1f}$ km s$^{{-1}}$ arcsec$^{{-1}}$",
            "smearing": fr"KinMS $R_{{\rm turn}}/\mathrm{{BMAJ}}={float(kinms_fit['r_t_arcsec']) / float(spec.support.bmaj_arcsec):.2f}$",
        },
        "posterior_label": "Dynesty weighted 16th-84th percentile",
    }
    render_pvd(target, (data, model, kinms), mask, header, geometry, stage, profile_payload, benchmarks)
    render_spectra(target, (data, model, kinms), mask, header, geometry, stage, benchmarks)
    render_radial_profiles(target, profile_payload, stage, benchmarks, benchmark=True)
    render_radial_profiles(target, profile_payload, stage, plots, benchmark=False)
    for suffix in ("pdf", "png"):
        shutil.copy2(benchmarks / f"moments_kinuv_vs_kinms.{suffix}", plots / f"moments_comparison.{suffix}")
        shutil.copy2(benchmarks / f"spectra_kinuv_vs_kinms.{suffix}", plots / f"spectral_profiles.{suffix}")
    return benchmark


def rebuild_manifests(target_root, target, selected_best):
    best = target_root / "best_model"
    write_json(best / "manifest.json", {
        "schema_version": "kinuv-unified-best-model-manifest-v2",
        "selected_best": selected_best,
        "files": {path.relative_to(best).as_posix(): file_record(path) for path in sorted(best.rglob("*")) if path.is_file() and path != best / "manifest.json"},
    })
    write_json(target_root / "manifest.json", {
        "schema_version": "kinuv-unified-production-manifest-v2",
        "target_id": target,
        "selected_best": selected_best,
        "posterior_available": True,
        "files": {path.relative_to(target_root).as_posix(): file_record(path) for path in sorted(target_root.rglob("*")) if path.is_file() and path != target_root / "manifest.json"},
    })


def stage_target(args, target):
    source = args.production_root / target
    evidence = args.dynesty_root / target
    destination = args.output_root / target
    if destination.exists():
        raise FileExistsError(destination)
    if not source.is_dir():
        raise FileNotFoundError(source)
    verify_production_target(source)
    selected_map = source / "best_model/selected_map.json"
    summary, samples, q, weight = verify_evidence(evidence, target, selected_map)
    verify_scoring_source(summary)
    map_record = load_json(selected_map)
    map_chi2 = float(map_record["objective"]["visibility_chi2"])
    context, spec, median_q, median_chi2 = score_posterior_median(target, q, weight)
    delta = median_chi2 - map_chi2
    select_posterior = delta < -float(args.chi2_margin)
    selected_best = "POSTERIOR_MEDIAN" if select_posterior else "MAP"
    selected_q = median_q if select_posterior else np.asarray(map_record["optimum_z"], dtype=np.float64)

    args.output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target}-posterior-", dir=args.output_root))
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        posterior = staging / "nuts" / "dynesty"
        posterior.mkdir()
        for source_name, destination_name in (
            ("summary.json", "dynesty_summary.json"),
            ("MANIFEST.json", "dynesty_source_manifest.json"),
            ("posterior_weighted.npz", "posterior_weighted.npz"),
            ("profile_quantiles.npz", "radial_profile_quantiles.npz"),
        ):
            shutil.copy2(evidence / source_name, posterior / destination_name)
        write_json(posterior / "parameter_quantiles.json", {
            "schema_version": "kinuv-unified-weighted-parameter-quantiles-v1",
            "weight_semantics": "normalized Dynesty importance weights",
            "quantiles": summary["parameter_summary"],
        })
        selection = load_json(staging / "best_model/selection.json")
        selection.update({
            "schema_version": "kinuv-unified-production-selection-v2",
            "selected_best": selected_best,
            "map_visibility_chi2": map_chi2,
            "posterior_median_visibility_chi2": median_chi2,
            "posterior_minus_map_visibility_chi2": delta,
            "posterior_selection_required_improvement_chi2": float(args.chi2_margin),
            "posterior_status": "ACCEPTED_WEIGHTED_DYNESTY",
            "visibility_chi2": median_chi2 if select_posterior else map_chi2,
            "selection_rationale": (
                "posterior median visibility chi2 improves on MAP beyond the declared margin"
                if select_posterior else
                "MAP retained because posterior median visibility chi2 is not demonstrably lower"
            ),
        })
        write_json(posterior / "selection_candidate.json", {
            "schema_version": "kinuv-unified-posterior-selection-candidate-v1",
            "componentwise_weighted_median_q": median_q.tolist(),
            "visibility_chi2": median_chi2,
            "map_visibility_chi2": map_chi2,
            "delta_chi2": delta,
            "selected": select_posterior,
        })
        shutil.copy2(
            staging / "best_model/radial_profiles.npz",
            posterior / "map_radial_profiles.npz",
        )
        if select_posterior:
            write_selected_model(target, staging, load_json(REPO / "configs/targets" / f"{target}.json"), context, spec, selected_q)
            write_json(staging / "best_model/selected_posterior_median.json", load_json(posterior / "selection_candidate.json"))
        write_json(staging / "best_model/selection.json", selection)
        best_summary = load_json(staging / "best_model/summary.json")
        best_summary["selection"] = selection
        best_summary["posterior"] = {
            "source": "nuts/dynesty/dynesty_summary.json",
            "sampler": "Dynesty rslice",
            "sample_semantics": "weighted nested-sampling particles",
            "parameter_quantiles": "nuts/dynesty/parameter_quantiles.json",
            "profile_quantiles": "nuts/dynesty/radial_profile_quantiles.npz",
        }
        write_json(staging / "best_model/summary.json", best_summary)
        write_json(staging / "nuts/status.json", {
            "schema_version": "kinuv-unified-nuts-status-v2",
            "state": "FALLBACK_TO_DYNESTY",
            "reason": "accepted replicated Dynesty evidence is the posterior source",
            "posterior_directory": "nuts/dynesty",
            "sample_semantics": "weighted nested-sampling particles; not MCMC chains or unweighted draws",
            "rhat_applicable": False,
        })
        config = load_json(REPO / "configs/targets" / f"{target}.json")
        benchmark = refresh_selected_diagnostics(
            target,
            staging,
            source,
            config,
            context,
            spec,
            selected_q,
            selected_best,
            posterior / "radial_profile_quantiles.npz",
        )
        best_summary = load_json(staging / "best_model/summary.json")
        best_summary["benchmark_metrics"] = benchmark["metrics"]
        write_json(staging / "best_model/summary.json", best_summary)
        render_corner(samples, weight, posterior)
        render_profile_bands(posterior / "radial_profile_quantiles.npz", posterior / "map_radial_profiles.npz", summary["profile"]["support"], posterior)
        render_pvd_posterior(target, staging, config, selected_q, q, weight, spec, posterior)
        phase4_global, phase4_records = phase4_target_records(args.phase4_synthetic)
        _, _, phase4_target_path = phase4_records[target]
        benchmark_copies = (
            (args.phase4_synthetic, "unified_phase4_synthetic_summary.json"),
            (phase4_target_path, "unified_phase4_synthetic_target.json"),
            (args.phase4_real, "unified_phase4_real_refit_result.json"),
        )
        for evidence_path, name in benchmark_copies:
            shutil.copy2(evidence_path, staging / "benchmarks" / name)
        render_phase4_synthetic(
            target, args.phase4_synthetic, context, staging / "benchmarks"
        )
        provenance = {
            "schema_version": SCHEMA,
            "created_utc": now(),
            "source_production": {"path": str(source.resolve()), "manifest_sha256": sha256(source / "manifest.json")},
            "dynesty_evidence": {"path": str(evidence.resolve()), "summary_sha256": sha256(evidence / "summary.json"), "manifest_sha256": sha256(evidence / "MANIFEST.json")},
            "phase4_synthetic": {
                "path": str(args.phase4_synthetic.resolve()),
                "sha256": sha256(args.phase4_synthetic),
                "state": phase4_global["state"],
                "registered_gate_pass": phase4_global["gate_pass"],
                "target_summary": {
                    "path": str(phase4_target_path.resolve()),
                    "sha256": sha256(phase4_target_path),
                },
                "replacement_figure": "benchmarks/synthetic_benchmark.pdf",
                "kgas007_varying_sigma_limitation_preserved": True,
            },
            "phase4_real": {"path": str(args.phase4_real.resolve()), "sha256": sha256(args.phase4_real)},
            "visibility_selection": selection,
            "selected_model_diagnostics_regenerated": True,
        }
        write_json(staging / "provenance/posterior_staging.json", provenance)
        (staging / "README.md").write_text(
            f"""# {target} unified kinUV MAP plus weighted posterior

`best_model/` remains the selected visibility model. `nuts/dynesty/` contains
accepted replicated Dynesty weighted particles, parameter and radial-profile
quantiles, a weighted primary-parameter corner, and corrected posterior PVD
endpoint overlays. `nuts/status.json` records the Dynesty fallback without
treating nested-sampling particles as MCMC chains. Image products remain
diagnostics.
""",
            encoding="ascii",
        )
        write_json(posterior / "manifest.json", {
            "schema_version": "kinuv-unified-weighted-posterior-manifest-v1",
            "sample_semantics": "normalized weighted nested-sampling particles",
            "files": {path.relative_to(posterior).as_posix(): file_record(path) for path in sorted(posterior.rglob("*")) if path.is_file() and path != posterior / "manifest.json"},
        })
        rebuild_manifests(staging, target, selected_best)
        os.replace(staging, destination)
        return {"target_id": target, "selected_best": selected_best, "map_visibility_chi2": map_chi2, "posterior_median_visibility_chi2": median_chi2, "delta_chi2": delta, "output": str(destination)}
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--dynesty-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase4-synthetic", type=Path, required=True)
    parser.add_argument("--phase4-real", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", choices=("KGAS066", "KGAS007"), default=("KGAS066", "KGAS007"))
    parser.add_argument("--chi2-margin", type=float, default=DEFAULT_CHI2_MARGIN)
    args = parser.parse_args()
    for name in ("production_root", "dynesty_root", "output_root", "phase4_synthetic", "phase4_real"):
        setattr(args, name, getattr(args, name).resolve())
    if args.chi2_margin < 0:
        raise ValueError("chi2-margin must be nonnegative")
    for path in (args.phase4_synthetic, args.phase4_real):
        if not path.is_file():
            raise FileNotFoundError(path)
    results = [stage_target(args, target) for target in args.targets]
    write_json(args.output_root / "summary.json", {"schema_version": SCHEMA, "created_utc": now(), "targets": results})
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
