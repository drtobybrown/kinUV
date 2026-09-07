#!/usr/bin/env python3
"""Render S3 selections and apply the PI-authorized real-target S4 gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

_repo = Path(__file__).resolve().parents[1]
_scratch_spec = importlib.util.spec_from_file_location(
    "_kinuv_scratch", _repo / "src/kinuv/scratch.py"
)
_scratch_module = importlib.util.module_from_spec(_scratch_spec)
_scratch_spec.loader.exec_module(_scratch_module)
_scratch_module.apply_scratch_env()

from kinuv.constants import F_REST_CO21_HZ, freq_to_velocity_kms
from kinuv.diagnostics.imaging import match_model_to_imaging, spectral_axis_kms
from kinuv.diagnostics.kinms_benchmark import write_cube_benchmark
from kinuv.forward.model import sky_cube
from kinuv.forward.sb import load_sb_template
from kinuv.infer.map import image_grid_for_vis
from kinuv.infer.s3 import build_positive_emissivity_basis
from kinuv.io.vis import (
    cube_vopt_window_kms,
    load_target_vis,
    load_visibility_table,
    optical_to_radio_kms,
    radio_to_optical_kms,
)
from kinuv.validation.groups import build_grouped_visibility_folds
from kinuv.validation.s4 import (
    channel_noise_from_integrated_error,
    common_reduced_chi2,
    projected_velocity_rmse,
    structured_line_free_diagnostics,
    topo_radio_to_lsrk_radio,
)


REPO = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _model_header(grid, velocity_lsrk_radio, data_header):
    velocity = np.asarray(velocity_lsrk_radio, dtype=np.float64)
    wcs = WCS(naxis=3)
    wcs.wcs.crpix = [grid.nx // 2 + 1, grid.ny // 2 + 1, 1.0]
    cell_deg = float(grid.cell_arcsec) / 3600.0
    wcs.wcs.cdelt = [-cell_deg, cell_deg, float(np.median(np.diff(velocity)))]
    wcs.wcs.crval = [
        float(data_header["CRVAL1"]),
        float(data_header["CRVAL2"]),
        float(velocity[0]),
    ]
    wcs.wcs.ctype = ["RA---SIN", "DEC--SIN", "VRAD"]
    wcs.wcs.cunit = ["deg", "deg", "km/s"]
    wcs.wcs.specsys = "LSRK"
    header = wcs.to_header()
    header["NAXIS"] = 3
    header["NAXIS1"] = int(grid.nx)
    header["NAXIS2"] = int(grid.ny)
    header["NAXIS3"] = int(velocity.size)
    header["RESTFRQ"] = float(F_REST_CO21_HZ)
    header["BUNIT"] = "Jy/pixel"
    header["ORIGIN"] = "kinUV S4 visibility-fit diagnostic"
    return header


def _ring_profile(knot_radii, projected_speeds, inclination_rad):
    radii = np.asarray(knot_radii, dtype=np.float64)
    speeds = np.asarray(projected_speeds, dtype=np.float64)
    sin_i = np.sin(float(inclination_rad))

    def profile(radius):
        r = np.asarray(radius)
        value = np.interp(r, radii, speeds)
        value = np.where(r < radii[0], speeds[0] * r / radii[0], value)
        value = np.where(r > radii[-1], speeds[-1], value)
        return value / sin_i

    return profile


def _two_zone_profile(knot_radii, sigma_inner, sigma_outer, bmaj_arcsec):
    transition = float(np.asarray(knot_radii)[1])
    width = 0.25 * float(bmaj_arcsec)

    def profile(radius):
        fraction = 0.5 * (1.0 + np.tanh((np.asarray(radius) - transition) / width))
        return float(sigma_inner) * (1.0 - fraction) + float(sigma_outer) * fraction

    return profile


def _active_parameter_count(preferred):
    return int(preferred["hessian"]["dimension"])


def _structured_diagnostics(config, covariance_metrics):
    target_id = config["target_id"]
    covariance_row = next(
        row for row in covariance_metrics["targets"] if row["target_id"] == target_id
    )
    source = next(iter(covariance_row["covariance"]["parameters"]["C1"].values()))
    table = load_visibility_table(config["visibility_npz"])
    folds = build_grouped_visibility_folds(
        table, n_folds=5, integrations_per_group=5
    )
    velocity = freq_to_velocity_kms(table.freqs)
    vopt_lo, vopt_hi = cube_vopt_window_kms(config["fit_window_cube"])
    vrad_lo = float(optical_to_radio_kms(vopt_lo))
    vrad_hi = float(optical_to_radio_kms(vopt_hi))
    line_free = (velocity < vrad_lo) | (velocity > vrad_hi)
    return structured_line_free_diagnostics(
        table,
        line_free,
        source["scale"],
        source["rho"],
        folds.row_fold_id,
    )


def run_target(config_path, s3_root, covariance_metrics, output):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    target_id = config["target_id"]
    target_s3 = Path(s3_root) / target_id / "ablations.json"
    s3 = json.loads(target_s3.read_text(encoding="utf-8"))
    preferred_name = s3["selection"]["preferred_candidate"]
    preferred = next(row for row in s3["fits"] if row["candidate"] == preferred_name)
    parameters = preferred["parameters"]
    s2_summary = json.loads(Path(s3["s2_summary_path"]).read_text(encoding="utf-8"))
    s2_parameters = next(
        row for row in s2_summary["targets"] if row["target_id"] == target_id
    )["best_joint"]["parameters"]

    phase = config.get("phase_center_deg")
    phase_rad = None if phase is None else np.radians(np.asarray(phase, dtype=float))
    data, load_metadata = load_target_vis(
        config["visibility_npz"],
        cube_path=config["fit_window_cube"],
        phase_dir_rad=phase_rad,
    )
    grid = image_grid_for_vis(data)
    base_template = load_sb_template(grid, Path(config["template_ico"]))
    basis = build_positive_emissivity_basis(
        base_template,
        grid,
        np.radians(float(s2_parameters["pa_deg"])),
        np.radians(float(s2_parameters["inclination_deg"])),
    )
    emissivity_weights = np.asarray(parameters["emissivity_weights"], dtype=np.float64)
    template = np.sum(emissivity_weights[:, None, None] * basis.images, axis=0)
    inclination_rad = np.radians(float(parameters["inclination_deg"]))
    knot_radii = np.asarray(s3["velocity_support"]["knot_radii_arcsec"])
    dispersion_parent = s3["selection"]["two_zone_dispersion_parent"]
    uses_rings = preferred_name == "supported_rings" or (
        preferred_name == "two_zone_dispersion"
        and dispersion_parent == "supported_rings"
    )
    velocity_profile = (
        _ring_profile(knot_radii, parameters["u_knots_kms"], inclination_rad)
        if uses_rings
        else None
    )
    dispersion_profile = (
        _two_zone_profile(
            knot_radii,
            parameters["sigma_inner_kms"],
            parameters["sigma_outer_kms"],
            config["diagnostic_beam"]["bmaj_arcsec"],
        )
        if preferred_name == "two_zone_dispersion"
        else None
    )
    cube_yxv = sky_cube(
        template,
        grid,
        data.freqs_native,
        flux=float(parameters["flux"]),
        pa_rad=np.radians(float(parameters["pa_deg"])),
        vsys_kms=float(parameters["vsys_kms"]),
        dx_arcsec=float(parameters["dx_arcsec"]),
        dy_arcsec=float(parameters["dy_arcsec"]),
        gas_sigma_kms=float(parameters["sigma_inner_kms"]),
        v0_kms=float(parameters["arctan_u_kms"]) / np.sin(inclination_rad),
        r_t_arcsec=float(parameters["turnover_over_bmaj"])
        * float(config["diagnostic_beam"]["bmaj_arcsec"]),
        i_rad=inclination_rad,
        velocity_profile=velocity_profile,
        dispersion_profile=dispersion_profile,
    )
    data_cube_path = Path(config["diagnostic_cube"])
    mask_path = Path(config["diagnostic_mask"])
    error_path = Path(config["diagnostic_ico_error"])
    with fits.open(data_cube_path, memmap=False) as hdul:
        observed = np.asarray(hdul[0].data, dtype=np.float64)
        data_header = hdul[0].header.copy()
    mask = np.asarray(fits.getdata(mask_path), dtype=np.float64) > 0.5
    correction = float(config["spectral_frame"]["frequency_correction_equivalent_kms"])
    native_radio_lsrk = topo_radio_to_lsrk_radio(data.vel_native, correction)
    model_header = _model_header(grid, native_radio_lsrk, data_header)
    model_fits_order = np.flip(np.moveaxis(cube_yxv, 2, 0), axis=2)
    model_k, _, dv_data = match_model_to_imaging(
        model_fits_order, model_header, data_header, undo_pb=True
    )
    destination = Path(output) / target_id
    destination.mkdir(parents=True, exist_ok=True)
    kinuv_cube_path = destination / "kinuv_model_k.fits"
    output_header = data_header.copy()
    output_header["BUNIT"] = "K"
    output_header["ORIGIN"] = "kinUV S4 visibility-fit model"
    fits.PrimaryHDU(model_k.astype(np.float32), output_header).writeto(
        kinuv_cube_path, overwrite=True
    )
    vsys_lsrk_radio = float(topo_radio_to_lsrk_radio(parameters["vsys_kms"], correction))
    vsys_lsrk_optical = float(radio_to_optical_kms(vsys_lsrk_radio))
    benchmark = write_cube_benchmark(
        target_id=target_id,
        data_cube=data_cube_path,
        mask_cube=mask_path,
        kinuv_cube=kinuv_cube_path,
        kinms_cube=Path(config["kinms"]["cube"]),
        output_dir=destination / "benchmark",
        pa_deg=float(parameters["pa_deg"]),
        inclination_deg=float(parameters["inclination_deg"]),
        vsys_kms=vsys_lsrk_optical,
        dx_arcsec=float(parameters["dx_arcsec"]),
        dy_arcsec=float(parameters["dy_arcsec"]),
    )
    kinms_k = np.asarray(
        fits.getdata(destination / "benchmark" / "kinms_model_k.fits"),
        dtype=np.float64,
    )
    error_moment0 = np.squeeze(np.asarray(fits.getdata(error_path), dtype=np.float64))
    sigma_channel, noise_pixels = channel_noise_from_integrated_error(
        error_moment0, mask, dv_data
    )
    kinuv_chi2 = common_reduced_chi2(
        observed, model_k, mask, sigma_channel, _active_parameter_count(preferred)
    )
    kinms_result = json.loads(Path(config["kinms"]["result"]).read_text(encoding="utf-8"))
    kinms_chi2 = common_reduced_chi2(
        observed,
        kinms_k,
        mask,
        sigma_channel,
        len(kinms_result.get("fitted", {})),
    )
    with np.load(destination / "benchmark" / "profiles.npz", allow_pickle=False) as z:
        profiles = {name: np.asarray(z[name]) for name in z.files}
    recovery = projected_velocity_rmse(profiles, parameters["inclination_deg"])
    structured = _structured_diagnostics(config, covariance_metrics)
    gates = {
        "projected_velocity_rmse_at_least_10_percent_better_than_kinms": recovery[
            "gate_eligible"
        ]
        and recovery["ratio_kinuv_over_kinms"] <= 0.90,
        "overall_reduced_chi2_better_than_kinms": kinuv_chi2["reduced_chi2"]
        < kinms_chi2["reduced_chi2"],
    }
    record = {
        "schema_version": "kinuv-s4-real-benchmark-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_id": target_id,
        "preferred_s3_candidate": preferred_name,
        "inputs": {
            "config": {"path": str(config_path.resolve()), "sha256": sha256(config_path)},
            "s3": {"path": str(target_s3.resolve()), "sha256": sha256(target_s3)},
            "data_cube": {"path": str(data_cube_path), "sha256": sha256(data_cube_path)},
            "mask_cube": {"path": str(mask_path), "sha256": sha256(mask_path)},
            "integrated_error": {"path": str(error_path), "sha256": sha256(error_path)},
            "kinms_cube": {
                "path": str(Path(config["kinms"]["cube"])),
                "sha256": sha256(Path(config["kinms"]["cube"])),
            },
            "kinms_result": {
                "path": str(Path(config["kinms"]["result"])),
                "sha256": sha256(Path(config["kinms"]["result"])),
            },
        },
        "load": load_metadata,
        "frame": {
            "likelihood": "native TOPO radio frequency",
            "diagnostic": "LSRK optical velocity",
            "frequency_equivalent_correction_kms": correction,
            "vsys_lsrk_radio_kms": vsys_lsrk_radio,
            "vsys_lsrk_optical_kms": vsys_lsrk_optical,
        },
        "noise": {
            "channel_rms_k": sigma_channel,
            "source_pixels": noise_pixels,
            "derivation": "median Ico_error/(abs(dv)*sqrt(masked_channel_count))",
            "spatial_covariance_note": "common scalar normalization; beam covariance cancels in method ratio",
        },
        "cube_metrics": benchmark["metrics"],
        "common_reduced_chi2": {"kinuv": kinuv_chi2, "kinms": kinms_chi2},
        "projected_velocity_recovery": recovery,
        "structured_line_free_residuals": structured,
        "gates": gates,
        "accepted": all(gates.values()),
    }
    write_json_atomic(destination / "metrics.json", record)
    print(
        json.dumps(
            {
                "target": target_id,
                "velocity_rmse_ratio": recovery["ratio_kinuv_over_kinms"],
                "reduced_chi2_ratio": kinuv_chi2["reduced_chi2"]
                / kinms_chi2["reduced_chi2"],
                "accepted": record["accepted"],
            },
            sort_keys=True,
            ensure_ascii=True,
        ),
        flush=True,
    )
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-config", action="append", type=Path, required=True)
    parser.add_argument("--s3-root", type=Path, required=True)
    parser.add_argument("--covariance-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = git_state()
    if state["branch"] != "dev" or state["dirty"]:
        raise RuntimeError("S4 benchmark requires a clean exact commit on dev")
    covariance = json.loads(args.covariance_metrics.read_text(encoding="utf-8"))
    records = [
        run_target(path, args.s3_root, covariance, args.output)
        for path in args.target_config
    ]
    summary = {
        "schema_version": "kinuv-s4-real-summary-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": state,
        "pi_gate": {
            "projected_velocity_rmse_ratio_max": 0.90,
            "overall_reduced_chi2_improvement_required": True,
            "targets": [row["target_id"] for row in records],
        },
        "targets": [
            {
                "target_id": row["target_id"],
                "accepted": row["accepted"],
                "gates": row["gates"],
                "projected_velocity_recovery": row["projected_velocity_recovery"],
                "common_reduced_chi2": row["common_reduced_chi2"],
            }
            for row in records
        ],
        "accepted": all(row["accepted"] for row in records),
    }
    write_json_atomic(args.output / "summary.json", summary)
    manifest = {
        "schema_version": "kinuv-s4-real-manifest-v1",
        "code_commit": state["commit"],
        "files": {},
    }
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            manifest["files"][str(path.relative_to(args.output))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    write_json_atomic(args.output / "MANIFEST.json", manifest)
    print(json.dumps({"stage": "S4", "accepted": summary["accepted"]}, sort_keys=True))
    if not summary["accepted"]:
        raise SystemExit("S4 PI superiority gate failed; escalate to Astra")


if __name__ == "__main__":
    main()
