#!/usr/bin/env python3
"""Conservative grouped visibility prediction audit for S4 recovery.

kinUV is refit on each training fold.  The frozen full-data stock KinMS fit is
then rendered by the validated continuum adapter and scored on exactly the
same held-out rows and C1 covariance.  Because KinMS has seen the held-out
image cube, this comparison gives the external baseline an information
advantage; a positive kinUV delta is therefore conservative.  It does not
replace the later strictly training-only stock-KinMS confirmation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from astropy.io import fits

from kinuv.diagnostics.comparator import sample_intrinsic_kinms
from kinuv.forward.sb import galaxy_r_phi, load_sb_template
from kinuv.infer.map import image_grid_for_vis, predict_binned
from kinuv.infer.s2 import correlated_chi2, propagate_equal_weight_ar1
from kinuv.infer.s3 import (
    EmissivityBasis,
    build_positive_emissivity_basis,
    fit_s3_candidate,
    initial_chart,
)
from kinuv.io.vis import (
    load_target_vis,
    load_visibility_table,
    optical_to_radio_kms,
)
from kinuv.profiles.rotation import arctan_vc
from kinuv.validation.groups import build_grouped_visibility_folds


REPO = Path(__file__).resolve().parents[1]
KINMS_PYTHON = Path("/scratch/kinuv-thbrown/s1-kinms/bin/python")
KINMS_WORKER = REPO / "external/_kinms_intrinsic_worker.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _git_state() -> dict:
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


def _covariance(metrics: dict, target_id: str, n_bin: int):
    target = next(row for row in metrics["targets"] if row["target_id"] == target_id)
    source = next(iter(target["covariance"]["parameters"]["C1"].values()))
    return propagate_equal_weight_ar1(source["scale"], source["rho"], n_bin)


def _hard_basis(template, grid, pa_rad, i_rad) -> EmissivityBasis:
    """Reconstruct the frozen pre-recovery hard-band basis for non-regression."""
    image = np.maximum(np.asarray(template, dtype=np.float64), 0.0)
    area = float(grid.cell_arcsec) ** 2
    image /= float(np.sum(image) * area)
    radius, _ = galaxy_r_phi(grid, pa_rad, i_rad)
    order = np.argsort(radius.ravel())
    cumulative = np.cumsum(image.ravel()[order])
    cumulative /= cumulative[-1]
    quantiles = np.interp(
        [1.0 / 3.0, 2.0 / 3.0, 0.95], cumulative, radius.ravel()[order]
    )
    q33, q67, r95 = quantiles
    edges = (0.0, q33, q67, np.inf)
    components = []
    natural = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        component = np.where((radius >= lower) & (radius < upper), image, 0.0)
        fraction = float(np.sum(component) * area)
        components.append(component / fraction)
        natural.append(fraction)
    natural = np.asarray(natural, dtype=np.float64)
    natural /= natural.sum()
    return EmissivityBasis(
        images=np.stack(components),
        natural_weights=natural,
        radial_edges_arcsec=np.asarray([0.0, q33, q67]),
        r95_arcsec=float(r95),
        transition_width_arcsec=0.0,
    )


def _chart_from_fit(fit, s3, bmaj, vsys_seed, dv_kms, basis):
    p = fit["parameters"]
    seed = {
        "flux": p["flux"],
        "pa_deg": p["pa_deg"],
        "vsys_kms": p["vsys_kms"],
        "gas_sigma_kms": p["sigma_inner_kms"],
        "dx_arcsec": p["dx_arcsec"],
        "dy_arcsec": p["dy_arcsec"],
        "u_kms": p["arctan_u_kms"],
        "inclination_deg": p["inclination_deg"],
        "r_t_arcsec": p["turnover_over_bmaj"] * bmaj,
    }
    z = initial_chart(
        seed,
        np.asarray(s3["velocity_support"]["knot_radii_arcsec"]),
        basis.natural_weights,
        vsys_seed_kms=vsys_seed,
        dv_kms=dv_kms,
        bmaj_arcsec=bmaj,
    )
    z[9:13] = np.asarray(p["u_knots_kms"]) / 100.0
    weights = np.asarray(p["emissivity_weights"])
    z[13:15] = np.log(weights[1:] / weights[0])
    z[15] = np.log(p["sigma_outer_kms"])
    return z


def _profiles(s3, fit, bmaj):
    p = fit["parameters"]
    candidate = fit["candidate"]
    knots = np.asarray(s3["velocity_support"]["knot_radii_arcsec"])
    parent = s3["selection"]["two_zone_dispersion_parent"]
    uses_rings = candidate == "supported_rings" or (
        candidate == "two_zone_dispersion" and parent == "supported_rings"
    )
    inclination = np.radians(float(p["inclination_deg"]))
    if uses_rings:
        speeds = np.asarray(p["u_knots_kms"])

        def velocity(radius):
            value = np.interp(radius, knots, speeds)
            value = np.where(radius < knots[0], speeds[0] * radius / knots[0], value)
            value = np.where(radius > knots[-1], speeds[-1], value)
            return value / np.sin(inclination)
    else:
        velocity = None
    if candidate == "two_zone_dispersion":
        transition = knots[1]
        width = 0.25 * bmaj

        def dispersion(radius):
            outer = 0.5 * (1.0 + np.tanh((radius - transition) / width))
            return p["sigma_inner_kms"] * (1.0 - outer) + p["sigma_outer_kms"] * outer
    else:
        dispersion = None
    return velocity, dispersion


def _predict(data, grid, template, basis, s3, fit):
    p = fit["parameters"]
    if fit["candidate"] == "baseline_arctan":
        image = template
    else:
        weights = np.asarray(p["emissivity_weights"], dtype=np.float64)
        image = np.sum(weights[:, None, None] * basis.images, axis=0)
    velocity, dispersion = _profiles(s3, fit, float(s3["bmaj_arcsec"]))
    params = {
        "flux": p["flux"],
        "pa_deg": p["pa_deg"],
        "vsys_kms": p["vsys_kms"],
        "gas_sigma_kms": p["sigma_inner_kms"],
        "dx_arcsec": p["dx_arcsec"],
        "dy_arcsec": p["dy_arcsec"],
        "v0_kms": p["arctan_u_kms"] / np.sin(np.radians(p["inclination_deg"])),
        "r_t_arcsec": p["turnover_over_bmaj"] * float(s3["bmaj_arcsec"]),
    }
    return np.asarray(
        predict_binned(
            data,
            params,
            image,
            grid,
            i_rad=np.radians(p["inclination_deg"]),
            velocity_profile=velocity,
            dispersion_profile=dispersion,
        )
    )


def _m0_radial_profile(config, fitted):
    cube = np.asarray(fits.getdata(config["diagnostic_cube"]), dtype=np.float64)
    header = fits.getheader(config["diagnostic_cube"])
    mask = np.asarray(fits.getdata(config["diagnostic_mask"]), dtype=float) > 0.5
    dv = abs(float(header["CDELT3"]))
    if str(header.get("CUNIT3", "km/s")).lower().replace(" ", "") in {"m/s", "ms-1"}:
        dv /= 1000.0
    m0 = np.sum(np.where(mask, cube, 0.0), axis=0) * dv
    nx, ny = int(header["NAXIS1"]), int(header["NAXIS2"])
    x = (np.arange(nx) + 1.0 - float(header["CRPIX1"])) * float(header["CDELT1"]) * 3600.0
    y = (np.arange(ny) + 1.0 - float(header["CRPIX2"])) * float(header["CDELT2"]) * 3600.0
    east = -x if float(header["CDELT1"]) < 0.0 else x
    xe, yn = np.meshgrid(east, y, indexing="xy")
    phase = json.loads(Path(config["kinms"]["result"]).read_text()).get(
        "phase_center_arcsec", [0.0, 0.0]
    )
    pa = np.radians(float(fitted["pa_deg"]))
    inc = np.radians(float(fitted["i_deg"]))
    dx, dy = xe - float(phase[0]), yn - float(phase[1])
    major = dx * np.sin(pa) + dy * np.cos(pa)
    minor = (dx * np.cos(pa) - dy * np.sin(pa)) / max(np.cos(inc), 1.0e-6)
    radius = np.hypot(major, minor)
    centres = np.linspace(0.05, max(12.0, float(np.nanmax(radius))), 64)
    half = 0.5 * (centres[1] - centres[0])
    profile = np.zeros_like(centres)
    for index, center in enumerate(centres):
        use = (np.abs(radius - center) <= half) & (m0 > 0.0)
        profile[index] = float(np.mean(m0[use])) if np.any(use) else 0.0
    peak = max(float(np.max(profile)), 1.0)
    return centres, np.maximum(profile, peak * 1.0e-12)


def _lsrk_optical_to_topo_radio(vopt, correction):
    lsrk_radio = float(optical_to_radio_kms(vopt))
    return (lsrk_radio + float(correction)) / (1.0 + float(correction) / 299792.458)


def _render_frozen_kinms(config, data, grid, output):
    result_path = Path(config["kinms"]["result"])
    result = json.loads(result_path.read_text())
    fitted = result["fitted"]
    radius, brightness = _m0_radial_profile(config, fitted)
    vr = np.linspace(0.0, max(15.0, float(radius[-1])), 257)
    correction = config["spectral_frame"]["frequency_correction_equivalent_kms"]
    artifact = output / "kinms_intrinsic.npz"
    worker_config = {
        "schema_version": "kinms-continuum-cube-v2",
        "output_npz": str(artifact),
        "artifact_id": f"{config['target_id']}/kinms_intrinsic.npz",
        "grid": {"nx": grid.nx, "ny": grid.ny, "cell_arcsec": grid.cell_arcsec},
        "velocity_centers_kms": np.asarray(data.vel_native).tolist(),
        "seed": 66007,
        "radial_samples": 256,
        "azimuth_samples": 512,
        "spectral_oversample": 2,
        "azimuth_phase_fractions": [0.0],
        "surface_brightness": {"radius_arcsec": radius.tolist(), "profile": brightness.tolist()},
        "parameters": {
            "velocity_radius_arcsec": vr.tolist(),
            "circular_speed_kms": np.asarray(arctan_vc(vr, fitted["v0_kms"], fitted["r_t_arcsec"])).tolist(),
            "inclination_deg": fitted["i_deg"],
            "pa_deg": fitted["pa_deg"],
            "gas_sigma_kms": fitted["gas_sigma_kms"],
            "flux_jy_kms": fitted["flux_scale"],
            "dx_arcsec": result.get("phase_center_arcsec", [0.0, 0.0])[0],
            "dy_arcsec": result.get("phase_center_arcsec", [0.0, 0.0])[1],
            "vsys_kms": _lsrk_optical_to_topo_radio(fitted["vsys_optical_kms"], correction),
        },
    }
    config_path = output / "kinms_intrinsic.config.json"
    _write_json(config_path, worker_config)
    process = subprocess.run(
        [str(KINMS_PYTHON), str(KINMS_WORKER), str(config_path)],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (output / "kinms_intrinsic.log").write_text(process.stdout, encoding="utf-8")
    if process.returncode or not artifact.is_file():
        raise RuntimeError(f"KinMS continuum render failed; see {output/'kinms_intrinsic.log'}")
    return artifact, result


def _bootstrap_delta(fold_rows, seed=4404, draws=20000):
    rng = np.random.default_rng(seed)
    delta = np.asarray([row["delta_chi2"] for row in fold_rows])
    count = np.asarray([row["n_real_components"] for row in fold_rows])
    indices = rng.integers(0, len(fold_rows), size=(draws, len(fold_rows)))
    samples = np.sum(delta[indices], axis=1) / np.sum(count[indices], axis=1)
    estimate = float(np.sum(delta) / np.sum(count))
    return {
        "delta_chi2_per_real_component": estimate,
        "lower_95_percent": float(np.quantile(samples, 0.025)),
        "upper_95_percent": float(np.quantile(samples, 0.975)),
        "bootstrap_draws": draws,
        "resampling_unit": "native-row time-block fold",
    }


def run_target(config_path, s3_root, old_s3_root, covariance_metrics, output, maxiter):
    config = json.loads(Path(config_path).read_text())
    target_id = config["target_id"]
    s3 = json.loads((Path(s3_root) / target_id / "ablations.json").read_text())
    old_s3 = json.loads((Path(old_s3_root) / target_id / "ablations.json").read_text())
    candidate = s3["selection"]["preferred_candidate"]
    full_fit = next(row for row in s3["fits"] if row["candidate"] == candidate)
    old_name = old_s3["selection"]["preferred_candidate"]
    old_fit = next(row for row in old_s3["fits"] if row["candidate"] == old_name)
    table = load_visibility_table(config["visibility_npz"])
    folds = build_grouped_visibility_folds(table, n_folds=5, integrations_per_group=5)
    phase = config.get("phase_center_deg")
    phase_rad = None if phase is None else np.radians(np.asarray(phase, dtype=float))
    full_data, _ = load_target_vis(
        table, cube_path=config["fit_window_cube"], phase_dir_rad=phase_rad
    )
    grid = image_grid_for_vis(full_data)
    template = load_sb_template(grid, Path(config["template_ico"]))
    s2_summary = json.loads(Path(s3["s2_summary_path"]).read_text())
    s2_parameters = next(
        row for row in s2_summary["targets"] if row["target_id"] == target_id
    )["best_joint"]["parameters"]
    pa = np.radians(s2_parameters["pa_deg"])
    inc = np.radians(s2_parameters["inclination_deg"])
    basis = build_positive_emissivity_basis(template, grid, pa, inc)
    hard_basis = _hard_basis(template, grid, pa, inc)
    bmaj = float(config["diagnostic_beam"]["bmaj_arcsec"])
    covariance = _covariance(covariance_metrics, target_id, full_data.n_bin)
    vsys_seed = float(config["stage_a"]["parameter_seed"]["vsys_kms"])
    two_zone_uses_rings = s3["selection"]["two_zone_dispersion_parent"] == "supported_rings"
    artifact, kinms_result = _render_frozen_kinms(config, full_data, grid, output)
    fold_rows = []
    for fold_id in range(folds.n_folds):
        embargo = float(2.0 * np.max(table.interval))
        train_mask = folds.training_mask(fold_id, embargo_s=embargo)
        test_mask = folds.validation_mask(fold_id)
        train, _ = load_target_vis(
            table, cube_path=config["fit_window_cube"], phase_dir_rad=phase_rad, row_mask=train_mask
        )
        held, _ = load_target_vis(
            table, cube_path=config["fit_window_cube"], phase_dir_rad=phase_rad, row_mask=test_mask
        )
        z0 = _chart_from_fit(full_fit, s3, bmaj, vsys_seed, train.dv_kms, basis)
        result, _ = fit_s3_candidate(
            train,
            template,
            basis,
            np.asarray(s3["velocity_support"]["knot_radii_arcsec"]),
            grid,
            covariance,
            z0,
            candidate=candidate,
            pa_seed_deg=float(config["geometry"]["pa_seed_deg"]),
            vsys_seed_kms=vsys_seed,
            bmaj_arcsec=bmaj,
            maxiter=maxiter,
            two_zone_uses_rings=two_zone_uses_rings,
        )
        fit = result.to_dict()
        kinuv_vis = _predict(held, grid, template, basis, s3, fit)
        kinms_vis, _, _ = sample_intrinsic_kinms(artifact, data=held, grid=grid)
        old_vis = _predict(held, grid, template, hard_basis, old_s3, old_fit)
        c_kinuv = correlated_chi2(held.vis, kinuv_vis, held.weights, covariance)
        c_kinms = correlated_chi2(held.vis, kinms_vis, held.weights, covariance)
        c_old = correlated_chi2(held.vis, old_vis, held.weights, covariance)
        n_component = int(2 * np.sum(held.weights > 0.0))
        row = {
            "fold_id": fold_id,
            "train_native_rows": int(np.sum(train_mask)),
            "held_native_rows": int(np.sum(test_mask)),
            "n_real_components": n_component,
            "chi2_kinuv": float(c_kinuv),
            "chi2_kinms": float(c_kinms),
            "chi2_frozen_old_kinuv": float(c_old),
            "delta_chi2": float(c_kinms - c_kinuv),
            "kinuv_minus_frozen_old_chi2": float(c_kinuv - c_old),
            "fit": fit,
        }
        fold_rows.append(row)
        print(json.dumps({"target": target_id, "fold": fold_id, "delta_per_component": row["delta_chi2"] / n_component}, sort_keys=True), flush=True)
    bootstrap = _bootstrap_delta(fold_rows)
    return {
        "target_id": target_id,
        "candidate": candidate,
        "comparator": {
            "stock_fit_result": str(Path(config["kinms"]["result"])),
            "stock_fit_sha256": _sha256(Path(config["kinms"]["result"])),
            "continuum_artifact": str(artifact),
            "full_data_fit_information_advantage": True,
            "interpretation": "conservative audit; strict training-only KinMS confirmation remains required for final promotion",
            "fitted": kinms_result["fitted"],
        },
        "folds": fold_rows,
        "aggregate": bootstrap,
        "gate": {
            "kinuv_beats_information_advantaged_kinms": bootstrap["lower_95_percent"] > 0.0,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-config", action="append", type=Path, required=True)
    parser.add_argument("--s3-root", type=Path, required=True)
    parser.add_argument("--old-s3-root", type=Path, required=True)
    parser.add_argument("--covariance-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maxiter", type=int, default=120)
    args = parser.parse_args()
    state = _git_state()
    if state["branch"] != "dev" or state["dirty"]:
        raise RuntimeError("S4 grouped prediction requires a clean exact commit on dev")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    covariance = json.loads(args.covariance_metrics.read_text())
    targets = [
        run_target(
            path,
            args.s3_root,
            args.old_s3_root,
            covariance,
            args.output / json.loads(path.read_text())["target_id"],
            args.maxiter,
        )
        for path in args.target_config
    ]
    strict_confirmation_required = True
    summary = {
        "schema_version": "kinuv-s4-grouped-prediction-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": state,
        "targets": targets,
        "conservative_audit_pass": all(
            row["gate"]["kinuv_beats_information_advantaged_kinms"] for row in targets
        ),
        "promotion_eligible": not strict_confirmation_required,
        "remaining_confirmation": "refit frozen stock KinMS on images made from each training fold only",
    }
    _write_json(args.output / "summary.json", summary)
    manifest = {"schema_version": "kinuv-s4-grouped-manifest-v1", "code_commit": state["commit"], "files": {}}
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            manifest["files"][str(path.relative_to(args.output))] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    _write_json(args.output / "MANIFEST.json", manifest)
    print(json.dumps({"stage": "S4-grouped", "audit_pass": summary["conservative_audit_pass"], "promotion_eligible": False}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
