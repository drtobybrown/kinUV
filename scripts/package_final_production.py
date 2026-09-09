#!/usr/bin/env python3
"""Stage the final target-centred production layout and diagnostic suite."""

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
import numpy as np
from astropy.io import fits

from kinuv.diagnostics.delivery import (
    render_moments,
    render_pvd,
    render_radial_profiles,
    render_spectra,
    render_synthetic,
)

REPO = Path(__file__).resolve().parents[1]
TARGETS = ("KGAS066", "KGAS007")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="ascii")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def git_state() -> dict:
    return {
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip()),
    }


def load_cube(path: Path):
    with fits.open(path, memmap=False) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float64).squeeze(), hdul[0].header.copy()


def arctan(radius, speed, turnover):
    return (2.0 / np.pi) * float(speed) * np.arctan(np.asarray(radius) / float(turnover))


def _posterior_profiles(target: str, source: Path, radius: np.ndarray, knot_radii: np.ndarray):
    posterior = source / "best_model" / "posterior" / "posterior_samples.npz"
    if not posterior.is_file():
        posterior = source / "nuts" / "posterior_samples.npz"
    if target != "KGAS007" or not posterior.is_file():
        return {}
    with np.load(posterior, allow_pickle=False) as archive:
        u = np.stack([archive[f"u_knot_{i}_kms"] for i in range(1, 5)], axis=-1).reshape(-1, 4)
        v = np.stack([archive[f"v_knot_{i}_kms"] for i in range(1, 5)], axis=-1).reshape(-1, 4)
        sigma = archive["sigma_inner_kms"].reshape(-1)
    projected = np.asarray([np.interp(radius, knot_radii, row, left=0.0, right=row[-1]) for row in u])
    intrinsic = np.asarray([np.interp(radius, knot_radii, row, left=0.0, right=row[-1]) for row in v])
    inner = radius < knot_radii[0]
    projected[:, inner] = u[:, :1] * radius[None, inner] / knot_radii[0]
    intrinsic[:, inner] = v[:, :1] * radius[None, inner] / knot_radii[0]
    return {
        "projected_lo": np.percentile(projected, 16, axis=0),
        "projected_hi": np.percentile(projected, 84, axis=0),
        "intrinsic_lo": np.percentile(intrinsic, 16, axis=0),
        "intrinsic_hi": np.percentile(intrinsic, 84, axis=0),
        "sigma_lo": np.full_like(radius, np.percentile(sigma, 16)),
        "sigma_hi": np.full_like(radius, np.percentile(sigma, 84)),
    }


def build_profiles(target: str, source: Path, header, parameters: dict, selected: dict, kinms: dict, moment0):
    with np.load(source / "best_model" / "rotation_curve.npz", allow_pickle=False) as archive:
        radius = np.asarray(archive["radius_arcsec"])
        projected = np.asarray(archive["projected_speed_kms"])
        intrinsic = np.asarray(archive["intrinsic_speed_kms"])
    checkpoint = load_json(Path(selected["selected"]["inputs"]["checkpoint"]["path"]))
    knot_radii = np.asarray(checkpoint["velocity_support"]["knot_radii_arcsec"], dtype=float)
    beam = float(header["BMAJ"]) * 3600.0
    candidate = parameters["candidate"]
    fitted = parameters["fitted_native_parameters"]
    if candidate == "two_zone_dispersion":
        transition = knot_radii[1]
        outer = 0.5 * (1.0 + np.tanh((radius - transition) / (0.25 * beam)))
        sigma_map = fitted["sigma_inner_kms"] * (1 - outer) + fitted["sigma_outer_kms"] * outer
        turnover = fitted["turnover_over_bmaj"] * beam
        asymptotic = float(fitted["arctan_u_kms"]) / np.sin(np.deg2rad(float(fitted["inclination_deg"])))
        gradient = (2.0 / np.pi) * asymptotic / turnover
        inflation = float(kinms["r_t_arcsec"]) / turnover
        profile_metrics = {
            "turnover": rf"$R_{{\rm turn}}={turnover:.3f}''$ ({turnover / beam:.2f} BMAJ)",
            "velocity": rf"$V_\infty={asymptotic:.1f}\ \mathrm{{km\ s^{{-1}}}}$",
            "inner_gradient": rf"$(dV/dr)_0={gradient:.1f}\ \mathrm{{km\ s^{{-1}}\ arcsec^{{-1}}}}$",
            "smearing": rf"KinMS $R_{{\rm turn}}$ inflation = {inflation:.2f}x",
        }
    else:
        sigma_map = np.full_like(radius, fitted["sigma_inner_kms"])
        turnover = None
        outer_speed = float(intrinsic[-1])
        gradient = float(intrinsic[1] / radius[1])
        profile_metrics = {
            "turnover": r"$R_{\rm turn}$: spline/knot model",
            "velocity": rf"$V_{{\rm knot,out}}={outer_speed:.1f}\ \mathrm{{km\ s^{{-1}}}}$",
            "inner_gradient": rf"$(dV/dr)_0={gradient:.1f}\ \mathrm{{km\ s^{{-1}}\ arcsec^{{-1}}}}$",
            "smearing": r"KinMS turnover inflation: not applicable",
        }
    result = {
        "radius": radius,
        "kinuv_projected": projected,
        "kinuv_intrinsic": intrinsic,
        "kinms_projected": arctan(radius, kinms["v0_kms"] * np.sin(np.deg2rad(kinms["i_deg"])), kinms["r_t_arcsec"]),
        "kinms_intrinsic": arctan(radius, kinms["v0_kms"], kinms["r_t_arcsec"]),
        "kinms_sigma": float(kinms["gas_sigma_kms"]),
        "kinms_vsys": float(kinms["vsys_optical_kms"]),
        "sigma_map": sigma_map,
        "turnover": turnover,
        "beam": beam,
        "moment0": moment0,
        "profile_metrics": profile_metrics,
    }
    result.update(_posterior_profiles(target, source, radius, knot_radii))
    return result


def copy_map_record(target, destination: Path, map_root: Path):
    source = map_root / target
    shutil.copytree(source, destination)
    logs = destination / "logs"
    logs.mkdir()
    for path in sorted((map_root / "logs").glob(f"{target}-start-*.log")):
        shutil.copy2(path, logs / path.name.replace(f"{target}-", ""))
    write_json(destination / "manifest.json", {
        "schema_version": "kinuv-map-stage-v1",
        "target_id": target,
        "files": {p.relative_to(destination).as_posix(): record(p) for p in sorted(destination.rglob("*")) if p.is_file() and p.name != "manifest.json"},
    })


def copy_nuts_record(target, source: Path, destination: Path, nuts_live_root: Path):
    destination.mkdir()
    accepted = source / "best_model" / "posterior"
    packaged = source / "nuts"
    if not accepted.is_dir() and (packaged / "status.json").is_file():
        if load_json(packaged / "status.json").get("state") == "ACCEPTED":
            accepted = packaged
    if accepted.is_dir():
        for path in accepted.iterdir():
            if path.is_file() and path.name not in {"manifest.json", "status.json"}:
                shutil.copy2(path, destination / path.name)
        for suffix in ("pdf", "png"):
            corner = source / "plots" / f"posterior_corner.{suffix}"
            if corner.is_file():
                shutil.copy2(corner, destination / corner.name)
        status = {"target_id": target, "state": "ACCEPTED", "source": str(accepted.resolve())}
    else:
        chains = []
        for chain in range(1, 5):
            path = nuts_live_root / target / f"chain-{chain}" / "status.json"
            chains.append({"status_path": str(path.resolve()), "status": load_json(path)})
        status = {
            "target_id": target,
            "state": "RUNNING_UNACCEPTED",
            "reason": "No posterior is promoted until all chains complete and convergence gates pass.",
            "chains": chains,
        }
    write_json(destination / "status.json", status)
    write_json(destination / "manifest.json", {
        "schema_version": "kinuv-nuts-stage-v1",
        "target_id": target,
        "state": status["state"],
        "files": {p.relative_to(destination).as_posix(): record(p) for p in sorted(destination.rglob("*")) if p.is_file() and p.name != "manifest.json"},
    })
    return status


def package_target(target, production_root, output_root, map_root, nuts_live_root, synthetic_root, subbeam_root, state):
    source = production_root / target
    destination = output_root / target
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(source, destination)
    for stale in (destination / "MANIFEST.json", destination / "best_model" / "MANIFEST.json"):
        stale.unlink(missing_ok=True)
    (destination / "best_model" / "posterior").exists() and shutil.rmtree(destination / "best_model" / "posterior")
    shutil.rmtree(destination / "map", ignore_errors=True); shutil.rmtree(destination / "nuts", ignore_errors=True)
    copy_map_record(target, destination / "map", map_root)
    nuts = copy_nuts_record(target, source, destination / "nuts", nuts_live_root)

    best = destination / "best_model"
    selected = load_json(best / "selected_map.json")
    parameters = load_json(best / "parameters.json")
    summary = load_json(best / "summary.json")
    geometry = parameters["diagnostic_geometry_lsrk_optical"]
    kinms_doc = load_json(destination / "benchmarks" / "kinms_fit_result.json")
    kinms = kinms_doc.get("fitted", kinms_doc)
    data, header = load_cube(Path(load_json(best / "config.json")["diagnostic_cube"]))
    mask, _ = load_cube(Path(load_json(best / "config.json")["diagnostic_mask"]))
    model, _ = load_cube(best / "model_on_science_grid.fits")
    kinms_cube, _ = load_cube(destination / "benchmarks" / "kinms_model_k.fits")
    with np.load(destination / "benchmarks" / "moments.npz", allow_pickle=False) as archive:
        moments = {name: np.asarray(archive[name]) for name in archive.files}
    profiles = build_profiles(target, source, header, parameters, selected, kinms, moments["data_moment0"])
    stage = "MAP"
    for directory in (destination / "plots", destination / "benchmarks"):
        for path in directory.glob("*.pdf"):
            path.unlink()
        for path in directory.glob("*.png"):
            path.unlink()
    benchmark = destination / "benchmarks"; plots = destination / "plots"
    render_moments(target, moments, header, geometry, stage, benchmark)
    render_pvd(target, (data, model, kinms_cube), mask > 0.5, header, geometry, stage, profiles, benchmark)
    render_spectra(target, (data, model, kinms_cube), mask > 0.5, header, geometry, stage, benchmark)
    render_radial_profiles(target, profiles, stage, plots, benchmark=False)
    render_radial_profiles(target, profiles, stage, benchmark, benchmark=True)
    render_synthetic(target, load_json(synthetic_root / target / "summary.json"), load_json(subbeam_root / target / "summary.json"), benchmark)
    for source_name, target_name in (("moments_kinuv_vs_kinms", "moments_comparison"), ("pvd_kinuv_vs_kinms", "pv_diagrams"), ("spectra_kinuv_vs_kinms", "spectral_profiles")):
        for suffix in ("pdf", "png"):
            shutil.copy2(benchmark / f"{source_name}.{suffix}", plots / f"{target_name}.{suffix}")
    if nuts["state"] == "ACCEPTED":
        for suffix in ("pdf", "png"):
            shutil.copy2(destination / "nuts" / f"posterior_corner.{suffix}", plots / f"posterior_corner.{suffix}")
    else:
        for suffix in ("pdf", "png"):
            (plots / f"posterior_corner.{suffix}").unlink(missing_ok=True)

    gates = None if nuts["state"] != "ACCEPTED" else load_json(destination / "nuts" / "summary.json")["gates"]
    selection = {
        "schema_version": "kinuv-production-selection-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_id": target,
        "selected_best": "MAP",
        "map_visibility_chi2": float(selected["selected"]["result"]["chi2"]),
        "map_selection": selected["selection"],
        "nuts_state": nuts["state"],
        "nuts_gates": gates,
        "rationale": (
            "MAP is the authoritative single model because it has the lowest validated visibility chi-square and defines the delivered cube. "
            + ("The accepted conditional NUTS posterior supplies uncertainty bands and covariance around this mode." if gates else
               "The active NUTS campaign is incomplete and currently exhibits retained tree-depth saturation, so it is not promoted.")
        ),
    }
    write_json(best / "selection.json", selection)
    summary["active_product_stage"] = "MAP"
    summary["nuts_state"] = nuts["state"]
    summary["posterior_available_for_selected_checkpoint"] = bool(gates)
    write_json(best / "summary.json", summary)
    (destination / "README.md").write_text(
        f"# {target} production products\n\n"
        "The authoritative point model is the visibility MAP in `best_model/`. "
        + ("The accepted conditional NUTS posterior is in `nuts/` and supplies the plotted 16th-84th percentile bands.\n\n" if gates else
           "The NUTS campaign remains unaccepted; `nuts/status.json` records its durable live state.\n\n")
        + "`map/` contains all four optimizer starts and logs. `plots/` contains standard diagnostics; `benchmarks/` contains matched KinMS and registered synthetic comparisons.\n",
        encoding="ascii",
    )
    files = {p.relative_to(destination).as_posix(): record(p) for p in sorted(destination.rglob("*"))
             if p.is_file() and p.name not in {"manifest.json", "MANIFEST.json"}}
    manifest = {
        "schema_version": "kinuv-production-layout-v5",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_id": target,
        "git": state,
        "selected_best": "MAP",
        "nuts_state": nuts["state"],
        "files": files,
    }
    write_json(best / "manifest.json", {**manifest, "schema_version": "kinuv-best-model-delivery-manifest-v1", "paths_relative_to": "target_root"})
    files["best_model/manifest.json"] = record(best / "manifest.json")
    write_json(destination / "manifest.json", {**manifest, "files": files})
    print(json.dumps({"target": target, "selected_best": "MAP", "nuts_state": nuts["state"], "files": len(files)}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--nuts-live-root", type=Path, required=True)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--subbeam-root", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=TARGETS)
    args = parser.parse_args()
    state = git_state()
    if state["branch"] != "dev" or state["dirty"]:
        raise RuntimeError("production packaging requires a clean dev checkout")
    for target in args.targets:
        package_target(target, args.production_root, args.output_root, args.map_root, args.nuts_live_root,
                       args.synthetic_root, args.subbeam_root, state)


if __name__ == "__main__":
    main()
