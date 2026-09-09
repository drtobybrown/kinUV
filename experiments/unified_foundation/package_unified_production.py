#!/usr/bin/env python3
"""Package a completed unified-chart MAP campaign for production review.

The fit remains visibility-domain.  Restored cubes and image products are
derived diagnostics generated from the selected frozen MAP vector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from astropy.io import fits


REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_production_figures as direct_figures  # noqa: E402
from unified_map_runner import build_problem  # noqa: E402
from kinuv.constants import freq_to_velocity_kms  # noqa: E402
from kinuv.diagnostics.delivery import (  # noqa: E402
    render_moments,
    render_pvd,
    render_radial_profiles,
    render_spectra,
)
from kinuv.diagnostics.imaging import (  # noqa: E402
    match_model_to_imaging,
    masked_moments,
    spectral_axis_kms,
)
from kinuv.diagnostics.kinms_benchmark import write_cube_benchmark  # noqa: E402
from kinuv.diagnostics.s1 import sky_cube_fits  # noqa: E402
from kinuv.forward.model import intrinsic_sky_cube, attenuate_intrinsic_cube  # noqa: E402
from kinuv.infer.unified import decode_unified_chart, unified_profile_callables  # noqa: E402
from kinuv.io.vis import radio_to_optical_kms  # noqa: E402
from kinuv.validation.s4 import topo_radio_to_lsrk_radio  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def select_results(campaign: Path, target: str) -> tuple[dict, list[dict]]:
    results = [load_json(path) for path in sorted((campaign / target).glob("start-*/result.json"))]
    if len(results) != 4:
        raise ValueError(f"{target}: expected four MAP results, found {len(results)}")
    eligible = [row for row in results if row["gate_pass"]]
    if not eligible:
        raise ValueError(f"{target}: no conditioning-gate-passed MAP result")
    return min(eligible, key=lambda row: row["objective"]["visibility_chi2"]), results


def profile_consensus(winner: dict, results: list[dict]) -> dict:
    radius = np.asarray(winner["profile"]["radius_arcsec"])
    selected = np.asarray(winner["profile"]["u_projected_kms"])
    r95 = float(winner["support"]["emission_flux_quantiles_arcsec"]["emission_r95"])
    support = radius <= r95
    rows = []
    for row in results:
        curve = np.asarray(row["profile"]["u_projected_kms"])
        relative_l2 = float(
            np.linalg.norm(curve[support] - selected[support])
            / max(np.linalg.norm(selected[support]), 1.0e-12)
        )
        rows.append(
            {
                "start_id": row["start_id"],
                "gate_pass": row["gate_pass"],
                "delta_visibility_chi2": float(
                    row["objective"]["visibility_chi2"]
                    - winner["objective"]["visibility_chi2"]
                ),
                "supported_u_relative_l2_to_selected": relative_l2,
            }
        )
    return {
        "emission_r95_arcsec": r95,
        "all_starts_supported_u_relative_l2_max": max(
            item["supported_u_relative_l2_to_selected"] for item in rows
        ),
        "starts": rows,
    }


def write_native_cube(path: Path, cube, grid, velocity_lsrk_radio, ico_header, *, intrinsic: bool) -> None:
    array, header = sky_cube_fits(cube, grid, velocity_lsrk_radio, ico_header)
    header["SPECSYS"] = "LSRK"
    header["BUNIT"] = "Jy/pixel"
    header["OBJECT"] = path.parent.parent.name
    header["ORIGIN"] = "kinUV unified-chart visibility MAP"
    header["KINUVSTG"] = "MAP"
    header["CUBEROLE"] = "INTRINSIC" if intrinsic else "PB_ATTENUATED"
    fits.PrimaryHDU(array.astype(np.float32), header).writeto(path, overwrite=True)


def copy_pair(source_dir: Path, source_stem: str, destination_dir: Path, destination_stem: str) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        source = source_dir / f"{source_stem}.{suffix}"
        if source.is_file():
            shutil.copy2(source, destination_dir / f"{destination_stem}.{suffix}")


def package_target(campaign: Path, source_production: Path, output_root: Path, target: str) -> dict:
    winner, results = select_results(campaign, target)
    source = source_production / target
    target_root = output_root / target
    if target_root.exists():
        shutil.rmtree(target_root)
    best, maps, nuts, plots, benchmarks, provenance = (
        target_root / name
        for name in ("best_model", "map", "nuts", "plots", "benchmarks", "provenance")
    )
    for directory in (best, maps, nuts, plots, benchmarks, provenance):
        directory.mkdir(parents=True)

    config_path = REPO / "configs" / "targets" / f"{target}.json"
    config = load_json(config_path)
    correction = float(config["spectral_frame"]["frequency_correction_equivalent_kms"])
    context, spec, _, _, _ = build_problem(target)
    optimum = np.asarray(winner["optimum_z"], dtype=np.float64)
    physical = decode_unified_chart(optimum, spec)
    velocity_profile, dispersion_profile = unified_profile_callables(optimum, spec)
    intrinsic = intrinsic_sky_cube(
        context.template,
        context.grid,
        context.data.freqs_native,
        flux=physical["flux"],
        pa_rad=physical["pa_rad"],
        vsys_kms=physical["vsys_kms"],
        dx_arcsec=physical["dx_arcsec"],
        dy_arcsec=physical["dy_arcsec"],
        gas_sigma_kms=physical["sigma0_kms"],
        v0_kms=physical["u_reference_kms"] / np.maximum(np.sin(physical["i_rad"]), 1.0e-6),
        r_t_arcsec=spec.support.outer_radius_arcsec,
        i_rad=physical["i_rad"],
        velocity_profile=velocity_profile,
        dispersion_profile=dispersion_profile,
    )
    attenuated = attenuate_intrinsic_cube(intrinsic, context.grid, context.data.freqs_native)
    velocity_topo_radio = freq_to_velocity_kms(context.data.freqs_native)
    velocity_lsrk_radio = topo_radio_to_lsrk_radio(velocity_topo_radio, correction)
    ico_header = fits.getheader(config["template_ico"])
    write_native_cube(best / "intrinsic_native_model.fits", intrinsic, context.grid, velocity_lsrk_radio, ico_header, intrinsic=True)
    native_pb_path = best / "native_pb_attenuated_model.fits"
    write_native_cube(native_pb_path, attenuated, context.grid, velocity_lsrk_radio, ico_header, intrinsic=False)

    with fits.open(config["diagnostic_cube"], memmap=False) as hdul:
        data_cube = np.asarray(hdul[0].data, dtype=np.float64).squeeze()
        data_header = hdul[0].header.copy()
    native_pb = np.asarray(fits.getdata(native_pb_path), dtype=np.float64)
    native_header = fits.getheader(native_pb_path)
    matched, velocity_optical, dv = match_model_to_imaging(
        native_pb,
        native_header,
        data_header,
        nu_hz=float(np.median(context.data.freqs_native)),
        undo_pb=True,
    )
    # Reprojection outside the native footprint is blank sky, not missing
    # science data.  Store it as zero so the delivered FITS cube is finite.
    matched = np.nan_to_num(matched, nan=0.0, posinf=0.0, neginf=0.0)
    model_header = data_header.copy()
    model_header["BUNIT"] = "K"
    model_header["ORIGIN"] = "kinUV unified-chart visibility MAP diagnostic"
    model_header["KINUVSTG"] = "MAP"
    model_path = best / "model_on_science_grid.fits"
    fits.PrimaryHDU(matched.astype(np.float32), model_header).writeto(model_path, overwrite=True)

    for row in results:
        directory = maps / f"start-{row['start_id']}"
        directory.mkdir()
        shutil.copy2(campaign / target / f"start-{row['start_id']}" / "result.json", directory / "result.json")
        for name in ("worker.log", "headless_exit.json"):
            path = campaign / target / f"start-{row['start_id']}" / name
            if path.is_file():
                shutil.copy2(path, directory / name)
    write_json(maps / "selected_map.json", winner)
    shutil.copy2(config_path, best / "config.json")
    shutil.copy2(maps / "selected_map.json", best / "selected_map.json")

    kinms_source = source / "benchmarks" / "kinms_model_k.fits"
    kinms_fit_source = source / "benchmarks" / "kinms_fit_result.json"
    shutil.copy2(kinms_fit_source, benchmarks / "kinms_fit_result.json")
    benchmark = write_cube_benchmark(
        target_id=target,
        data_cube=Path(config["diagnostic_cube"]),
        mask_cube=Path(config["diagnostic_mask"]),
        kinuv_cube=model_path,
        kinms_cube=kinms_source,
        output_dir=benchmarks,
        pa_deg=float(np.degrees(physical["pa_rad"])),
        inclination_deg=float(np.degrees(physical["i_rad"])),
        vsys_kms=float(radio_to_optical_kms(topo_radio_to_lsrk_radio(float(physical["vsys_kms"]), correction))),
        dx_arcsec=float(physical["dx_arcsec"]),
        dy_arcsec=float(physical["dy_arcsec"]),
    )
    for obsolete in (
        "moments_comparison.png", "spectra_comparison.png", "pvd_major_comparison.png",
        "rotation_curve_comparison.png", "channel_maps_comparison.png",
    ):
        (benchmarks / obsolete).unlink(missing_ok=True)

    mask = np.asarray(fits.getdata(config["diagnostic_mask"]), dtype=np.float64).squeeze() > 0.5
    kinms = np.asarray(fits.getdata(benchmarks / "kinms_model_k.fits"), dtype=np.float64).squeeze()
    with np.load(benchmarks / "moments.npz", allow_pickle=False) as archive:
        moments = {key: np.asarray(archive[key]) for key in archive.files}
    geometry = {
        "pa_deg": float(np.degrees(physical["pa_rad"])) % 360.0,
        "inclination_deg": float(np.degrees(physical["i_rad"])),
        "vsys_kms": float(radio_to_optical_kms(topo_radio_to_lsrk_radio(float(physical["vsys_kms"]), correction))),
        "dx_arcsec": float(physical["dx_arcsec"]),
        "dy_arcsec": float(physical["dy_arcsec"]),
    }
    direct_figures.pv_figure(target, data_cube, matched, mask, data_header, geometry, plots)
    direct_figures.spectral_figure(target, data_cube, matched, mask, data_header, geometry, plots)
    render_moments(target, moments, data_header, geometry, "MAP unified", benchmarks)

    profile = winner["profile"]
    radius = np.asarray(profile["radius_arcsec"])
    kinuv_projected = np.asarray(profile["u_projected_kms"])
    kinuv_intrinsic = np.asarray(profile["v_rot_kms"])
    sigma = np.asarray(profile["sigma_kms"])
    kinms_document = load_json(kinms_fit_source)
    kinms_fit = kinms_document.get("fitted", kinms_document)
    kinms_intrinsic = float(kinms_fit["v0_kms"]) * (2.0 / np.pi) * np.arctan(
        radius / float(kinms_fit["r_t_arcsec"])
    )
    kinms_projected = kinms_intrinsic * np.sin(
        np.radians(float(kinms_fit.get("i_deg", kinms_fit.get("inclination_deg"))))
    )
    slope = float((kinuv_projected[1] - kinuv_projected[0]) / (radius[1] - radius[0]))
    profile_payload = {
        "radius": radius,
        "beam": float(spec.support.bmaj_arcsec),
        "kinuv_projected": kinuv_projected,
        "kinuv_intrinsic": kinuv_intrinsic,
        "sigma_map": sigma,
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
    }
    render_pvd(target, (data_cube, matched, kinms), mask, data_header, geometry, "MAP unified", profile_payload, benchmarks)
    render_spectra(target, (data_cube, matched, kinms), mask, data_header, geometry, "MAP unified", benchmarks)
    render_radial_profiles(target, profile_payload, "MAP unified", benchmarks, benchmark=True)
    render_radial_profiles(target, profile_payload, "MAP unified", plots, benchmark=False)
    copy_pair(benchmarks, "moments_kinuv_vs_kinms", plots, "moments_comparison")
    copy_pair(benchmarks, "spectra_kinuv_vs_kinms", plots, "spectral_profiles")

    for name in ("synthetic_benchmark.pdf", "synthetic_benchmark.png"):
        existing = source / "benchmarks" / name
        if existing.is_file():
            shutil.copy2(existing, benchmarks / name)

    np.savez(
        best / "radial_profiles.npz",
        radius_arcsec=radius,
        projected_velocity_kms=kinuv_projected,
        intrinsic_rotation_kms=kinuv_intrinsic,
        dispersion_kms=sigma,
    )
    old_chi2 = float(load_json(source / "best_model" / "selection.json")["map_visibility_chi2"])
    consensus = profile_consensus(winner, results)
    selection = {
        "schema_version": "kinuv-unified-production-selection-v1",
        "target_id": target,
        "selected_best": "MAP",
        "selected_start_id": winner["start_id"],
        "model_chart": "unified_direct_projected_velocity_pspline",
        "visibility_chi2": winner["objective"]["visibility_chi2"],
        "superseded_visibility_chi2": old_chi2,
        "delta_visibility_chi2": winner["objective"]["visibility_chi2"] - old_chi2,
        "gradient_inf_per_complex": winner["objective"]["gradient_inf_per_complex"],
        "selection_rationale": "lowest visibility chi2 among conditioning-gate-passed unified MAP starts",
        "posterior_status": "NOT_RUN_FOR_UNIFIED_CHART",
        "r50_status": winner["profile"]["flags"]["R50_status"],
        "profile_consensus": consensus,
    }
    write_json(best / "selection.json", selection)
    write_json(best / "parameters.json", {"native_TOPO_radio": winner["physical"], "diagnostic_optical_LSRK": geometry})
    write_json(best / "summary.json", {"selection": selection, "benchmark_metrics": benchmark["metrics"]})
    write_json(nuts / "status.json", {"state": "NOT_RUN", "reason": "unified chart campaign requested MAP conditioning and production diagnostics"})
    write_json(provenance / "campaign.json", {
        "campaign_root": str(campaign.resolve()),
        "code_commit": winner["git"]["commit"],
        "diagnostic_frame_fix_commit": "1e8ea5bf645a3c3bd00144fb364c3acea024cf81",
        "visibility_fit_only": True,
        "image_products_are_diagnostics": True,
        "input_files": {
            key: {"path": str(Path(config[key]).resolve()), **file_record(Path(config[key]))}
            for key in ("visibility_npz", "template_ico", "diagnostic_cube", "diagnostic_mask")
        },
    })
    write_json(best / "manifest.json", {
        "schema_version": "kinuv-unified-best-model-manifest-v1",
        "files": {
            path.relative_to(best).as_posix(): file_record(path)
            for path in sorted(best.rglob("*")) if path.is_file() and path != best / "manifest.json"
        },
    })
    write_json(target_root / "manifest.json", {
        "schema_version": "kinuv-unified-production-manifest-v1",
        "target_id": target,
        "selected_best": "MAP",
        "files": {
            path.relative_to(target_root).as_posix(): file_record(path)
            for path in sorted(target_root.rglob("*")) if path.is_file() and path != target_root / "manifest.json"
        },
    })
    return selection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--source-production", type=Path, default=PROJECT / "results" / "production")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", choices=("KGAS007", "KGAS066"), default=("KGAS007", "KGAS066"))
    args = parser.parse_args()
    summary = {}
    for target in args.targets:
        summary[target] = package_target(args.campaign, args.source_production, args.output_root, target)
        print(json.dumps({"target": target, "state": "PACKAGED", "chi2": summary[target]["visibility_chi2"]}, sort_keys=True), flush=True)
    write_json(args.output_root / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
