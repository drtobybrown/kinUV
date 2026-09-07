#!/usr/bin/env python3
"""Merge completed collaborator NUTS chains and compute posterior diagnostics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from kinuv.io.vis import load_target_vis

from run_collaborator_nuts_chain import REPO, setup_problem


TARGETS = ("KGAS066", "KGAS007")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )


def bounded_draws(transform, fixed_full, unconstrained):
    output = np.broadcast_to(fixed_full, unconstrained.shape[:-1] + fixed_full.shape).copy()
    bounds = transform.solver_bounds()
    for position, index in enumerate(transform.active):
        lo, hi = bounds[index]
        fraction = 1.0 / (1.0 + np.exp(-unconstrained[..., position]))
        value = lo + (hi - lo) * fraction
        if index in transform.log_kinematic_indices:
            value = np.exp(value)
        output[..., index] = value
    return output


def physical_draws(selected_path: Path, unconstrained: np.ndarray) -> dict[str, np.ndarray]:
    selected_doc, transform, fixed_full, _ = setup_problem(selected_path)
    z = bounded_draws(transform, fixed_full, unconstrained)
    target = selected_doc["target_id"]
    config = json.loads((REPO / "configs/targets" / f"{target}.json").read_text(encoding="utf-8"))
    phase = config.get("phase_center_deg")
    phase_rad = None if phase is None else np.radians(np.asarray(phase, dtype=float))
    data, _ = load_target_vis(
        config["visibility_npz"],
        cube_path=config["fit_window_cube"],
        phase_dir_rad=phase_rad,
    )
    bmaj = float(config["diagnostic_beam"]["bmaj_arcsec"])
    vsys_seed = float(config["stage_a"]["parameter_seed"]["vsys_kms"])
    inclination = np.degrees(np.arccos(z[..., 6]))
    pa_map = float(selected_doc["selected"]["result"]["parameters"]["pa_deg"])
    pa = np.degrees(z[..., 1]) % 360.0
    pa_unwrapped = pa_map + ((pa - pa_map + 180.0) % 360.0 - 180.0)
    values = {
        "flux_jy_kms": np.exp(z[..., 0]),
        "pa_deg": pa_unwrapped,
        "vsys_native_kms": vsys_seed + z[..., 2] * data.dv_kms,
        "sigma_inner_kms": np.exp(z[..., 3]),
        "dx_arcsec": z[..., 4] * bmaj,
        "dy_arcsec": z[..., 5] * bmaj,
        "inclination_deg": inclination,
    }
    sine = np.sin(np.radians(inclination))
    if selected_doc["selected"]["selected_parent"] == "supported_rings" or selected_doc["selected"].get("two_zone_uses_rings", False):
        for index in range(4):
            projected = 100.0 * z[..., 9 + index]
            values[f"u_knot_{index + 1}_kms"] = projected
            values[f"v_knot_{index + 1}_kms"] = projected / sine
    else:
        projected = 100.0 * z[..., 8]
        values["u_infinity_kms"] = projected
        values["v_flat_kms"] = projected / sine
        values["r_turn_arcsec"] = z[..., 7] * bmaj
    if 15 in transform.active:
        values["sigma_outer_kms"] = np.exp(z[..., 15])
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--nuts-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    import arviz as az
    import xarray as xr

    controller = json.loads((args.nuts_root / "controller_status.json").read_text(encoding="utf-8"))
    if controller["state"] != "SUCCEEDED" or any(row["exit_code"] != 0 for row in controller["completed"]):
        raise RuntimeError("all eight NUTS workers must succeed before posterior finalization")
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    for target in TARGETS:
        chains = []
        sample_stats = {name: [] for name in ("diverging", "num_steps", "accept_prob", "energy")}
        source_records = []
        for chain in range(1, 5):
            status_path = args.nuts_root / target / f"chain-{chain}" / "status.json"
            draws_path = args.nuts_root / target / f"chain-{chain}" / "draws.npz"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status["state"] != "SUCCEEDED" or status["completed_draws"] != 1000:
                raise RuntimeError(f"incomplete chain: {target} {chain}")
            with np.load(draws_path, allow_pickle=False) as archive:
                chains.append(np.asarray(archive["unconstrained"], dtype=np.float64))
                for name in sample_stats:
                    sample_stats[name].append(np.asarray(archive[name]))
            source_records.append(
                {"chain": chain, "draws": {"path": str(draws_path.resolve()), "sha256": sha256(draws_path)}, "status": {"path": str(status_path.resolve()), "sha256": sha256(status_path)}}
            )
        unconstrained = np.stack(chains, axis=0)
        physical = physical_draws(args.map_root / target / "selected_map.json", unconstrained)
        stats = {name: np.stack(parts, axis=0) for name, parts in sample_stats.items()}
        inference = az.from_dict(
            posterior=physical,
            sample_stats={
                "diverging": stats["diverging"].astype(bool),
                "tree_depth": np.ceil(np.log2(stats["num_steps"] + 1)).astype(int),
                "acceptance_rate": stats["accept_prob"],
                "energy": stats["energy"],
            },
        )
        summary_frame = az.summary(inference, kind="all", round_to=None)
        summary = {
            name: {
                "p16": float(np.percentile(values, 16.0)),
                "p50": float(np.percentile(values, 50.0)),
                "p84": float(np.percentile(values, 84.0)),
                "r_hat": float(summary_frame.loc[name, "r_hat"]),
                "ess_bulk": float(summary_frame.loc[name, "ess_bulk"]),
                "ess_tail": float(summary_frame.loc[name, "ess_tail"]),
            }
            for name, values in physical.items()
        }
        names = list(physical)
        matrix = np.stack([physical[name].reshape(-1) for name in names], axis=0)
        covariance = np.cov(matrix)
        correlation = np.corrcoef(matrix)
        output = args.output_root / target
        output.mkdir(parents=True, exist_ok=True)
        samples_path = output / "posterior_samples.npz"
        np.savez(samples_path, **physical, chain=np.arange(1, 5), draw=np.arange(1000))
        xr.Dataset(
            {
                name: (("chain", "draw"), values)
                for name, values in physical.items()
            },
            coords={"chain": np.arange(1, 5), "draw": np.arange(1000)},
        ).to_netcdf(output / "trace.nc", engine="h5netcdf")
        np.savez(output / "covariance.npz", names=np.asarray(names), covariance=covariance, correlation=correlation)
        divergence_count = int(np.sum(stats["diverging"]))
        bfmi = np.asarray(az.bfmi(inference), dtype=float)
        primary = [name for name in names if name not in {"flux_jy_kms", "dx_arcsec", "dy_arcsec"}]
        gates = {
            "r_hat_primary_max": max(summary[name]["r_hat"] for name in primary),
            "ess_bulk_primary_min": min(summary[name]["ess_bulk"] for name in primary),
            "ess_tail_primary_min": min(summary[name]["ess_tail"] for name in primary),
            "divergences": divergence_count,
            "bfmi_min": float(np.min(bfmi)),
            "tree_depth_saturation_count": int(np.sum(stats["num_steps"] >= 1023)),
        }
        gates["accepted"] = bool(
            gates["r_hat_primary_max"] <= 1.05
            and gates["ess_bulk_primary_min"] >= 400
            and gates["ess_tail_primary_min"] >= 400
            and divergence_count == 0
            and gates["bfmi_min"] >= 0.30
            and gates["tree_depth_saturation_count"] == 0
        )
        record = {
            "schema_version": "kinuv-collaborator-posterior-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "target_id": target,
            "git_commit": git_commit,
            "conditional_on_fixed_emissivity": True,
            "chains": 4,
            "warmup_per_chain": 1000,
            "draws_per_chain": 1000,
            "summary": summary,
            "gates": gates,
            "bfmi_by_chain": bfmi.tolist(),
            "sources": source_records,
            "packages": {"arviz": az.__version__, "xarray": xr.__version__},
        }
        write_json(output / "summary.json", record)
        write_json(
            output / "MANIFEST.json",
            {
                "schema_version": "kinuv-collaborator-posterior-manifest-v1",
                "git_commit": git_commit,
                "files": {
                    path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                    for path in sorted(output.iterdir())
                    if path.is_file() and path.name != "MANIFEST.json"
                },
            },
        )
        print(json.dumps({"target": target, **gates}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
