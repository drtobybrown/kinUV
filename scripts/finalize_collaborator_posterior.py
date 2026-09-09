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
from scipy import __version__ as scipy_version
from scipy.io import netcdf_file
from scipy.special import ndtri
from scipy.stats import rankdata

from kinuv.infer.posterior import ess_bulk, ess_tail, split_rhat
from kinuv.io.vis import load_target_vis

from run_collaborator_nuts_chain import REPO, setup_problem


TARGETS = ("KGAS066", "KGAS007")


def rank_normalize(chains: np.ndarray) -> np.ndarray:
    """Transform pooled draws to normal scores while retaining chain shape."""
    values = np.asarray(chains, dtype=np.float64)
    flat = values.reshape(-1, values.shape[-1])
    normalized = np.empty_like(flat)
    count = flat.shape[0]
    for index in range(flat.shape[1]):
        ranks = rankdata(flat[:, index], method="average")
        normalized[:, index] = ndtri((ranks - 3.0 / 8.0) / (count + 1.0 / 4.0))
    return normalized.reshape(values.shape)


def rank_diagnostics(chains: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized = rank_normalize(chains)
    folded = rank_normalize(np.abs(chains - np.median(chains, axis=(0, 1))))
    rhat = np.maximum(split_rhat(normalized), split_rhat(folded))
    return rhat, ess_bulk(normalized), ess_tail(chains)


def energy_bfmi(energy: np.ndarray) -> np.ndarray:
    energy = np.asarray(energy, dtype=np.float64)
    numerator = np.mean(np.diff(energy, axis=1) ** 2, axis=1)
    denominator = np.var(energy, axis=1, ddof=1)
    return numerator / denominator


def write_trace(path: Path, physical: dict[str, np.ndarray], draw_count: int) -> None:
    """Write a dependency-light NetCDF3 trace with explicit chain/draw axes."""
    with netcdf_file(path, mode="w") as dataset:
        dataset.createDimension("chain", 4)
        dataset.createDimension("draw", draw_count)
        for name, values in physical.items():
            variable = dataset.createVariable(name, "f8", ("chain", "draw"))
            variable[:] = np.asarray(values, dtype=np.float64)


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
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=TARGETS)
    args = parser.parse_args()
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    for target in args.targets:
        chains = []
        sample_stats = {name: [] for name in ("diverging", "num_steps", "accept_prob", "energy")}
        source_records = []
        warmup_counts = set()
        draw_counts = set()
        sampling_depths = set()
        for chain in range(1, 5):
            status_path = args.nuts_root / target / f"chain-{chain}" / "status.json"
            draws_path = args.nuts_root / target / f"chain-{chain}" / "draws.npz"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status["state"] != "SUCCEEDED" or int(status["completed_draws"]) <= 0:
                raise RuntimeError(f"incomplete chain: {target} {chain}")
            with np.load(draws_path, allow_pickle=False) as archive:
                unconstrained = np.asarray(archive["unconstrained"], dtype=np.float64)
                if unconstrained.shape[0] != int(status["completed_draws"]):
                    raise RuntimeError(f"draw/status mismatch: {target} {chain}")
                chains.append(unconstrained)
                for name in sample_stats:
                    sample_stats[name].append(np.asarray(archive[name]))
            warmup_counts.add(int(status["completed_warmup"]))
            draw_counts.add(int(status["completed_draws"]))
            sampling_depths.add(int(status.get("sampler_contract", {}).get("max_tree_depth", 10)))
            source_records.append(
                {"chain": chain, "draws": {"path": str(draws_path.resolve()), "sha256": sha256(draws_path)}, "status": {"path": str(status_path.resolve()), "sha256": sha256(status_path)}}
            )
        if len(warmup_counts) != 1 or len(draw_counts) != 1 or len(sampling_depths) != 1:
            raise RuntimeError(f"inconsistent sampler contracts across {target} chains")
        warmup_count = warmup_counts.pop()
        draw_count = draw_counts.pop()
        max_tree_depth = sampling_depths.pop()
        unconstrained = np.stack(chains, axis=0)
        physical = physical_draws(args.map_root / target / "selected_map.json", unconstrained)
        stats = {name: np.stack(parts, axis=0) for name, parts in sample_stats.items()}
        names = list(physical)
        physical_matrix = np.stack([physical[name] for name in names], axis=-1)
        rhat, bulk_ess, tail_ess = rank_diagnostics(physical_matrix)
        summary = {
            name: {
                "p16": float(np.percentile(values, 16.0)),
                "p50": float(np.percentile(values, 50.0)),
                "p84": float(np.percentile(values, 84.0)),
                "r_hat": float(rhat[index]),
                "ess_bulk": float(bulk_ess[index]),
                "ess_tail": float(tail_ess[index]),
            }
            for index, (name, values) in enumerate(physical.items())
        }
        matrix = np.stack([physical[name].reshape(-1) for name in names], axis=0)
        covariance = np.cov(matrix)
        correlation = np.corrcoef(matrix)
        output = args.output_root / target
        output.mkdir(parents=True, exist_ok=True)
        samples_path = output / "posterior_samples.npz"
        np.savez(samples_path, **physical, chain=np.arange(1, 5), draw=np.arange(draw_count))
        write_trace(output / "trace.nc", physical, draw_count)
        np.savez(output / "covariance.npz", names=np.asarray(names), covariance=covariance, correlation=correlation)
        divergence_count = int(np.sum(stats["diverging"]))
        bfmi = energy_bfmi(stats["energy"])
        primary = [name for name in names if name not in {"flux_jy_kms", "dx_arcsec", "dy_arcsec"}]
        gates = {
            "r_hat_primary_max": max(summary[name]["r_hat"] for name in primary),
            "ess_bulk_primary_min": min(summary[name]["ess_bulk"] for name in primary),
            "ess_tail_primary_min": min(summary[name]["ess_tail"] for name in primary),
            "divergences": divergence_count,
            "bfmi_min": float(np.min(bfmi)),
            "tree_depth_saturation_count": int(
                np.sum(stats["num_steps"] >= (2**max_tree_depth - 1))
            ),
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
            "warmup_per_chain": warmup_count,
            "draws_per_chain": draw_count,
            "sampling_max_tree_depth": max_tree_depth,
            "summary": summary,
            "gates": gates,
            "bfmi_by_chain": bfmi.tolist(),
            "sources": source_records,
            "packages": {"numpy": np.__version__, "scipy": scipy_version},
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
