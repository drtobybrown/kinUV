#!/usr/bin/env python3
"""Validate four unified chains and atomically create an incoming evidence packet."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import expit, ndtri
from scipy.stats import rankdata

from kinuv.infer.posterior import ess_bulk, ess_tail, split_rhat
from kinuv.infer.unified import decode_unified_chart
from kinuv.profiles.unified import projected_velocity, velocity_dispersion
from unified_map_runner import build_problem


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank_normalize(chains):
    values = np.asarray(chains, dtype=np.float64)
    flat = values.reshape(-1, values.shape[-1])
    normalized = np.empty_like(flat)
    count = flat.shape[0]
    for index in range(flat.shape[1]):
        ranks = rankdata(flat[:, index], method="average")
        normalized[:, index] = ndtri((ranks - 3.0 / 8.0) / (count + 1.0 / 4.0))
    return normalized.reshape(values.shape)


def rank_diagnostics(chains):
    normalized = rank_normalize(chains)
    folded = rank_normalize(np.abs(chains - np.median(chains, axis=(0, 1))))
    return (
        np.maximum(split_rhat(normalized), split_rhat(folded)),
        ess_bulk(normalized),
        ess_tail(chains),
    )


def energy_bfmi(energy):
    return np.mean(np.diff(energy, axis=1) ** 2, axis=1) / np.var(energy, axis=1, ddof=1)


def physical_draws(q, spec):
    pa = spec.pa_reference_rad - np.pi + 2.0 * np.pi * expit(q[..., 1])
    pa_deg = np.degrees(pa)
    reference_deg = np.degrees(spec.pa_reference_rad)
    pa_deg = reference_deg + ((pa_deg - reference_deg + 180.0) % 360.0 - 180.0)
    cos_i = spec.cos_inclination_min + (
        spec.cos_inclination_max - spec.cos_inclination_min
    ) * expit(q[..., 5])
    result = {
        "flux_jy_kms": np.exp(q[..., 0]),
        "pa_deg": pa_deg,
        "vsys_native_kms": spec.vsys_reference_kms + q[..., 2] * spec.dv_kms,
        "dx_arcsec": q[..., 3] * spec.support.bmaj_arcsec,
        "dy_arcsec": q[..., 4] * spec.support.bmaj_arcsec,
        "inclination_deg": np.degrees(np.arccos(cos_i)),
        "u_reference_kms": np.exp(q[..., 6]),
        "sigma0_kms": np.exp(q[..., 11]),
    }
    for index in range(4):
        result[f"rotation_shape_{index + 1}"] = q[..., 7 + index]
    for index in range(2):
        result[f"dispersion_shape_{index + 1}"] = q[..., 12 + index]
    return result


def profile_draws(q, spec, radius):
    def one(sample):
        physical = decode_unified_chart(sample, spec)
        u = projected_velocity(radius, sample[6], sample[7:11], spec.support)
        vc = u / jnp.maximum(jnp.sin(physical["i_rad"]), 1e-8)
        sigma = velocity_dispersion(radius, sample[11], sample[12:14], spec.support)
        return u, vc, sigma
    flat = jnp.asarray(q.reshape(-1, q.shape[-1]))
    u, vc, sigma = jax.jit(jax.vmap(one))(flat)
    shape = q.shape[:2] + (radius.size,)
    return np.asarray(u).reshape(shape), np.asarray(vc).reshape(shape), np.asarray(sigma).reshape(shape)


def write_npz(path, **arrays):
    np.savez(path, **arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("KGAS066", "KGAS007"), required=True)
    parser.add_argument("--nuts-root", type=Path, required=True)
    parser.add_argument("--map-result", type=Path, required=True)
    parser.add_argument("--map-commit", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable evidence destination exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        q_chains, stats, sources = [], {name: [] for name in ("num_steps", "diverging", "accept_prob", "energy", "step_size")}, []
        common_contract = None
        for chain in range(1, 5):
            root = args.nuts_root / args.target / f"chain-{chain}"
            paths = {name: root / name for name in ("status.json", "draws.npz", "headless_exit.json")}
            for path in paths.values():
                if not path.is_file():
                    raise FileNotFoundError(path)
            status = json.loads(paths["status.json"].read_text(encoding="ascii"))
            exit_doc = json.loads(paths["headless_exit.json"].read_text(encoding="ascii"))
            if status.get("state") != "SUCCEEDED" or exit_doc.get("exit_code") != 0:
                raise RuntimeError(f"chain {chain} is incomplete")
            contract = status["contract"]
            if contract["code_commit"] != args.code_commit or contract["map_commit"] != args.map_commit:
                raise RuntimeError(f"chain {chain} provenance mismatch")
            if contract["map_sha256"] != sha256(args.map_result):
                raise RuntimeError(f"chain {chain} MAP hash mismatch")
            common_contract = contract if common_contract is None else common_contract
            if contract != common_contract:
                raise RuntimeError("chain sampler contracts differ")
            with np.load(paths["draws.npz"], allow_pickle=False) as archive:
                q = np.asarray(archive["q_unified"], dtype=np.float64)
                if q.shape != (500, 14):
                    raise RuntimeError(f"chain {chain} has shape {q.shape}, expected (500, 14)")
                q_chains.append(q)
                for name in stats:
                    stats[name].append(np.asarray(archive[name]))
            sources.append({"chain": chain, **{name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in paths.items()}})
        q = np.stack(q_chains)
        stats = {name: np.stack(value) for name, value in stats.items()}
        _, spec, _, _, provenance = build_problem(args.target)
        physical = physical_draws(q, spec)
        names = list(physical)
        matrix = np.stack([physical[name] for name in names], axis=-1)
        rhat, bulk, tail = rank_diagnostics(matrix)
        parameter_summary = {
            name: {"q16": float(np.percentile(values, 16)), "q50": float(np.percentile(values, 50)),
                   "q84": float(np.percentile(values, 84)), "r_hat": float(rhat[index]),
                   "ess_bulk": float(bulk[index]), "ess_tail": float(tail[index])}
            for index, (name, values) in enumerate(physical.items())
        }
        primary = [name for name in names if name != "flux_jy_kms"]
        radius = np.linspace(0.0, spec.support.outer_radius_arcsec, 256)
        u, vc, sigma = profile_draws(q, spec, radius)
        profile_quantiles = {
            "radius_arcsec": radius,
            "u_projected_q16": np.percentile(u, 16, axis=(0, 1)),
            "u_projected_q50": np.percentile(u, 50, axis=(0, 1)),
            "u_projected_q84": np.percentile(u, 84, axis=(0, 1)),
            "v_rot_q16": np.percentile(vc, 16, axis=(0, 1)),
            "v_rot_q50": np.percentile(vc, 50, axis=(0, 1)),
            "v_rot_q84": np.percentile(vc, 84, axis=(0, 1)),
            "sigma_q16": np.percentile(sigma, 16, axis=(0, 1)),
            "sigma_q50": np.percentile(sigma, 50, axis=(0, 1)),
            "sigma_q84": np.percentile(sigma, 84, axis=(0, 1)),
        }
        tree_depth = np.ceil(np.log2(stats["num_steps"] + 1)).astype(int)
        bfmi = energy_bfmi(stats["energy"])
        gates = {
            "r_hat_primary_max": max(parameter_summary[name]["r_hat"] for name in primary),
            "r_hat_threshold": 1.05,
            "ess_bulk_primary_min": min(parameter_summary[name]["ess_bulk"] for name in primary),
            "ess_bulk_threshold": 400.0,
            "divergence_count": int(np.sum(stats["diverging"])),
            "tree_depth_saturation_count": int(
                np.sum(stats["num_steps"] >= 2 ** common_contract["max_tree_depth"] - 1)
            ),
            "bfmi_min": float(np.min(bfmi)),
        }
        gates["accepted"] = bool(
            gates["r_hat_primary_max"] <= gates["r_hat_threshold"]
            and gates["ess_bulk_primary_min"] >= gates["ess_bulk_threshold"]
            and gates["divergence_count"] == 0
            and gates["tree_depth_saturation_count"] == 0
            and gates["bfmi_min"] >= 0.30
        )
        write_npz(staging / "posterior_samples.npz", q_unified=q, **physical,
                  chain=np.arange(1, 5), draw=np.arange(500))
        write_npz(staging / "profile_samples.npz", u_projected_kms=u,
                  v_rot_kms=vc, sigma_kms=sigma, **profile_quantiles)
        write_npz(staging / "sample_stats.npz", **stats, tree_depth=tree_depth, bfmi=bfmi)
        summary = {
            "schema_version": "kinuv-unified-posterior-evidence-v1", "created_utc": utc_now(),
            "state": "ACCEPTED" if gates["accepted"] else "REJECTED_DIAGNOSTICS",
            "target_id": args.target, "chains": 4, "warmup_per_chain": 200, "draws_per_chain": 500,
            "conditional_on_fixed_emissivity": True, "posterior_calibration": "NOT_ESTABLISHED",
            "parameter_summary": parameter_summary, "gates": gates, "bfmi_by_chain": bfmi.tolist(),
            "profile": {"support": spec.support.metadata(), "quantiles_file": "profile_samples.npz",
                        "R50": None, "R50_status": "UNRESOLVED_PLATEAU",
                        "R50_reason": "outer continuation is computational and supplies no plateau-identifiability evidence"},
            "contract": common_contract, "model_provenance": provenance,
            "sources": {"map": {"path": str(args.map_result.resolve()), "sha256": sha256(args.map_result)},
                        "chains": sources},
        }
        (staging / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="ascii")
        files = {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(staging.iterdir())}
        manifest = {"schema_version": "kinuv-unified-posterior-evidence-manifest-v1",
                    "created_utc": utc_now(), "code_commit": args.code_commit, "files": files}
        (staging / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="ascii")
        os.replace(staging, args.output)
        print(json.dumps({"target": args.target, **gates}, sort_keys=True))
        return 0 if gates["accepted"] else 1
    except BaseException:
        for path in sorted(staging.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        staging.rmdir()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
