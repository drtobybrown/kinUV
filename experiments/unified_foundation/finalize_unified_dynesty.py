#!/usr/bin/env python3
"""Validate two dynesty replicates and atomically create weighted evidence."""

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
from scipy.special import expit

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


def weighted_quantile(values, weights, probabilities=(0.16, 0.50, 0.84)):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape[0] != weights.size:
        raise ValueError("weights must align with first sample axis")
    result = np.empty((len(probabilities),) + values.shape[1:], dtype=np.float64)
    for index in np.ndindex(values.shape[1:]):
        column = values[(slice(None),) + index]
        order = np.argsort(column)
        cumulative = np.cumsum(weights[order])
        cumulative /= cumulative[-1]
        result[(slice(None),) + index] = np.interp(probabilities, cumulative, column[order])
    return result


def scalar_quantiles(values, weights):
    return {
        key: float(value)
        for key, value in zip(("q16", "q50", "q84"), weighted_quantile(values, weights))
    }


def physical_draws(q, spec):
    pa = spec.pa_reference_rad - np.pi + 2.0 * np.pi * expit(q[:, 1])
    pa_deg = np.degrees(pa)
    reference = np.degrees(spec.pa_reference_rad)
    pa_deg = reference + ((pa_deg - reference + 180.0) % 360.0 - 180.0)
    cos_i = spec.cos_inclination_min + (
        spec.cos_inclination_max - spec.cos_inclination_min
    ) * expit(q[:, 5])
    output = {
        "flux_jy_kms": np.exp(q[:, 0]), "pa_deg": pa_deg,
        "vsys_native_kms": spec.vsys_reference_kms + q[:, 2] * spec.dv_kms,
        "dx_arcsec": q[:, 3] * spec.support.bmaj_arcsec,
        "dy_arcsec": q[:, 4] * spec.support.bmaj_arcsec,
        "inclination_deg": np.degrees(np.arccos(cos_i)),
        "u_reference_kms": np.exp(q[:, 6]), "sigma0_kms": np.exp(q[:, 11]),
    }
    for index in range(4):
        output[f"rotation_shape_{index + 1}"] = q[:, 7 + index]
    for index in range(2):
        output[f"dispersion_shape_{index + 1}"] = q[:, 12 + index]
    return output


def profiles(q, spec, radius):
    def one(sample):
        physical = decode_unified_chart(sample, spec)
        u = projected_velocity(radius, sample[6], sample[7:11], spec.support)
        vc = u / jnp.maximum(jnp.sin(physical["i_rad"]), 1e-8)
        sigma = velocity_dispersion(radius, sample[11], sample[12:14], spec.support)
        return u, vc, sigma
    evaluate = jax.jit(jax.vmap(one))
    pieces = [evaluate(jnp.asarray(q[start:start + 2048])) for start in range(0, len(q), 2048)]
    return tuple(np.concatenate([np.asarray(piece[index]) for piece in pieces]) for index in range(3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("KGAS066", "KGAS007"), required=True)
    parser.add_argument("--dynesty-root", type=Path, required=True)
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
        records, samples, weights, replicate_ids, sources = [], [], [], [], []
        common_contract = None
        for replicate in (1, 2):
            root = args.dynesty_root / args.target / f"replicate-{replicate}"
            paths = {name: root / name for name in ("status.json", "result.json", "posterior_weighted.npz", "headless_exit.json")}
            for path in paths.values():
                if not path.is_file():
                    raise FileNotFoundError(path)
            status = json.loads(paths["status.json"].read_text(encoding="ascii"))
            result = json.loads(paths["result.json"].read_text(encoding="ascii"))
            exit_doc = json.loads(paths["headless_exit.json"].read_text(encoding="ascii"))
            if status.get("state") != "SUCCEEDED" or result.get("state") != "SUCCEEDED" or exit_doc.get("exit_code") != 0:
                raise RuntimeError(f"replicate {replicate} is incomplete")
            contract = result["contract"]
            if contract["code_commit"] != args.code_commit or contract["map_commit"] != args.map_commit:
                raise RuntimeError(f"replicate {replicate} provenance mismatch")
            if contract["map_sha256"] != sha256(args.map_result):
                raise RuntimeError(f"replicate {replicate} MAP hash mismatch")
            common_contract = contract if common_contract is None else common_contract
            if contract != common_contract:
                raise RuntimeError("replicate contracts differ")
            with np.load(paths["posterior_weighted.npz"], allow_pickle=False) as archive:
                q = np.asarray(archive["q_unified"], dtype=np.float64)
                weight = np.asarray(archive["weight"], dtype=np.float64)
            if q.ndim != 2 or q.shape[1] != 14 or weight.shape != (q.shape[0],):
                raise RuntimeError(f"replicate {replicate} sample shape mismatch")
            if not np.all(np.isfinite(weight)) or np.any(weight < 0) or not np.isclose(weight.sum(), 1.0, atol=1e-10):
                raise RuntimeError(f"replicate {replicate} weights are not normalized")
            records.append(result)
            samples.append(q)
            weights.append(weight)
            replicate_ids.append(np.full(q.shape[0], replicate, dtype=np.int16))
            sources.append({"replicate": replicate, **{name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in paths.items()}})
        _, spec, _, _, provenance = build_problem(args.target)
        physical_by_rep = [physical_draws(q, spec) for q in samples]
        primary = [name for name in physical_by_rep[0] if name != "flux_jy_kms"]
        replicate_summary = []
        for replicate, (physical, weight, record) in enumerate(zip(physical_by_rep, weights, records), 1):
            replicate_summary.append({
                "replicate": replicate, "weighted_ess": record["weighted_ess"],
                "log_evidence": record["log_evidence"], "log_evidence_error": record["log_evidence_error"],
                "parameters": {name: scalar_quantiles(value, weight)
                               for name, value in physical.items()},
            })
        median_comparison = {}
        for name in primary:
            first = replicate_summary[0]["parameters"][name]
            second = replicate_summary[1]["parameters"][name]
            scale = max(0.5 * ((first["q84"] - first["q16"]) + (second["q84"] - second["q16"])), 1e-10)
            median_comparison[name] = {
                "absolute_difference": float(abs(first["q50"] - second["q50"])),
                "difference_over_mean_68_width": float(abs(first["q50"] - second["q50"]) / scale),
            }
        radius = np.linspace(0.0, spec.support.outer_radius_arcsec, 256)
        profile_by_rep = [profiles(q, spec, radius) for q in samples]
        profile_comparison = {}
        for index, name in enumerate(("u_projected_kms", "v_rot_kms", "sigma_kms")):
            medians = [weighted_quantile(profile[index], weight)[1] for profile, weight in zip(profile_by_rep, weights)]
            profile_comparison[name] = {
                "relative_l2_median_difference": float(np.linalg.norm(medians[0] - medians[1]) / max(np.linalg.norm(0.5 * (medians[0] + medians[1])), 1e-12))
            }
        combined_q = np.concatenate(samples)
        combined_weight = np.concatenate([0.5 * value for value in weights])
        combined_replicate = np.concatenate(replicate_ids)
        physical = physical_draws(combined_q, spec)
        combined_profiles = tuple(np.concatenate([profile_by_rep[0][i], profile_by_rep[1][i]]) for i in range(3))
        parameter_summary = {
            name: scalar_quantiles(value, combined_weight)
            for name, value in physical.items()
        }
        profile_quantiles = {}
        for values, name in zip(combined_profiles, ("u_projected", "v_rot", "sigma")):
            quantiles = weighted_quantile(values, combined_weight)
            for index, label in enumerate(("q16", "q50", "q84")):
                profile_quantiles[f"{name}_{label}"] = quantiles[index]
        evidence_delta = abs(records[0]["log_evidence"] - records[1]["log_evidence"])
        evidence_sigma = float(np.hypot(records[0]["log_evidence_error"], records[1]["log_evidence_error"]))
        gates = {
            "replicate_weighted_ess_min": min(record["weighted_ess"] for record in records),
            "weighted_ess_threshold": 400.0,
            "log_evidence_error_max": max(record["log_evidence_error"] for record in records),
            "log_evidence_error_threshold": 0.5,
            "log_evidence_difference": evidence_delta,
            "log_evidence_stability_3sigma": evidence_delta <= 3.0 * evidence_sigma,
            "primary_median_difference_over_68_width_max": max(value["difference_over_mean_68_width"] for value in median_comparison.values()),
            "primary_median_threshold": 0.5,
            "profile_relative_l2_median_difference_max": max(value["relative_l2_median_difference"] for value in profile_comparison.values()),
            "profile_relative_l2_threshold": 0.10,
        }
        gates["accepted"] = bool(
            gates["replicate_weighted_ess_min"] >= gates["weighted_ess_threshold"]
            and gates["log_evidence_error_max"] <= gates["log_evidence_error_threshold"]
            and gates["log_evidence_stability_3sigma"]
            and gates["primary_median_difference_over_68_width_max"] <= gates["primary_median_threshold"]
            and gates["profile_relative_l2_median_difference_max"] <= gates["profile_relative_l2_threshold"]
        )
        np.savez(staging / "posterior_weighted.npz", q_unified=combined_q, weight=combined_weight,
                 replicate=combined_replicate, r50_defined=np.zeros(len(combined_q), dtype=bool), **physical)
        np.savez(staging / "profile_quantiles.npz", radius_arcsec=radius, **profile_quantiles)
        summary = {
            "schema_version": "kinuv-unified-dynesty-evidence-v1", "created_utc": utc_now(),
            "state": "ACCEPTED" if gates["accepted"] else "REJECTED_REPLICATE_DIAGNOSTICS",
            "target_id": args.target, "replicates": replicate_summary, "gates": gates,
            "replicate_parameter_median_comparison": median_comparison,
            "replicate_profile_median_comparison": profile_comparison,
            "parameter_summary": parameter_summary,
            "profile": {"quantiles_file": "profile_quantiles.npz", "support": spec.support.metadata(),
                        "R50": None, "samplewise_defined_count": 0, "R50_status": "UNRESOLVED_PLATEAU",
                        "R50_reason": "outer continuation supplies no plateau-identifiability evidence"},
            "contract": common_contract, "model_provenance": provenance,
            "sources": {"map": {"path": str(args.map_result.resolve()), "sha256": sha256(args.map_result)},
                        "replicates": sources},
        }
        (staging / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="ascii")
        files = {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(staging.iterdir())}
        manifest = {"schema_version": "kinuv-unified-dynesty-evidence-manifest-v1",
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
