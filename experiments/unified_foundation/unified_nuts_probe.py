#!/usr/bin/env python3
"""Bounded one-chain preconditioned NUTS probe on the unified 14D posterior."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import signal
import tempfile
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np

from unified_map_runner import atomic_json, build_problem


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def posterior_covariance(value_gradient, point, relative_step=1e-2, eigen_floor=1e-6):
    point = np.asarray(point, dtype=np.float64)
    hessian = np.empty((point.size, point.size), dtype=np.float64)
    for index in range(point.size):
        step = relative_step * max(1.0, abs(float(point[index])))
        plus, minus = point.copy(), point.copy()
        plus[index] += step
        minus[index] -= step
        _, gp = value_gradient(jnp.asarray(plus))
        _, gm = value_gradient(jnp.asarray(minus))
        hessian[:, index] = (np.asarray(gp) - np.asarray(gm)) / (2.0 * step)
    hessian = 0.5 * (hessian + hessian.T)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).eps)
    material_negative = eigenvalues < -1e-3 * scale
    if np.any(material_negative):
        raise RuntimeError(
            f"MAP posterior Hessian has {int(np.sum(material_negative))} materially negative eigenvalues"
        )
    floor = scale * eigen_floor
    regularized = np.maximum(eigenvalues, floor)
    covariance = (eigenvectors * (1.0 / regularized)) @ eigenvectors.T
    return 0.5 * (covariance + covariance.T), {
        "method": "central finite difference of JIT posterior gradient",
        "gradient_calls": 2 * point.size,
        "relative_step": relative_step,
        "raw_min_eigenvalue": float(eigenvalues.min()),
        "raw_max_eigenvalue": float(eigenvalues.max()),
        "negative_eigenvalues": int(np.sum(eigenvalues < 0.0)),
        "material_negative_threshold_relative": -1e-3,
        "material_negative_eigenvalues": int(np.sum(material_negative)),
        "nonpositive_modes_floored": int(np.sum(eigenvalues <= floor)),
        "regularized_min_eigenvalue": float(regularized.min()),
        "regularized_condition": float(regularized.max() / regularized.min()),
        "metric_convention": "q=q_MAP+Lz with LLT=regularized H^-1; NumPyro inverse_mass_matrix=I in z",
    }


def run(args):
    output = args.output
    status_path = output / "status.json"
    output.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": "kinuv-unified-nuts-probe-status-v1",
        "state": "SETUP",
        "target_id": args.target,
        "session_id": os.environ.get("SKAHA_SESSION_ID", os.environ.get("skaha_sessionid", platform.node())),
        "pid": os.getpid(),
        "code_commit": args.code_commit,
        "map_result": str(args.map_result),
        "warmup": args.warmup,
        "draws": args.draws,
        "started_utc": utc_now(),
        "completed_warmup": 0,
        "completed_draws": 0,
    }
    atomic_json(status_path, state)
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(55.0):
            state["updated_utc"] = utc_now()
            atomic_json(status_path, state)

    threading.Thread(target=heartbeat, daemon=True).start()

    def terminate(signum, _frame):
        state.update({"state": "TIME_LIMIT", "signal": int(signum), "updated_utc": utc_now()})
        atomic_json(status_path, state)
        raise SystemExit(124)

    signal.signal(signal.SIGTERM, terminate)
    started = time.perf_counter()
    try:
        map_doc = json.loads(args.map_result.read_text(encoding="utf-8"))
        if map_doc.get("target_id") != args.target or not map_doc.get("gate_pass"):
            raise RuntimeError("selected unified MAP must match target and pass its MAP gate")
        if map_doc.get("git", {}).get("commit") != args.map_commit:
            raise RuntimeError("selected unified MAP commit does not match --map-commit")
        q_map = np.asarray(map_doc["optimum_z"], dtype=np.float64)
        context, spec, density, _, provenance = build_problem(args.target)
        if q_map.shape != (14,):
            raise RuntimeError(f"expected 14D unified MAP, got {q_map.shape}")
        potential = jax.jit(lambda q: -density.log_posterior(q))
        value_gradient = jax.jit(jax.value_and_grad(lambda q: -density.log_posterior(q)))
        compile_started = time.perf_counter()
        map_value, map_gradient = value_gradient(jnp.asarray(q_map))
        jax.block_until_ready(map_gradient)
        compile_s = time.perf_counter() - compile_started
        state.update({"state": "HESSIAN", "map_energy": float(map_value), "updated_utc": utc_now()})
        atomic_json(status_path, state)
        hessian_started = time.perf_counter()
        covariance, metric = posterior_covariance(value_gradient, q_map)
        hessian_s = time.perf_counter() - hessian_started
        chol = np.linalg.cholesky(covariance)
        q0, lmat = jnp.asarray(q_map), jnp.asarray(chol)
        def whitened_potential(z):
            return potential(q0 + lmat @ z)
        whitened = jax.jit(whitened_potential)

        from numpyro.infer import NUTS

        kernel = NUTS(
            potential_fn=whitened,
            inverse_mass_matrix=jnp.eye(q_map.size),
            dense_mass=True,
            adapt_mass_matrix=False,
            target_accept_prob=args.target_accept,
            max_tree_depth=(args.max_tree_depth, args.max_tree_depth),
            find_heuristic_step_size=True,
        )
        init_started = time.perf_counter()
        sampler_state = kernel.init(
            jax.random.PRNGKey(args.seed), args.warmup, init_params=jnp.zeros(q_map.size)
        )
        sample_once = jax.jit(lambda current: kernel.sample(current, (), {}))
        init_s = time.perf_counter() - init_started
        state.update({"state": "WARMUP", "metric": metric, "updated_utc": utc_now()})
        atomic_json(status_path, state)
        warmup_started = time.perf_counter()
        for index in range(args.warmup):
            sampler_state = sample_once(sampler_state)
            state.update({
                "completed_warmup": index + 1,
                "current_num_steps": int(np.asarray(sampler_state.num_steps)),
                "current_diverging": bool(np.asarray(sampler_state.diverging)),
            })
        warmup_s = time.perf_counter() - warmup_started
        state["state"] = "SAMPLING"
        atomic_json(status_path, state)
        z_draws, num_steps, divergences, accept_prob, energies, step_sizes = [], [], [], [], [], []
        sampling_started = time.perf_counter()
        for index in range(args.draws):
            sampler_state = sample_once(sampler_state)
            z_draws.append(np.asarray(sampler_state.z, dtype=np.float64))
            num_steps.append(int(np.asarray(sampler_state.num_steps)))
            divergences.append(bool(np.asarray(sampler_state.diverging)))
            accept_prob.append(float(np.asarray(sampler_state.accept_prob)))
            energies.append(float(np.asarray(sampler_state.energy)))
            step_sizes.append(float(np.asarray(sampler_state.adapt_state.step_size)))
            state.update({"completed_draws": index + 1, "current_num_steps": num_steps[-1],
                          "current_diverging": divergences[-1]})
        sampling_s = time.perf_counter() - sampling_started
        z_draws = np.asarray(z_draws)
        q_draws = q_map[None, :] + z_draws @ chol.T
        num_steps_array = np.asarray(num_steps)
        tree_depth = np.ceil(np.log2(num_steps_array + 1)).astype(int)
        saturation = num_steps_array >= 2**args.max_tree_depth - 1
        saturation_fraction = float(np.mean(saturation))
        finite_step_size = bool(np.all(np.isfinite(step_sizes)) and np.all(np.asarray(step_sizes) > 0.0))
        step_size_relative_range = float(np.ptp(step_sizes) / np.mean(step_sizes)) if finite_step_size else float("inf")
        diagnostic_gate = {
            "mean_tree_depth_le_7": float(np.mean(tree_depth)) <= 7.0,
            "zero_divergences": int(np.sum(divergences)) == 0,
            "finite_stable_step_size": finite_step_size and step_size_relative_range <= 1e-6,
        }
        diagnostic = {
            "tree_depth_cap": args.max_tree_depth,
            "tree_depth_saturation_count": int(saturation.sum()),
            "tree_depth_saturation_fraction": saturation_fraction,
            "tree_depth_trigger_fraction": 0.20,
            "mean_tree_depth": float(np.mean(tree_depth)),
            "fraction_tree_depth_gt_7": float(np.mean(tree_depth > 7)),
            "divergence_count": int(np.sum(divergences)),
            "mean_num_steps": float(np.mean(num_steps)),
            "max_num_steps": int(np.max(num_steps)),
            "final_step_size": step_sizes[-1],
            "step_size_relative_range": step_size_relative_range,
            "mean_accept_prob": float(np.mean(accept_prob)),
            "finite_energy": bool(np.all(np.isfinite(energies))),
            "single_chain_rhat": "NOT_DEFINED",
            "posterior_fidelity": "UNRESOLVED_BOUNDED_PILOT",
            "gates": diagnostic_gate,
        }
        state_name = (
            "TREE_SATURATION_TRIGGER"
            if saturation_fraction > 0.20
            else "COMPLETED_PILOT" if all(diagnostic_gate.values()) else "DIAGNOSTIC_GATE_FAILED"
        )
        result = {
            "schema_version": "kinuv-unified-nuts-probe-v1",
            "state": state_name,
            "created_utc": utc_now(), "target_id": args.target, "code_commit": args.code_commit,
            "map_commit": args.map_commit, "map_result": str(args.map_result),
            "sampler": {"name": "NumPyro NUTS", "warmup": args.warmup, "draws": args.draws,
                        "seed": args.seed, "target_accept": args.target_accept,
                        "preconditioning": "fixed MAP posterior covariance; step-size adaptation only"},
            "diagnostics": diagnostic, "metric": metric,
            "timing": {"compile_map_value_gradient_s": compile_s, "hessian_s": hessian_s,
                       "sampler_init_s": init_s, "warmup_s": warmup_s, "sampling_s": sampling_s,
                       "total_s": time.perf_counter() - started},
            "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "prior_likelihood_contract": "unchanged proper unified chart prior plus sole fixed-C1 visibility likelihood",
            "provenance": provenance,
        }
        atomic_npz(output / "draws.npz", z_whitened=z_draws, q_unified=q_draws,
                   num_steps=num_steps_array, tree_depth=tree_depth, diverging=np.asarray(divergences),
                   accept_prob=np.asarray(accept_prob), energy=np.asarray(energies),
                   step_size=np.asarray(step_sizes), covariance=covariance)
        atomic_json(output / "result.json", result)
        state.update({"state": result["state"], "diagnostics": diagnostic,
                      "completed_utc": utc_now(), "exit_code": 0})
        atomic_json(status_path, state)
        print(json.dumps({"state": result["state"], **diagnostic}, sort_keys=True), flush=True)
        return 0
    except SystemExit:
        raise
    except BaseException as error:
        state.update({"state": "FAILED", "error": f"{type(error).__name__}: {error}",
                      "completed_utc": utc_now(), "exit_code": 1})
        atomic_json(status_path, state)
        raise
    finally:
        stop.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("KGAS066", "KGAS007"), default="KGAS066")
    parser.add_argument("--map-result", type=Path, required=True)
    parser.add_argument("--map-commit", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=150)
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--max-tree-depth", type=int, default=8)
    parser.add_argument("--target-accept", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=930066)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
