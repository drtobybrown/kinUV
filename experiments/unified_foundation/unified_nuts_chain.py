#!/usr/bin/env python3
"""Run one resumable production-length chain on the unified posterior."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import shutil
import signal
import tempfile
import threading

import jax
import jax.numpy as jnp
import numpy as np

from unified_map_runner import atomic_json, build_problem
from unified_nuts_probe import posterior_covariance


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scientific_input_hashes(map_doc):
    provenance = map_doc["provenance"]
    config_path = Path(provenance["config_path"])
    checkpoint_path = Path(provenance["checkpoint_path"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    paths = {
        "target_config": config_path,
        "unified_parent_checkpoint": checkpoint_path,
        "c1_covariance": Path(provenance["covariance_path"]),
        "s2_emissivity": Path(checkpoint["s2_summary_path"]),
        "visibility": Path(config["visibility_npz"]),
        "template_ico": Path(config["template_ico"]),
        "fit_window_cube": Path(config["fit_window_cube"]),
    }
    return {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in paths.items()}


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


def atomic_pickle(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def publish(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def write_heartbeat(path, state):
    line = (
        f"utc={utc_now()} target={state['target_id']} chain={state['chain_id']} "
        f"session={state['session_id']} state={state['state']} "
        f"warmup={state['completed_warmup']}/{state['warmup']} "
        f"draws={state['completed_draws']}/{state['draws']} "
        f"num_steps={state.get('current_num_steps', 'NA')}\n"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(line, encoding="ascii")
    os.replace(temporary, path)


def bounded_start(potential, direction, maximum_delta):
    origin = np.zeros_like(direction)
    base = float(potential(jnp.asarray(origin)))
    scale = 0.05
    while scale >= 1e-8:
        candidate = scale * direction
        energy = float(potential(jnp.asarray(candidate)))
        if np.isfinite(energy) and energy - base <= maximum_delta:
            return candidate, base, energy, scale
        scale *= 0.5
    return origin, base, base, 0.0


def run(args):
    scratch = args.scratch / f"chain-{args.chain_id}"
    durable = args.durable / args.target / f"chain-{args.chain_id}"
    scratch.mkdir(parents=True, exist_ok=True)
    durable.mkdir(parents=True, exist_ok=True)
    status_path = durable / "status.json"
    heartbeat_path = durable / "heartbeat.txt"
    map_doc = json.loads(args.map_result.read_text(encoding="utf-8"))
    if map_doc.get("target_id") != args.target or not map_doc.get("gate_pass"):
        raise RuntimeError("MAP result must match target and pass its gate")
    if map_doc.get("git", {}).get("commit") != args.map_commit:
        raise RuntimeError("MAP result commit mismatch")
    q_map = np.asarray(map_doc["optimum_z"], dtype=np.float64)
    if q_map.shape != (14,):
        raise RuntimeError("unified MAP must contain 14 coordinates")
    contract = {
        "code_commit": args.code_commit,
        "map_commit": args.map_commit,
        "map_sha256": sha256(args.map_result),
        "scientific_inputs": scientific_input_hashes(map_doc),
        "warmup": args.warmup,
        "draws": args.draws,
        "chunk": args.chunk,
        "max_tree_depth": args.max_tree_depth,
        "target_accept": args.target_accept,
        "fixed_map_hessian_whitening": True,
        "adapt_mass_matrix": False,
    }
    prior = json.loads(status_path.read_text(encoding="ascii")) if status_path.is_file() else None
    compatible = bool(
        prior
        and prior.get("target_id") == args.target
        and prior.get("chain_id") == args.chain_id
        and prior.get("seed") == args.seed
        and prior.get("contract") == contract
        and (durable / "sampler_state.pkl").is_file()
    )
    if prior and prior.get("state") == "SUCCEEDED":
        return 0
    if prior and not compatible:
        raise RuntimeError("existing incomplete chain is not resume compatible")
    state = {
        "schema_version": "kinuv-unified-nuts-chain-status-v1",
        "state": "SETUP",
        "target_id": args.target,
        "chain_id": args.chain_id,
        "seed": args.seed,
        "session_id": os.environ.get("SKAHA_SESSION_ID", os.environ.get("skaha_sessionid", platform.node())),
        "pid": os.getpid(),
        "warmup": args.warmup,
        "draws": args.draws,
        "completed_warmup": int(prior.get("completed_warmup", 0)) if compatible else 0,
        "completed_draws": int(prior.get("completed_draws", 0)) if compatible else 0,
        "started_utc": prior.get("started_utc", utc_now()) if compatible else utc_now(),
        "resumed": compatible,
        "contract": contract,
    }
    atomic_json(status_path, state)
    write_heartbeat(heartbeat_path, state)
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(55):
            state["updated_utc"] = utc_now()
            atomic_json(status_path, state)
            write_heartbeat(heartbeat_path, state)

    threading.Thread(target=heartbeat, daemon=True).start()

    def terminate(signum, _frame):
        state.update({"state": "INTERRUPTED_RESUMABLE", "signal": int(signum), "updated_utc": utc_now()})
        atomic_json(status_path, state)
        write_heartbeat(heartbeat_path, state)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        context, spec, density, _, provenance = build_problem(args.target)
        potential_q = jax.jit(lambda q: -density.log_posterior(q))
        value_gradient = jax.jit(jax.value_and_grad(lambda q: -density.log_posterior(q)))
        value, gradient = value_gradient(jnp.asarray(q_map))
        jax.block_until_ready(gradient)
        state.update({"state": "HESSIAN", "map_energy": float(value)})
        atomic_json(status_path, state)
        covariance, metric = posterior_covariance(value_gradient, q_map)
        chol = np.linalg.cholesky(covariance)
        q0, lmat = jnp.asarray(q_map), jnp.asarray(chol)
        potential = jax.jit(lambda z: potential_q(q0 + lmat @ z))
        rng = np.random.default_rng(args.seed)
        initial, map_energy, initial_energy, jitter_scale = bounded_start(
            potential, rng.standard_normal(14), args.max_initial_energy_delta
        )
        from numpyro.infer import NUTS
        kernel = NUTS(
            potential_fn=potential,
            inverse_mass_matrix=jnp.eye(14),
            dense_mass=True,
            adapt_mass_matrix=False,
            target_accept_prob=args.target_accept,
            max_tree_depth=(args.max_tree_depth, args.max_tree_depth),
            find_heuristic_step_size=True,
        )
        sample_once = jax.jit(lambda current: kernel.sample(current, (), {}))
        if compatible:
            with (durable / "sampler_state.pkl").open("rb") as stream:
                sampler_state = pickle.load(stream)
        else:
            sampler_state = kernel.init(
                jax.random.PRNGKey(args.seed), args.warmup, init_params=jnp.asarray(initial)
            )
        state.update({
            "state": "WARMUP", "metric": metric, "map_energy": map_energy,
            "initial_energy": initial_energy, "initial_jitter_scale": jitter_scale,
            "provenance": provenance,
        })
        atomic_json(status_path, state)
        for index in range(state["completed_warmup"], args.warmup):
            sampler_state = sample_once(sampler_state)
            completed = index + 1
            state["current_num_steps"] = int(np.asarray(sampler_state.num_steps))
            if completed % args.chunk == 0 or completed == args.warmup:
                atomic_pickle(scratch / "sampler_state.pkl", sampler_state)
                publish(scratch / "sampler_state.pkl", durable / "sampler_state.pkl")
                state.update({"completed_warmup": completed, "updated_utc": utc_now()})
                atomic_json(status_path, state)
                write_heartbeat(heartbeat_path, state)
        state["state"] = "SAMPLING"
        atomic_json(status_path, state)
        arrays = {name: [] for name in ("q_unified", "z_whitened", "num_steps", "diverging", "accept_prob", "energy", "step_size")}
        if compatible and state["completed_draws"]:
            with np.load(durable / "draws.npz", allow_pickle=False) as archive:
                for name in arrays:
                    arrays[name].append(np.asarray(archive[name]))
        pending = {name: [] for name in arrays}
        for index in range(state["completed_draws"], args.draws):
            sampler_state = sample_once(sampler_state)
            z = np.asarray(sampler_state.z, dtype=np.float64)
            pending["z_whitened"].append(z)
            pending["q_unified"].append(q_map + chol @ z)
            pending["num_steps"].append(int(np.asarray(sampler_state.num_steps)))
            pending["diverging"].append(bool(np.asarray(sampler_state.diverging)))
            pending["accept_prob"].append(float(np.asarray(sampler_state.accept_prob)))
            pending["energy"].append(float(np.asarray(sampler_state.energy)))
            pending["step_size"].append(float(np.asarray(sampler_state.adapt_state.step_size)))
            completed = index + 1
            state["current_num_steps"] = pending["num_steps"][-1]
            if completed % args.chunk != 0 and completed != args.draws:
                continue
            for name in arrays:
                arrays[name].append(np.asarray(pending[name]))
            pending = {name: [] for name in arrays}
            merged = {name: np.concatenate(parts, axis=0) for name, parts in arrays.items()}
            atomic_npz(scratch / "draws.npz", **merged, covariance=covariance)
            atomic_pickle(scratch / "sampler_state.pkl", sampler_state)
            publish(scratch / "draws.npz", durable / "draws.npz")
            publish(scratch / "sampler_state.pkl", durable / "sampler_state.pkl")
            state.update({"completed_draws": completed, "updated_utc": utc_now()})
            atomic_json(status_path, state)
            write_heartbeat(heartbeat_path, state)
        state.update({"state": "SUCCEEDED", "exit_code": 0, "completed_utc": utc_now()})
        atomic_json(status_path, state)
        write_heartbeat(heartbeat_path, state)
        return 0
    except SystemExit:
        raise
    except BaseException as error:
        state.update({"state": "FAILED_RESUMABLE", "error": f"{type(error).__name__}: {error}",
                      "exit_code": 1, "completed_utc": utc_now()})
        atomic_json(status_path, state)
        write_heartbeat(heartbeat_path, state)
        raise
    finally:
        stop.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("KGAS066", "KGAS007"), required=True)
    parser.add_argument("--chain-id", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--map-result", type=Path, required=True)
    parser.add_argument("--map-commit", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--durable", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--chunk", type=int, default=25)
    parser.add_argument("--max-tree-depth", type=int, default=8)
    parser.add_argument("--target-accept", type=float, default=0.90)
    parser.add_argument("--max-initial-energy-delta", type=float, default=10.0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
