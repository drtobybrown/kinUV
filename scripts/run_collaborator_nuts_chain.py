#!/usr/bin/env python3
"""Run one checkpointed headless collaborator-delivery NUTS chain."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import pickle
import shutil
import tempfile
import threading
import time

import numpy as np

from kinuv.forward.sb import load_sb_template
from kinuv.infer.collaborator import campaign_transform
from kinuv.infer.map import image_grid_for_vis
from kinuv.infer.s3 import (
    build_positive_emissivity_basis,
    build_s3_objective,
    chart_bounds,
)
from kinuv.io.vis import load_target_vis

from run_collaborator_map import REPO, covariance_for


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_pickle(path: Path, value):
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


def copy_checkpoint(scratch: Path, durable: Path):
    durable.parent.mkdir(parents=True, exist_ok=True)
    temporary = durable.with_name(f".{durable.name}.tmp")
    shutil.copy2(scratch, temporary)
    os.replace(temporary, durable)


def setup_problem(selected_path: Path):
    selected_doc = json.loads(selected_path.read_text(encoding="utf-8"))
    selected = selected_doc["selected"]
    target_id = selected_doc["target_id"]
    config = json.loads((REPO / "configs/targets" / f"{target_id}.json").read_text(encoding="utf-8"))
    checkpoint = json.loads(Path(selected["inputs"]["checkpoint"]["path"]).read_text(encoding="utf-8"))
    phase = config.get("phase_center_deg")
    phase_rad = None if phase is None else np.radians(np.asarray(phase, dtype=float))
    data, _ = load_target_vis(config["visibility_npz"], cube_path=config["fit_window_cube"], phase_dir_rad=phase_rad)
    grid = image_grid_for_vis(data)
    base = load_sb_template(grid, Path(config["template_ico"]))
    s2 = json.loads(Path(checkpoint["s2_summary_path"]).read_text(encoding="utf-8"))
    s2_parameters = next(row for row in s2["targets"] if row["target_id"] == target_id)["best_joint"]["parameters"]
    basis = build_positive_emissivity_basis(base, grid, np.radians(s2_parameters["pa_deg"]), np.radians(s2_parameters["inclination_deg"]))
    bmaj = float(config["diagnostic_beam"]["bmaj_arcsec"])
    vsys_seed = float(config["stage_a"]["parameter_seed"]["vsys_kms"])
    candidate = selected["selected_parent"]
    uses_rings = candidate == "supported_rings" or (candidate == "two_zone_dispersion" and selected["two_zone_uses_rings"])
    fixed_weights = np.asarray(selected["emissivity_weights"], dtype=np.float64)
    initial_logits = np.log(basis.natural_weights[1:] / basis.natural_weights[0])
    objective, _ = build_s3_objective(
        data, base, basis, checkpoint["velocity_support"]["knot_radii_arcsec"], grid,
        covariance_for(target_id, data.n_bin), candidate=candidate,
        vsys_seed_kms=vsys_seed, bmaj_arcsec=bmaj,
        initial_emissivity_logits=initial_logits,
        two_zone_uses_rings=selected["two_zone_uses_rings"],
        fixed_emissivity_weights=fixed_weights,
    )
    bounds, active = chart_bounds(float(config["geometry"]["pa_seed_deg"]), data.dv_kms, bmaj, candidate, two_zone_uses_rings=selected["two_zone_uses_rings"])
    transform = campaign_transform(bounds, active, uses_rings=uses_rings)
    fixed_full = np.asarray(selected["optimum_chart"], dtype=np.float64)

    def potential(y):
        full, log_measure = transform.unconstrained_to_full_jax(y, fixed_full)
        return objective(full) - log_measure

    return selected_doc, transform, fixed_full, potential


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-map", type=Path, required=True)
    parser.add_argument("--chain-id", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--durable", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--chunk", type=int, default=100)
    parser.add_argument("--target-accept", type=float, default=0.90)
    parser.add_argument("--max-tree-depth", type=int, default=10)
    args = parser.parse_args()
    chain = f"chain-{args.chain_id}"
    scratch = args.scratch / chain
    durable = args.durable / chain
    scratch.mkdir(parents=True, exist_ok=True)
    durable.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(durable / "worker.log"), logging.StreamHandler()])
    log = logging.getLogger(chain)
    selected, transform, fixed_full, potential = setup_problem(args.selected_map)
    target = selected["target_id"]
    y0 = transform.active_initial_unconstrained(fixed_full)
    rng = np.random.default_rng(args.seed)
    y0 = y0 + 0.01 * rng.standard_normal(y0.shape)
    state_doc = {"state": "RUNNING", "target": target, "chain_id": args.chain_id, "seed": args.seed, "pid": os.getpid(), "started_utc": utc_now(), "warmup": args.warmup, "samples": args.samples, "completed_warmup": 0, "completed_draws": 0}
    atomic_json(durable / "status.json", state_doc)
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(60.0):
            log.info("heartbeat target=%s chain=%d state=%s draws=%d pid=%d", target, args.chain_id, state_doc["state"], state_doc["completed_draws"], os.getpid())
            state_doc["updated_utc"] = utc_now()
            atomic_json(durable / "status.json", state_doc)

    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        import jax
        import jax.numpy as jnp
        from numpyro.infer import NUTS

        compiled = jax.jit(potential)
        initial_energy = float(compiled(jnp.asarray(y0)))
        if not np.isfinite(initial_energy):
            raise RuntimeError("non-finite NUTS energy at initialization")
        kernel = NUTS(potential_fn=compiled, target_accept_prob=args.target_accept, max_tree_depth=args.max_tree_depth, adapt_mass_matrix=True)
        key = jax.random.PRNGKey(args.seed)
        state_doc["state"] = "WARMUP"
        atomic_json(durable / "status.json", state_doc)
        sampler_state = kernel.init(
            key,
            args.warmup,
            init_params=jnp.asarray(y0),
            model_args=(),
            model_kwargs={},
        )
        sample_once = jax.jit(lambda current: kernel.sample(current, (), {}))
        for warmup_index in range(args.warmup):
            sampler_state = sample_once(sampler_state)
            completed = warmup_index + 1
            if completed % args.chunk == 0 or completed == args.warmup:
                atomic_pickle(scratch / "sampler_state.pkl", sampler_state)
                copy_checkpoint(scratch / "sampler_state.pkl", durable / "sampler_state.pkl")
                state_doc["completed_warmup"] = completed
                state_doc["updated_utc"] = utc_now()
                atomic_json(durable / "status.json", state_doc)
        state_doc["state"] = "SAMPLING"
        draws = []
        extras = {name: [] for name in ("diverging", "num_steps", "accept_prob", "energy")}
        chunk_draws = []
        chunk_extras = {name: [] for name in extras}
        for draw_index in range(args.samples):
            sampler_state = sample_once(sampler_state)
            chunk_draws.append(np.asarray(sampler_state.z, dtype=np.float64))
            chunk_extras["diverging"].append(np.asarray(sampler_state.diverging))
            chunk_extras["num_steps"].append(np.asarray(sampler_state.num_steps))
            chunk_extras["accept_prob"].append(np.asarray(sampler_state.accept_prob))
            chunk_extras["energy"].append(np.asarray(sampler_state.energy))
            completed = draw_index + 1
            if completed % args.chunk != 0 and completed != args.samples:
                continue
            draws.append(np.stack(chunk_draws, axis=0))
            for name in extras:
                extras[name].append(np.stack(chunk_extras[name], axis=0))
            chunk_draws = []
            chunk_extras = {name: [] for name in extras}
            all_draws = np.concatenate(draws, axis=0)
            all_extras = {name: np.concatenate(parts, axis=0) for name, parts in extras.items()}
            atomic_npz(scratch / "draws.npz", unconstrained=all_draws, **all_extras)
            atomic_pickle(scratch / "sampler_state.pkl", sampler_state)
            copy_checkpoint(scratch / "draws.npz", durable / "draws.npz")
            copy_checkpoint(scratch / "sampler_state.pkl", durable / "sampler_state.pkl")
            state_doc["completed_draws"] = completed
            state_doc["updated_utc"] = utc_now()
            atomic_json(durable / "status.json", state_doc)
        state_doc.update({"state": "SUCCEEDED", "exit_code": 0, "completed_utc": utc_now()})
        atomic_json(durable / "status.json", state_doc)
        log.info("complete target=%s chain=%d draws=%d", target, args.chain_id, state_doc["completed_draws"])
        return 0
    except BaseException as error:
        state_doc.update({"state": "FAILED", "exit_code": 1, "error": f"{type(error).__name__}: {error}", "completed_utc": utc_now()})
        atomic_json(durable / "status.json", state_doc)
        log.exception("chain failed")
        return 1
    finally:
        stop.set()


if __name__ == "__main__":
    raise SystemExit(main())
