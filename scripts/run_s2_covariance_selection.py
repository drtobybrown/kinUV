#!/usr/bin/env python3
"""Run fold-safe S2 C0/C1 selection on provenance-complete visibility data."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from kinuv.constants import freq_to_velocity_kms
from kinuv.io.vis import (
    cube_vopt_window_kms,
    load_visibility_table,
    optical_to_radio_kms,
    require_s2_provenance,
)
from kinuv.validation.covariance import (
    fit_covariance_by_stratum,
    whitened_innovations,
    whitening_diagnostics,
    select_grouped_covariance,
)
from kinuv.validation.groups import build_grouped_visibility_folds


REPO = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _line_free_mask(table, cube_path: Path, margin_native: int):
    velocity = freq_to_velocity_kms(table.freqs)
    optical_lo, optical_hi = cube_vopt_window_kms(cube_path)
    radio_bounds = sorted(
        [
            float(optical_to_radio_kms(optical_lo)),
            float(optical_to_radio_kms(optical_hi)),
        ]
    )
    channel_step = float(np.median(np.abs(np.diff(velocity))))
    lower = radio_bounds[0] - int(margin_native) * channel_step
    upper = radio_bounds[1] + int(margin_native) * channel_step
    return (velocity < lower) | (velocity > upper), {
        "line_window_radio_kms": [lower, upper],
        "native_channel_step_kms": channel_step,
        "margin_native_channels": int(margin_native),
    }


def _stratum_mask(table, key):
    return (
        (table.observation_id == key[0])
        & (table.array_id == key[1])
        & (table.field_id == key[2])
        & (table.data_desc_id == key[3])
    )


def _json_parameters(parameters):
    return {
        "/".join(str(value) for value in key): asdict(value)
        for key, value in parameters.items()
    }


def _run_target(config_path: Path, n_folds: int, margin_native: int):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    visibility_path = Path(config["visibility_npz"])
    cube_path = Path(config["fit_window_cube"])
    if not cube_path.is_file():
        raise FileNotFoundError(cube_path)
    table = load_visibility_table(visibility_path)
    require_s2_provenance(table)
    line_free, window = _line_free_mask(table, cube_path, margin_native)
    if int(np.sum(line_free)) < 2:
        raise ValueError("fewer than two line-free native channels remain")
    folds = build_grouped_visibility_folds(table, n_folds=n_folds)
    embargo_s = float(2.0 * np.max(table.interval))
    selection = select_grouped_covariance(
        table, folds, line_free, embargo_s=embargo_s
    )
    fits = {
        model: fit_covariance_by_stratum(table, line_free, model=model)
        for model in ("C0", "C1")
    }
    diagnostics = {}
    for key, parameters in fits[selection.selected].items():
        rows = _stratum_mask(table, key)
        innovations = whitened_innovations(
            table.vis, table.weights, line_free, rows, parameters
        )
        diagnostics["/".join(str(value) for value in key)] = (
            whitening_diagnostics(innovations)
        )
    fold_rows = {
        str(fold_id): {
            "validation_rows": int(np.sum(folds.validation_mask(fold_id))),
            "training_rows_after_embargo": int(
                np.sum(folds.training_mask(fold_id, embargo_s=embargo_s))
            ),
        }
        for fold_id in range(n_folds)
    }
    gates = {
        "fold_count": folds.n_folds >= 5,
        "all_folds_populated": all(
            item["validation_rows"] > 0 and item["training_rows_after_embargo"] > 0
            for item in fold_rows.values()
        ),
        "whitened_mean": all(item["mean_pass"] for item in diagnostics.values()),
        "whitened_variance": all(
            item["variance_pass"] for item in diagnostics.values()
        ),
        "whitened_lag_one": all(
            item["lag_one_pass"] for item in diagnostics.values()
        ),
    }
    return {
        "target_id": config["target_id"],
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "visibility_path": str(visibility_path.resolve()),
        "visibility_sha256": _sha256(visibility_path),
        "input": {
            "schema": table.schema,
            "n_rows": int(table.vis.shape[0]),
            "n_channels": int(table.vis.shape[1]),
            "n_line_free_channels": int(np.sum(line_free)),
            "n_scan_groups": len(folds.groups),
            "frequency_frame": table.frequency_frame,
            "visibility_unit": table.visibility_unit,
            "weight_convention": table.weight_convention,
            **window,
        },
        "folds": fold_rows,
        "embargo_s": embargo_s,
        "groups": [asdict(group) for group in folds.groups],
        "covariance": {
            "selected": selection.selected,
            "mean_loglike_advantage_per_complex": (
                selection.mean_loglike_advantage_per_complex
            ),
            "standard_error_per_complex": selection.standard_error_per_complex,
            "group_ids": selection.group_ids.tolist(),
            "group_advantages_per_complex": (
                selection.group_advantages_per_complex.tolist()
            ),
            "parameters": {
                model: _json_parameters(parameters)
                for model, parameters in fits.items()
            },
            "whitening": diagnostics,
        },
        "gates": gates,
        "accepted": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target_configs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--line-margin-native", type=int, default=3)
    args = parser.parse_args()
    git = _git_state()
    if git["branch"] != "dev" or git["dirty"]:
        raise RuntimeError("S2 validation requires a clean exact commit on dev")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    results = [
        _run_target(config, args.n_folds, args.line_margin_native)
        for config in args.target_configs
    ]
    payload = {
        "schema_version": "kinuv-s2-covariance-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": git,
        "targets": results,
        "accepted": all(result["accepted"] for result in results),
    }
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "kinuv-s2-covariance-manifest-v1",
        "code_commit": payload["git"]["commit"],
        "files": {
            "metrics.json": {
                "bytes": metrics_path.stat().st_size,
                "sha256": _sha256(metrics_path),
            }
        },
    }
    (args.output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if not payload["accepted"]:
        raise SystemExit("S2 covariance gates did not all pass")


if __name__ == "__main__":
    main()
