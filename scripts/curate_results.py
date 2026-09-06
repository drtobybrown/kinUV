#!/usr/bin/env python3
"""Curate the 2026-09-06 legacy kinUV result trees into one indexed layout.

This is intentionally a one-shot migration.  It refuses to run when the new
``results/production`` or ``results/archive`` trees already exist.  Production
bundles are built and archives are byte-verified before any source is removed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tarfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


WORKSPACE = Path(__file__).resolve().parents[2]
REPO = WORKSPACE / "kinUV"
RESULTS = WORKSPACE / "results"
RUNS = WORKSPACE / "kinuv_runs"
STAGE = RESULTS / ".curation_staging"
PRODUCTION = STAGE / "production"
ARCHIVE = STAGE / "archive"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_bytes(path: Path) -> int:
    if path.is_symlink():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


def tree_bytes(path: Path) -> int:
    return file_bytes(path) if path.exists() else 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy(path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)


def remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def archive_source(source: Path, destination: Path, arcname: str) -> dict:
    """Create and byte-verify one gzip archive without removing its source."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz", compresslevel=6) as tf:
        tf.add(source, arcname=arcname, recursive=True)

    expected = {}
    if source.is_file():
        expected[arcname] = sha256(source)
    else:
        for item in source.rglob("*"):
            if item.is_file() and not item.is_symlink():
                expected[f"{arcname}/{item.relative_to(source)}"] = sha256(item)

    observed = {}
    with tarfile.open(destination, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            extracted = tf.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"cannot read archived member {member.name}")
            digest = hashlib.sha256()
            for block in iter(lambda: extracted.read(1024 * 1024), b""):
                digest.update(block)
            observed[member.name] = digest.hexdigest()
    if observed != expected:
        raise RuntimeError(f"archive verification failed for {source}")
    return {
        "source": str(source.relative_to(WORKSPACE)),
        "archive": str(destination.relative_to(STAGE)),
        "source_bytes": file_bytes(source),
        "archive_bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "file_count": len(expected),
    }


def yaml_scalar(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return json.dumps(str(value))


def write_config(path: Path, *, target: str, run_type: str, commit: str | None,
                 model: str, source: str, record: dict) -> None:
    keys = (
        "flux", "pa_deg", "vsys_kms", "gas_sigma_kms", "dx_arcsec",
        "dy_arcsec", "v0_kms", "r_t_arcsec", "i_deg_frozen",
    )
    lines = [
        "schema_version: kinuv-curated-config-v1",
        f"target_id: {target}",
        f"run_type: {run_type}",
        f"model: {model}",
        f"code_commit: {yaml_scalar(commit)}",
        "provenance:",
        f"  source: {yaml_scalar(source)}",
        "  reconstructed_from_legacy_record: true",
        "  reconstructed_at: 2026-09-06",
        "parameters:",
    ]
    for key in keys:
        if key in record:
            lines.append(f"  {key}: {yaml_scalar(record[key])}")
    lines.extend(
        [
            "data_contract:",
            f"  n_row: {yaml_scalar(record.get('n_row'))}",
            f"  n_chan: {yaml_scalar(record.get('n_chan'))}",
            f"  channel_bin: {yaml_scalar(record.get('n_bin'))}",
            f"  channel_width_kms: {yaml_scalar(record.get('dv_kms'))}",
            f"  weight_scale: {yaml_scalar(record.get('s'))}",
            "limitations:",
            "  - Original runs predated the immutable configuration schema.",
            "  - This file records recovered provenance and is not represented as the original launch configuration.",
        ]
    )
    if run_type == "nuts":
        lines.extend(
            [
                "sampler:",
                "  algorithm: nuts",
                "  chains: 4",
                "  draws_per_chain: 600",
                "  intervals_calibrated: false",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parameter_rows_from_map(record: dict) -> list[dict]:
    units = {
        "flux": "legacy flux unit", "pa_deg": "deg", "vsys_kms": "km/s",
        "gas_sigma_kms": "km/s", "dx_arcsec": "arcsec",
        "dy_arcsec": "arcsec", "v0_kms": "km/s", "r_t_arcsec": "arcsec",
    }
    return [
        {"name": name, "estimate": float(record[name]), "unit": units[name],
         "uncertainty": None}
        for name in units if name in record
    ]


def parameter_rows_from_draws(record: dict) -> list[dict]:
    draws = np.asarray(record["draws"], dtype=np.float64)
    flat = draws.reshape(-1, draws.shape[-1])
    quantiles = np.quantile(flat, [0.16, 0.5, 0.84], axis=0)
    units = {
        "flux": "legacy flux unit", "pa_deg": "deg", "vsys_kms": "km/s",
        "gas_sigma_kms": "km/s", "dx_arcsec": "arcsec",
        "dy_arcsec": "arcsec", "v0_kms": "km/s", "r_t_arcsec": "arcsec",
    }
    rows = []
    for index, name in enumerate(record["param_names"]):
        rows.append(
            {
                "name": name,
                "q16": float(quantiles[0, index]),
                "median": float(quantiles[1, index]),
                "q84": float(quantiles[2, index]),
                "unit": units.get(name),
                "interval_calibrated": False,
            }
        )
    return rows


def finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None


def write_metrics(path: Path, summary: dict) -> None:
    lines = [
        "# Run metrics",
        "",
        f"- **Target:** {summary['target_id']}",
        f"- **Run:** `{summary['run_id']}`",
        f"- **Type:** {summary['run_type']}",
        f"- **Acceptance:** {summary['acceptance']['status']}",
        f"- **Publication ready:** {str(summary['acceptance']['publication_ready']).lower()}",
        "",
        "| Parameter | Estimate/median | q16 | q84 | Unit |",
        "|---|---:|---:|---:|---|",
    ]
    for row in summary["parameters"]:
        estimate = row.get("median", row.get("estimate"))
        q16 = row.get("q16")
        q84 = row.get("q84")
        lines.append(
            f"| {row['name']} | {estimate:.8g} | "
            f"{'' if q16 is None else f'{q16:.8g}'} | "
            f"{'' if q84 is None else f'{q84:.8g}'} | {row.get('unit') or ''} |"
        )
    metrics = summary["metrics"]
    lines.extend(["", "## Fit and convergence", ""])
    for key in ("chi2", "chi2_zero", "delta_chi2_vs_zero", "max_rhat", "min_bulk_ess", "min_tail_ess"):
        lines.append(f"- **{key}:** {metrics.get(key)}")
    if summary["limitations"]:
        lines.extend(["", "## Limitations", ""])
        lines.extend(f"- {item}" for item in summary["limitations"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rotation_curve(path: Path, rows: list[dict], *, posterior: dict | None = None) -> None:
    values = {row["name"]: row.get("median", row.get("estimate")) for row in rows}
    radius = np.linspace(0.0, 8.0, 300)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    if posterior is None:
        velocity = (2.0 / np.pi) * values["v0_kms"] * np.arctan(radius / values["r_t_arcsec"])
        ax.plot(radius, velocity, color="#155d8b", lw=2.0)
    else:
        draws = np.asarray(posterior["draws"], dtype=np.float64).reshape(-1, 8)
        v0 = draws[:, posterior["param_names"].index("v0_kms")]
        rt = draws[:, posterior["param_names"].index("r_t_arcsec")]
        subset = np.linspace(0, len(draws) - 1, min(800, len(draws)), dtype=int)
        curves = (2.0 / np.pi) * v0[subset, None] * np.arctan(radius[None, :] / rt[subset, None])
        q16, q50, q84 = np.quantile(curves, [0.16, 0.5, 0.84], axis=0)
        ax.fill_between(radius, q16, q84, color="#8ecae6", alpha=0.55,
                        label="q16–q84 (uncalibrated)")
        ax.plot(radius, q50, color="#155d8b", lw=2.0, label="posterior median")
        ax.legend(frameon=False)
    ax.set(xlabel="Radius (arcsec)", ylabel="Circular velocity (km/s)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def checksums(directory: Path) -> None:
    rows = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "CHECKSUMS.sha256":
            rows.append(f"{sha256(path)}  {path.relative_to(directory)}")
    (directory / "CHECKSUMS.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def copy_diagnostic_set(source: Path, destination: Path) -> None:
    for name in ("moments.png", "spectra.png", "pv_major.png", "pv_minor.png", "leftover_chi2.png"):
        path = source / name
        if path.is_file():
            target = "residual_visibility_chi2.png" if name == "leftover_chi2.png" else name
            copy(path, destination / target)


def build_map_product(target: str, source: Path, run_id: str, commit: str | None,
                      diagnostic_source: Path | None = None) -> dict:
    destination = PRODUCTION / target / run_id
    destination.mkdir(parents=True)
    plots = destination / "plots"
    plots.mkdir()
    map_record = read_json(source / "stage_a_map.json")
    copy(source / "stage_a_map.json", destination / "map_result.json")
    (destination / "stage_a_map.json").symlink_to("map_result.json")
    for name in ("stage_b_map.json", "inventory.json", "leftover_chi2.json", "leftover_chi2.npz"):
        if (source / name).is_file():
            copy(source / name, destination / name)
    for name in ("stage_a_model_cube.fits", "stage_b_model_cube.fits", "stage_b_model_on_10kms.fits"):
        if (source / name).is_file():
            copy(source / name, plots / name)
            (destination / name).symlink_to(Path("plots") / name)
    if (source / "leftover_chi2.png").is_file():
        copy(source / "leftover_chi2.png", plots / "residual_visibility_chi2.png")
    if diagnostic_source is not None:
        copy_diagnostic_set(diagnostic_source, plots)

    parameters = parameter_rows_from_map(map_record)
    summary = {
        "schema_version": "kinuv-production-summary-v1",
        "target_id": target,
        "run_id": run_id,
        "run_type": "MAP",
        "model": "arctangent Stage A" + (" with Stage B rings" if (source / "stage_b_map.json").is_file() else ""),
        "acceptance": {
            "status": "ACCEPTED_LEGACY_PRODUCT",
            "publication_ready": target == "KGAS066",
            "basis": "docs/PRODUCTION_RECORD.md current products",
        },
        "parameters": parameters,
        "metrics": {
            "chi2": float(map_record["chi2_map"]),
            "chi2_zero": float(map_record["chi2_zero"]),
            "delta_chi2_vs_zero": float(map_record["delta_chi2"]),
            "max_rhat": None,
            "min_bulk_ess": None,
            "min_tail_ess": None,
        },
        "gates": {
            "optimizer_converged": bool(map_record.get("success")),
            "null_comparison": "PASS",
            "posterior_convergence": "NOT_APPLICABLE",
            "interval_calibration": "NOT_APPLICABLE",
        },
        "limitations": (
            ["Turnover radius reached its configured lower bound; do not quote an inner slope."]
            if target == "KGAS066" else
            ["Registered as diagnostic_only in the original record.",
             "Used as NUTS initialization; full cross-target mock recovery was waived.",
             "Turnover radius reached its configured lower bound; do not quote an inner slope."]
        ),
        "source_paths": [str(source.relative_to(WORKSPACE))],
        "curated_at": "2026-09-06",
    }
    write_json(destination / "summary.json", summary)
    write_metrics(destination / "METRICS.md", summary)
    write_config(destination / "config.yaml", target=target, run_type="map", commit=commit,
                 model="arctangent-stage-a", source=str(source.relative_to(WORKSPACE)),
                 record=map_record)
    rotation_curve(plots / "rotation_curve.png", parameters)
    available = sorted(p.name for p in plots.iterdir() if p.is_file())
    (plots / "README.md").write_text(
        "# Plot inventory\n\n" + "\n".join(f"- `{name}`" for name in available) +
        "\n\nMissing plots are not implied to have passed. See `../summary.json`.\n",
        encoding="utf-8",
    )
    checksums(destination)
    return {"target": target, "run_id": run_id, "path": str(destination.relative_to(STAGE)),
            "type": "MAP", "summary": summary}


def build_nuts_product(target: str, source_record: Path, run_id: str,
                       commit: str, source_paths: list[str], plots_source: Path,
                       provenance: list[tuple[Path, str]], chi2: float | None,
                       chi2_zero: float | None, limitations: list[str]) -> dict:
    destination = PRODUCTION / target / run_id
    destination.mkdir(parents=True)
    plots = destination / "plots"
    plots.mkdir()
    record = read_json(source_record)
    copy(source_record, destination / "posterior_samples.json")
    for source_path, relative in provenance:
        if source_path.is_file():
            copy(source_path, destination / "provenance" / relative)
    copy_diagnostic_set(plots_source, plots)
    if (plots_source / "corner.png").is_file():
        copy(plots_source / "corner.png", plots / "corner.png")
    for old, new in (
        ("stage_a_nuts_mean.fits", "model_cube.fits"),
        ("model_on_10kms.fits", "model_on_10kms.fits"),
        ("leftover_chi2.npz", "residual_visibility_chi2.npz"),
        ("nuts_mean_params.json", "posterior_mean_parameters.json"),
    ):
        if (plots_source / old).is_file():
            copy(plots_source / old, plots / new)

    parameters = parameter_rows_from_draws(record)
    mixing = record["mixing"]
    max_rhat = max(float(value["rhat"]) for value in mixing.values())
    min_ess = min(float(value["ess"]) for value in mixing.values())
    min_tail = min(float(value["ess_tail"]) for value in mixing.values())
    delta = None if chi2 is None or chi2_zero is None else chi2_zero - chi2
    summary = {
        "schema_version": "kinuv-production-summary-v1",
        "target_id": target,
        "run_id": run_id,
        "run_type": "NUTS",
        "model": "arctangent Stage A posterior with astrometric offsets fixed at MAP",
        "acceptance": {
            "status": "ACCEPTED_COMPUTATIONAL_POSTERIOR",
            "publication_ready": False,
            "basis": "docs/PRODUCTION_RECORD.md current products and mixing checks",
        },
        "parameters": parameters,
        "metrics": {
            "chi2": chi2,
            "chi2_zero": chi2_zero,
            "delta_chi2_vs_zero": delta,
            "max_rhat": max_rhat,
            "min_bulk_ess": min_ess,
            "min_tail_ess": min_tail,
            "chains": int(np.asarray(record["draws"]).shape[0]),
            "draws_per_chain": int(np.asarray(record["draws"]).shape[1]),
        },
        "gates": {
            "posterior_finite": bool(np.isfinite(np.asarray(record["draws"])).all()),
            "mixing": "PASS" if record.get("mixing_pass") else "FAIL",
            "interval_calibration": "FAIL",
        },
        "limitations": limitations,
        "source_paths": source_paths,
        "curated_at": "2026-09-06",
    }
    write_json(destination / "summary.json", summary)
    write_metrics(destination / "METRICS.md", summary)
    config_record = {
        "pa_deg": record.get("pa_init_deg"),
        "dx_arcsec": record.get("dx_arcsec"),
        "dy_arcsec": record.get("dy_arcsec"),
    }
    write_config(destination / "config.yaml", target=target, run_type="nuts", commit=commit,
                 model="arctangent-stage-a-nuts", source=", ".join(source_paths),
                 record=config_record)
    rotation_curve(plots / "rotation_curve.png", parameters, posterior=record)
    available = sorted(p.name for p in plots.iterdir() if p.is_file())
    (plots / "README.md").write_text(
        "# Plot inventory\n\n" + "\n".join(f"- `{name}`" for name in available) +
        "\n\nPosterior q16–q84 bands are uncalibrated. Missing diagnostics are listed in `../summary.json`.\n",
        encoding="utf-8",
    )
    checksums(destination)
    return {"target": target, "run_id": run_id, "path": str(destination.relative_to(STAGE)),
            "type": "NUTS", "summary": summary}


def verify_merged_shards(record_path: Path, shard_paths: list[Path]) -> None:
    from kinuv.infer.nuts import physical_sampled_from_z6

    record = read_json(record_path)
    merged = np.asarray(record["draws"], dtype=np.float64)
    z6 = []
    for path in shard_paths:
        with np.load(path, allow_pickle=False) as data:
            z6.append(np.asarray(data["z6"], dtype=np.float64))
    rebuilt = physical_sampled_from_z6(
        np.stack(z6), float(record["dx_arcsec"]), float(record["dy_arcsec"])
    )
    if not np.array_equal(merged, rebuilt, equal_nan=True):
        raise RuntimeError(f"merged posterior does not reproduce shards: {record_path}")


def render_manifest(production: list[dict], archives: list[dict], removed: list[dict],
                    initial_bytes: int) -> str:
    archive_bytes = sum(item["archive_bytes"] for item in archives)
    archived_source = sum(item["source_bytes"] for item in archives)
    purged_bytes = sum(
        item["bytes"] for item in removed
        if item["classification"] in {"EPHEMERAL_TRASH", "REDUNDANT_AFTER_MERGE"}
    )
    lines = [
        "# kinUV result manifest",
        "",
        "Curated 2026-09-06. `results/production/` is the only accepted-product tree. "
        "Accepted computational output does not remove the scientific limitations recorded below.",
        "",
        "## Production products",
        "",
        "| Target | Run | Type | Gate status | Headline result |",
        "|---|---|---|---|---|",
    ]
    for item in production:
        summary = item["summary"]
        metrics = summary["metrics"]
        if item["type"] == "MAP":
            headline = f"chi2={metrics['chi2']:.3f}; delta chi2={metrics['delta_chi2_vs_zero']:.3f}"
        else:
            headline = (
                f"max Rhat={metrics['max_rhat']:.5f}; min ESS={metrics['min_bulk_ess']:.0f}; "
                "intervals uncalibrated"
            )
        status = summary["acceptance"]["status"]
        lines.append(
            f"| {item['target']} | [`{item['run_id']}`]({item['path'].replace('production/', 'production/', 1)}/) "
            f"| {item['type']} | {status} | {headline} |"
        )
    lines.extend(
        [
            "",
            "NUTS mixing passed for both targets, but simulation-based interval calibration did not. "
            "Their q16/q50/q84 values are retained as uncalibrated posterior quantiles and must not be reported as calibrated 1-sigma measurements. KGAS007 also lacks a completed posterior residual/imaging suite.",
            "",
            "## Archived intermediate runs",
            "",
            "| Original path | Reason | Archive | Original size | Archive size | SHA-256 |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for item in archives:
        lines.append(
            f"| `{item['source']}` | {item['reason']} | [`{Path(item['archive']).name}`]({item['archive']}) | "
            f"{item['source_bytes']} B | {item['archive_bytes']} B | `{item['sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## Removed legacy paths",
            "",
            "| Original path | Classification | Size | Reason |",
            "|---|---|---:|---|",
        ]
    )
    for item in removed:
        lines.append(
            f"| `{item['path']}` | {item['classification']} | "
            f"{item['bytes']} B | {item['reason']} |"
        )
    lines.extend(
        [
            "",
            "## Storage accounting",
            "",
            f"- Original logical size of `results/` plus `kinuv_runs/`: **{initial_bytes} bytes**.",
            f"- Intermediate material compressed: **{archived_source} bytes → {archive_bytes} bytes**.",
            f"- Ephemeral/redundant material purged after verification: **{purged_bytes} bytes**.",
            "- Final on-disk size and net reclaimed space are recorded in `CURATION.json` after migration.",
            "",
            "## Reading rules",
            "",
            "1. Start with this manifest, then the selected run's `summary.json` and `METRICS.md`.",
            "2. `config.yaml` is reconstructed provenance because these legacy runs predate the frozen configuration schema.",
            "3. Validate files with the run-local `CHECKSUMS.sha256` before external transfer.",
            "4. Archives are historical evidence and are never selected by a production runner.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    if (RESULTS / "production").exists() or (RESULTS / "archive").exists() or STAGE.exists():
        raise SystemExit("refusing: curated output or staging tree already exists")
    initial_bytes = tree_bytes(RESULTS) + tree_bytes(RUNS)
    PRODUCTION.mkdir(parents=True)
    ARCHIVE.mkdir(parents=True)

    k66_map_source = RESULTS / "KILOGAS066" / "kinuv-KGAS066-uvsign-map"
    k07_map_source = RESULTS / "KILOGAS007" / "kinuv-KGAS007-stage-a-map"
    k66_nuts_source = RUNS / "KGAS066-20260831T194009Z-nuts"
    k66_posterior = k66_nuts_source / "posteriors" / "kgas066_nuts.json"
    k07_artifact = REPO / "docs/reviews/artifacts/2026-09-05-kgas007-nuts"
    k07_posterior = k07_artifact / "kgas007_nuts.json"

    k66_shards = [k66_nuts_source / "checkpoints" / f"chain_{n}.npz" for n in range(1, 5)]
    k07_shard_dirs = [
        RUNS / "KGAS007-20260905T141720Z-nuts-kgas007-c1",
        RUNS / "KGAS007-20260906T003036Z-nuts-kgas007-c2",
        RUNS / "KGAS007-20260905T141754Z-nuts-kgas007-c3",
        RUNS / "KGAS007-20260905T141801Z-nuts-kgas007-c4",
    ]
    k07_shards = [directory / "checkpoints" / f"chain_{index}.npz"
                  for directory, index in zip(k07_shard_dirs, (1, 2, 3, 4), strict=True)]
    verify_merged_shards(k66_posterior, k66_shards)
    verify_merged_shards(k07_posterior, k07_shards)

    production = []
    production.append(
        build_map_product(
            "KGAS066", k66_map_source, "kinuv-KGAS066-uvsign-map", None,
            REPO / "docs/reviews/artifacts/2026-09-02-kgas066-leftover-and-modes/stage-a-map",
        )
    )
    production.append(
        build_nuts_product(
            "KGAS066", k66_posterior, "kinuv-KGAS066-3de838-nuts", "3de838",
            [str(k66_nuts_source.relative_to(WORKSPACE))], k66_nuts_source / "plots",
            [
                (k66_nuts_source / "manifest.json", "original_manifest.json"),
                (k66_nuts_source / "status.json", "original_status.json"),
                (k66_nuts_source / "posteriors/summary.json", "original_summary.json"),
            ],
            167486.7639374534, 204228.2478024876,
            [
                "Posterior intervals failed the earlier simulation-based calibration and are not calibrated 1-sigma measurements.",
                "Residual chi-square remains structured with velocity; surface-brightness mismatch is indicated.",
                "Do not quote an inner rotation-curve slope.",
            ],
        )
    )
    production.append(
        build_map_product(
            "KGAS007", k07_map_source, "kinuv-KGAS007-stage-a-map", "680b45", None,
        )
    )
    k07_provenance = [
        (k07_artifact / name, name)
        for name in ("summary.json", "identity_chi2.json", "dispatch.json", "wall.json", "nuts_mean_params.json")
    ]
    for directory, index in zip(k07_shard_dirs, (1, 2, 3, 4), strict=True):
        prefix = f"shards/chain-{index}"
        k07_provenance.extend(
            [
                (directory / "manifest.json", f"{prefix}/manifest.json"),
                (directory / "status.json", f"{prefix}/status.json"),
                (directory / "logs" / f"chain_{index}.json", f"{prefix}/chain.json"),
                (directory / "identity_chi2.json", f"{prefix}/identity_chi2.json"),
            ]
        )
    production.append(
        build_nuts_product(
            "KGAS007", k07_posterior, "kinuv-KGAS007-32cbbd-nuts", "32cbbd",
            [str(path.relative_to(WORKSPACE)) for path in k07_shard_dirs] +
            [str(k07_artifact.relative_to(WORKSPACE))],
            k07_artifact, k07_provenance, None, 128282.39213001104,
            [
                "Posterior intervals are not calibrated 1-sigma measurements.",
                "The original campaign waived target-specific mock recovery.",
                "Posterior-mean residual and image-domain diagnostic suites were not evaluated.",
                "Do not quote an inner rotation-curve slope.",
            ],
        )
    )

    archive_specs = [
        (RESULTS / "KILOGAS066/kinuv-KGAS066-f47bc9-map", "KGAS066/20260819_kinuv-KGAS066-f47bc9-map.tar.gz", "Superseded pre-sign MAP and duplicate model cubes."),
        (RESULTS / "KILOGAS066/kinuv-KGAS066-f47bc9-lambda", "KGAS066/20260820_kinuv-KGAS066-f47bc9-lambda.tar.gz", "Completed Gate-4 regularization exploration; not the accepted product."),
        (RESULTS / "KILOGAS066/kinuv-KGAS066-f47bc9-lambda-resid", "KGAS066/20260821_kinuv-KGAS066-f47bc9-lambda-resid.tar.gz", "Residual-metric exploration retained for historical diagnosis."),
        (RESULTS / "KILOGAS066/kinuv-KGAS066-pa201-map", "KGAS066/20260828_kinuv-KGAS066-pa201-map.tar.gz", "Position-angle probe superseded by the official MAP."),
        (RUNS / "KGAS066-20260902T085027Z-nuts-pa25", "KGAS066/20260902_kinuv-KGAS066-a00657-nuts-pa25.tar.gz", "Incomplete approaching-mode NUTS experiment with failed mixing."),
        (RUNS / "KGAS066-20260902T170918Z-nuts-pa25-c1", "KGAS066/20260902_kinuv-KGAS066-628e4b-nuts-pa25-c1.tar.gz", "Approaching-mode chain retained as failed-mode evidence."),
        (RUNS / "KGAS066-20260902T170918Z-nuts-pa25-c2", "KGAS066/20260902_kinuv-KGAS066-628e4b-nuts-pa25-c2.tar.gz", "Approaching-mode chain retained as failed-mode evidence."),
        (RUNS / "KGAS066-20260902T170918Z-nuts-pa25-c3", "KGAS066/20260902_kinuv-KGAS066-628e4b-nuts-pa25-c3.tar.gz", "Approaching-mode chain retained as failed-mode evidence."),
        (RUNS / "KGAS066-20260902T170918Z-nuts-pa25-c4", "KGAS066/20260902_kinuv-KGAS066-628e4b-nuts-pa25-c4.tar.gz", "Approaching-mode chain retained as failed-mode evidence."),
        (RUNS / "KGAS066-20260902T170918Z-pa25-parallel", "KGAS066/20260902_kinuv-KGAS066-628e4b-pa25-coordinator.tar.gz", "Coordinator logs for the rejected approaching-mode campaign."),
        (RUNS / "KGAS066-20260903T211338Z-map-pa25", "KGAS066/20260903_kinuv-KGAS066-b4f5f9-map-pa25.tar.gz", "Recovery MAP proving return to the accepted receding solution."),
        (RESULTS / "KILOGAS007/bestfit_cube.fits", "KGAS007/20260905_orphan-bestfit-cube.tar.gz", "Orphan cube with no kinUV run manifest and incompatible placeholder WCS."),
    ]
    archives = []
    for source, relative, reason in archive_specs:
        item = archive_source(source, ARCHIVE / relative, str(source.relative_to(WORKSPACE)))
        item["classification"] = "INTERMEDIATE_ARCHIVE"
        item["reason"] = reason
        archives.append(item)

    purge_specs = [
        (RESULTS / "KILOGAS007/kinuv-KGAS007-nuts", "REDUNDANT_AFTER_MERGE", "Redundant merge marker; records incorporated into the curated NUTS product."),
        (RUNS / "KGAS007-20260905T141705Z-nuts-kgas007-c1", "EPHEMERAL_TRASH", "Aborted SUBMITTING stub with no samples."),
        (RUNS / "KGAS007-20260905T141746Z-nuts-kgas007-c2", "EPHEMERAL_TRASH", "Crashed chain replaced by verified chain 2."),
        (RUNS / "KGAS007-20260906T003014Z-nuts-kgas007-c2", "EPHEMERAL_TRASH", "Aborted SUBMITTING relaunch stub with no samples."),
    ]
    for directory in k07_shard_dirs:
        purge_specs.append((directory, "REDUNDANT_AFTER_MERGE", "Redundant checkpoint shard; merged posterior reconstructed exactly and compact provenance retained."))
    purge_specs.extend(
        [
            (RUNS / "KGAS066-latest", "EPHEMERAL_TRASH", "Obsolete symlink to the former run-root layout."),
            (k66_nuts_source, "ACCEPTED_RELOCATED", "Accepted source reorganized into the curated production product; merged posterior exactly reproduces all checkpoints."),
            (k66_map_source, "ACCEPTED_RELOCATED", "Accepted source reorganized into the curated production product."),
            (k07_map_source, "ACCEPTED_RELOCATED", "Accepted source reorganized into the curated production product."),
        ]
    )
    removed = [
        {"path": str(path.relative_to(WORKSPACE)), "classification": classification,
         "bytes": file_bytes(path), "reason": reason}
        for path, classification, reason in purge_specs
    ]

    # All outputs now exist and all tar members and merged shards have been verified.
    for source, _, _ in archive_specs:
        remove(source)
    for path, _, _ in purge_specs:
        if path.exists() or path.is_symlink():
            remove(path)
    for legacy in (RESULTS / "KILOGAS066", RESULTS / "KILOGAS007", RUNS):
        if legacy.is_dir() and not any(legacy.iterdir()):
            legacy.rmdir()

    shutil.move(str(PRODUCTION), str(RESULTS / "production"))
    shutil.move(str(ARCHIVE), str(RESULTS / "archive"))
    STAGE.rmdir()

    manifest = render_manifest(production, archives, removed, initial_bytes)
    (RESULTS / "MANIFEST.md").write_text(manifest, encoding="utf-8")
    final_bytes = tree_bytes(RESULTS)
    curation = {
        "schema_version": "kinuv-curation-v1",
        "curated_at": "2026-09-06",
        "initial_bytes": initial_bytes,
        "final_bytes_before_this_record": final_bytes,
        "net_bytes_reclaimed_before_this_record": initial_bytes - final_bytes,
        "production_products": len(production),
        "archives": len(archives),
        "purged_entries": len(removed),
        "archives_detail": archives,
        "removed_detail": removed,
        "merged_shard_verification": {
            "KGAS066": "exact physical reconstruction from 4 x 600 x 6 checkpoints",
            "KGAS007": "exact physical reconstruction from 4 x 600 x 6 checkpoints",
        },
    }
    write_json(RESULTS / "CURATION.json", curation)
    print(json.dumps(curation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
