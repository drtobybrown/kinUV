#!/usr/bin/env python3
"""Report sub-beam turnover recovery from accepted S4 synthetic fits.

This is an additive S5 publication diagnostic. It reads the retained S4 fit
parameters, performs no refit, and does not alter or reopen the S4 gate.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kinuv.validation.s4 import projected_arctan_speed, subbeam_turnover_recovery


REPO = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def git_state() -> dict:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=REPO, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO, text=True
        ).strip()
    )
    return {"commit": commit, "branch": branch, "dirty": dirty}


def ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        raise ValueError("reference error must be positive")
    return float(numerator / denominator)


def target_record(summary_path: Path, config_path: Path, target_output: Path) -> dict:
    source = json.loads(summary_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    target_id = source["target_id"]
    if config["target_id"] != target_id:
        raise ValueError(f"target mismatch: {summary_path} and {config_path}")

    beam = float(config["diagnostic_beam"]["bmaj_arcsec"])
    radius = np.asarray(source["generation"]["profile_radius_arcsec"], dtype=float)
    weight = radius * np.exp(-radius / (1.35 * beam))
    truth = source["truth"]
    realizations = []

    figure, axis = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    inner = radius <= beam
    axis.plot(
        radius[inner] / beam,
        projected_arctan_speed(truth, radius[inner]),
        color="black",
        linewidth=2.4,
        label="truth",
        zorder=5,
    )
    for index, source_row in enumerate(source["realizations"]):
        kinuv_parameters = source_row["kinuv"]["parameters"]
        kinms_parameters = source_row["kinms"]["parameters"]
        kinuv = subbeam_turnover_recovery(
            truth, kinuv_parameters, radius, weight, beam
        )
        kinms = subbeam_turnover_recovery(
            truth, kinms_parameters, radius, weight, beam
        )
        if not kinuv["eligible"] or not kinms["eligible"]:
            raise ValueError(f"{target_id} does not meet R_turn < BMAJ")
        realizations.append(
            {
                "seed": source_row["seed"],
                "kinuv": kinuv,
                "kinms": kinms,
                "absolute_turnover_error_ratio_kinuv_over_kinms": ratio(
                    kinuv["absolute_turnover_error_arcsec"],
                    kinms["absolute_turnover_error_arcsec"],
                ),
                "inner_rmse_ratio_kinuv_over_kinms": ratio(
                    kinuv["inner_projected_velocity_rmse_kms"],
                    kinms["inner_projected_velocity_rmse_kms"],
                ),
            }
        )
        axis.plot(
            radius[inner] / beam,
            projected_arctan_speed(kinuv_parameters, radius[inner]),
            color="#1261a0",
            alpha=0.55,
            linewidth=1.3,
            label="kinUV fits" if index == 0 else None,
        )
        axis.plot(
            radius[inner] / beam,
            projected_arctan_speed(kinms_parameters, radius[inner]),
            color="#e67e22",
            alpha=0.55,
            linewidth=1.3,
            label="KinMS fits" if index == 0 else None,
        )

    kinuv_turnover = np.asarray(
        [row["kinuv"]["absolute_turnover_error_arcsec"] for row in realizations]
    )
    kinms_turnover = np.asarray(
        [row["kinms"]["absolute_turnover_error_arcsec"] for row in realizations]
    )
    kinuv_inner = np.asarray(
        [row["kinuv"]["inner_projected_velocity_rmse_kms"] for row in realizations]
    )
    kinms_inner = np.asarray(
        [row["kinms"]["inner_projected_velocity_rmse_kms"] for row in realizations]
    )
    aggregate = {
        "eligible": True,
        "eligibility_rule": "truth r_t_arcsec < 1.0 * BMAJ",
        "truth_turnover_over_bmaj": float(truth["r_t_arcsec"] / beam),
        "kinuv_mean_absolute_turnover_error_arcsec": float(np.mean(kinuv_turnover)),
        "kinms_mean_absolute_turnover_error_arcsec": float(np.mean(kinms_turnover)),
        "absolute_turnover_error_ratio_kinuv_over_kinms": ratio(
            float(np.mean(kinuv_turnover)), float(np.mean(kinms_turnover))
        ),
        "kinuv_inner_rms_rmse_kms": float(np.sqrt(np.mean(kinuv_inner**2))),
        "kinms_inner_rms_rmse_kms": float(np.sqrt(np.mean(kinms_inner**2))),
        "inner_rmse_ratio_kinuv_over_kinms": ratio(
            float(np.sqrt(np.mean(kinuv_inner**2))),
            float(np.sqrt(np.mean(kinms_inner**2))),
        ),
        "n_realizations": len(realizations),
        "gate_role": "supporting publication diagnostic; does not gate S4",
    }

    axis.axvline(
        aggregate["truth_turnover_over_bmaj"],
        color="black",
        linestyle=":",
        linewidth=1.2,
        label="true Rturn / BMAJ",
    )
    axis.axvline(1.0, color="0.4", linestyle="--", linewidth=1.0, label="BMAJ")
    axis.set(
        title=f"{target_id}: projected inner rotation rise",
        xlabel="Radius / BMAJ",
        ylabel="Projected rotation speed u(r) [km/s]",
        xlim=(0.0, 1.03),
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8)
    target_output.mkdir(parents=True, exist_ok=True)
    figure.savefig(target_output / "subbeam_profiles.png", dpi=180)
    plt.close(figure)

    record = {
        "schema_version": "kinuv-s5-subbeam-target-v1",
        "target_id": target_id,
        "bmaj_arcsec": beam,
        "source": {
            "accepted_s4_summary": str(summary_path.resolve()),
            "accepted_s4_summary_sha256": sha256(summary_path),
            "target_config": str(config_path.resolve()),
            "target_config_sha256": sha256(config_path),
        },
        "method": {
            "turnover_error": "absolute fitted minus true R_turn",
            "inner_velocity_domain": "stored S4 profile radii with r <= BMAJ",
            "inner_velocity_weight": "r * exp(-r / (1.35 * BMAJ))",
            "refit_performed": False,
            "s4_gate_reopened": False,
        },
        "aggregate": aggregate,
        "realizations": realizations,
    }
    write_json(target_output / "summary.json", record)
    return record


def summary_figure(records: list[dict], output: Path) -> None:
    labels = [record["target_id"] for record in records]
    position = np.arange(len(labels), dtype=float)
    width = 0.34
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), constrained_layout=True)
    turnover_kinuv = [
        record["aggregate"]["kinuv_mean_absolute_turnover_error_arcsec"]
        / record["bmaj_arcsec"]
        for record in records
    ]
    turnover_kinms = [
        record["aggregate"]["kinms_mean_absolute_turnover_error_arcsec"]
        / record["bmaj_arcsec"]
        for record in records
    ]
    inner_kinuv = [record["aggregate"]["kinuv_inner_rms_rmse_kms"] for record in records]
    inner_kinms = [record["aggregate"]["kinms_inner_rms_rmse_kms"] for record in records]
    for axis, first, second, ylabel, title in (
        (
            axes[0],
            turnover_kinuv,
            turnover_kinms,
            "Mean |Delta Rturn| / BMAJ",
            "Sub-beam turnover error",
        ),
        (
            axes[1],
            inner_kinuv,
            inner_kinms,
            "Inner u(r) RMSE [km/s]",
            "Inner projected-velocity error",
        ),
    ):
        axis.bar(position - width / 2, first, width, label="kinUV", color="#1261a0")
        axis.bar(position + width / 2, second, width, label="KinMS", color="#e67e22")
        axis.set_xticks(position, labels)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def markdown_summary(records: list[dict], code_commit: str) -> str:
    rows = []
    for record in records:
        aggregate = record["aggregate"]
        rows.append(
            "| {target} | {scale:.4f} | {ku_rt:.6f} | {km_rt:.6f} | {rt_ratio:.4f} "
            "| {ku_inner:.6f} | {km_inner:.6f} | {inner_ratio:.4f} |".format(
                target=record["target_id"],
                scale=aggregate["truth_turnover_over_bmaj"],
                ku_rt=aggregate["kinuv_mean_absolute_turnover_error_arcsec"],
                km_rt=aggregate["kinms_mean_absolute_turnover_error_arcsec"],
                rt_ratio=aggregate["absolute_turnover_error_ratio_kinuv_over_kinms"],
                ku_inner=aggregate["kinuv_inner_rms_rmse_kms"],
                km_inner=aggregate["kinms_inner_rms_rmse_kms"],
                inner_ratio=aggregate["inner_rmse_ratio_kinuv_over_kinms"],
            )
        )
    return "\n".join(
        [
            "# S5 sub-beam turnover diagnostic",
            "",
            "This supporting publication diagnostic reuses accepted S4 fitted parameters; it performs no refit and does not reopen the S4 gate.",
            "",
            f"Code commit: `{code_commit}`",
            "",
            "| Target | true Rturn/BMAJ | kinUV mean abs(Delta Rturn) [arcsec] | KinMS mean abs(Delta Rturn) [arcsec] | error ratio | kinUV inner RMSE [km/s] | KinMS inner RMSE [km/s] | RMSE ratio |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "The inner RMSE uses only stored profile samples at `r <= BMAJ`, with the same radial weighting as the accepted full-disk S4 metric.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--target-config", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = git_state()
    if state["branch"] != "dev" or state["dirty"]:
        raise SystemExit("reporting requires a clean dev checkout")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    configs = {
        json.loads(path.read_text(encoding="utf-8"))["target_id"]: path
        for path in args.target_config
    }
    summaries = sorted(args.synthetic_root.glob("*/summary.json"))
    if not summaries:
        raise SystemExit(f"no accepted summaries found under {args.synthetic_root}")
    args.output.mkdir(parents=True)
    records = []
    for summary_path in summaries:
        target_id = json.loads(summary_path.read_text(encoding="utf-8"))["target_id"]
        if target_id not in configs:
            raise SystemExit(f"missing target config for {target_id}")
        records.append(
            target_record(summary_path, configs[target_id], args.output / target_id)
        )
    summary_figure(records, args.output / "subbeam_summary.png")
    metrics = {
        "schema_version": "kinuv-s5-subbeam-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": state,
        "status": "supporting-diagnostic-complete",
        "gate_role": "supporting publication diagnostic; does not gate or reopen S4",
        "s4_gate_reopened": False,
        "source_s4_root": str(args.synthetic_root.resolve()),
        "targets": records,
    }
    write_json(args.output / "metrics.json", metrics)
    (args.output / "summary.md").write_text(
        markdown_summary(records, state["commit"]), encoding="utf-8"
    )
    manifest = {
        "schema_version": "kinuv-s5-subbeam-manifest-v1",
        "code_commit": state["commit"],
        "files": {},
    }
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            manifest["files"][str(path.relative_to(args.output))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    write_json(args.output / "MANIFEST.json", manifest)
    for record in records:
        aggregate = record["aggregate"]
        print(
            f"{record['target_id']}: Rturn error ratio="
            f"{aggregate['absolute_turnover_error_ratio_kinuv_over_kinms']:.6f}, "
            f"inner RMSE ratio={aggregate['inner_rmse_ratio_kinuv_over_kinms']:.6f}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
