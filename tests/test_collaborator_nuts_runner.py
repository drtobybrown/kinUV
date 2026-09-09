from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

from run_collaborator_postprocess_controller import aggregate_chain_statuses
from run_collaborator_nuts_race import campaign_status


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_collaborator_nuts_chain.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("run_collaborator_nuts_chain", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_initial_jitter_is_shrunk_to_remain_in_map_basin():
    energy = lambda value: 1000.0 * float(np.sum(np.asarray(value) ** 2))
    point, map_energy, initial_energy, scale = MODULE.bounded_initial_point(
        energy,
        np.zeros(2),
        np.ones(2),
        requested_scale=0.1,
        max_delta=10.0,
    )

    assert 0.0 < scale < 0.1
    assert initial_energy - map_energy <= 10.0
    np.testing.assert_allclose(point, scale * np.ones(2))


def test_regularized_inverse_mass_matches_quadratic_map_covariance():
    curvature = np.asarray([[4.0, 1.0], [1.0, 2.0]])

    def value_gradient(point):
        point = np.asarray(point)
        return 0.5 * point @ curvature @ point, curvature @ point

    metric, diagnostics = MODULE.regularized_inverse_mass(
        value_gradient,
        np.asarray([0.2, -0.1]),
        relative_step=1.0e-3,
    )

    np.testing.assert_allclose(metric, np.linalg.inv(curvature), rtol=1.0e-10, atol=1.0e-12)
    assert diagnostics["hessian_negative_eigenvalues"] == 0
    assert np.linalg.eigvalsh(metric).min() > 0.0


def test_regularized_inverse_mass_repairs_indefinite_numeric_curvature():
    curvature = np.asarray([[-1.0, 0.0], [0.0, 2.0]])

    def value_gradient(point):
        point = np.asarray(point)
        return 0.5 * point @ curvature @ point, curvature @ point

    metric, diagnostics = MODULE.regularized_inverse_mass(
        value_gradient,
        np.zeros(2),
        relative_step=1.0e-3,
    )

    assert diagnostics["hessian_negative_eigenvalues"] == 1
    assert np.linalg.eigvalsh(metric).min() > 0.0


def test_postprocessor_aggregates_independent_chain_sessions(tmp_path):
    for target in ("KGAS066", "KGAS007"):
        for chain in range(1, 5):
            path = tmp_path / target / f"chain-{chain}" / "status.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                '{"state":"SUCCEEDED","exit_code":0,"target":"%s",'
                '"chain_id":%d,"seed":%d,"pid":11,"started_utc":"x"}\n'
                % (target, chain, chain)
            )
    result = aggregate_chain_statuses(tmp_path, ("KGAS066", "KGAS007"))
    assert result["state"] == "SUCCEEDED"
    assert len(result["completed"]) == 8
    assert not result["active"]
    assert not result["queued"]


def test_race_status_waits_for_requested_extension_draws(tmp_path):
    for target in ("KGAS066", "KGAS007"):
        for chain in range(1, 5):
            path = tmp_path / target / f"chain-{chain}" / "status.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                '{"state":"SUCCEEDED","completed_warmup":200,"completed_draws":500}\n'
            )

    initial = campaign_status(tmp_path, {"KGAS066": 500, "KGAS007": 500})
    extended = campaign_status(tmp_path, {"KGAS066": 1000, "KGAS007": 500})

    assert initial["state"] == "SUCCEEDED"
    assert extended["state"] == "RUNNING"
    assert extended["chains"][0]["state"] == "AWAITING_EXTENSION"
