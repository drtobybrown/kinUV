"""Pure-array tests for the downstream KinMS comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kinuv.diagnostics.kinms_benchmark import aperture_spectrum, cube_metrics

ROOT = Path(__file__).resolve().parents[1]


def test_cube_metrics_do_not_refit_flux():
    data = np.ones((3, 4, 5))
    model = 0.75 * data
    mask = np.ones_like(data, dtype=bool)
    got = cube_metrics(data, model, mask)
    assert got["n_voxel"] == data.size
    assert got["residual_rms_k"] == pytest.approx(0.25)
    assert got["normalized_rmse"] == pytest.approx(0.25)
    assert got["model_to_data_flux"] == pytest.approx(0.75)


def test_cube_metrics_respect_mask_and_shapes():
    data = np.arange(8.0).reshape(2, 2, 2)
    model = data.copy()
    mask = np.zeros_like(data, dtype=bool)
    mask[:, 0, 0] = True
    assert cube_metrics(data, model, mask)["residual_rms_k"] == 0.0
    with pytest.raises(ValueError, match="share a cube shape"):
        cube_metrics(data, model[:, :, :1], mask)
    with pytest.raises(ValueError, match="selects no finite voxels"):
        cube_metrics(data, model, np.zeros_like(mask))


def test_aperture_spectrum_uses_projected_source_mask():
    cube = np.arange(12.0).reshape(3, 2, 2)
    mask = np.zeros_like(cube, dtype=bool)
    mask[0, 0, 1] = True
    np.testing.assert_allclose(aperture_spectrum(cube, mask), cube[:, 0, 1])


def test_canonical_runner_defaults_to_both_targets_without_inference_calls():
    config = json.loads(
        (ROOT / "configs/benchmarks/canonical-kinms.json").read_text()
    )
    assert [item["target_id"] for item in config["targets"]] == [
        "KGAS066",
        "KGAS007",
    ]
    source = (ROOT / "scripts/run_canonical_kinms_benchmark.py").read_text()
    assert 'DEFAULT_TARGETS = ("KGAS066", "KGAS007")' in source
    for forbidden in ("kinuv.infer", "run_stage_a_map", "run_nuts", "launch_headless"):
        assert forbidden not in source
