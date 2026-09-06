import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[1] / "external" / "run_image_benchmarks.py"
    spec = importlib.util.spec_from_file_location("run_image_benchmarks", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_mock_recovery_schema_requires_corrected_field_for_both_fitters():
    validate = _module()._validate_mock_recovery_schema
    valid = {
        name: {"status": "ran", "mock_turnover_radius_recovered": True}
        for name in ("kinuv_mock", "kinms_mock")
    }
    validate(valid)
    for name in ("kinuv_mock", "kinms_mock"):
        broken = {key: dict(value) for key, value in valid.items()}
        broken[name].pop("mock_turnover_radius_recovered")
        with pytest.raises(RuntimeError, match=name):
            validate(broken)
