import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[1] / "scripts" / "run_production_milestone.py"
    spec = importlib.util.spec_from_file_location("run_production_milestone", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v2_target_configs_name_the_authoritative_rotation_gate():
    module = _module()
    root = Path(__file__).parents[1]
    for target in ("KGAS066", "KGAS007"):
        config = json.loads((root / "configs" / "targets" / f"{target}.json").read_text())
        module._validate_target_config(config)
        assert config["schema_version"] == module.TARGET_CONFIG_SCHEMA


def test_v1_or_unknown_rotation_gate_is_rejected():
    module = _module()
    config = {
        "schema_version": module.TARGET_CONFIG_SCHEMA,
        "acceptance": {"rotation_gate_id": module.ROTATION_GATE_ID},
    }
    for mutation in (
        {"schema_version": "kinuv-production-target-v1"},
        {"acceptance": {"rotation_gate_id": "arbitrary"}},
    ):
        broken = {**config, **mutation}
        with pytest.raises(ValueError):
            module._validate_target_config(broken)
