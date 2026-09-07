"""Guard executable configuration and package-level project invariants."""

from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
def test_configuration_templates_define_separate_target_and_campaign_state():
    target = (REPO / "configs" / "targets" / "_template.yaml").read_text()
    campaign = (REPO / "configs" / "campaigns" / "_template.yaml").read_text()
    assert "schema_version: kinuv-target-v1" in target
    assert "schema_version: kinuv-campaign-v1" in campaign
    assert "target_config:" in campaign
    assert "run_root_env: KINUV_RUN_ROOT" in campaign


def test_project_metadata_describes_a_general_modeling_engine():
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8").lower()
    description = next(
        line for line in text.splitlines() if line.startswith("description =")
    )
    assert "standalone visibility-plane kinematic modeling engine" in description
    assert "kgas" not in description
