"""Guard the data-agnostic production field guide and authority model."""

from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
GUIDE = REPO / "field-guide" / "index.md"
OPERATIONAL_DOCS = (
    REPO / "AGENTS.md",
    GUIDE,
    REPO / "docs" / "decisions" / "DEC-066-INDEX.md",
    REPO / "docs" / "decisions" / "DEC-066-AGENTS.md",
    REPO / "docs" / "decisions" / "DEC-067-RUNNER.md",
    REPO / "docs" / "reviews" / "BOARD.md",
    REPO / "docs" / "diagnostics" / "scratch.md",
)


def test_field_guide_has_required_operational_sections():
    text = GUIDE.read_text(encoding="utf-8")
    required = (
        "Agent hierarchy and authority boundaries",
        "Configuration boundary",
        "Architectural invariants",
        "Generalized scientific pipeline gates",
        "Analytic closure",
        "Mock recovery",
        "Null and baseline comparisons",
        "Covariance and residual adequacy",
        "Posterior sampling and convergence",
        "Storage tiering",
        "Review and promotion workflow",
    )
    for heading in required:
        assert heading in text


def test_operational_standard_contains_no_target_identity():
    banned = (
        "kgas066",
        "kgas007",
        "kilogas066",
        "kilogas007",
        "/by_galaxy/",
        "43.9°",
        "205.2°",
    )
    for path in OPERATIONAL_DOCS:
        text = path.read_text(encoding="utf-8").lower()
        for token in banned:
            assert token.lower() not in text, f"{path} contains {token}"


def test_hierarchy_does_not_delegate_scientific_authority_to_implementer():
    text = GUIDE.read_text(encoding="utf-8")
    assert "Consultant: Lead Architect / frontier model" in text
    assert "Senior Registrar" in text
    assert "may not lower the thresholds" in text
    assert "Only Astra may waive" in text


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
