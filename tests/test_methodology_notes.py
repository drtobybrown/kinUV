"""Executable boundary checks retained from the historical methodology suite."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
def test_kgas007_script_is_map_only():
    text = (REPO / "scripts/run_kgas007_stage_a_map.py").read_text(encoding="utf-8")
    assert "launch_headless" not in text
    assert "steal_latest" not in text
    assert "point_latest" not in text
    assert "numpyro" not in text
    assert "KINUV_KIND" not in text
    assert "diagnostic_only" in text
    assert "dec_066_target_amended" in text
