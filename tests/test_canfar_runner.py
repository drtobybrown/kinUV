"""CANFAR runner helpers. No live submit."""

from __future__ import annotations

from kinuv.runner.canfar import parse_info_status, parse_session_id


def test_parse_session_id():
    text = "Successfully created session 'kinuv-KGAS066-e14d58-nuts' (ID: ab12cd34)"
    assert parse_session_id(text) == "ab12cd34"
    assert parse_session_id("no id here") is None


def test_parse_info_status():
    text = "Session ID    xyz\n  Name          kinuv-KGAS066-e14d58-nuts\n  Status        Running\n"
    got = parse_info_status(text)
    assert got["state"] == "Running"
    assert got["name"] == "kinuv-KGAS066-e14d58-nuts"


def test_dec_067_on_disk():
    from kinuv.decisions import load_decision_index

    index = load_decision_index()
    assert index["DEC-067-RUNNER"] == "accepted"


def test_entrypoint_uses_scratch_and_venv():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    text = (repo / "scripts/canfar_entrypoint.sh").read_text()
    assert "/scratch/kinuv-" in text
    assert "kinuv-venv-recovery" in text
    assert "run_kgas066_nuts_headless.py" in text
    assert "canfar create" not in text
