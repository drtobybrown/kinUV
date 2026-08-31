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
    from pathlib import Path

    from kinuv.decisions import load_decision_index

    index = load_decision_index()
    assert index["DEC-067-RUNNER"] == "accepted"
    text = (Path(__file__).resolve().parents[1] / "docs/decisions/DEC-067-RUNNER.md").read_text()
    assert "1 hour" in text
    assert "worker.log" in text
    assert "/arc/projects/KILOGAS/analysis/toby_sandbox/kinuv_runs" in text
    assert "/arc/home/thbrown/kinuv_runs" not in text


def test_entrypoint_uses_scratch_and_venv():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    text = (repo / "scripts/canfar_entrypoint.sh").read_text()
    assert "/scratch/kinuv-" in text
    assert "kinuv-venv-recovery" in text
    assert "run_kgas066_nuts_headless.py" in text
    assert "canfar create" not in text
    assert "worker.log" in text
    assert "tee -a" in text
    assert "PYTHONUNBUFFERED" in text
    assert "/arc/projects/KILOGAS/analysis/toby_sandbox/kinuv_runs" in text
    assert "/arc/home/thbrown/kinuv_runs" not in text
    assert "checkpoints" in text
    assert "SCRATCH_LOG" in text
    assert "ARC_LOG" in text


def test_watch_script_exists():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    text = (repo / "scripts/watch_headless.py").read_text()
    assert "canfar logs expire" in text or "Persist canfar logs" in text
    assert "stream_logs" in text
    assert "stream_events" in text


def test_default_runs_root_is_project_not_home():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "src/kinuv/runner/canfar.py").read_text()
    assert 'PROJECT_ROOT / "kinuv_runs"' in src
    assert '"/arc/home/thbrown/kinuv_runs"' not in src


def test_job_log_and_archive(tmp_path, monkeypatch):
    import json

    import kinuv.runner.canfar as canfar
    from kinuv.runner.canfar import archive_run, write_json, write_status
    from kinuv.runner.log import append_log, setup_worker_logging

    monkeypatch.setattr(canfar, "RUNS_ROOT", tmp_path)
    monkeypatch.setenv("KINUV_RUNS", str(tmp_path))
    import kinuv.runner.log as logmod

    monkeypatch.setattr(logmod, "RUNS_ROOT", tmp_path)

    write_status("job1", {"state": "RUNNING"})
    append_log(tmp_path / "job1" / "worker.log", "hello from worker")
    logger = setup_worker_logging("job1")
    logger.info("structured")
    write_json(tmp_path / "job1" / "logs" / "chain_1.json", {"chain": 1})
    dest = archive_run("job1")
    assert dest is not None
    assert dest.name.startswith("job1.archive.")
    assert (dest / "worker.log").read_text().strip() == "hello from worker"
    assert "structured" in (dest / "logs" / "run.log").read_text()
    rec = json.loads((dest / "status.json").read_text())
    assert rec["state"] == "RUNNING"
    assert not (tmp_path / "job1").exists()
    assert archive_run("missing") is None
