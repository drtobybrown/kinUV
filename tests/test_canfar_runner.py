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
    assert "YYYYMMDDTHHMMSSZ" in text
    assert "symlink to the newest run" in text
    assert "Agent Run Status" in text


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
    assert 'tee -a "${SCRATCH_LOG}" "${ARC_LOG}"' not in text
    assert "copy_worker_log" in text
    assert "PYTHONUNBUFFERED" in text
    assert "/arc/projects/KILOGAS/analysis/toby_sandbox/kinuv_runs" in text
    assert "/arc/home/thbrown/kinuv_runs" not in text
    assert "checkpoints" in text
    assert "SCRATCH_LOG" in text
    assert "ARC_LOG" in text
    assert "scratchcopy" not in text


def test_watch_script_exists():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    text = (repo / "scripts/watch_headless.py").read_text()
    assert "canfar logs expire" in text or "Persist canfar logs" in text
    assert "stream_logs" in text
    assert "stream_events" in text
    assert "append_log(platform, text" not in text
    assert "log-interval" in text
    assert "log_interval" in text


def test_default_runs_root_is_project_not_home():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "src/kinuv/runner/canfar.py").read_text()
    assert 'PROJECT_ROOT / "kinuv_runs"' in src
    assert '"/arc/home/thbrown/kinuv_runs"' not in src


def test_make_run_id_and_latest(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    import kinuv.runner.canfar as canfar
    from kinuv.runner.canfar import galaxy_tag, make_run_id, point_latest

    monkeypatch.setattr(canfar, "RUNS_ROOT", tmp_path)
    assert galaxy_tag("kgas066") == "KGAS066"
    assert galaxy_tag("KILOGAS066") == "KGAS066"
    when = datetime(2026, 8, 31, 8, 1, 12, tzinfo=timezone.utc)
    rid = make_run_id("KGAS066", "nuts", when=when)
    assert rid == "KGAS066-20260831T080112Z-nuts"
    (tmp_path / rid).mkdir()
    link = point_latest("KGAS066", rid)
    assert link.name == "KGAS066-latest"
    assert link.resolve() == (tmp_path / rid).resolve()
    rid2 = make_run_id("066", "nuts", when=when)
    assert rid2.startswith("KGAS066-")


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


def test_save_npz_atomic_does_not_append_extra_npz(tmp_path):
    import numpy as np

    from kinuv.runner.checkpoint import dual_checkpoint, flush_scratch_to_arc, save_npz_atomic

    dest = tmp_path / "chain_1.npz"
    save_npz_atomic(dest, z6=np.arange(6.0), mean_steps=np.array(1.5))
    assert dest.is_file()
    leftover = [p.name for p in tmp_path.iterdir() if p.suffix == ".npz" or ".npz" in p.name]
    assert leftover == ["chain_1.npz"]
    got = np.load(dest)
    np.testing.assert_array_equal(got["z6"], np.arange(6.0))
    assert float(got["mean_steps"]) == 1.5

    scratch, arc = tmp_path / "scratch", tmp_path / "arc"
    s, a = dual_checkpoint(scratch, arc, "chain_2.npz", z6=np.ones(3))
    assert s.is_file() and a is not None and a.is_file()
    np.testing.assert_array_equal(np.load(a)["z6"], np.ones(3))
    extra = list(scratch.glob("*"))
    assert [p.name for p in extra] == ["chain_2.npz"]
    (scratch / "chain_3.npz").write_bytes(s.read_bytes())
    copied = flush_scratch_to_arc(scratch, arc)
    assert any(p.name == "chain_3.npz" for p in copied)
    assert (arc / "chain_3.npz").is_file()


def test_patch_agent_run_status_does_not_touch_mailbox(tmp_path, monkeypatch):
    from kinuv.runner.status_md import patch_agent_run_status, write_job_status_md

    dest = tmp_path / "docs" / "architecture"
    dest.mkdir(parents=True)
    path = dest / "STATUS.md"
    path.write_text(
        "---\npending: [\"old-job\"]\n---\n\n"
        "## Agent Run Status\n\n"
        "* **Phase:** waiting\n"
        "* **Last Action:** launched\n"
        "* **Decisions Made:** none yet\n"
        "* **Blockers / Gates:** sentinel\n"
        "* **Next Step:** wait\n\n"
        "# Architecture mailbox\n\n"
        "**2026-08-31 (keep).** Parent physics line.\n"
    )
    patch_agent_run_status(
        path,
        {
            "Phase": "done",
            "Last Action": "SUCCEEDED",
            "Decisions Made": "worker wrote STATUS",
            "Blockers / Gates": "none",
            "Next Step": "corner",
        },
        pending=[],
    )
    text = path.read_text()
    assert "pending: []" in text
    assert "* **Phase:** done" in text
    assert "* **Last Action:** SUCCEEDED" in text
    assert "**2026-08-31 (keep).** Parent physics line." in text
    monkeypatch.setenv("KINUV_REPO", str(tmp_path))
    got = write_job_status_md(
        run_id="KGAS066-test",
        session_id="abc",
        state="SUCCEEDED",
        mixing_pass=True,
        sampler="nuts",
        elapsed_s=12.0,
    )
    assert got == path
    out = path.read_text()
    assert "SUCCEEDED" in out
    assert "pending: []" in out
    assert "Parent physics line" in out

