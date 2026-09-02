"""CANFAR runner helpers. No live submit."""

from __future__ import annotations

from pathlib import Path

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
    assert "corner" in text or "PNG" in text


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
    assert "MPLBACKEND" in text


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


def test_write_nuts_product_plots_corner_only(tmp_path):
    from pathlib import Path

    import numpy as np

    from kinuv.runner.plots import write_nuts_product_plots

    rng = np.random.default_rng(0)
    draws = rng.normal(size=(2, 30, 8))
    draws[..., 4] = 0.091
    draws[..., 5] = 0.019
    rec = {"sampler": "nuts", "draws": draws}
    dest = tmp_path / "run"
    (dest / "posteriors").mkdir(parents=True)
    art = tmp_path / "art"
    written = write_nuts_product_plots(
        rec, dest, artifact_dir=art, leftover=False, imaging=False
    )
    assert Path(written["corner"]).is_file()
    assert (art / "corner.png").is_file()
    assert not (dest / "plots" / "moments.png").is_file()


def test_pa25_env_does_not_gpu_or_clobber_g3(tmp_path):
    from kinuv.runner.canfar import (
        ARTIFACT_G3_REL,
        ARTIFACT_PA25_REL,
        headless_job_env,
        steal_latest,
    )

    env = headless_job_env(
        run_id="KGAS066-test-nuts-pa25",
        galaxy="KGAS066",
        kind="nuts-pa25",
        gpu=None,
        repo=tmp_path,
        runs_root=tmp_path / "runs",
    )
    assert env["KINUV_PA_INIT"] == "25.2"
    assert env["JAX_PLATFORMS"] == "cpu"
    assert "KINUV_PA_INIT" in env
    assert ARTIFACT_G3_REL not in env["KINUV_ARTIFACT_DIR"]
    assert ARTIFACT_PA25_REL in env["KINUV_ARTIFACT_DIR"]
    assert env["KINUV_KIND"] == "nuts-pa25"
    assert steal_latest("nuts-pa25") is False
    assert steal_latest("nuts") is True


def test_write_nuts_product_plots_pa25_does_not_touch_g3(tmp_path, monkeypatch):
    from pathlib import Path

    import numpy as np

    import kinuv.runner.plots as plots

    g3 = tmp_path / "g3-sentinel"
    g3.mkdir()
    marker = g3 / "keep.txt"
    marker.write_text("untouched\n")
    monkeypatch.setattr(plots, "ARTIFACT_G3", g3)
    rng = np.random.default_rng(1)
    draws = rng.normal(size=(2, 20, 8))
    rec = {"sampler": "nuts", "draws": draws, "pa_init_deg": 25.2}
    dest = tmp_path / "run"
    (dest / "posteriors").mkdir(parents=True)
    art = tmp_path / "leftover" / "pa25"
    plots.write_nuts_product_plots(
        rec, dest, artifact_dir=art, leftover=False, imaging=False
    )
    assert marker.read_text() == "untouched\n"
    assert list(g3.iterdir()) == [marker]
    assert (art / "corner.png").is_file()


def test_job_status_pa25_does_not_name_g3(tmp_path, monkeypatch):
    from kinuv.runner.status_md import write_job_status_md

    path = tmp_path / "docs" / "architecture" / "STATUS.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "---\npending: [x]\n---\n\n## Agent Run Status\n\n"
        "* **Phase:** old\n* **Last Action:** old\n"
        "* **Decisions Made:** old\n* **Blockers / Gates:** old\n"
        "* **Next Step:** old\n\n# Architecture mailbox\n\nkeep.\n"
    )
    monkeypatch.setenv("KINUV_REPO", str(tmp_path))
    write_job_status_md(
        run_id="KGAS066-20260902T000000Z-nuts-pa25",
        session_id="abc",
        state="SUCCEEDED",
        mixing_pass=True,
        sampler="nuts",
        elapsed_s=1.0,
        kind="nuts-pa25",
    )
    out = path.read_text()
    assert "2026-08-30-g3-nuts" not in out
    assert "leftover-and-modes" in out
    assert "keep." in out


def test_product_record_does_not_force_leftover_true():
    import numpy as np

    from kinuv.infer.nuts import product_record

    draws = np.zeros((2, 4, 8))
    rec = product_record(
        draws8=draws,
        mix={"flux": {"rhat": 1.0, "ess": 500, "ess_tail": 500}},
        pa_init_deg=25.2,
        dx_map=0.09,
        dy_map=0.02,
        autodiff_ok=True,
        mixing_pass=True,
        leftover_chi2_structured=False,
        r_t_at_floor=False,
        mean_num_steps=10.0,
        eval_s=1.0,
        note="test",
    )
    assert rec["leftover_chi2_structured"] is False
    assert rec["pa_init_deg"] == 25.2


def test_headless_worker_does_not_hardcode_leftover_true():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    worker = (repo / "scripts" / "run_kgas066_nuts_headless.py").read_text()
    assert "leftover_chi2_structured=True" not in worker
    assert "KINUV_PA_INIT" in worker
    entry = (repo / "scripts" / "canfar_entrypoint.sh").read_text()
    assert "--pa-init" in entry
    assert "KINUV_PA_INIT" in entry
    assert "KINUV_CHAIN_ID" in entry
    assert "--chain-id" in entry


def test_gpu_kind_does_not_steal_or_clobber_g3():
    from kinuv.runner.canfar import (
        ARTIFACT_G3_REL,
        ARTIFACT_GPU_REL,
        headless_job_env,
        require_pinned_gpu,
        steal_latest,
    )
    import pytest

    assert steal_latest("nuts-gpu") is False
    assert steal_latest("nuts-gpu-c1") is False
    assert steal_latest("nuts") is True
    env = headless_job_env(
        run_id="KGAS066-test-nuts-gpu-c1",
        galaxy="KGAS066",
        kind="nuts-gpu",
        gpu=1,
        chain_id=1,
        repo=Path("/tmp"),
        runs_root=Path("/tmp/runs"),
    )
    assert env["JAX_PLATFORMS"] == "cuda"
    assert "kinuv-cuda" in env["KINUV_VENV"]
    assert "kinuv-venv-recovery" not in env["KINUV_VENV"]
    assert ARTIFACT_G3_REL not in env["KINUV_ARTIFACT_DIR"]
    assert ARTIFACT_GPU_REL in env["KINUV_ARTIFACT_DIR"]
    assert env["KINUV_CHAIN_ID"] == "1"
    cpu_env = headless_job_env(
        run_id="KGAS066-test-nuts",
        galaxy="KGAS066",
        kind="nuts",
        gpu=None,
        repo=Path("/tmp"),
        runs_root=Path("/tmp/runs"),
    )
    assert cpu_env["JAX_PLATFORMS"] == "cpu"
    assert "kinuv-venv-recovery" in cpu_env["KINUV_VENV"]
    with pytest.raises(ValueError):
        require_pinned_gpu(1, None, None)
    with pytest.raises(ValueError):
        require_pinned_gpu(1, 4, None)
    with pytest.raises(ValueError):
        require_pinned_gpu(1, None, 16)
    require_pinned_gpu(1, 4, 16)
    require_pinned_gpu(None, None, None)
    with pytest.raises(ValueError):
        headless_job_env(
            run_id="x",
            galaxy="KGAS066",
            kind="nuts-gpu",
            gpu=1,
            venv="/arc/home/thbrown/kinuv-venv-recovery",
        )


def test_gpu_worker_requires_chain_id_and_skips_status():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    worker = (repo / "scripts" / "run_kgas066_nuts_headless.py").read_text()
    assert "KINUV_CHAIN_ID" in worker
    assert "GPU / nuts-gpu requires KINUV_CHAIN_ID" in worker
    assert "pending_merge" in worker
    merge = (repo / "scripts" / "merge_nuts_chains.py").read_text()
    assert "ess_min=400.0" in merge
    assert "ARTIFACT_G3_REL" in merge
    launcher = (repo / "scripts" / "launch_headless.py").read_text()
    assert "DEFAULT_GPU_IMAGE" in launcher
    assert "require_pinned_gpu" in launcher


def test_job_status_gpu_does_not_patch(tmp_path, monkeypatch):
    from kinuv.runner.status_md import write_job_status_md

    path = tmp_path / "docs" / "architecture" / "STATUS.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "---\npending: []\n---\n\n## Agent Run Status\n\n"
        "* **Phase:** leftover identity landed; approaching NUTS Running (`xgepg7qy`)\n"
        "* **Last Action:** old\n* **Decisions Made:** old\n"
        "* **Blockers / Gates:** old\n* **Next Step:** Wait for `xgepg7qy`\n\n"
        "# Architecture mailbox\n\nkeep.\n"
    )
    monkeypatch.setenv("KINUV_REPO", str(tmp_path))
    got = write_job_status_md(
        run_id="KGAS066-t-nuts-gpu-c1",
        session_id="abc",
        state="SUCCEEDED",
        mixing_pass=True,
        sampler="nuts",
        elapsed_s=1.0,
        kind="nuts-gpu",
    )
    assert got is None
    out = path.read_text()
    assert "xgepg7qy" in out
    assert "2026-08-30-g3-nuts" not in out
    assert "keep." in out


def test_merge_tenx_constant():
    from kinuv.runner.kind import SERIAL_CPU_WALL_S, TENX_WALL_S

    assert abs(SERIAL_CPU_WALL_S - 17440.032) < 1e-3
    assert abs(TENX_WALL_S - 1744.0032) < 1e-3

