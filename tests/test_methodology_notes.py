"""Grep gate: methodology notes are not a KinMS posterior table."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTES = REPO / "docs/architecture/notes/2026-09-02-kinematic-methodology-review.md"
DIRTY = (
    REPO
    / "docs/reviews/artifacts/2026-09-02-kgas066-leftover-and-modes/dirty-residuals/README.md"
)
FORBID = "Not KinMS. These PNGs are CLEAN-matched cubes of vis models, not vis inversions."


def test_notes_rank_and_not_kinms_posterior():
    text = NOTES.read_text(encoding="utf-8")
    assert text.lstrip().startswith("# Rank")
    assert "Not an ADR" in text
    assert FORBID in text
    assert "| KinMS |" not in text
    assert "|KinMS|" not in text
    lower = text.lower()
    assert "inflow" not in lower
    assert "\\dot{m}" not in lower
    assert FORBID in text.split("## A. Interferometry", 1)[1]


def test_dirty_residuals_readme_opens_not_kinms():
    body = [ln for ln in DIRTY.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert body[0].startswith("#")
    assert body[1] == FORBID


def test_kgas007_script_is_map_only():
    text = (REPO / "scripts/run_kgas007_stage_a_map.py").read_text(encoding="utf-8")
    assert "launch_headless" not in text
    assert "steal_latest" not in text
    assert "point_latest" not in text
    assert "numpyro" not in text
    assert "KINUV_KIND" not in text
    assert "diagnostic_only" in text
    assert "dec_066_target_amended" in text


def test_s3_readme_vis_chi2_is_the_fit():
    path = (
        REPO
        / "docs/reviews/artifacts/2026-09-05-kgas066-s3-image-benchmark/README.md"
    )
    body = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert body[0].startswith("#")
    assert "vis" in body[1].lower() and "fit" in body[1].lower()
    assert "s1" not in body[1].lower()
    assert "s_1" not in path.read_text(encoding="utf-8")
    assert "c_3" not in path.read_text(encoding="utf-8")
