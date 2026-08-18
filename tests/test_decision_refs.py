"""Every @requires id must exist on disk and not be superseded/conflicted."""

from pathlib import Path

import pytest

from kinuv.decisions import load_decision_index, requires

ROOT = Path(__file__).resolve().parents[1] / "docs" / "decisions"


def test_decision_files_have_unique_ids():
    index = load_decision_index(ROOT)
    assert "DEC-066-INDEX" in index
    assert "DEC-066-AGENTS" in index
    for status in index.values():
        assert status in {"accepted", "proposed"}


def test_requires_unknown_id_raises():
    @requires("DEC-DOES-NOT-EXIST")
    def _fn():
        return 1

    with pytest.raises(LookupError, match="unknown"):
        _fn()


def test_requires_accepted_id_calls():
    @requires("DEC-066-GRID")
    def _fn():
        return 7

    assert _fn() == 7
