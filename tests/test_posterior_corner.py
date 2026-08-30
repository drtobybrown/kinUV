"""Corner plotter provenance: refuse laplace_mh and S2 interval tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kinuv.diagnostics.figures import plot_posterior_corner
from kinuv.scratch import kinuv_scratch_root

REPO = Path(__file__).resolve().parents[1]
S2_JSON = REPO / "docs/reviews/artifacts/2026-08-29-s2/s2_mock_mcmc.json"


def test_scratch_root_is_not_arc():
    root = kinuv_scratch_root()
    assert "/arc/" not in str(root.resolve()) + "/"
    assert str(root).startswith("/scratch/") or str(root).startswith("/tmp/")


def test_s2_json_is_laplace_mh_and_corner_raises():
    rec = json.loads(S2_JSON.read_text())
    assert rec["sampler"] == "laplace_mh"
    with pytest.raises(ValueError, match="nuts"):
        plot_posterior_corner(rec, Path("/tmp/nope.png"))
    with pytest.raises(ValueError, match="nuts"):
        plot_posterior_corner({"sampler": rec["sampler"], "intervals": rec["intervals"]}, Path("/tmp/nope.png"))
    with pytest.raises(ValueError, match="draws"):
        plot_posterior_corner(
            {"sampler": "nuts", "intervals": rec["intervals"]},
            Path("/tmp/nope.png"),
        )
    with pytest.raises(ValueError, match="path"):
        plot_posterior_corner(S2_JSON, Path("/tmp/nope.png"))


def test_synthetic_nuts_draws_write_png(tmp_path):
    rng = np.random.default_rng(0)
    draws = rng.normal(size=(200, 8))
    out = tmp_path / "corner.png"
    rec = {"sampler": "nuts", "draws": draws}
    got = plot_posterior_corner(
        rec,
        out,
        title="synthetic nuts fixture; not 066; not laplace_mh",
    )
    assert got == out
    assert out.is_file() and out.stat().st_size > 1000
