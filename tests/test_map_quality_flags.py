"""G0 MAP quality flags. No /arc FITS; uses committed leftover npz."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kinuv.diagnostics.flags import map_quality_flags

REPO = Path(__file__).resolve().parents[1]
LEFTOVER = REPO / "docs/reviews/artifacts/2026-08-30-final-fit/leftover_chi2.npz"
FLAGS_SRC = REPO / "src/kinuv/diagnostics/flags.py"

OFFICIAL = {
    "r_t_arcsec": 0.5,
    "pa_deg": 199.72980072503037,
    "delta_chi2": 35552.65225039818,
    "v0_kms": 267.67,
    "gas_sigma_kms": 12.05,
}


def test_flags_source_stays_off_the_fitter():
    imports = [
        line
        for line in FLAGS_SRC.read_text().splitlines()
        if line.startswith("import ") or line.startswith("from ")
    ]
    blob = "\n".join(imports)
    assert "run_stage_a_map" not in blob
    assert "kinuv.infer" not in blob
    assert "matplotlib" not in blob


def test_official_066_flags():
    assert LEFTOVER.is_file()
    flags = map_quality_flags(OFFICIAL, leftover_npz=LEFTOVER)
    assert flags["r_t_at_floor"] is True
    assert flags["beats_zero"] is True
    assert flags["delta_chi2_vs_zero_fail"] is False
    assert flags["delta_chi2"] > 3.0e4
    assert abs(flags["delta_chi2"] - 35552.65225039818) < 1.0
    assert flags["pa_alias"] is False
    assert flags["i_held_fixed"] is True
    assert flags["h_z_in_model"] is False
    assert flags["axisym_assumed"] is True
    assert flags["leftover_chi2_structured"] is True
    assert flags["leftover_vel_span"] > flags["leftover_uv_span"]
    assert flags["quote_inner_slope"] is False


def test_off_floor_rt_does_not_quote_inner_slope_while_leftover_structured():
    rec = dict(OFFICIAL)
    rec["r_t_arcsec"] = 0.224
    flags = map_quality_flags(rec, leftover_npz=LEFTOVER)
    assert flags["r_t_at_floor"] is False
    assert flags["leftover_chi2_structured"] is True
    assert flags["quote_inner_slope"] is False


def test_unmeasured_leftover_does_not_quote_inner_slope():
    rec = dict(OFFICIAL)
    rec["r_t_arcsec"] = 0.224
    flags = map_quality_flags(rec)
    assert flags["quote_inner_slope"] is False


def test_leftover_json_resolves_sibling_npz():
    flags = map_quality_flags(
        OFFICIAL,
        leftover_npz=LEFTOVER.with_suffix(".json"),
    )
    assert flags["leftover_chi2_structured"] is True


def test_pa_21p9_fires_alias():
    rec = dict(OFFICIAL)
    rec["pa_deg"] = 21.9
    flags = map_quality_flags(rec)
    assert flags["pa_alias"] is True
    assert flags["leftover_chi2_structured"] is False
    assert flags["beats_zero"] is True


def test_uv_bowl_does_not_fire_velocity_structure():
    rng = np.random.default_rng(0)
    b = np.linspace(10.0, 400.0, 881)
    row = 400.0 * np.exp(-b / 40.0) + rng.normal(0.0, 1.0, size=881)
    chan = 1775.0 + rng.normal(0.0, 5.0, size=95)
    flags = map_quality_flags(
        OFFICIAL,
        leftover_npz={"baseline_m": b, "chi2_row": row, "chi2_chan": chan},
    )
    assert flags["leftover_chi2_structured"] is False
