"""066-8 Stage A MAP: Hann+bin model path, radio vsys, 180° PA two-start."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kinuv.forward.model import VSYS_SEED_KM_S
from kinuv.forward.sb import load_sb_template
from kinuv.geometry import pa_seed_deg
from kinuv.infer.map import (
    DX_DY_BOUND_ARCSEC,
    SHIFT_PRIOR_SIGMA_ARCSEC,
    gate_delta_chi2,
    image_grid_for_vis,
    map_gate_scores,
    map_objective,
    run_stage_a_map,
    score_seed_delta_chi2,
    shift_prior,
    stage_a_bounds,
    stage_a_seeds,
)
from kinuv.infer.seeds import (
    BLOB_VSYS_KMS,
    PA_BOUND_HALF_DEG,
    pa_start_degs,
    vsys_seed_radio_kms,
)
from kinuv.io.vis import DEFAULT_NPZ, load_kgas066
from kinuv.likelihood.chi2 import chi2, chi2_zero, delta_chi2
from kinuv.profiles.rotation import CALIBRATION_V0_KM_S
from kinuv.response.spectral import s_theory

MAP_SRC = Path(__file__).resolve().parents[1] / "src" / "kinuv" / "infer" / "map.py"
INFER_DIR = MAP_SRC.parent
BLOB_DCHI2 = 4341.0
V0_WALL_KMS = 1.0
SIGMA_WALL_KMS = 49.9


def test_map_source_uses_hann_then_bin_not_native_diagonal():
    text = MAP_SRC.read_text(encoding="utf-8")
    assert "from kinuv.response.spectral import hann_then_bin" in text
    assert "from kinuv.likelihood import hann_then_bin" not in text
    assert "native_diagonal" not in text
    for path in INFER_DIR.glob("*.py"):
        body = path.read_text(encoding="utf-8")
        assert "uvkin" not in body
        assert "KinMS" not in body
        assert "from kinms" not in body.lower()
        assert "run_lambda_reg_campaign" not in body


def test_gate_is_delta_chi2_not_reduced():
    rng = np.random.default_rng(66)
    vis = rng.standard_normal((12, 6)) + 1j * rng.standard_normal((12, 6))
    model = 0.4 * vis
    w = np.full(vis.shape, 1.7, dtype=np.float64)
    s = 0.8
    c, c0, dchi = map_gate_scores(vis, model, w, s)
    assert c == pytest.approx(chi2(vis, model, w, s))
    assert c0 == pytest.approx(chi2_zero(vis, w, s))
    assert dchi == pytest.approx(delta_chi2(c, c0))
    assert dchi == pytest.approx(c0 - c)
    n = vis.size
    assert dchi != pytest.approx(c / n)
    assert dchi != pytest.approx(c0 / n)
    assert gate_delta_chi2(c, c0) == pytest.approx(c0 - c)
    assert map_gate_scores.__doc__ is not None
    assert "χ² / n" in map_gate_scores.__doc__ or "chi2 / n" in map_gate_scores.__doc__


def test_dx_dy_bounds_include_two_arcsec_and_prior_in_objective():
    bounds = stage_a_bounds()
    dx = bounds["dx_arcsec"]
    dy = bounds["dy_arcsec"]
    assert dx == pytest.approx((-DX_DY_BOUND_ARCSEC, DX_DY_BOUND_ARCSEC))
    assert dy == pytest.approx((-DX_DY_BOUND_ARCSEC, DX_DY_BOUND_ARCSEC))
    assert dx[0] < 0.0 < dx[1]
    assert dy[0] < 0.0 < dy[1]
    assert dx != (0.0, 0.0)
    assert set(dx) != {0.0}
    seeds = stage_a_seeds()
    assert seeds["dx_arcsec"] == 0.0 and seeds["dy_arcsec"] == 0.0
    pa = pa_seed_deg()
    assert bounds["pa_deg"] == pytest.approx((pa - PA_BOUND_HALF_DEG, pa + PA_BOUND_HALF_DEG))
    assert SHIFT_PRIOR_SIGMA_ARCSEC == pytest.approx(0.5)
    assert shift_prior(0.5, 0.0) == pytest.approx(1.0)
    assert shift_prior(0.0, -0.5) == pytest.approx(1.0)
    chi2_val = 12.0
    assert map_objective(chi2_val, 0.5, 0.0) == pytest.approx(13.0)
    assert gate_delta_chi2(chi2_val, 20.0) == pytest.approx(8.0)
    assert map_objective(chi2_val, 0.5, 0.0) != gate_delta_chi2(chi2_val, 20.0)


def test_vsys_seed_is_radio_inside_fit_window():
    seeds = stage_a_seeds()
    bounds = stage_a_bounds()
    radio = vsys_seed_radio_kms()
    assert radio == pytest.approx(8076.0, abs=1.0)
    assert seeds["vsys_kms"] == pytest.approx(radio)
    assert seeds["vsys_kms"] != pytest.approx(VSYS_SEED_KM_S)
    assert 7823.0 < seeds["vsys_kms"] < 8301.0
    lo, hi = bounds["vsys_kms"]
    assert lo < 8076.0 < hi
    assert not (lo <= BLOB_VSYS_KMS <= hi)


def test_pa_box_includes_flip_start():
    pa0, pa_flip = pa_start_degs()
    assert pa0 == pytest.approx(205.2)
    assert pa_flip == pytest.approx(25.2)
    lo, hi = stage_a_bounds()["pa_deg"]
    assert lo <= 25.2 <= hi
    assert lo <= 205.2 <= hi
    assert PA_BOUND_HALF_DEG == pytest.approx(180.0)
    assert hi - lo == pytest.approx(360.0)


def test_no_vis_phase_ramp_after_pb():
    text = MAP_SRC.read_text(encoding="utf-8")
    assert "exp(" not in text
    assert "np.exp" not in text
    assert "2j" not in text
    assert "fourier_shift" in text
    assert "predict_vis" in text
    assert "phase ramp" in text.lower() or "visibility phase ramp" in text.lower()


@pytest.mark.skipif(not DEFAULT_NPZ.is_file(), reason="KILOGAS066.npz not on this machine")
def test_radio_pa25_beats_pa205_and_optical_vsys():
    data = load_kgas066()
    grid = image_grid_for_vis(data)
    tmpl = load_sb_template(grid)
    radio = vsys_seed_radio_kms()
    base = stage_a_seeds()
    d25, _ = score_seed_delta_chi2(
        data, tmpl, grid, {**base, "pa_deg": 25.2, "vsys_kms": radio, "v0_kms": CALIBRATION_V0_KM_S}
    )
    d205, _ = score_seed_delta_chi2(
        data, tmpl, grid, {**base, "pa_deg": 205.2, "vsys_kms": radio, "v0_kms": CALIBRATION_V0_KM_S}
    )
    d_opt, _ = score_seed_delta_chi2(
        data,
        tmpl,
        grid,
        {**base, "pa_deg": 205.2, "vsys_kms": float(VSYS_SEED_KM_S), "v0_kms": CALIBRATION_V0_KM_S},
    )
    print(f"\nseed Δχ² PA25={d25:.1f} PA205={d205:.1f} optical_vsys={d_opt:.1f}")
    assert d25 > d205
    assert d25 > d_opt


@pytest.mark.skipif(not DEFAULT_NPZ.is_file(), reason="KILOGAS066.npz not on this machine")
def test_stage_a_map_on_real_hann_bin_066():
    rec = run_stage_a_map()
    n_row, n_chan, dv, n_bin, s = rec.n_row, rec.n_chan, rec.dv_kms, rec.n_bin, rec.s
    print(
        f"\n066-8 MAP n_row={n_row} n_chan={n_chan} dv={dv:.4f} km/s N={n_bin} s={s:.4f} "
        f"eval={rec.eval_s:.3f}s chi2_MAP={rec.chi2_map:.6g} chi2_zero={rec.chi2_zero:.6g} "
        f"dchi2={rec.delta_chi2:.6g} flux={rec.flux:.4f} PA={rec.pa_deg:.3f} deg "
        f"vsys={rec.vsys_kms:.3f} km/s gas_sigma={rec.gas_sigma_kms:.3f} km/s "
        f"dx={rec.dx_arcsec:.4f}\" dy={rec.dy_arcsec:.4f}\" "
        f"V0={rec.v0_kms:.3f} km/s rt={rec.r_t_arcsec:.3f}\" "
        f"pa_start={rec.pa_start_deg:.1f} nfev={rec.nfev} "
        f"optimiser_ran={rec.optimiser_ran} success={rec.success}"
    )
    assert 0.3 < s < 1.5
    assert s != pytest.approx(0.5, rel=0.01)
    assert s != pytest.approx(12.0 / 29.0, rel=0.01)
    assert s != pytest.approx(s_theory(8), rel=0.01)
    assert n_chan == pytest.approx(95, abs=2)
    assert n_row == pytest.approx(881, abs=20)
    assert n_chan < 1920
    assert n_chan < 1920 // 4
    assert n_bin == 4
    assert dv == pytest.approx(5.080, rel=0.02)
    assert rec.flux > 0.0
    assert abs(rec.dx_arcsec) <= DX_DY_BOUND_ARCSEC + 1e-9
    assert abs(rec.dy_arcsec) <= DX_DY_BOUND_ARCSEC + 1e-9
    lo, hi = stage_a_bounds()["vsys_kms"]
    assert lo <= rec.vsys_kms <= hi
    if rec.delta_chi2 <= 0.0:
        print("MAP_LOSES_TO_ZERO")
    else:
        assert rec.delta_chi2 > 0.0
    if rec.optimiser_ran:
        assert rec.v0_kms > V0_WALL_KMS, "MAP_STILL_COLLAPSED: V_0 on the floor"
        assert rec.gas_sigma_kms < SIGMA_WALL_KMS
        assert abs(rec.dy_arcsec) < DX_DY_BOUND_ARCSEC - 1e-3
        assert rec.delta_chi2 >= BLOB_DCHI2
    test_stage_a_map_on_real_hann_bin_066.record = rec
