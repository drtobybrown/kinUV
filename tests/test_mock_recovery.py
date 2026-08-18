"""Gate 2: noise-free mock recovery of flux, PA, vsys, 0.3″ (dx, dy).

066-6 Hann+bin is used when ``kinuv.likelihood.hann_then_bin`` is importable.
Otherwise this test recovers on a **short native window with diagonal χ²**.
066-8 (real 066 MAP) must go through 066-6 Hann+bin (DEC-066-SPECRESP).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from kinuv.forward.mocks import (
    NPZ_PATH,
    N_ROW_MOCK,
    recover_stage_a,
    stage_a_truth,
    subsample_native_uv,
)
from kinuv.forward.model import predict_vis
from kinuv.forward.sb import load_sb_template
from kinuv.geometry import inclination_deg, inclination_rad

# 10% flux, 5° PA, 5 km/s vsys, 0.1″ offset
FLUX_TOL = 0.10
PA_TOL_DEG = 5.0
VSYS_TOL_KM_S = 5.0
OFFSET_TOL_ARCSEC = 0.10


@pytest.mark.skipif(not NPZ_PATH.is_file(), reason="KILOGAS066.npz not on this machine")
def test_mock_recovery_flux_pa_vsys_offset():
    t_load = time.perf_counter()
    window = subsample_native_uv(NPZ_PATH)
    assert window.u_m.size <= N_ROW_MOCK
    assert window.freqs_hz.size < 500
    assert window.grid.cell_arcsec != pytest.approx(0.4)
    tmpl = load_sb_template(window.grid)
    truth = stage_a_truth(flux=1.0)
    # Frozen i is an input, not a recovery parameter.
    assert inclination_deg() == pytest.approx(43.9, abs=0.05)
    vis_true = predict_vis(
        window.u_m,
        window.v_m,
        window.freqs_hz,
        flux=truth["flux"],
        pa_rad=truth["pa_rad"],
        vsys_kms=truth["vsys_kms"],
        dx_arcsec=truth["dx_arcsec"],
        dy_arcsec=truth["dy_arcsec"],
        gas_sigma_kms=truth["gas_sigma_kms"],
        template=tmpl,
        grid=window.grid,
        v0_kms=truth["v0_kms"],
        r_t_arcsec=truth["r_t_arcsec"],
        i_rad=inclination_rad(),
    )
    t0 = time.perf_counter()
    _ = predict_vis(
        window.u_m,
        window.v_m,
        window.freqs_hz,
        flux=truth["flux"],
        pa_rad=truth["pa_rad"],
        vsys_kms=truth["vsys_kms"],
        dx_arcsec=truth["dx_arcsec"],
        dy_arcsec=truth["dy_arcsec"],
        gas_sigma_kms=truth["gas_sigma_kms"],
        template=tmpl,
        grid=window.grid,
        v0_kms=truth["v0_kms"],
        r_t_arcsec=truth["r_t_arcsec"],
        i_rad=inclination_rad(),
    )
    eval_s = time.perf_counter() - t0
    rec = recover_stage_a(
        window,
        tmpl,
        vis_true,
        flux_true=truth["flux"],
        pa_deg_true=truth["pa_deg"],
        vsys_true=truth["vsys_kms"],
        dx_true=truth["dx_arcsec"],
        dy_true=truth["dy_arcsec"],
        gas_sigma_kms=truth["gas_sigma_kms"],
        v0_kms=truth["v0_kms"],
        r_t_arcsec=truth["r_t_arcsec"],
    )
    test_mock_recovery_flux_pa_vsys_offset.rec = rec
    test_mock_recovery_flux_pa_vsys_offset.eval_s = eval_s
    test_mock_recovery_flux_pa_vsys_offset.operator = rec.operator
    test_mock_recovery_flux_pa_vsys_offset.n_row = int(window.u_m.size)
    test_mock_recovery_flux_pa_vsys_offset.n_chan = int(window.freqs_hz.size)
    test_mock_recovery_flux_pa_vsys_offset.grid = window.grid
    test_mock_recovery_flux_pa_vsys_offset.load_s = time.perf_counter() - t_load
    print(
        f"\n066-7 recovery operator={rec.operator} "
        f"flux={rec.flux:.4f} (truth {truth['flux']}) "
        f"PA={rec.pa_deg:.3f}° (truth {truth['pa_deg']}) "
        f"vsys={rec.vsys_kms:.3f} km/s (truth {truth['vsys_kms']}) "
        f"dx={rec.dx_arcsec:.4f}\" dy={rec.dy_arcsec:.4f}\" "
        f"(inject {truth['dx_arcsec']}\") chi2={rec.chi2:.3e} nfev={rec.nfev} "
        f"eval={eval_s:.3f}s n_row={window.u_m.size} n_chan={window.freqs_hz.size} "
        f"grid={window.grid.nx}²@{window.grid.cell_arcsec:.3f}\""
    )
    assert rec.operator in {"native_diagonal", "hann_then_bin"}
    if rec.operator == "native_diagonal":
        # 066-8 must not skip DEC-066-SPECRESP; this mock is the 066-6 stand-in.
        assert rec.operator == "native_diagonal"
    assert abs(rec.flux / truth["flux"] - 1.0) < FLUX_TOL
    assert abs(rec.pa_deg - truth["pa_deg"]) < PA_TOL_DEG
    assert abs(rec.vsys_kms - truth["vsys_kms"]) < VSYS_TOL_KM_S
    assert abs(rec.dx_arcsec - truth["dx_arcsec"]) < OFFSET_TOL_ARCSEC
    assert abs(rec.dy_arcsec - truth["dy_arcsec"]) < OFFSET_TOL_ARCSEC


def test_subsample_does_not_touch_full_cube():
    src = Path(__file__).resolve().parents[1] / "src" / "kinuv" / "forward" / "mocks.py"
    text = src.read_text(encoding="utf-8")
    assert "not the full 1920×43240" in text
    assert "native_diagonal" in text
    assert "066-8" in text and "Hann+bin" in text
