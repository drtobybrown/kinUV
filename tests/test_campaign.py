"""Stub OSCMETRIC loop (066-12). Does not run real uv."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from kinuv.infer.campaign import (
    _prior_checkpoint,
    add_complex_noise,
    calibrate_lambda_reg,
)
from kinuv.infer.stage_b import StageBResult


def _fake_stage_b(*_args, **kwargs):
    lam = float(kwargs["lam_reg"])
    omega = 0.05 if lam >= 0.1 else 0.9
    return StageBResult(
        v_knots_kms=(100.0,) * 7,
        r_knots_arcsec=tuple(np.linspace(0.65, 7.5, 7)),
        lam_reg=lam,
        chi2_map=10.0,
        chi2_zero=20.0,
        delta_chi2=10.0,
        chi2_stage_a=12.0,
        aic_stage_a=16.0,
        aic_stage_b=24.0,
        keep_stage_a=True,
        n_rings=7,
        nfev=1,
        success=True,
        message="stub",
        v0_recovered=200.0,
        r_t_recovered=3.0,
        max_omega=omega,
        dv_kms=5.08,
    )


def test_add_complex_noise_zero_mean_variance():
    rng = np.random.default_rng(1)
    vis = np.zeros((8, 4), dtype=np.complex128)
    w = np.full(vis.shape, 4.0)
    s = 0.5
    noisy = add_complex_noise(vis, w, s, rng)
    mag = noisy.real**2 + noisy.imag**2
    mean_chi = float(np.mean(s * w * mag))
    assert 1.0 < mean_chi < 3.5


def test_calibrate_lambda_reg_stub_early_exit(monkeypatch):
    data = SimpleNamespace(
        vis=np.zeros((881, 95), dtype=np.complex128),
        weights=np.ones((881, 95)),
        s=0.5,
        n_guard=1,
        n_bin=4,
        dv_kms=5.08,
        u_m=np.zeros(881),
        v_m=np.zeros(881),
        freqs_native=np.linspace(1e11, 1.01e11, 99),
        vel_native=np.linspace(8000, 8500, 99),
        weights_native=np.ones((881, 99)),
    )
    monkeypatch.setattr(
        "kinuv.infer.campaign.predict_binned",
        lambda *a, **k: np.zeros((881, 95), dtype=np.complex128),
    )
    monkeypatch.setattr(
        "kinuv.infer.campaign.fit_v0_rt",
        lambda *a, **k: {
            "v0_kms": 200.0,
            "r_t_arcsec": 3.0,
            "chi2": 10.0,
            "chi2_zero": 20.0,
            "delta_chi2": 10.0,
            "nfev": 1,
            "success": True,
            "message": "stub",
        },
    )
    monkeypatch.setattr("kinuv.infer.campaign.run_stage_b_map", _fake_stage_b)
    out = calibrate_lambda_reg(
        data=data,
        template=np.ones((8, 8)),
        grid=object(),
        n_mock=4,
        lambdas=(0.01, 0.1, 1.0),
        n_rings=7,
        seed=1,
        smoke=False,
    )
    assert out["chosen_lambda"] == pytest.approx(0.1)
    assert out["lambdas_tried"] == [0.01, 0.1]
    assert out["omega_mode"] == "residual"


def test_prior_checkpoint_refuses_absolute_omega(tmp_path):
    payload = {
        "n_mock": 4,
        "n_rings": 7,
        "v0_stage_a": [200.0] * 4,
        "rows": [],
    }
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(payload) + "\n")
    assert _prior_checkpoint(tmp_path, 7, 4, False) is None
    payload["omega_mode"] = "absolute"
    path.write_text(json.dumps(payload) + "\n")
    assert _prior_checkpoint(tmp_path, 7, 4, False) is None
    payload["omega_mode"] = "residual"
    path.write_text(json.dumps(payload) + "\n")
    loaded = _prior_checkpoint(tmp_path, 7, 4, False)
    assert loaded is not None
    assert loaded["omega_mode"] == "residual"
