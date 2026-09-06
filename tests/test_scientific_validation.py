"""Prospective scientific accounting tests."""

from types import SimpleNamespace

import numpy as np
import pytest

from kinuv.infer.nulls import fit_nonrotating_emission
from kinuv.validation import rotation_test_metrics


def test_rotation_gate_requires_fitted_nonrot_and_bootstrap():
    pending = rotation_test_metrics(
        chi2_blank=200.0,
        chi2_nonrot=150.0,
        chi2_rot=100.0,
    )
    assert pending.delta_chi2_blank == pytest.approx(100.0)
    assert pending.delta_chi2_nonrot == pytest.approx(50.0)
    assert pending.status == "pending_bootstrap"
    assert pending.bootstrap_arithmetic_pass is None
    assert pending.scientific_gate_pass is False

    passed = rotation_test_metrics(
        chi2_blank=200.0,
        chi2_nonrot=150.0,
        chi2_rot=100.0,
        bootstrap_exceedances=1,
        bootstrap_trials=199,
    )
    assert passed.bootstrap_p == pytest.approx(0.01)
    assert passed.bootstrap_arithmetic_pass is True
    assert passed.scientific_gate_pass is False
    assert passed.status == "arithmetic_pass_evidence_unverified"


def test_rotation_gate_rejects_short_or_partial_bootstrap():
    with pytest.raises(ValueError, match="pair"):
        rotation_test_metrics(
            chi2_blank=3.0,
            chi2_nonrot=2.0,
            chi2_rot=1.0,
            bootstrap_trials=199,
        )
    with pytest.raises(ValueError, match=">= 199"):
        rotation_test_metrics(
            chi2_blank=3.0,
            chi2_nonrot=2.0,
            chi2_rot=1.0,
            bootstrap_exceedances=0,
            bootstrap_trials=198,
        )


def test_nonrotating_fit_emits_and_fixes_rotation(monkeypatch):
    seen = []

    def fake_predict(_data, params, _template, _grid, *, i_rad, xla):
        seen.append((params["v0_kms"], i_rad, xla))
        return np.full((2, 3), params["flux"], dtype=np.complex128)

    monkeypatch.setattr("kinuv.infer.nulls.predict_binned", fake_predict)
    data = SimpleNamespace(
        vis=np.full((2, 3), 2.0 + 0.0j),
        weights=np.ones((2, 3)),
        s=1.0,
    )
    result = fit_nonrotating_emission(
        data,
        np.ones((2, 2)),
        object(),
        seeds={"flux": 1.0},
        bounds={
            "flux": (1.0e-8, 5.0),
            "vsys_kms": (7900.0, 8300.0),
            "gas_sigma_kms": (2.0, 50.0),
            "dx_arcsec": (-2.0, 2.0),
            "dy_arcsec": (-2.0, 2.0),
        },
        maxiter=3,
    )
    assert result.flux == pytest.approx(2.0, rel=1.0e-3)
    assert result.chi2_nonrot < result.chi2_blank
    assert seen and all(v0 == 0.0 and i_rad == 0.0 for v0, i_rad, _ in seen)
