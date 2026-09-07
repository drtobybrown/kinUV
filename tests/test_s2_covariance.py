"""C0/C1 covariance fitting and grouped predictive selection."""

from __future__ import annotations

import numpy as np

from kinuv.io.vis import MS2KINUV_PROVENANCE_SCHEMA_VERSION, NativeVisTable
from kinuv.validation.covariance import (
    fit_covariance_stratum,
    select_grouped_covariance,
    whitened_innovations,
    whitening_diagnostics,
)
from kinuv.validation.groups import build_grouped_visibility_folds


def _ar1_complex(rng, n_row, n_chan, rho):
    values = np.empty((n_row, n_chan), dtype=np.complex128)
    scale = np.sqrt(1.0 - rho**2)
    for component in ("real", "imag"):
        noise = rng.normal(size=(n_row, n_chan))
        series = np.empty_like(noise)
        series[:, 0] = noise[:, 0]
        for channel in range(1, n_chan):
            series[:, channel] = rho * series[:, channel - 1] + scale * noise[:, channel]
        if component == "real":
            values.real = series
        else:
            values.imag = series
    return values


def _grouped_table(residual):
    n_row, n_chan = residual.shape
    rows_per_scan = 20
    scan = np.repeat(np.arange(n_row // rows_per_scan), rows_per_scan)
    time = scan.astype(float) * 100.0 + np.tile(
        np.arange(rows_per_scan), scan.max() + 1
    )
    ant1 = np.arange(n_row, dtype=np.int64) % 8
    ant2 = (ant1 + 1) % 9
    return NativeVisTable(
        u_m=np.zeros(n_row),
        v_m=np.zeros(n_row),
        vis=residual,
        weights=np.ones((n_row, n_chan)),
        freqs=np.arange(n_chan, dtype=float),
        time=time,
        baseline=(ant1 << np.int64(32)) | ant2,
        phase_dir_rad=np.zeros(2),
        schema=MS2KINUV_PROVENANCE_SCHEMA_VERSION,
        uv_ref_hz=None,
        row_id=np.arange(n_row),
        antenna1=ant1,
        antenna2=ant2,
        scan_number=scan,
        observation_id=np.zeros(n_row, dtype=int),
        array_id=np.zeros(n_row, dtype=int),
        state_id=np.zeros(n_row, dtype=int),
        field_id=np.zeros(n_row, dtype=int),
        data_desc_id=np.zeros(n_row, dtype=int),
        time_centroid=time,
        interval=np.ones(n_row),
        visibility_unit="Jy",
        weight_convention="ms_weight_equals_2_over_complex_noise_variance",
        extraction={
            "source_row_identity_preserved": True,
            "row_averaging": "none",
            "channel_averaging": "none",
        },
    )


def test_c0_scale_uses_both_complex_components():
    residual = np.array([[1.0 + 2.0j, 3.0 + 4.0j]])
    weights = np.ones((1, 2))
    fit = fit_covariance_stratum(
        residual,
        weights,
        np.ones(2, dtype=bool),
        np.ones(1, dtype=bool),
        model="C0",
    )
    assert fit.scale == 7.5
    assert fit.rho == 0.0
    assert fit.n_complex == 2


def test_c1_recovers_adjacent_channel_correlation():
    rng = np.random.default_rng(20260907)
    residual = _ar1_complex(rng, 600, 24, 0.3)
    fit = fit_covariance_stratum(
        residual,
        np.ones(residual.shape),
        np.ones(residual.shape[1], dtype=bool),
        np.ones(residual.shape[0], dtype=bool),
        model="C1",
    )
    assert abs(fit.rho - 0.3) < 0.03
    assert abs(fit.scale - 1.0) < 0.05
    innovations = whitened_innovations(
        residual,
        np.ones(residual.shape),
        np.ones(residual.shape[1], dtype=bool),
        np.ones(residual.shape[0], dtype=bool),
        fit,
    )
    diagnostics = whitening_diagnostics(innovations)
    assert diagnostics["mean_pass"]
    assert diagnostics["variance_pass"]
    assert diagnostics["lag_one_pass"]


def test_c1_respects_registered_rho_bound():
    rng = np.random.default_rng(17)
    residual = _ar1_complex(rng, 400, 16, 0.8)
    fit = fit_covariance_stratum(
        residual,
        np.ones(residual.shape),
        np.ones(residual.shape[1], dtype=bool),
        np.ones(residual.shape[0], dtype=bool),
        model="C1",
    )
    assert 0.49 < fit.rho <= 0.5


def test_grouped_predictive_selection_prefers_real_ar1_signal():
    rng = np.random.default_rng(4407)
    table = _grouped_table(_ar1_complex(rng, 500, 20, 0.35))
    folds = build_grouped_visibility_folds(table, n_folds=5)
    result = select_grouped_covariance(
        table, folds, np.ones(20, dtype=bool)
    )
    assert result.selected == "C1"
    assert result.mean_loglike_advantage_per_complex > result.standard_error_per_complex
