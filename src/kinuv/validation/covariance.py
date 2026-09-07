"""Grouped line-free C0/C1 covariance selection for native visibilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from kinuv.io.vis import NativeVisTable, require_s2_provenance
from kinuv.validation.groups import GroupedVisibilityFolds


@dataclass(frozen=True)
class CovarianceParameters:
    """Scale and adjacent-channel correlation for one MS stratum."""

    scale: float
    rho: float
    n_complex: int


@dataclass(frozen=True)
class CovarianceSelection:
    """Grouped predictive comparison between diagonal C0 and AR(1) C1."""

    selected: str
    mean_loglike_advantage_per_complex: float
    standard_error_per_complex: float
    group_advantages_per_complex: np.ndarray
    group_ids: np.ndarray


@dataclass(frozen=True)
class _AR1Statistics:
    start_power: float
    previous_power: float
    current_power: float
    cross_power: float
    n_complex: int
    n_adjacent: int
    log_weight: float


def whitened_innovations(
    residual, weights, line_free_mask, row_mask, parameters: CovarianceParameters
) -> np.ndarray:
    """Return complex AR innovations on the native row/channel grid."""
    residual = np.asarray(residual, dtype=np.complex128)
    weights = np.asarray(weights, dtype=np.float64)
    channels = np.asarray(line_free_mask, dtype=bool).ravel()
    rows = np.asarray(row_mask, dtype=bool).ravel()
    x = residual[rows]
    w = weights[rows]
    good = (
        channels[None, :]
        & np.isfinite(x.real)
        & np.isfinite(x.imag)
        & np.isfinite(w)
        & (w > 0.0)
    )
    y = np.sqrt(np.where(good, w, 0.0)) * x
    starts = good.copy()
    starts[:, 1:] &= ~good[:, :-1]
    adjacent = good[:, 1:] & good[:, :-1]
    output = np.full(x.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    output[starts] = y[starts] / np.sqrt(parameters.scale)
    later = (y[:, 1:] - parameters.rho * y[:, :-1])
    later /= np.sqrt(parameters.scale * (1.0 - parameters.rho**2))
    output_later = output[:, 1:]
    output_later[adjacent] = later[adjacent]
    return output


def whitening_diagnostics(innovations) -> dict[str, float | bool | int]:
    """Evaluate the frozen mean, variance, and lag-one whitening gates."""
    z = np.asarray(innovations, dtype=np.complex128)
    if z.ndim != 2:
        raise ValueError("innovations must retain native row and channel axes")
    good = np.isfinite(z.real) & np.isfinite(z.imag)
    values = np.concatenate([z.real[good], z.imag[good]])
    if values.size < 4:
        raise ValueError("whitening diagnostics require at least four finite values")
    mean = float(np.mean(values))
    variance = float(np.var(values, ddof=1))
    mean_limit = float(3.0 / np.sqrt(values.size))
    variance_limit = float(max(0.05, 3.0 * np.sqrt(2.0 / values.size)))
    adjacent = good[:, 1:] & good[:, :-1]
    lag_real = float(np.corrcoef(z.real[:, :-1][adjacent], z.real[:, 1:][adjacent])[0, 1])
    lag_imag = float(np.corrcoef(z.imag[:, :-1][adjacent], z.imag[:, 1:][adjacent])[0, 1])
    lag_one = max(abs(lag_real), abs(lag_imag))
    lag_pairs = 2 * int(np.sum(adjacent))
    if lag_pairs < 2:
        raise ValueError("whitening lag diagnostic requires adjacent valid channels")
    lag_limit = float(max(0.05, 3.0 / np.sqrt(lag_pairs)))
    return {
        "n_component": int(values.size),
        "mean": mean,
        "mean_abs_limit": mean_limit,
        "mean_pass": abs(mean) <= mean_limit,
        "variance": variance,
        "variance_abs_error_limit": variance_limit,
        "variance_pass": abs(variance - 1.0) <= variance_limit,
        "lag_one_real": lag_real,
        "lag_one_imag": lag_imag,
        "lag_one_max_abs": lag_one,
        "lag_one_abs_limit": lag_limit,
        "lag_one_pass": lag_one <= lag_limit,
    }


def _stratum_rows(table: NativeVisTable) -> dict[tuple[int, ...], np.ndarray]:
    keys = np.column_stack(
        [
            table.observation_id,
            table.array_id,
            table.field_id,
            table.data_desc_id,
        ]
    ).astype(np.int64, copy=False)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    return {
        tuple(int(value) for value in key): inverse == index
        for index, key in enumerate(unique)
    }


def _ar1_statistics(residual, weights, line_free_mask, row_mask):
    residual = np.asarray(residual, dtype=np.complex128)
    weights = np.asarray(weights, dtype=np.float64)
    channels = np.asarray(line_free_mask, dtype=bool).ravel()
    rows = np.asarray(row_mask, dtype=bool).ravel()
    if residual.shape != weights.shape:
        raise ValueError("residual and weights must have the same shape")
    if channels.shape != (residual.shape[1],) or rows.shape != (residual.shape[0],):
        raise ValueError("line-free and row masks must match the visibility axes")
    x = residual[rows]
    w = weights[rows]
    good = (
        channels[None, :]
        & np.isfinite(x.real)
        & np.isfinite(x.imag)
        & np.isfinite(w)
        & (w > 0.0)
    )
    count = int(np.sum(good))
    if count == 0:
        raise ValueError("no positive-weight line-free cells in covariance subset")
    y = np.sqrt(np.where(good, w, 0.0)) * x
    starts = good.copy()
    starts[:, 1:] &= ~good[:, :-1]
    adjacent = good[:, 1:] & good[:, :-1]
    previous = y[:, :-1][adjacent]
    current = y[:, 1:][adjacent]
    return _AR1Statistics(
        start_power=float(np.sum(np.abs(y[starts]) ** 2, dtype=np.float64)),
        previous_power=float(np.sum(np.abs(previous) ** 2, dtype=np.float64)),
        current_power=float(np.sum(np.abs(current) ** 2, dtype=np.float64)),
        cross_power=float(
            np.sum(np.real(current * np.conjugate(previous)), dtype=np.float64)
        ),
        n_complex=count,
        n_adjacent=int(np.sum(adjacent)),
        log_weight=2.0 * float(np.sum(np.log(w[good]), dtype=np.float64)),
    )


def _ar1_terms(statistics: _AR1Statistics, rho):
    if not -0.5 <= float(rho) <= 0.5:
        raise ValueError("rho must lie within the registered [-0.5, 0.5] bound")
    denominator = 1.0 - float(rho) ** 2
    q_adjacent = (
        statistics.current_power
        + float(rho) ** 2 * statistics.previous_power
        - 2.0 * float(rho) * statistics.cross_power
    ) / denominator
    return (
        statistics.start_power + q_adjacent,
        statistics.n_complex,
        2 * statistics.n_complex,
        2.0 * statistics.n_adjacent * np.log(denominator),
        statistics.log_weight,
    )


def _profiled_nll(statistics, rho):
    q, n_complex, n_component, logdet_correlation, log_weight = _ar1_terms(
        statistics, rho
    )
    scale = q / n_component
    if not np.isfinite(scale) or scale <= 0.0:
        return np.inf, scale, n_complex
    nll = 0.5 * (
        n_component * (1.0 + np.log(2.0 * np.pi * scale))
        + logdet_correlation
        - log_weight
    )
    return float(nll), float(scale), int(n_complex)


def fit_covariance_stratum(
    residual,
    weights,
    line_free_mask,
    row_mask,
    *,
    model: str,
) -> CovarianceParameters:
    """Fit C0 or bounded spectral AR(1) C1 by profiled Gaussian likelihood."""
    statistics = _ar1_statistics(residual, weights, line_free_mask, row_mask)
    if model == "C0":
        _, scale, n_complex = _profiled_nll(statistics, 0.0)
        return CovarianceParameters(scale=scale, rho=0.0, n_complex=n_complex)
    if model != "C1":
        raise ValueError("model must be 'C0' or 'C1'")

    def objective(rho):
        return _profiled_nll(statistics, float(rho))[0]

    fit = minimize_scalar(
        objective,
        bounds=(-0.5, 0.5),
        method="bounded",
        options={"xatol": 1.0e-6},
    )
    if not fit.success:
        raise RuntimeError(f"C1 covariance optimization failed: {fit.message}")
    _, scale, n_complex = _profiled_nll(statistics, float(fit.x))
    return CovarianceParameters(scale=scale, rho=float(fit.x), n_complex=n_complex)


def fit_covariance_by_stratum(
    table: NativeVisTable,
    line_free_mask,
    *,
    model: str,
    residual=None,
    row_mask=None,
) -> dict[tuple[int, ...], CovarianceParameters]:
    """Fit the selected covariance independently in each observation stratum."""
    require_s2_provenance(table)
    values = table.vis if residual is None else np.asarray(residual)
    selected_rows = (
        np.ones(table.vis.shape[0], dtype=bool)
        if row_mask is None
        else np.asarray(row_mask, dtype=bool).ravel()
    )
    return {
        key: fit_covariance_stratum(
            values,
            table.weights,
            line_free_mask,
            selected_rows & rows,
            model=model,
        )
        for key, rows in _stratum_rows(table).items()
        if np.any(selected_rows & rows)
    }


def _fixed_nll(residual, weights, line_free_mask, row_mask, parameters):
    statistics = _ar1_statistics(residual, weights, line_free_mask, row_mask)
    return _fixed_nll_from_statistics(statistics, parameters)


def _fixed_nll_from_statistics(statistics, parameters):
    q, n_complex, n_component, logdet_correlation, log_weight = _ar1_terms(
        statistics,
        parameters.rho,
    )
    nll = 0.5 * (
        q / parameters.scale
        + n_component * np.log(2.0 * np.pi * parameters.scale)
        + logdet_correlation
        - log_weight
    )
    return float(nll), int(n_complex)


def select_grouped_covariance(
    table: NativeVisTable,
    folds: GroupedVisibilityFolds,
    line_free_mask,
    *,
    residual=None,
    embargo_s: float = 0.0,
) -> CovarianceSelection:
    """Choose C0/C1 with grouped predictive likelihood and one-SE parsimony."""
    require_s2_provenance(table)
    values = table.vis if residual is None else np.asarray(residual)
    strata = _stratum_rows(table)
    advantages = []
    group_ids = []
    for fold_id in range(folds.n_folds):
        train = folds.training_mask(fold_id, embargo_s=embargo_s)
        fitted = {"C0": {}, "C1": {}}
        for key, stratum in strata.items():
            train_stratum = train & stratum
            if np.any(train_stratum):
                for model in ("C0", "C1"):
                    fitted[model][key] = fit_covariance_stratum(
                        values,
                        table.weights,
                        line_free_mask,
                        train_stratum,
                        model=model,
                    )
        for group in folds.groups:
            if group.fold_id != fold_id:
                continue
            held = folds.row_group_id == group.group_id
            key = (
                group.observation_id,
                group.array_id,
                group.field_id,
                group.data_desc_id,
            )
            if key not in fitted["C0"] or key not in fitted["C1"]:
                raise ValueError(
                    "held-out group has no training rows in its covariance stratum"
                )
            held_statistics = _ar1_statistics(
                values,
                table.weights,
                line_free_mask,
                held,
            )
            nll0, n_complex = _fixed_nll_from_statistics(
                held_statistics, fitted["C0"][key]
            )
            nll1, n_complex_1 = _fixed_nll_from_statistics(
                held_statistics, fitted["C1"][key]
            )
            if n_complex != n_complex_1:
                raise RuntimeError("C0/C1 scored different held-out cells")
            advantages.append((nll0 - nll1) / n_complex)
            group_ids.append(group.group_id)
    advantage = np.asarray(advantages, dtype=np.float64)
    if advantage.size < 2:
        raise ValueError("one-standard-error selection requires at least two groups")
    mean = float(np.mean(advantage))
    standard_error = float(np.std(advantage, ddof=1) / np.sqrt(advantage.size))
    selected = "C1" if mean - standard_error > 0.0 else "C0"
    return CovarianceSelection(
        selected=selected,
        mean_loglike_advantage_per_complex=mean,
        standard_error_per_complex=standard_error,
        group_advantages_per_complex=advantage,
        group_ids=np.asarray(group_ids, dtype=np.int64),
    )
