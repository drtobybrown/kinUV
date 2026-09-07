"""Generic S2 geometry optimization with a frozen spectral covariance.

The likelihood remains entirely in visibility space.  The fitted velocity
amplitude is the observable projected quantity ``u = v_c sin(i)``; inclination
is used for deprojection but intrinsic circular speed is not promoted when the
orientation prior is isotropic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.optimize import minimize

from kinuv.infer.map import predict_binned


S2_PARAMETER_NAMES = (
    "log_flux",
    "pa_rad",
    "vsys_channel_offset",
    "log_gas_sigma",
    "dx_over_bmaj",
    "dy_over_bmaj",
    "u_over_100",
    "cos_inclination",
    "turnover_over_bmaj",
)


@dataclass(frozen=True)
class FitCovariance:
    """Stationary fit-channel AR(1) approximation frozen before fitting."""

    scale: float
    rho: float
    source_scale: float
    source_rho: float
    software_bin: int


@dataclass(frozen=True)
class S2FitResult:
    """One deterministic S2 optimizer result."""

    start_id: int
    fixed_turnover_over_bmaj: float | None
    parameters: dict[str, float]
    objective: float
    chi2: float
    prior: float
    regularization: float
    projected_gradient_inf: float
    nfev: int
    njev: int
    nit: int
    success: bool
    message: str
    boundary_parameters: tuple[str, ...]

    def to_dict(self) -> dict:
        out = asdict(self)
        out["boundary_parameters"] = list(self.boundary_parameters)
        return out


def propagate_equal_weight_ar1(source_scale: float, source_rho: float, n_bin: int) -> FitCovariance:
    """Propagate stationary AR(1) covariance through disjoint mean bins.

    Returned ``scale`` is relative to the summed inverse-variance weight used
    for a binned visibility.  ``rho`` is the correlation between adjacent
    output bins.  Row averaging preserves both quantities for independent
    rows under inverse-variance weighting.
    """

    n = int(n_bin)
    if n < 1:
        raise ValueError("n_bin must be positive")
    rho = float(source_rho)
    if not -1.0 < rho < 1.0:
        raise ValueError("source_rho must lie in (-1, 1)")
    idx = np.arange(n)
    within = rho ** np.abs(idx[:, None] - idx[None, :])
    across = rho ** np.abs(idx[:, None] - (idx[None, :] + n))
    scale_factor = float(np.sum(within) / n)
    rho_out = float(np.sum(across) / np.sum(within))
    return FitCovariance(
        scale=float(source_scale) * scale_factor,
        rho=rho_out,
        source_scale=float(source_scale),
        source_rho=rho,
        software_bin=n,
    )


def correlated_chi2(vis, model, weights, covariance: FitCovariance):
    """AR(1) quadratic form for both components with flags/gaps respected."""

    from kinuv.xp import numpy_or_jax

    xp = numpy_or_jax(vis, model, weights)
    data = xp.asarray(vis)
    pred = xp.asarray(model)
    weight = xp.asarray(weights)
    good = xp.isfinite(data.real) & xp.isfinite(data.imag) & (weight > 0.0)
    residual = xp.sqrt(xp.where(good, weight, 0.0)) * (data - pred)
    starts = good.at[:, 1:].set(good[:, 1:] & ~good[:, :-1]) if hasattr(good, "at") else good.copy()
    if not hasattr(good, "at"):
        starts[:, 1:] = good[:, 1:] & ~good[:, :-1]
    adjacent = good[:, 1:] & good[:, :-1]
    q_start = xp.sum(xp.where(starts, residual.real**2 + residual.imag**2, 0.0))
    innovation = residual[:, 1:] - float(covariance.rho) * residual[:, :-1]
    q_adjacent = xp.sum(
        xp.where(adjacent, innovation.real**2 + innovation.imag**2, 0.0)
    ) / (1.0 - float(covariance.rho) ** 2)
    return (q_start + q_adjacent) / float(covariance.scale)


def deterministic_start_grid(pa_seed_deg: float, vsys_seed_kms: float, bmaj_arcsec: float, turnover_ratio: float):
    """Twelve fixed starts spanning PA sign, projected speed, and inclination."""

    rows = []
    start_id = 0
    for pa_shift in (0.0, -180.0):
        for u_kms in (75.0, 175.0, 300.0):
            for inclination_deg in (30.0, 60.0):
                rows.append(
                    {
                        "start_id": start_id,
                        "flux": 1.0,
                        "pa_deg": float(pa_seed_deg) + pa_shift,
                        "vsys_kms": float(vsys_seed_kms),
                        "gas_sigma_kms": 12.0,
                        "dx_arcsec": 0.0,
                        "dy_arcsec": 0.0,
                        "u_kms": u_kms,
                        "inclination_deg": inclination_deg,
                        "r_t_arcsec": float(turnover_ratio) * float(bmaj_arcsec),
                    }
                )
                start_id += 1
    return rows


def pack_s2(parameters: dict[str, float], *, vsys_seed_kms: float, dv_kms: float, bmaj_arcsec: float) -> np.ndarray:
    return np.array(
        [
            np.log(parameters["flux"]),
            np.radians(parameters["pa_deg"]),
            (parameters["vsys_kms"] - vsys_seed_kms) / dv_kms,
            np.log(parameters["gas_sigma_kms"]),
            parameters["dx_arcsec"] / bmaj_arcsec,
            parameters["dy_arcsec"] / bmaj_arcsec,
            parameters["u_kms"] / 100.0,
            np.cos(np.radians(parameters["inclination_deg"])),
            parameters["r_t_arcsec"] / bmaj_arcsec,
        ],
        dtype=np.float64,
    )


def _unpack_s2(z, *, vsys_seed_kms: float, dv_kms: float, bmaj_arcsec: float):
    from kinuv.xp import numpy_or_jax

    xp = numpy_or_jax(z)
    values = xp.asarray(z)
    cos_i = values[7]
    i_rad = xp.arccos(cos_i)
    sin_i = xp.sqrt(xp.maximum(1.0 - cos_i * cos_i, 1.0e-12))
    u_kms = 100.0 * values[6]
    model_parameters = {
        "flux": xp.exp(values[0]),
        "pa_deg": values[1] * (180.0 / np.pi),
        "vsys_kms": float(vsys_seed_kms) + values[2] * float(dv_kms),
        "gas_sigma_kms": xp.exp(values[3]),
        "dx_arcsec": values[4] * float(bmaj_arcsec),
        "dy_arcsec": values[5] * float(bmaj_arcsec),
        "v0_kms": u_kms / sin_i,
        "r_t_arcsec": values[8] * float(bmaj_arcsec),
    }
    return model_parameters, i_rad, u_kms


def unpack_s2(z, *, vsys_seed_kms: float, dv_kms: float, bmaj_arcsec: float) -> dict[str, float]:
    p, i_rad, u_kms = _unpack_s2(
        np.asarray(z),
        vsys_seed_kms=vsys_seed_kms,
        dv_kms=dv_kms,
        bmaj_arcsec=bmaj_arcsec,
    )
    return {
        "flux": float(p["flux"]),
        "pa_deg": float(p["pa_deg"] % 360.0),
        "vsys_kms": float(p["vsys_kms"]),
        "gas_sigma_kms": float(p["gas_sigma_kms"]),
        "dx_arcsec": float(p["dx_arcsec"]),
        "dy_arcsec": float(p["dy_arcsec"]),
        "u_kms": float(u_kms),
        "inclination_deg": float(np.degrees(i_rad)),
        "v0_kms_diagnostic": float(p["v0_kms"]),
        "r_t_arcsec": float(p["r_t_arcsec"]),
        "turnover_over_bmaj": float(np.asarray(z)[8]),
    }


def s2_bounds(
    *,
    pa_seed_deg: float,
    dv_kms: float,
    bmaj_arcsec: float,
    turnover_ratio: float | None,
):
    """Registered finite computational envelope in dimensionless coordinates."""

    pa = math.radians(float(pa_seed_deg))
    rt = (0.05, 1.6) if turnover_ratio is None else (float(turnover_ratio), float(turnover_ratio))
    shift = 2.0 / float(bmaj_arcsec)
    return [
        (math.log(1.0e-8), math.log(100.0)),
        (pa - math.pi, pa + math.pi),
        (-100.0 / float(dv_kms), 100.0 / float(dv_kms)),
        (math.log(2.0), math.log(50.0)),
        (-shift, shift),
        (-shift, shift),
        (0.0, 5.0),
        (1.0e-3, 0.999),
        rt,
    ]


def projected_gradient(z, gradient, bounds) -> np.ndarray:
    values = np.asarray(z, dtype=np.float64)
    grad = np.asarray(gradient, dtype=np.float64)
    lower = np.array([item[0] for item in bounds], dtype=np.float64)
    upper = np.array([item[1] for item in bounds], dtype=np.float64)
    return values - np.clip(values - grad, lower, upper)


def build_s2_value_gradient(
    data,
    template,
    grid,
    covariance: FitCovariance,
    *,
    vsys_seed_kms: float,
    bmaj_arcsec: float,
):
    """Compile the target objective once for reuse by all deterministic starts."""

    import jax
    import jax.numpy as jnp

    vis = jnp.asarray(data.vis)
    weights = jnp.asarray(data.weights)

    def objective(z):
        params, i_rad, _ = _unpack_s2(
            z,
            vsys_seed_kms=vsys_seed_kms,
            dv_kms=data.dv_kms,
            bmaj_arcsec=bmaj_arcsec,
        )
        model = predict_binned(data, params, template, grid, i_rad=i_rad, xla=True)
        c2 = correlated_chi2(vis, model, weights, covariance)
        prior = (params["dx_arcsec"] / 0.5) ** 2 + (params["dy_arcsec"] / 0.5) ** 2
        return 0.5 * c2 + 0.5 * prior

    return jax.jit(jax.value_and_grad(objective))


def fit_s2_start(
    data,
    template,
    grid,
    start: dict[str, float],
    covariance: FitCovariance,
    *,
    pa_seed_deg: float,
    vsys_seed_kms: float,
    bmaj_arcsec: float,
    fixed_turnover_over_bmaj: float | None,
    maxiter: int = 80,
    value_gradient=None,
) -> S2FitResult:
    """Run one exact-gradient visibility MAP fit from a registered start."""

    import jax
    import jax.numpy as jnp

    z0 = pack_s2(
        start,
        vsys_seed_kms=vsys_seed_kms,
        dv_kms=data.dv_kms,
        bmaj_arcsec=bmaj_arcsec,
    )
    bounds = s2_bounds(
        pa_seed_deg=pa_seed_deg,
        dv_kms=data.dv_kms,
        bmaj_arcsec=bmaj_arcsec,
        turnover_ratio=fixed_turnover_over_bmaj,
    )
    if fixed_turnover_over_bmaj is not None:
        z0[8] = float(fixed_turnover_over_bmaj)

    if value_gradient is None:
        value_gradient = build_s2_value_gradient(
            data,
            template,
            grid,
            covariance,
            vsys_seed_kms=vsys_seed_kms,
            bmaj_arcsec=bmaj_arcsec,
        )
    _ = value_gradient(jnp.asarray(z0))

    optimization_scale = 1.0 / max(1, int(data.vis.size))

    def raw_value_gradient(z):
        value, gradient = value_gradient(jnp.asarray(z))
        return float(value), np.asarray(gradient, dtype=np.float64)

    def scipy_value_gradient(z):
        value, gradient = raw_value_gradient(z)
        return optimization_scale * value, optimization_scale * gradient

    opt = minimize(
        scipy_value_gradient,
        z0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": int(maxiter), "ftol": 1.0e-15, "gtol": 1.0e-8, "maxls": 40},
    )
    value, gradient = raw_value_gradient(opt.x)
    params = unpack_s2(
        opt.x,
        vsys_seed_kms=vsys_seed_kms,
        dv_kms=data.dv_kms,
        bmaj_arcsec=bmaj_arcsec,
    )
    prior = (params["dx_arcsec"] / 0.5) ** 2 + (params["dy_arcsec"] / 0.5) ** 2
    pg = projected_gradient(opt.x, gradient, bounds)
    boundary = []
    for index, name in enumerate(S2_PARAMETER_NAMES):
        lo, hi = bounds[index]
        if lo == hi:
            continue
        span = hi - lo
        if opt.x[index] - lo <= 1.0e-4 * span or hi - opt.x[index] <= 1.0e-4 * span:
            boundary.append(name)
    return S2FitResult(
        start_id=int(start["start_id"]),
        fixed_turnover_over_bmaj=(
            None if fixed_turnover_over_bmaj is None else float(fixed_turnover_over_bmaj)
        ),
        parameters=params,
        objective=float(value),
        chi2=float(2.0 * value - prior),
        prior=float(prior),
        regularization=0.0,
        projected_gradient_inf=float(np.max(np.abs(pg))),
        nfev=int(opt.nfev),
        njev=int(opt.njev),
        nit=int(opt.nit),
        success=bool(opt.success),
        message=str(opt.message),
        boundary_parameters=tuple(boundary),
    )


__all__ = [
    "FitCovariance",
    "S2FitResult",
    "S2_PARAMETER_NAMES",
    "build_s2_value_gradient",
    "correlated_chi2",
    "deterministic_start_grid",
    "fit_s2_start",
    "pack_s2",
    "projected_gradient",
    "propagate_equal_weight_ar1",
    "s2_bounds",
    "unpack_s2",
]
