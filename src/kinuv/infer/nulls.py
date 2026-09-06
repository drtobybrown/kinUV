"""Fitted null models for scientific comparison in visibility space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from kinuv.io.vis import VisData
from kinuv.likelihood.chi2 import chi2, chi2_blank, chi2_nonrot

from .map import FD_STEP, _optimal_flux, map_objective, predict_binned
from .seeds import stage_a_bounds, stage_a_seeds

NONROT_PARAM_NAMES = (
    "flux",
    "vsys_kms",
    "gas_sigma_kms",
    "dx_arcsec",
    "dy_arcsec",
)
_NONROT_SCALES = np.asarray([1.0, 20.0, 5.0, 0.5, 0.5], dtype=np.float64)


@dataclass(frozen=True)
class NonRotatingResult:
    """Best-fit emitting disk with circular speed fixed exactly to zero."""

    flux: float
    vsys_kms: float
    gas_sigma_kms: float
    dx_arcsec: float
    dy_arcsec: float
    chi2_nonrot: float
    chi2_blank: float
    delta_chi2_nonrot_vs_blank: float
    map_objective: float
    nfev: int
    success: bool
    message: str


def _forward_params(params: dict[str, float]) -> dict[str, float]:
    """Complete the forward-model vector while fixing rotation to zero."""
    return {
        **params,
        "pa_deg": 0.0,
        "v0_kms": 0.0,
        "r_t_arcsec": 1.0,
    }


def fit_nonrotating_emission(
    data: VisData,
    template,
    grid,
    *,
    seeds: dict[str, float] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
    maxiter: int = 80,
    xla: bool = False,
) -> NonRotatingResult:
    """Fit an emitting non-rotating disk with Stage A nuisance treatment.

    Flux, systemic velocity, positive constant dispersion, and sky offsets are
    optimized with the same L-BFGS-B machinery and center prior as Stage A.
    The returned chi-square scores exclude the center prior. The surface
    brightness template and every sampling/response operator are identical to
    the rotating path because both call :func:`infer.map.predict_binned`.
    """
    seed_all = stage_a_seeds()
    if seeds:
        seed_all.update({key: float(value) for key, value in seeds.items()})
    bound_all = stage_a_bounds()
    if bounds:
        bound_all.update(bounds)
    seed = np.asarray([seed_all[key] for key in NONROT_PARAM_NAMES], dtype=np.float64)

    def unpack(z) -> dict[str, float]:
        physical = seed + np.asarray(z, dtype=np.float64) * _NONROT_SCALES
        return {key: float(physical[i]) for i, key in enumerate(NONROT_PARAM_NAMES)}

    z_bounds = []
    for i, key in enumerate(NONROT_PARAM_NAMES):
        lo, hi = bound_all[key]
        z_bounds.append(
            ((float(lo) - seed[i]) / _NONROT_SCALES[i],
             (float(hi) - seed[i]) / _NONROT_SCALES[i])
        )

    initial = {key: float(seed_all[key]) for key in NONROT_PARAM_NAMES}
    unit_params = _forward_params({**initial, "flux": 1.0})
    unit = np.asarray(predict_binned(data, unit_params, template, grid, i_rad=0.0, xla=xla))
    initial["flux"] = _optimal_flux(data.vis, unit, data.weights)
    z0 = (np.asarray([initial[key] for key in NONROT_PARAM_NAMES]) - seed) / _NONROT_SCALES
    nfev = {"n": 0}

    def objective(z):
        params = unpack(z)
        model = np.asarray(
            predict_binned(
                data, _forward_params(params), template, grid, i_rad=0.0, xla=xla
            )
        )
        score = chi2(data.vis, model, data.weights, data.s)
        return map_objective(score, params["dx_arcsec"], params["dy_arcsec"])

    def counted(z):
        nfev["n"] += 1
        return objective(z)

    def jac(z):
        z = np.asarray(z, dtype=np.float64)
        base = counted(z)
        grad = np.empty(z.size, dtype=np.float64)
        for index in range(z.size):
            shifted = z.copy()
            shifted[index] += FD_STEP
            grad[index] = (counted(shifted) - base) / FD_STEP
        return grad

    opt = minimize(
        counted,
        z0,
        method="L-BFGS-B",
        jac=jac,
        bounds=z_bounds,
        options={"maxiter": int(maxiter), "ftol": 1.0e-9},
    )
    fitted = unpack(opt.x)
    model = np.asarray(
        predict_binned(
            data, _forward_params(fitted), template, grid, i_rad=0.0, xla=xla
        )
    )
    c_nonrot = chi2_nonrot(data.vis, model, data.weights, data.s)
    c_blank = chi2_blank(data.vis, data.weights, data.s)
    return NonRotatingResult(
        **fitted,
        chi2_nonrot=float(c_nonrot),
        chi2_blank=float(c_blank),
        delta_chi2_nonrot_vs_blank=float(c_blank - c_nonrot),
        map_objective=float(
            map_objective(c_nonrot, fitted["dx_arcsec"], fitted["dy_arcsec"])
        ),
        nfev=int(nfev["n"]),
        success=bool(opt.success),
        message=str(opt.message),
    )
