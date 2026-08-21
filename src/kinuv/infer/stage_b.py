"""Stage B ring MAP: fit ``V_k`` only (066-12). No NUTS. No PA search.

Nuisance (flux, PA, vsys, σ, dx, dy) is frozen. ``V_c`` is ``ring_vc``.
Objective is ``χ² + ring_regulariser``; the V=0 gate is likelihood Δχ².
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares, minimize

from kinuv.decisions import requires
from kinuv.forward.model import predict_vis
from kinuv.geometry import inclination_rad
from kinuv.io.vis import VisData
from kinuv.likelihood.chi2 import chi2
from kinuv.profiles.rotation import (
    CALIBRATION_RT_ARCSEC,
    CALIBRATION_V0_KM_S,
    DISK_RADIUS_ARCSEC,
    V_K_MAX_KM_S,
    V_K_MIN_KM_S,
    aic_keep_stage_a,
    arctan_vc,
    omega_residual,
    ring_regulariser,
    ring_vc,
    rings_from_arctan,
    uniform_knot_radii,
)
from kinuv.response.spectral import hann_then_bin

from .map import map_gate_scores
from .seeds import RT_BOUNDS_ARCSEC, V0_BOUNDS_KM_S

FD_STEP = 1.0e-3
MAXITER_STAGE_B = 40
N_RINGS_DEFAULT = 7


@dataclass
class StageBResult:
    """Ring MAP at frozen nuisance. ``keep_stage_a`` is the AIC gate."""

    v_knots_kms: tuple[float, ...]
    r_knots_arcsec: tuple[float, ...]
    lam_reg: float
    chi2_map: float
    chi2_zero: float
    delta_chi2: float
    chi2_stage_a: float
    aic_stage_a: float
    aic_stage_b: float
    keep_stage_a: bool
    n_rings: int
    nfev: int
    success: bool
    message: str
    v0_recovered: float
    r_t_recovered: float
    max_omega: float
    dv_kms: float


def nuisance_from_params(params: dict[str, float]) -> dict[str, float]:
    """Geometry/flux freeze vector. Not ``V_0`` / ``r_t``."""
    keys = (
        "flux",
        "pa_deg",
        "vsys_kms",
        "gas_sigma_kms",
        "dx_arcsec",
        "dy_arcsec",
    )
    return {k: float(params[k]) for k in keys}


def predict_binned(
    data: VisData,
    nuisance: dict[str, float],
    template,
    grid,
    *,
    v0_kms: float | None = None,
    r_t_arcsec: float | None = None,
    r_knots_arcsec=None,
    v_knots_kms=None,
):
    """Native ``predict_vis`` → Hann+bin. Arctan or rings, not both."""
    n_g = int(data.n_guard)
    model_native = predict_vis(
        data.u_m,
        data.v_m,
        data.freqs_native,
        flux=nuisance["flux"],
        pa_rad=np.radians(nuisance["pa_deg"]),
        vsys_kms=nuisance["vsys_kms"],
        dx_arcsec=nuisance["dx_arcsec"],
        dy_arcsec=nuisance["dy_arcsec"],
        gas_sigma_kms=nuisance["gas_sigma_kms"],
        template=template,
        grid=grid,
        v0_kms=CALIBRATION_V0_KM_S if v0_kms is None else float(v0_kms),
        r_t_arcsec=CALIBRATION_RT_ARCSEC if r_t_arcsec is None else float(r_t_arcsec),
        i_rad=inclination_rad(),
        r_knots_arcsec=r_knots_arcsec,
        v_knots_kms=v_knots_kms,
    )
    vel_trim = data.vel_native[n_g:-n_g]
    freqs_trim = data.freqs_native[n_g:-n_g]
    model_binned = hann_then_bin(
        model_native,
        data.n_bin,
        n_guard=n_g,
        weights=data.weights_native,
        vel=vel_trim,
        freqs=freqs_trim,
    )
    if model_binned.shape != data.vis.shape:
        raise ValueError(
            f"binned model {model_binned.shape} != data vis {data.vis.shape}"
        )
    return model_binned


def recover_arctan_from_rings(r_knots_arcsec, v_knots_kms):
    """Least-squares arctan to Stage B ``V_c(r)``. Not a second vis MAP."""
    r_k = np.asarray(r_knots_arcsec, dtype=np.float64)
    v_k = np.asarray(v_knots_kms, dtype=np.float64)
    r = np.linspace(max(float(r_k[0]) * 0.5, 0.05), float(r_k[-1]) + 2.0, 200)
    v = ring_vc(r, r_k, v_k)

    def resid(p):
        return arctan_vc(r, p[0], p[1]) - v

    lo = [V0_BOUNDS_KM_S[0], RT_BOUNDS_ARCSEC[0]]
    hi = [V0_BOUNDS_KM_S[1], RT_BOUNDS_ARCSEC[1]]
    seed = np.array([float(v_k[-1]), 3.0], dtype=np.float64)
    fit = least_squares(resid, seed, bounds=(lo, hi), max_nfev=200)
    return float(fit.x[0]), float(fit.x[1])


def fit_v0_rt(
    data: VisData,
    nuisance: dict[str, float],
    template,
    grid,
    *,
    v0_seed: float = CALIBRATION_V0_KM_S,
    rt_seed: float = CALIBRATION_RT_ARCSEC,
    maxiter: int = MAXITER_STAGE_B,
):
    """Stage A kinematics only: ``(V_0, r_t)`` at frozen nuisance."""
    scales = np.array([50.0, 1.0], dtype=np.float64)
    seed = np.array([float(v0_seed), float(rt_seed)], dtype=np.float64)
    nfev = {"n": 0}

    def unpack(z):
        p = seed + np.asarray(z, dtype=np.float64) * scales
        return float(p[0]), float(p[1])

    def fun(z):
        v0, rt = unpack(z)
        model = predict_binned(
            data, nuisance, template, grid, v0_kms=v0, r_t_arcsec=rt
        )
        return chi2(data.vis, model, data.weights, data.s)

    def fun_count(z):
        nfev["n"] += 1
        return fun(z)

    def jac(z):
        z = np.asarray(z, dtype=np.float64)
        f0 = fun_count(z)
        g = np.empty(z.size, dtype=np.float64)
        for i in range(z.size):
            zp = z.copy()
            zp[i] += FD_STEP
            g[i] = (fun_count(zp) - f0) / FD_STEP
        return g

    z0 = np.zeros(2, dtype=np.float64)
    lo = (np.array([V0_BOUNDS_KM_S[0], RT_BOUNDS_ARCSEC[0]]) - seed) / scales
    hi = (np.array([V0_BOUNDS_KM_S[1], RT_BOUNDS_ARCSEC[1]]) - seed) / scales
    opt = minimize(
        fun_count,
        z0,
        method="L-BFGS-B",
        jac=jac,
        bounds=list(zip(lo, hi)),
        options={"maxiter": int(maxiter), "ftol": 1e-9},
    )
    v0, rt = unpack(opt.x)
    model = predict_binned(data, nuisance, template, grid, v0_kms=v0, r_t_arcsec=rt)
    c, c0, dchi = map_gate_scores(data.vis, model, data.weights, data.s)
    return {
        "v0_kms": v0,
        "r_t_arcsec": rt,
        "chi2": c,
        "chi2_zero": c0,
        "delta_chi2": dchi,
        "nfev": nfev["n"],
        "success": bool(opt.success),
        "message": str(opt.message),
    }


@requires("DEC-066-VC", "DEC-066-OSCMETRIC", "DEC-066-INFER")
def run_stage_b_map(
    data: VisData,
    nuisance: dict[str, float],
    template,
    grid,
    *,
    lam_reg: float,
    v0_init: float = CALIBRATION_V0_KM_S,
    rt_init: float = CALIBRATION_RT_ARCSEC,
    n_rings: int = N_RINGS_DEFAULT,
    chi2_stage_a: float | None = None,
    maxiter: int = MAXITER_STAGE_B,
) -> StageBResult:
    """L-BFGS on ``V_k``. Init ``rings_from_arctan``. Freeze nuisance."""
    r_k = uniform_knot_radii(int(n_rings), r_last_arcsec=DISK_RADIUS_ARCSEC)
    v_init = rings_from_arctan(r_k, v0_init, rt_init)
    scales = np.full(r_k.size, 20.0, dtype=np.float64)
    nfev = {"n": 0}

    def unpack(z):
        return v_init + np.asarray(z, dtype=np.float64) * scales

    def fun(z):
        v_k = unpack(z)
        model = predict_binned(
            data,
            nuisance,
            template,
            grid,
            r_knots_arcsec=r_k,
            v_knots_kms=v_k,
        )
        c_z = chi2(data.vis, model, data.weights, data.s)
        return float(c_z) + ring_regulariser(v_k, lam_reg)

    def fun_count(z):
        nfev["n"] += 1
        return fun(z)

    def jac(z):
        z = np.asarray(z, dtype=np.float64)
        f0 = fun_count(z)
        g = np.empty(z.size, dtype=np.float64)
        for i in range(z.size):
            zp = z.copy()
            zp[i] += FD_STEP
            g[i] = (fun_count(zp) - f0) / FD_STEP
        return g

    lo = (V_K_MIN_KM_S - v_init) / scales
    hi = (V_K_MAX_KM_S - v_init) / scales
    opt = minimize(
        fun_count,
        np.zeros(r_k.size, dtype=np.float64),
        method="L-BFGS-B",
        jac=jac,
        bounds=list(zip(lo, hi)),
        options={"maxiter": int(maxiter), "ftol": 1e-9},
    )
    v_k = unpack(opt.x)
    model = predict_binned(
        data, nuisance, template, grid, r_knots_arcsec=r_k, v_knots_kms=v_k
    )
    c, c0, dchi = map_gate_scores(data.vis, model, data.weights, data.s)
    if chi2_stage_a is None:
        model_a = predict_binned(
            data, nuisance, template, grid, v0_kms=v0_init, r_t_arcsec=rt_init
        )
        chi2_a = chi2(data.vis, model_a, data.weights, data.s)
    else:
        chi2_a = float(chi2_stage_a)
    n = int(n_rings)
    aic_a = chi2_a + 2.0 * 2.0
    aic_b = float(c) + 2.0 * n
    v0_r, rt_r = recover_arctan_from_rings(r_k, v_k)
    om = omega_residual(v_k, v_init, data.dv_kms)
    return StageBResult(
        v_knots_kms=tuple(float(x) for x in v_k),
        r_knots_arcsec=tuple(float(x) for x in r_k),
        lam_reg=float(lam_reg),
        chi2_map=float(c),
        chi2_zero=float(c0),
        delta_chi2=float(dchi),
        chi2_stage_a=chi2_a,
        aic_stage_a=aic_a,
        aic_stage_b=aic_b,
        keep_stage_a=bool(aic_keep_stage_a(aic_a, aic_b, n)),
        n_rings=n,
        nfev=nfev["n"],
        success=bool(opt.success),
        message=str(opt.message),
        v0_recovered=v0_r,
        r_t_recovered=rt_r,
        max_omega=float(np.max(om)),
        dv_kms=float(data.dv_kms),
    )
