"""Stage A L-BFGS MAP on real Hann+bin KGAS066 visibilities (066-8).

Fit flux, PA, vsys, gas σ, (dx, dy), V_0, r_t. Freeze i. Shift lives inside
``predict_vis`` via ``fourier_shift`` then PB — no visibility phase ramp.
Model path is native ``predict_vis`` → ``hann_then_bin`` (DEC-066-SPECRESP).
Gate is likelihood ``Δχ² = χ²_zero − χ²``; the (dx, dy) prior is MAP-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from kinuv.decisions import requires
from kinuv.forward.model import (
    GAS_SIGMA_SEED_KM_S,
    VSYS_SEED_KM_S,
    predict_vis,
)
from kinuv.forward.sb import load_sb_template
from kinuv.geometry import inclination_rad, pa_seed_deg
from kinuv.io.vis import VisData, load_kgas066
from kinuv.likelihood.chi2 import chi2, chi2_zero, delta_chi2
from kinuv.profiles.rotation import CALIBRATION_RT_ARCSEC, CALIBRATION_V0_KM_S
from kinuv.response.spectral import hann_then_bin
from kinuv.transforms.grid import (
    fov_co_plus_pb_arcsec,
    image_grid_from_uv,
    max_baseline_lambda,
)

SHIFT_PRIOR_SIGMA_ARCSEC = 0.5
DX_DY_BOUND_ARCSEC = 2.0
FLUX_SEED_JY = 1.0
PA_BOUND_HALF_DEG = 30.0
VSYS_BOUND_HALF_KM_S = 100.0
GAS_SIGMA_BOUNDS_KM_S = (2.0, 50.0)
V0_BOUNDS_KM_S = (0.0, 400.0)
RT_BOUNDS_ARCSEC = (0.5, 15.0)
FLUX_BOUNDS_JY = (1.0e-8, 100.0)
MAXITER_STAGE_A = 80
ABORT_EVAL_S = 20.0
FD_STEP = 1.0e-3

# Scaled z so L-BFGS steps are O(1). Physical = offset + z * scale.
_SCALES = np.array(
    [1.0, 10.0, 20.0, 5.0, 0.5, 0.5, 50.0, 1.0], dtype=np.float64
)
PARAM_NAMES = (
    "flux",
    "pa_deg",
    "vsys_kms",
    "gas_sigma_kms",
    "dx_arcsec",
    "dy_arcsec",
    "v0_kms",
    "r_t_arcsec",
)


@dataclass
class MapResult:
    """Stage A MAP. ``delta_chi2`` is the V=0 gate (prior not included)."""

    flux: float
    pa_deg: float
    vsys_kms: float
    gas_sigma_kms: float
    dx_arcsec: float
    dy_arcsec: float
    v0_kms: float
    r_t_arcsec: float
    chi2_map: float
    chi2_zero: float
    delta_chi2: float
    n_row: int
    n_chan: int
    dv_kms: float
    n_bin: int
    s: float
    eval_s: float
    nfev: int
    success: bool
    optimiser_ran: bool
    message: str

    @property
    def beats_zero(self) -> bool:
        return self.delta_chi2 > 0.0


def stage_a_seeds() -> dict[str, float]:
    """YAML / ADR seeds. ``(dx, dy) = (0, 0)`` is a seed, not a freeze."""
    return {
        "flux": FLUX_SEED_JY,
        "pa_deg": pa_seed_deg(),
        "vsys_kms": VSYS_SEED_KM_S,
        "gas_sigma_kms": GAS_SIGMA_SEED_KM_S,
        "dx_arcsec": 0.0,
        "dy_arcsec": 0.0,
        "v0_kms": CALIBRATION_V0_KM_S,
        "r_t_arcsec": CALIBRATION_RT_ARCSEC,
    }


def stage_a_bounds() -> dict[str, tuple[float, float]]:
    """L-BFGS-B box in physical units. ``(dx, dy)`` support is ±2″, not ``{0}``."""
    pa = pa_seed_deg()
    vsys = VSYS_SEED_KM_S
    box = float(DX_DY_BOUND_ARCSEC)
    return {
        "flux": FLUX_BOUNDS_JY,
        "pa_deg": (pa - PA_BOUND_HALF_DEG, pa + PA_BOUND_HALF_DEG),
        "vsys_kms": (vsys - VSYS_BOUND_HALF_KM_S, vsys + VSYS_BOUND_HALF_KM_S),
        "gas_sigma_kms": GAS_SIGMA_BOUNDS_KM_S,
        "dx_arcsec": (-box, box),
        "dy_arcsec": (-box, box),
        "v0_kms": V0_BOUNDS_KM_S,
        "r_t_arcsec": RT_BOUNDS_ARCSEC,
    }


@requires("DEC-066-SHIFT")
def shift_prior(
    dx_arcsec,
    dy_arcsec,
    sigma_arcsec: float = SHIFT_PRIOR_SIGMA_ARCSEC,
) -> float:
    """``(dx/σ)² + (dy/σ)²`` with σ = 0.5″ (DEC-066-SHIFT)."""
    sig = float(sigma_arcsec)
    return (float(dx_arcsec) / sig) ** 2 + (float(dy_arcsec) / sig) ** 2


def map_objective(chi2_val: float, dx_arcsec: float, dy_arcsec: float) -> float:
    """MAP objective ``χ² + prior``. Do not use this as the V=0 gate."""
    return float(chi2_val) + shift_prior(dx_arcsec, dy_arcsec)


@requires("DEC-066-ZEROMODEL")
def gate_delta_chi2(chi2_map: float, chi2_zero_val: float) -> float:
    """Likelihood gate ``χ²_zero − χ²``. Not reduced χ²; prior is not folded in."""
    return delta_chi2(chi2_map, chi2_zero_val)


@requires("DEC-066-WEIGHT", "DEC-066-ZEROMODEL")
def map_gate_scores(vis, model, weights, s):
    """Return ``(χ², χ²_zero, Δχ²)``. The score is not ``χ² / n``."""
    c = chi2(vis, model, weights, s)
    c0 = chi2_zero(vis, weights, s)
    return c, c0, gate_delta_chi2(c, c0)


def _offsets(seeds: dict[str, float]) -> np.ndarray:
    return np.array(
        [
            0.0,
            seeds["pa_deg"],
            seeds["vsys_kms"],
            seeds["gas_sigma_kms"],
            0.0,
            0.0,
            seeds["v0_kms"],
            seeds["r_t_arcsec"],
        ],
        dtype=np.float64,
    )


def _unpack(z, offsets: np.ndarray) -> dict[str, float]:
    phys = offsets + np.asarray(z, dtype=np.float64) * _SCALES
    return {name: float(phys[i]) for i, name in enumerate(PARAM_NAMES)}


def _pack(params: dict[str, float], offsets: np.ndarray) -> np.ndarray:
    phys = np.array([params[n] for n in PARAM_NAMES], dtype=np.float64)
    return (phys - offsets) / _SCALES


def _z_bounds(offsets: np.ndarray) -> list[tuple[float, float]]:
    b = stage_a_bounds()
    out = []
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = b[name]
        out.append(
            ((lo - offsets[i]) / _SCALES[i], (hi - offsets[i]) / _SCALES[i])
        )
    return out


def image_grid_for_vis(data: VisData):
    """Nyquist grid from aggregated uv and the native model axis, not Ico CDELT."""
    mb = max_baseline_lambda(data.u_m, data.v_m, data.freqs_native)
    return image_grid_from_uv(mb, fov_co_plus_pb_arcsec())


@requires("DEC-066-SPECRESP", "DEC-066-PB", "DEC-066-SHIFT", "DEC-066-GRID")
def predict_binned(data: VisData, params: dict[str, float], template, grid):
    """Native ``predict_vis`` (guards in) → Hann+bin to the fit array."""
    n_g = int(data.n_guard)
    model_native = predict_vis(
        data.u_m,
        data.v_m,
        data.freqs_native,
        flux=params["flux"],
        pa_rad=np.radians(params["pa_deg"]),
        vsys_kms=params["vsys_kms"],
        dx_arcsec=params["dx_arcsec"],
        dy_arcsec=params["dy_arcsec"],
        gas_sigma_kms=params["gas_sigma_kms"],
        template=template,
        grid=grid,
        v0_kms=params["v0_kms"],
        r_t_arcsec=params["r_t_arcsec"],
        i_rad=inclination_rad(),
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


def _optimal_flux(vis, model_unit, weights) -> float:
    """Closed-form amplitude at fixed kinematics. Do not leave flux at 0."""
    m = np.asarray(model_unit)
    d = np.asarray(vis)
    w = np.asarray(weights, dtype=np.float64)
    numer = float(np.sum(w * (d.real * m.real + d.imag * m.imag), dtype=np.float64))
    denom = float(np.sum(w * (m.real * m.real + m.imag * m.imag), dtype=np.float64))
    lo, hi = FLUX_BOUNDS_JY
    if denom <= 0.0 or not np.isfinite(numer):
        return FLUX_SEED_JY
    flux = numer / denom
    if not np.isfinite(flux) or flux < 1.0e-3:
        return FLUX_SEED_JY
    return float(np.clip(flux, lo, hi))


def _result(params, c, c0, dchi, data, eval_s, nfev, success, ran, message):
    n_row, n_chan = data.vis.shape
    return MapResult(
        flux=params["flux"],
        pa_deg=params["pa_deg"],
        vsys_kms=params["vsys_kms"],
        gas_sigma_kms=params["gas_sigma_kms"],
        dx_arcsec=params["dx_arcsec"],
        dy_arcsec=params["dy_arcsec"],
        v0_kms=params["v0_kms"],
        r_t_arcsec=params["r_t_arcsec"],
        chi2_map=float(c),
        chi2_zero=float(c0),
        delta_chi2=float(dchi),
        n_row=int(n_row),
        n_chan=int(n_chan),
        dv_kms=float(data.dv_kms),
        n_bin=int(data.n_bin),
        s=float(data.s),
        eval_s=float(eval_s),
        nfev=int(nfev),
        success=bool(success),
        optimiser_ran=bool(ran),
        message=str(message),
    )


@requires(
    "DEC-066-INFER",
    "DEC-066-ZEROMODEL",
    "DEC-066-SHIFT",
    "DEC-066-SPECRESP",
    "DEC-066-VC",
    "DEC-066-WEIGHT",
)
def run_stage_a_map(
    data: VisData | None = None,
    *,
    template=None,
    grid=None,
    maxiter: int = MAXITER_STAGE_A,
    abort_eval_s: float = ABORT_EVAL_S,
) -> MapResult:
    """L-BFGS-B Stage A MAP. Times one aggregated eval; aborts if it is tens of s."""
    if data is None:
        data = load_kgas066()
    if grid is None:
        grid = image_grid_for_vis(data)
    if template is None:
        template = load_sb_template(grid)

    seeds = stage_a_seeds()
    offsets = _offsets(seeds)
    params = dict(seeds)
    t0 = perf_counter()
    model_unit = predict_binned(data, params, template, grid)
    eval_s = perf_counter() - t0
    params["flux"] = _optimal_flux(data.vis, model_unit, data.weights)
    model = params["flux"] * model_unit
    c, c0, dchi = map_gate_scores(data.vis, model, data.weights, data.s)

    if eval_s >= float(abort_eval_s):
        return _result(
            params,
            c,
            c0,
            dchi,
            data,
            eval_s,
            1,
            False,
            False,
            f"single eval {eval_s:.3f}s >= {abort_eval_s:.1f}s; optimiser skipped",
        )

    nfev = {"n": 0}

    def predict_from_z(z):
        p = _unpack(z, offsets)
        return p, predict_binned(data, p, template, grid)

    def fun(z):
        p, model_z = predict_from_z(z)
        c_z = chi2(data.vis, model_z, data.weights, data.s)
        return map_objective(c_z, p["dx_arcsec"], p["dy_arcsec"])

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

    z0 = _pack(params, offsets)
    opt = minimize(
        fun_count,
        z0,
        method="L-BFGS-B",
        jac=jac,
        bounds=_z_bounds(offsets),
        options={"maxiter": int(maxiter), "ftol": 1e-9},
    )
    params = _unpack(opt.x, offsets)
    model = predict_binned(data, params, template, grid)
    c, c0, dchi = map_gate_scores(data.vis, model, data.weights, data.s)
    return _result(
        params,
        c,
        c0,
        dchi,
        data,
        eval_s,
        nfev["n"],
        bool(opt.success),
        True,
        str(opt.message),
    )
