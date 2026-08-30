"""Stage A L-BFGS MAP on real Hann+bin KGAS066 visibilities (066-8).

Fit flux, PA, vsys, gas σ, (dx, dy), V_0, r_t. Freeze i. Shift lives inside
``predict_vis`` via ``fourier_shift`` then PB — no visibility phase ramp.
Model path is native ``predict_vis`` → ``hann_then_bin`` (DEC-066-SPECRESP).
Gate is likelihood ``Δχ² = χ²_zero − χ²``; the (dx, dy) prior is MAP-only.
Two-start PA: 205.2° and 25.2°. Keep the larger likelihood Δχ².
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from kinuv.decisions import requires
from kinuv.forward.model import predict_vis
from kinuv.forward.sb import load_sb_template
from kinuv.geometry import inclination_rad
from kinuv.io.vis import VisData, load_kgas066
from kinuv.likelihood.chi2 import chi2, chi2_zero, delta_chi2
from kinuv.response.spectral import hann_then_bin
from kinuv.transforms.grid import (
    fov_co_plus_pb_arcsec,
    image_grid_from_uv,
    max_baseline_lambda,
)

from .seeds import (
    DX_DY_BOUND_ARCSEC,
    FLUX_BOUNDS_JY,
    FLUX_SEED_JY,
    SHIFT_PRIOR_SIGMA_ARCSEC,
    pa_start_degs,
    stage_a_bounds,
    stage_a_seeds,
)

MAXITER_STAGE_A = 80
ABORT_EVAL_S = 20.0
FD_STEP = 1.0e-3

_SCALES = np.array(
    [1.0, 10.0, 20.0, 5.0, 0.5, 0.5, 50.0, 1.0], dtype=np.float64
)
PARAM_NAMES = (
    "flux", "pa_deg", "vsys_kms", "gas_sigma_kms",
    "dx_arcsec", "dy_arcsec", "v0_kms", "r_t_arcsec",
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
    pa_start_deg: float = 0.0

    @property
    def beats_zero(self) -> bool:
        return self.delta_chi2 > 0.0


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
        [0.0, seeds["pa_deg"], seeds["vsys_kms"], seeds["gas_sigma_kms"],
         0.0, 0.0, seeds["v0_kms"], seeds["r_t_arcsec"]],
        dtype=np.float64,
    )


def _unpack(z, offsets: np.ndarray) -> dict[str, float]:
    phys = offsets + np.asarray(z, dtype=np.float64) * _SCALES
    return {name: float(phys[i]) for i, name in enumerate(PARAM_NAMES)}


def _pack(params: dict[str, float], offsets: np.ndarray) -> np.ndarray:
    phys = np.array([params[n] for n in PARAM_NAMES], dtype=np.float64)
    return (phys - offsets) / _SCALES


def _z_bounds(
    offsets: np.ndarray, extra_bounds: dict[str, tuple[float, float]] | None = None
) -> list[tuple[float, float]]:
    b = dict(stage_a_bounds())
    if extra_bounds:
        b.update(extra_bounds)
    out = []
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = b[name]
        out.append(((lo - offsets[i]) / _SCALES[i], (hi - offsets[i]) / _SCALES[i]))
    return out


def image_grid_for_vis(data: VisData):
    """Nyquist grid from aggregated uv and the native model axis, not Ico CDELT."""
    mb = max_baseline_lambda(data.u_m, data.v_m, data.freqs_native)
    return image_grid_from_uv(mb, fov_co_plus_pb_arcsec())


@requires("DEC-066-SPECRESP", "DEC-066-PB", "DEC-066-SHIFT", "DEC-066-GRID")
def predict_binned(
    data: VisData, params: dict[str, float], template, grid, *, i_rad=None, xla=False
):
    """Native ``predict_vis`` (guards in) → Hann+bin to the fit array."""
    if xla:
        from kinuv.xp import is_jax, numpy_or_jax
        import jax.numpy as jnp

        tmpl = jnp.asarray(template)
        n_g = int(data.n_guard)
        i_use = inclination_rad() if i_rad is None else float(i_rad)
        xp = numpy_or_jax(
            params["flux"],
            params["pa_deg"],
            params["vsys_kms"],
            params["gas_sigma_kms"],
            params["v0_kms"],
            params["r_t_arcsec"],
        )
        pa_rad = xp.asarray(params["pa_deg"]) * (np.pi / 180.0)
        model_native = predict_vis(
            data.u_m,
            data.v_m,
            data.freqs_native,
            flux=params["flux"],
            pa_rad=pa_rad,
            vsys_kms=params["vsys_kms"],
            dx_arcsec=params["dx_arcsec"],
            dy_arcsec=params["dy_arcsec"],
            gas_sigma_kms=params["gas_sigma_kms"],
            template=tmpl,
            grid=grid,
            v0_kms=params["v0_kms"],
            r_t_arcsec=params["r_t_arcsec"],
            i_rad=i_use,
        )
        if not is_jax(model_native):
            raise RuntimeError("xla predict_binned host-bounced before Hann")
        vel_trim = data.vel_native[n_g:-n_g]
        freqs_trim = data.freqs_native[n_g:-n_g]
        model_binned = hann_then_bin(
            model_native,
            data.n_bin,
            n_guard=n_g,
            weights=jnp.asarray(data.weights_native),
            vel=jnp.asarray(vel_trim),
            freqs=jnp.asarray(freqs_trim),
        )
        if tuple(model_binned.shape) != tuple(data.vis.shape):
            raise ValueError(
                f"binned model {model_binned.shape} != data vis {data.vis.shape}"
            )
        return model_binned
    n_g = int(data.n_guard)
    i_use = inclination_rad() if i_rad is None else float(i_rad)
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
        i_rad=i_use,
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


def score_seed_delta_chi2(data: VisData, template, grid, params: dict[str, float]):
    """Optimal-flux Δχ² at a frozen kinematics vector. Not the MAP product."""
    p = dict(params)
    p["flux"] = 1.0
    unit = predict_binned(data, p, template, grid)
    p["flux"] = _optimal_flux(data.vis, unit, data.weights)
    _, _, dchi = map_gate_scores(data.vis, p["flux"] * unit, data.weights, data.s)
    return float(dchi), float(p["flux"])


def _result(params, c, c0, dchi, data, eval_s, nfev, success, ran, message, pa_start):
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
        pa_start_deg=float(pa_start),
    )


def _lbfgs_one_start(
    data, template, grid, seeds, eval_s, maxiter, pa_start, extra_bounds=None
):
    offsets = _offsets(seeds)
    params = dict(seeds)
    unit = predict_binned(data, params, template, grid)
    params["flux"] = _optimal_flux(data.vis, unit, data.weights)
    nfev = {"n": 0}

    def fun(z):
        p = _unpack(z, offsets)
        model_z = predict_binned(data, p, template, grid)
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

    opt = minimize(
        fun_count,
        _pack(params, offsets),
        method="L-BFGS-B",
        jac=jac,
        bounds=_z_bounds(offsets, extra_bounds),
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
        pa_start,
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
    rt_bounds_arcsec: tuple[float, float] | None = None,
) -> MapResult:
    """Two-start L-BFGS-B. Keep the start with larger Δχ²."""
    if data is None:
        data = load_kgas066()
    if grid is None:
        grid = image_grid_for_vis(data)
    if template is None:
        template = load_sb_template(grid)
    extra_bounds = (
        None
        if rt_bounds_arcsec is None
        else {"r_t_arcsec": (float(rt_bounds_arcsec[0]), float(rt_bounds_arcsec[1]))}
    )

    seeds = stage_a_seeds()
    t0 = perf_counter()
    _ = predict_binned(data, seeds, template, grid)
    eval_s = perf_counter() - t0
    if eval_s >= float(abort_eval_s):
        unit = predict_binned(data, seeds, template, grid)
        seeds["flux"] = _optimal_flux(data.vis, unit, data.weights)
        c, c0, dchi = map_gate_scores(
            data.vis, seeds["flux"] * unit, data.weights, data.s
        )
        return _result(
            seeds,
            c,
            c0,
            dchi,
            data,
            eval_s,
            1,
            False,
            False,
            f"single eval {eval_s:.3f}s >= {abort_eval_s:.1f}s; optimiser skipped",
            seeds["pa_deg"],
        )

    runs = []
    for pa in pa_start_degs():
        runs.append(
            _lbfgs_one_start(
                data,
                template,
                grid,
                stage_a_seeds(pa_deg=pa),
                eval_s,
                maxiter,
                pa,
                extra_bounds=extra_bounds,
            )
        )
    winner = max(runs, key=lambda r: r.delta_chi2)
    nfev = sum(r.nfev for r in runs)
    msg = (
        f"{winner.message}; starts "
        + ", ".join(f"PA={r.pa_start_deg:.1f} Δχ²={r.delta_chi2:.1f}" for r in runs)
    )
    return _result(
        {
            "flux": winner.flux,
            "pa_deg": winner.pa_deg,
            "vsys_kms": winner.vsys_kms,
            "gas_sigma_kms": winner.gas_sigma_kms,
            "dx_arcsec": winner.dx_arcsec,
            "dy_arcsec": winner.dy_arcsec,
            "v0_kms": winner.v0_kms,
            "r_t_arcsec": winner.r_t_arcsec,
        },
        winner.chi2_map,
        winner.chi2_zero,
        winner.delta_chi2,
        data,
        eval_s,
        nfev,
        winner.success,
        True,
        msg,
        winner.pa_start_deg,
    )
