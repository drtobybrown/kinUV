"""066-7 mock recovery on a native uv window.

If ``kinuv.likelihood.hann_then_bin`` exists it is applied before χ². Otherwise
recovery uses **diagonal χ² on a short native window**. 066-8 (real-data MAP)
must go through 066-6 Hann+bin (DEC-066-SPECRESP); this path is not that MAP.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from kinuv.constants import freq_to_velocity_kms
from kinuv.decisions import requires
from kinuv.geometry import inclination_rad, pa_seed_deg, pa_seed_rad
from kinuv.profiles.rotation import CALIBRATION_RT_ARCSEC, CALIBRATION_V0_KM_S
from kinuv.transforms.grid import (
    ImageGrid,
    fov_co_plus_pb_arcsec,
    image_grid_from_uv,
    max_baseline_lambda,
)

from .model import (
    GAS_SIGMA_SEED_KM_S,
    INJECT_OFFSET_ARCSEC,
    LINE_V_MAX_KM_S,
    LINE_V_MIN_KM_S,
    VSYS_SEED_KM_S,
    predict_vis,
)

NPZ_PATH = Path("/Users/thbrown/kilogas/DR1/visibilities/KILOGAS066.npz")

# Gate 2 is recovery, not a 30-minute bench: few thousand rows, line window.
N_ROW_MOCK = 2048
CHAN_STRIDE_MOCK = 4


@dataclass(frozen=True)
class NativeUvWindow:
    u_m: np.ndarray
    v_m: np.ndarray
    freqs_hz: np.ndarray
    weights: np.ndarray
    vis: np.ndarray
    grid: ImageGrid
    operator: str


@dataclass(frozen=True)
class RecoveryResult:
    flux: float
    pa_deg: float
    vsys_kms: float
    dx_arcsec: float
    dy_arcsec: float
    chi2: float
    nfev: int
    success: bool
    operator: str


def _hann_then_bin_if_present(vis, weights):
    """Use 066-6 when imported; else native vis (document for 066-8)."""
    try:
        from kinuv.likelihood import hann_then_bin
    except ImportError:
        return (
            np.asarray(vis),
            np.asarray(weights, dtype=np.float64),
            "native_diagonal",
        )
    out = hann_then_bin(vis, weights)
    if isinstance(out, tuple) and len(out) >= 2:
        return out[0], out[1], "hann_then_bin"
    return out, np.asarray(weights, dtype=np.float64), "hann_then_bin"


@requires("DEC-066-GRID")
def subsample_native_uv(
    npz_path: Path | None = None,
    *,
    v_min_kms: float = LINE_V_MIN_KM_S,
    v_max_kms: float = LINE_V_MAX_KM_S,
    n_row: int = N_ROW_MOCK,
    chan_stride: int = CHAN_STRIDE_MOCK,
    rng_seed: int = 66,
    min_baseline_m: float = 1.0,
) -> NativeUvWindow:
    """Real 066 ``(u,v,ν)`` on the Ico line window; not the full 1920×43240 cube."""
    path = NPZ_PATH if npz_path is None else Path(npz_path)
    z = np.load(path)
    u = np.asarray(z["u_m"], dtype=np.float64)
    v = np.asarray(z["v_m"], dtype=np.float64)
    freqs = np.asarray(z["freqs"], dtype=np.float64)
    weights = np.asarray(z["weights"], dtype=np.float64)
    vis = np.asarray(z["vis"])
    vel = freq_to_velocity_kms(freqs)
    chan = np.flatnonzero((vel >= v_min_kms) & (vel <= v_max_kms))
    if chan.size == 0:
        raise ValueError("no channels in the requested velocity window")
    chan = chan[:: max(int(chan_stride), 1)]
    b = np.hypot(u, v)
    good = (b >= float(min_baseline_m)) & np.any(weights[:, chan] > 0.0, axis=1)
    idx = np.flatnonzero(good)
    if idx.size == 0:
        raise ValueError("no finite-weight rows in the line window")
    n_take = min(int(n_row), int(idx.size))
    rng = np.random.default_rng(rng_seed)
    pick = rng.choice(idx, size=n_take, replace=False)
    pick.sort()
    u_s, v_s = u[pick], v[pick]
    f_s = freqs[chan]
    w_s = weights[np.ix_(pick, chan)]
    vis_s, w_s, op = _hann_then_bin_if_present(vis[np.ix_(pick, chan)], w_s)
    mb = max_baseline_lambda(u_s, v_s, f_s)
    grid = image_grid_from_uv(mb, fov_co_plus_pb_arcsec())
    return NativeUvWindow(
        u_m=u_s,
        v_m=v_s,
        freqs_hz=f_s,
        weights=np.asarray(w_s, dtype=np.float64),
        vis=np.asarray(vis_s),
        grid=grid,
        operator=op,
    )


def diagonal_chi2(v_model, vis, weights) -> float:
    """Σ w |V_m − V|². Native stand-in; 066-8 must use Hann+bin χ²."""
    r = np.asarray(v_model) - np.asarray(vis)
    w = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    return float(np.sum(w * (r.real * r.real + r.imag * r.imag)))


@requires("DEC-066-PB", "DEC-066-SHIFT", "DEC-066-GRID", "DEC-066-VC", "DEC-066-PA")
def recover_stage_a(
    window: NativeUvWindow,
    template,
    vis_true,
    *,
    flux_true: float,
    pa_deg_true: float = None,
    vsys_true: float = VSYS_SEED_KM_S,
    dx_true: float = INJECT_OFFSET_ARCSEC,
    dy_true: float = INJECT_OFFSET_ARCSEC,
    gas_sigma_kms: float = GAS_SIGMA_SEED_KM_S,
    v0_kms: float = CALIBRATION_V0_KM_S,
    r_t_arcsec: float = CALIBRATION_RT_ARCSEC,
    start=None,
):
    """L-BFGS-B on ``(flux, PA, vsys, dx, dy)``. Frozen *i* is not a recovery test."""
    pa0 = pa_seed_deg() if pa_deg_true is None else float(pa_deg_true)
    vis_t, w_t, _ = _hann_then_bin_if_present(vis_true, window.weights)
    # Scaled z: flux/flux0, ΔPA/5°, Δvsys/5 km/s, dx/0.1″, dy/0.1″
    scales = np.array(
        [float(flux_true), 5.0, 5.0, 0.1, 0.1], dtype=np.float64
    )

    def unpack(z):
        z = np.asarray(z, dtype=np.float64)
        flux = z[0] * scales[0]
        pa_deg = pa0 + z[1] * scales[1]
        vsys = vsys_true + z[2] * scales[2]
        dx = z[3] * scales[3]
        dy = z[4] * scales[4]
        return flux, pa_deg, vsys, dx, dy

    def predict_from_z(z):
        flux, pa_deg, vsys, dx, dy = unpack(z)
        vm = predict_vis(
            window.u_m,
            window.v_m,
            window.freqs_hz,
            flux=flux,
            pa_rad=np.radians(pa_deg),
            vsys_kms=vsys,
            dx_arcsec=dx,
            dy_arcsec=dy,
            gas_sigma_kms=gas_sigma_kms,
            template=template,
            grid=window.grid,
            v0_kms=v0_kms,
            r_t_arcsec=r_t_arcsec,
            i_rad=inclination_rad(),
        )
        vm, _, _ = _hann_then_bin_if_present(vm, window.weights)
        return vm

    def fun(z):
        return diagonal_chi2(predict_from_z(z), vis_t, w_t)

    if start is None:
        z0 = np.array([0.75, 8.0 / 5.0, 12.0 / 5.0, 0.0, 0.0], dtype=np.float64)
    else:
        flux_s, pa_s, vsys_s, dx_s, dy_s = start
        z0 = np.array(
            [
                flux_s / scales[0],
                (pa_s - pa0) / scales[1],
                (vsys_s - vsys_true) / scales[2],
                dx_s / scales[3],
                dy_s / scales[4],
            ],
            dtype=np.float64,
        )
    nfev = {"n": 0}

    def fun_count(z):
        nfev["n"] += 1
        return fun(z)

    def jac(z):
        z = np.asarray(z, dtype=np.float64)
        f0 = fun_count(z)
        g = np.empty(z.size, dtype=np.float64)
        step = 1.0e-3
        for i in range(z.size):
            zp = z.copy()
            zp[i] += step
            g[i] = (fun_count(zp) - f0) / step
        return g

    opt = minimize(
        fun_count,
        z0,
        method="L-BFGS-B",
        jac=jac,
        bounds=[(0.05, 4.0), (-8.0, 8.0), (-8.0, 8.0), (-20.0, 20.0), (-20.0, 20.0)],
        options={"maxiter": 40, "ftol": 1e-12},
    )
    flux, pa_deg, vsys, dx, dy = unpack(opt.x)
    return RecoveryResult(
        flux=float(flux),
        pa_deg=float(pa_deg),
        vsys_kms=float(vsys),
        dx_arcsec=float(dx),
        dy_arcsec=float(dy),
        chi2=float(opt.fun),
        nfev=int(nfev["n"]),
        success=bool(opt.success),
        operator=window.operator,
    )


def stage_a_truth(*, flux: float = 1.0):
    """Injected Stage A vector for gate 2."""
    return {
        "flux": float(flux),
        "pa_rad": pa_seed_rad(),
        "pa_deg": pa_seed_deg(),
        "vsys_kms": VSYS_SEED_KM_S,
        "dx_arcsec": INJECT_OFFSET_ARCSEC,
        "dy_arcsec": INJECT_OFFSET_ARCSEC,
        "gas_sigma_kms": GAS_SIGMA_SEED_KM_S,
        "i_rad": inclination_rad(),
        "v0_kms": CALIBRATION_V0_KM_S,
        "r_t_arcsec": CALIBRATION_RT_ARCSEC,
    }
