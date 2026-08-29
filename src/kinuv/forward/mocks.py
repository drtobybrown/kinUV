"""066-7 mock recovery on a native uv window through Hann+bin.

Spectral axis is contiguous native channels plus guards, then
``kinuv.response.spectral.hann_then_bin`` (DEC-066-SPECRESP). Row subsample
only. Not the full 1920×43240 cube. 066-8 (real 066 MAP) uses the same
operator on the aggregated fit array via ``predict_binned``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from kinuv.constants import freq_to_velocity_kms
from kinuv.decisions import requires
from kinuv.geometry import inclination_rad, pa_seed_deg, pa_seed_rad
from kinuv.io.vis import N_BIN, N_GUARD, optical_to_radio_kms
from kinuv.profiles.rotation import CALIBRATION_RT_ARCSEC, CALIBRATION_V0_KM_S
from kinuv.response.spectral import bin_channels, hann_then_bin
from kinuv.transforms.grid import (
    ImageGrid,
    fov_co_plus_pb_arcsec,
    image_grid_from_uv,
    max_baseline_lambda,
)

from .model import (
    GAS_SIGMA_SEED_KM_S,
    INJECT_OFFSET_ARCSEC,
    VSYS_SEED_KM_S,
)

_LAPTOP_NPZ = Path("/Users/thbrown/kilogas/DR1/visibilities/KILOGAS066.npz")
_CANFAR_NPZ = Path(
    "/arc/projects/KILOGAS/analysis/toby_sandbox/visibilities/KILOGAS066.npz"
)
NPZ_PATH = _LAPTOP_NPZ if _LAPTOP_NPZ.is_file() else _CANFAR_NPZ

PIPELINE_KERNEL = "hann_then_bin"

# Gate 2 is recovery, not a 30-minute bench: few thousand rows, short line core.
N_ROW_MOCK = 2048
N_BINNED_CHAN_MOCK = 20


@dataclass(frozen=True)
class NativeUvWindow:
    u_m: np.ndarray
    v_m: np.ndarray
    freqs_native: np.ndarray
    vel_native: np.ndarray
    weights_native: np.ndarray
    weights: np.ndarray
    grid: ImageGrid
    operator: str
    n_bin: int = N_BIN
    n_guard: int = N_GUARD

    @property
    def freqs_hz(self) -> np.ndarray:
        """Native frequencies (guards in). Alias for older call sites."""
        return self.freqs_native


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


def _assert_hann_bin_operator(name: str) -> None:
    if name != PIPELINE_KERNEL:
        raise AssertionError(
            f"pipeline_kernel must be {PIPELINE_KERNEL!r} (Hann+bin); got {name!r}"
        )


def apply_hann_then_bin(vis_native, window: NativeUvWindow):
    """SPECRESP operator: Hann native (guards in), trim, bin ``N``."""
    n_g = int(window.n_guard)
    vel_trim = window.vel_native[n_g:-n_g]
    freqs_trim = window.freqs_native[n_g:-n_g]
    return hann_then_bin(
        vis_native,
        window.n_bin,
        n_guard=n_g,
        weights=window.weights_native,
        vel=vel_trim,
        freqs=freqs_trim,
    )


@requires("DEC-066-GRID")
def subsample_native_uv(
    npz_path: Path | None = None,
    *,
    n_row: int = N_ROW_MOCK,
    n_binned: int = N_BINNED_CHAN_MOCK,
    rng_seed: int = 66,
    min_baseline_m: float = 1.0,
) -> NativeUvWindow:
    """Real 066 ``(u,v,ν)``; contiguous native core + guards; Hann+bin.

    Row subsample only; not the full 1920×43240 cube.
    """
    path = NPZ_PATH if npz_path is None else Path(npz_path)
    z = np.load(path)
    u = np.asarray(z["u_m"], dtype=np.float64)
    v = np.asarray(z["v_m"], dtype=np.float64)
    freqs = np.asarray(z["freqs"], dtype=np.float64)
    weights = np.asarray(z["weights"], dtype=np.float64)
    vel = freq_to_velocity_kms(freqs)
    v_c = float(optical_to_radio_kms(VSYS_SEED_KM_S))
    n_core = int(n_binned) * int(N_BIN)
    n_need = n_core + 2 * int(N_GUARD)
    if freqs.size < n_need:
        raise ValueError(f"native axis shorter than Hann+bin window ({n_need})")
    mid = int(np.argmin(np.abs(vel - v_c)))
    i0 = mid - n_need // 2
    i0 = max(0, min(i0, int(freqs.size) - n_need))
    i1 = i0 + n_need
    chan = np.arange(i0, i1)
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
    f_native = freqs[chan]
    vel_native = vel[chan]
    w_full = weights[np.ix_(pick, chan)]
    n_g = int(N_GUARD)
    w_trim = w_full[:, n_g:-n_g]
    vel_trim = vel_native[n_g:-n_g]
    freqs_trim = f_native[n_g:-n_g]
    dummy = np.zeros_like(w_trim, dtype=np.complex128)
    _, w_b, _, _, _ = bin_channels(dummy, w_trim, vel_trim, freqs_trim, int(N_BIN))
    mb = max_baseline_lambda(u_s, v_s, f_native)
    grid = image_grid_from_uv(mb, fov_co_plus_pb_arcsec())
    _assert_hann_bin_operator(PIPELINE_KERNEL)
    return NativeUvWindow(
        u_m=u_s,
        v_m=v_s,
        freqs_native=f_native,
        vel_native=vel_native,
        weights_native=np.asarray(w_trim, dtype=np.float64),
        weights=np.asarray(w_b, dtype=np.float64),
        grid=grid,
        operator=PIPELINE_KERNEL,
        n_bin=int(N_BIN),
        n_guard=n_g,
    )


def diagonal_chi2(v_model, vis, weights) -> float:
    """Σ w |V_m − V|² on the **binned** array (after Hann+bin)."""
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
    vsys_true: float | None = None,
    dx_true: float = INJECT_OFFSET_ARCSEC,
    dy_true: float = INJECT_OFFSET_ARCSEC,
    gas_sigma_kms: float = GAS_SIGMA_SEED_KM_S,
    v0_kms: float = CALIBRATION_V0_KM_S,
    r_t_arcsec: float = CALIBRATION_RT_ARCSEC,
    start=None,
):
    """L-BFGS-B on ``(flux, PA, vsys, dx, dy)``. Frozen *i* is not a recovery test."""
    _assert_hann_bin_operator(window.operator)
    pa0 = pa_seed_deg() if pa_deg_true is None else float(pa_deg_true)
    if vsys_true is None:
        vsys_true = float(optical_to_radio_kms(VSYS_SEED_KM_S))
    vis_t = apply_hann_then_bin(vis_true, window)
    w_t = window.weights
    scales = np.array(
        [float(flux_true), 5.0, 5.0, 0.1, 0.1], dtype=np.float64
    )
    from .model import predict_vis

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
            window.freqs_native,
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
        return apply_hann_then_bin(vm, window)

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
        "vsys_kms": float(optical_to_radio_kms(VSYS_SEED_KM_S)),
        "dx_arcsec": INJECT_OFFSET_ARCSEC,
        "dy_arcsec": INJECT_OFFSET_ARCSEC,
        "gas_sigma_kms": GAS_SIGMA_SEED_KM_S,
        "i_rad": inclination_rad(),
        "v0_kms": CALIBRATION_V0_KM_S,
        "r_t_arcsec": CALIBRATION_RT_ARCSEC,
    }
