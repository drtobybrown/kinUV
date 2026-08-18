"""KGAS066 visibility load, cube-window trim, time/uv aggregate (DEC-066-VIS).

Source is the native ``43240×1920`` npz. Fit product is time-averaged 30 s,
uv-binned 10 m, then software-binned ``N=4``. Data are already
correlator-Hann'd — this module never Hanns visibilities.

The Ico cube is ``VOPT``; visibilities are radio vs rest CO via
:func:`kinuv.constants.freq_to_velocity_kms`. The trim converts the cube
window to radio so the same sky frequencies are selected. YAML
``obs_freq_range`` is not used (it clips the receding side).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kinuv.constants import C_LIGHT_KM_S, freq_to_velocity_kms
from kinuv.decisions import requires
from kinuv.likelihood.chi2 import empirical_s
from kinuv.response.spectral import bin_channels

DEFAULT_NPZ = Path("/Users/thbrown/kilogas/DR1/visibilities/KILOGAS066.npz")
DEFAULT_CUBE = Path(
    "/Users/thbrown/kilogas/analysis/kinms_test/kgas066/KGAS66_clipped_cube.fits"
)

# KGAS66_clipped_cube.fits CTYPE3=VOPT-W2W channel-centre span (dispatch 8034–8536).
CUBE_VOPT_LO_KMS = 8034.059711054924
CUBE_VOPT_HI_KMS = 8535.65720660806

N_BIN = 4
TIME_BIN_S = 30.0
UV_BIN_M = 10.0
TRIM_MARGIN_NATIVE = 3
N_GUARD = 1
NATIVE_N_ROW = 43240
NATIVE_N_CHAN = 1920


@dataclass
class VisData:
    """Fit-array visibilities plus the native model axis (trim + guards)."""

    u_m: np.ndarray
    v_m: np.ndarray
    vis: np.ndarray
    weights: np.ndarray
    freqs: np.ndarray
    vel: np.ndarray
    freqs_native: np.ndarray
    vel_native: np.ndarray
    n_bin: int
    dv_kms: float
    s: float
    phase_dir_rad: np.ndarray
    line_free_mask: np.ndarray
    n_guard: int = N_GUARD
    weights_native: np.ndarray | None = None
    v_lo_line: float = 0.0
    v_hi_line: float = 0.0


def optical_to_radio_kms(v_opt_kms, c_kms: float = C_LIGHT_KM_S):
    """``v_radio = v_opt / (1 + v_opt/c)``. Cube is VOPT; vis are radio."""
    v = np.asarray(v_opt_kms, dtype=np.float64)
    return v / (1.0 + v / c_kms)


def cube_vopt_window_kms(cube_path: Path | None = DEFAULT_CUBE) -> tuple[float, float]:
    """Channel-centre VOPT min/max from the Ico cube, or the dispatch span."""
    path = Path(cube_path) if cube_path is not None else DEFAULT_CUBE
    if path.is_file():
        try:
            from astropy.io import fits
        except ImportError:
            return CUBE_VOPT_LO_KMS, CUBE_VOPT_HI_KMS
        header = fits.getheader(path)
        n = int(header["NAXIS3"])
        vel = float(header["CRVAL3"]) + (
            np.arange(1, n + 1, dtype=np.float64) - float(header["CRPIX3"])
        ) * float(header["CDELT3"])
        return float(np.min(vel)), float(np.max(vel))
    return CUBE_VOPT_LO_KMS, CUBE_VOPT_HI_KMS


def _weighted_row_reduce(u_m, v_m, vis, weights, inv, n_grp):
    """Weighted mean vis per group ``inv``; uv centroid uses row-sum weights."""
    u64 = np.asarray(u_m, dtype=np.float64)
    v64 = np.asarray(v_m, dtype=np.float64)
    w64 = np.asarray(weights, dtype=np.float64)
    vis_c = np.asarray(vis, dtype=np.complex128)
    w_row = np.sum(w64, axis=1)
    u_acc = np.zeros(n_grp, dtype=np.float64)
    v_acc = np.zeros(n_grp, dtype=np.float64)
    w_uv = np.zeros(n_grp, dtype=np.float64)
    np.add.at(u_acc, inv, u64 * w_row)
    np.add.at(v_acc, inv, v64 * w_row)
    np.add.at(w_uv, inv, w_row)
    safe = np.maximum(w_uv, 1e-40)
    u_out = u_acc / safe
    v_out = v_acc / safe
    numer = np.zeros((n_grp, vis_c.shape[1]), dtype=np.complex128)
    denom = np.zeros((n_grp, vis_c.shape[1]), dtype=np.float64)
    np.add.at(numer, inv, vis_c * w64)
    np.add.at(denom, inv, w64)
    vis_out = np.divide(
        numer, denom, out=np.zeros_like(numer), where=denom > 0.0
    )
    return u_out, v_out, vis_out, denom


def average_time_steps(
    u_m,
    v_m,
    vis,
    weights,
    time_s,
    bin_s: float,
    baseline_ids,
):
    """Average visibilities in ``bin_s``-second bins per physical baseline."""
    if bin_s <= 0.0:
        raise ValueError(f"bin_s must be positive; got {bin_s}")
    vis = np.asarray(vis)
    weights = np.asarray(weights)
    if vis.shape != weights.shape:
        raise ValueError("vis and weights must have the same shape")
    nrow = vis.shape[0]
    if u_m.shape != (nrow,) or v_m.shape != (nrow,):
        raise ValueError("u_m, v_m must be 1D with length n_row")
    time_s = np.asarray(time_s, dtype=np.float64).ravel()
    baseline_ids = np.asarray(baseline_ids, dtype=np.int64).ravel()
    if time_s.shape[0] != nrow or baseline_ids.shape[0] != nrow:
        raise ValueError("time_s and baseline_ids must match vis row count")

    t_rel = time_s - np.min(time_s)
    tb = np.floor(t_rel / bin_s).astype(np.int64)
    stack = np.column_stack([tb, baseline_ids])
    _, inv = np.unique(stack, axis=0, return_inverse=True)
    n_grp = int(inv.max()) + 1 if inv.size else 0
    return _weighted_row_reduce(u_m, v_m, vis, weights, inv, n_grp)


def bin_uv_plane(u_m, v_m, vis, weights, bin_size_m: float):
    """Grid-average visibilities in ``bin_size_m``-metre UV cells."""
    if bin_size_m <= 0.0:
        raise ValueError(f"bin_size_m must be positive; got {bin_size_m}")
    vis = np.asarray(vis)
    weights = np.asarray(weights)
    if vis.shape != weights.shape:
        raise ValueError("vis and weights must have the same shape")
    nrow = vis.shape[0]
    if u_m.shape != (nrow,) or v_m.shape != (nrow,):
        raise ValueError("u_m, v_m must be 1D with length n_row")

    u64 = np.asarray(u_m, dtype=np.float64)
    v64 = np.asarray(v_m, dtype=np.float64)
    iu = np.floor(u64 / bin_size_m).astype(np.int64)
    iv = np.floor(v64 / bin_size_m).astype(np.int64)
    pairs = np.column_stack([iu, iv])
    _, inv = np.unique(pairs, axis=0, return_inverse=True)
    n_bins = int(inv.max()) + 1 if inv.size else 0
    return _weighted_row_reduce(u_m, v_m, vis, weights, inv, n_bins)


def _trim_and_guard_indices(vel, v_lo_line, v_hi_line, *, margin, n_guard):
    dv = float(np.median(np.abs(np.diff(vel)))) if vel.size > 1 else 1.0
    v_lo_trim = float(v_lo_line) - margin * dv
    v_hi_trim = float(v_hi_line) + margin * dv
    trim = (vel >= v_lo_trim) & (vel <= v_hi_trim)
    if int(np.sum(trim)) < 2:
        raise ValueError(
            f"Cube trim [{v_lo_trim:.1f}, {v_hi_trim:.1f}] km/s leaves "
            f"{int(np.sum(trim))} native channels"
        )
    idx = np.flatnonzero(trim)
    i0, i1 = int(idx[0]), int(idx[-1])
    n_chan = int(vel.shape[0])
    g0 = i0 - n_guard
    g1 = i1 + n_guard
    dnu = float(np.median(np.diff(vel))) if vel.size > 1 else dv
    extra_lo = extra_hi = 0
    if g0 < 0:
        extra_lo = -g0
        g0 = 0
    if g1 > n_chan - 1:
        extra_hi = g1 - (n_chan - 1)
        g1 = n_chan - 1
    return i0, i1, g0, g1, dv, extra_lo, extra_hi, dnu


def _extend_axis(freq_core, vel_core, extra_lo, extra_hi, dvel):
    """If the npz has no extra channels, evaluate ν_edge ± Δν_native."""
    freqs = np.asarray(freq_core, dtype=np.float64)
    vel = np.asarray(vel_core, dtype=np.float64)
    if extra_lo:
        df = float(np.median(np.diff(freqs))) if freqs.size > 1 else 0.0
        lo_f = freqs[0] + df * np.arange(-extra_lo, 0, dtype=np.float64)
        lo_v = vel[0] + dvel * np.arange(-extra_lo, 0, dtype=np.float64)
        freqs = np.concatenate([lo_f, freqs])
        vel = np.concatenate([lo_v, vel])
    if extra_hi:
        df = float(np.median(np.diff(freqs))) if freqs.size > 1 else 0.0
        hi_f = freqs[-1] + df * np.arange(1, extra_hi + 1, dtype=np.float64)
        hi_v = vel[-1] + dvel * np.arange(1, extra_hi + 1, dtype=np.float64)
        freqs = np.concatenate([freqs, hi_f])
        vel = np.concatenate([vel, hi_v])
    return freqs, vel


@requires("DEC-066-VIS", "DEC-066-SPECRESP", "DEC-066-WEIGHT", "DEC-066-ZEROMODEL")
def load_kgas066(
    path=DEFAULT_NPZ,
    *,
    cube_path=DEFAULT_CUBE,
    n_bin: int = N_BIN,
    time_bin_s: float = TIME_BIN_S,
    uv_bin_m: float = UV_BIN_M,
    trim_margin: int = TRIM_MARGIN_NATIVE,
    n_guard: int = N_GUARD,
) -> VisData:
    """Load native KGAS066, trim to the Ico cube + margin, aggregate, bin ``N``.

    Does **not** Hann the data. Records ``(n_row, n_chan, Δv_kms, N)`` on the
    returned :class:`VisData`. ``s`` is measured on line-free **fit** channels.
    """
    npz_path = Path(path)
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)

    z = np.load(npz_path, mmap_mode="r")
    required = ("u_m", "v_m", "vis", "weights", "freqs", "time", "baseline")
    missing = [k for k in required if k not in z.files]
    if missing:
        raise KeyError(f"{npz_path} missing keys {missing}")

    freqs_all = np.asarray(z["freqs"], dtype=np.float64).ravel()
    vel_all = freq_to_velocity_kms(freqs_all)
    v_lo_opt, v_hi_opt = cube_vopt_window_kms(cube_path)
    v_lo_line = float(optical_to_radio_kms(v_lo_opt))
    v_hi_line = float(optical_to_radio_kms(v_hi_opt))
    i0, i1, g0, g1, dv_native, extra_lo, extra_hi, dvel = _trim_and_guard_indices(
        vel_all,
        v_lo_line,
        v_hi_line,
        margin=int(trim_margin),
        n_guard=int(n_guard),
    )

    sl = slice(i0, i1 + 1)
    vis = np.asarray(z["vis"][:, sl], dtype=np.complex128)
    weights = np.asarray(z["weights"][:, sl], dtype=np.float64)
    u_m = np.asarray(z["u_m"], dtype=np.float64)
    v_m = np.asarray(z["v_m"], dtype=np.float64)
    time_s = np.asarray(z["time"], dtype=np.float64).ravel()
    baseline = np.asarray(z["baseline"], dtype=np.int64).ravel()
    phase_dir = np.asarray(z["phase_dir_rad"], dtype=np.float64).ravel()
    freqs_trim = freqs_all[sl]
    vel_trim = vel_all[sl]
    freqs_native = freqs_all[g0 : g1 + 1]
    vel_native = vel_all[g0 : g1 + 1]
    freqs_native, vel_native = _extend_axis(
        freqs_native, vel_native, extra_lo, extra_hi, dvel
    )

    u_m, v_m, vis, weights = average_time_steps(
        u_m, v_m, vis, weights, time_s, float(time_bin_s), baseline
    )
    u_m, v_m, vis, weights = bin_uv_plane(u_m, v_m, vis, weights, float(uv_bin_m))

    vis_b, w_b, vel_b, freqs_b, _ = bin_channels(
        vis, weights, vel_trim, freqs_trim, int(n_bin)
    )
    dv_kms = (
        float(np.median(np.abs(np.diff(vel_b)))) if vel_b.size > 1 else float(n_bin) * dv_native
    )
    line_free = (vel_b < v_lo_line) | (vel_b > v_hi_line)
    s = empirical_s(vis_b, w_b, line_free)
    return VisData(
        u_m=u_m,
        v_m=v_m,
        vis=vis_b,
        weights=w_b,
        freqs=freqs_b,
        vel=vel_b,
        freqs_native=freqs_native,
        vel_native=vel_native,
        n_bin=int(n_bin),
        dv_kms=dv_kms,
        s=s,
        phase_dir_rad=phase_dir,
        line_free_mask=line_free,
        n_guard=int(n_guard),
        weights_native=weights,
        v_lo_line=v_lo_line,
        v_hi_line=v_hi_line,
    )
