"""Stage A arctan and Stage B ring V_c(r) (DEC-066-VC, DEC-066-OSCMETRIC).

Stage A is analytic for every radius. Stage B is piecewise-linear between
knots, solid-body inside ``r_0``, and flat for ``R > r_last``. The curvature
term is a discrete second-difference (cubic-spline / integrated-Wiener)
penalty, not a GP / SE / Matérn kernel. L-BFGS is not run here; bounds are
encoded as constants for a later fitter.
"""

from __future__ import annotations

import numpy as np

from kinuv.decisions import requires

# DEC-066-OSCMETRIC: Ico restoring beam and innermost knot.
BMAJ_ICO_ARCSEC = 1.30
R0_MIN_OVER_BMAJ = 0.5
DISK_RADIUS_ARCSEC = 7.5
N_RINGS_MIN = 6
N_RINGS_MAX = 8
DR_TARGET_MIN_ARCSEC = 1.0
DR_TARGET_MAX_ARCSEC = 1.3

# L-BFGS box on ring amplitudes (DEC-066-VC). Spec only; no optimiser here.
V_K_MIN_KM_S = 0.0
V_K_MAX_KM_S = 400.0

# Local 066 npz inventory: radio vs rest CO, native N=1, n_chan=1920.
# Callers pass replica ~5.3 or bin-8 ~10.6 later; do not hardcode 10.6.
DV_CHAN_NATIVE_KM_S = 1.270

# Calibration truth and acceptance (DEC-066-OSCMETRIC). Campaign is later.
CALIBRATION_V0_KM_S = 200.0
CALIBRATION_RT_ARCSEC = 3.0
OMEGA_ACCEPT_MAX = 0.3
OMEGA_PASS_FRACTION = 0.95
RECOVERY_1SIGMA_FRACTION = 0.68

_INNER_SOLID = "solid_body"
_INNER_FLAT = "flat"


def _f64(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def r0_min_arcsec(bmaj_arcsec: float = BMAJ_ICO_ARCSEC) -> float:
    """Innermost knot floor: ``r_0 >= 0.5 BMAJ``."""
    return R0_MIN_OVER_BMAJ * float(bmaj_arcsec)


@requires("DEC-066-OSCMETRIC")
def uniform_knot_radii(
    n_rings: int = 7,
    *,
    r0_arcsec: float | None = None,
    r_last_arcsec: float = DISK_RADIUS_ARCSEC,
    bmaj_arcsec: float = BMAJ_ICO_ARCSEC,
) -> np.ndarray:
    """Uniform knots on ``[r_0, r_last]`` with ADR placement constraints."""
    if not N_RINGS_MIN <= int(n_rings) <= N_RINGS_MAX:
        raise ValueError(
            f"N_rings must be {N_RINGS_MIN}–{N_RINGS_MAX}, got {n_rings}"
        )
    r0_floor = r0_min_arcsec(bmaj_arcsec)
    r0 = r0_floor if r0_arcsec is None else float(r0_arcsec)
    r_last = float(r_last_arcsec)
    if r0 < r0_floor:
        raise ValueError(
            f"r_0={r0}″ < 0.5 BMAJ={r0_floor}″ (BMAJ={bmaj_arcsec}″)"
        )
    if r_last <= r0:
        raise ValueError(f"r_last={r_last}″ must exceed r_0={r0}″")
    return np.linspace(r0, r_last, int(n_rings), dtype=np.float64)


@requires("DEC-066-VC")
def arctan_vc(radius_arcsec, v0_kms: float, r_t_arcsec: float):
    """``V_c(r) = V_0 (2/π) arctan(r/r_t)`` for all ``R``. Do not flatten."""
    r = _f64(radius_arcsec)
    if np.any(r < 0.0):
        raise ValueError("radius_arcsec must be >= 0")
    if r_t_arcsec <= 0.0:
        raise ValueError("r_t_arcsec must be positive")
    return float(v0_kms) * (2.0 / np.pi) * np.arctan(r / float(r_t_arcsec))


@requires("DEC-066-VC")
def rings_from_arctan(r_knots_arcsec, v0_kms: float, r_t_arcsec: float) -> np.ndarray:
    """Initialise ring amplitudes on the Stage A arctan. No cold start."""
    return np.asarray(arctan_vc(r_knots_arcsec, v0_kms, r_t_arcsec), dtype=np.float64)


def _validate_knots(r_knots: np.ndarray, v_knots: np.ndarray) -> None:
    if r_knots.ndim != 1 or v_knots.ndim != 1:
        raise ValueError("knots must be 1-d")
    if r_knots.size != v_knots.size:
        raise ValueError("r_knots and v_knots length mismatch")
    n = r_knots.size
    if not N_RINGS_MIN <= n <= N_RINGS_MAX:
        raise ValueError(f"N_rings must be {N_RINGS_MIN}–{N_RINGS_MAX}, got {n}")
    if np.any(np.diff(r_knots) <= 0.0):
        raise ValueError("r_knots must be strictly increasing")
    if r_knots[0] < 0.0:
        raise ValueError("r_0 must be >= 0")


@requires("DEC-066-VC", "DEC-066-OSCMETRIC")
def ring_vc(
    radius_arcsec,
    r_knots_arcsec,
    v_knots_kms,
    *,
    inner_bc: str = _INNER_SOLID,
):
    """Evaluate Stage B rings: solid-body inner, linear knots, flat outer.

    ``inner_bc='flat'`` is a diagnostic control only (not the 066 evaluation).
    """
    r_in = _f64(radius_arcsec)
    r_k = _f64(r_knots_arcsec)
    v_k = _f64(v_knots_kms)
    if np.any(r_in < 0.0):
        raise ValueError("radius_arcsec must be >= 0")
    _validate_knots(r_k, v_k)
    if inner_bc not in (_INNER_SOLID, _INNER_FLAT):
        raise ValueError(f"inner_bc must be {_INNER_SOLID!r} or {_INNER_FLAT!r}")
    r0, v0 = r_k[0], v_k[0]
    r_last, v_last = r_k[-1], v_k[-1]
    r = np.atleast_1d(r_in)
    out = np.empty_like(r)
    inner = r < r0
    outer = r > r_last
    mid = ~inner & ~outer
    if inner_bc == _INNER_SOLID:
        if r0 <= 0.0:
            raise ValueError("solid-body inner BC needs r_0 > 0")
        out[inner] = v0 * (r[inner] / r0)
    else:
        out[inner] = v0
    out[mid] = np.interp(r[mid], r_k, v_k)
    out[outer] = v_last
    if r_in.ndim == 0:
        return out[0]
    return out


def ring_velocity_bounds(n_rings: int) -> np.ndarray:
    """L-BFGS box ``V_k ∈ [0, 400]`` km/s, shape ``(N_rings, 2)``."""
    if not N_RINGS_MIN <= int(n_rings) <= N_RINGS_MAX:
        raise ValueError(
            f"N_rings must be {N_RINGS_MIN}–{N_RINGS_MAX}, got {n_rings}"
        )
    return np.tile(
        np.array([V_K_MIN_KM_S, V_K_MAX_KM_S], dtype=np.float64),
        (int(n_rings), 1),
    )


def _second_differences(v_knots_kms) -> np.ndarray:
    v = _f64(v_knots_kms)
    if v.ndim != 1 or v.size < 3:
        raise ValueError("need >= 3 ring amplitudes for second differences")
    return v[2:] - 2.0 * v[1:-1] + v[:-2]


@requires("DEC-066-OSCMETRIC")
def curvature_penalty(v_knots_kms, lam_reg: float) -> float:
    """``λ_reg Σ (V_{k+1} − 2 V_k + V_{k-1})²``. Not a GP."""
    d2 = _second_differences(v_knots_kms)
    return float(lam_reg) * float(np.sum(d2 * d2))


@requires("DEC-066-OSCMETRIC")
def monotonicity_penalty(v_knots_kms, mu_mono: float) -> float:
    """Soft ``Φ_mono = μ_mono Σ max(0, V_k − V_{k+1})²`` inside the last knot.

    The outermost ring may decline without penalty. ``μ_mono = λ_reg`` on 066.
    """
    v = _f64(v_knots_kms)
    if v.ndim != 1 or v.size < 2:
        raise ValueError("need >= 2 ring amplitudes")
    decline = v[:-1] - v[1:]
    interior = decline[:-1]
    return float(mu_mono) * float(np.sum(np.maximum(0.0, interior) ** 2))


@requires("DEC-066-OSCMETRIC")
def ring_regulariser(v_knots_kms, lam_reg: float) -> float:
    """Curvature plus 066 heuristic ``μ_mono = λ_reg`` monotonicity."""
    return curvature_penalty(v_knots_kms, lam_reg) + monotonicity_penalty(
        v_knots_kms, lam_reg
    )


@requires("DEC-066-OSCMETRIC")
def omega_k(v_knots_kms, dv_chan_kms: float = DV_CHAN_NATIVE_KM_S) -> np.ndarray:
    """``Ω_k = |V_{k+1} − 2 V_k + V_{k-1}| / Δv_chan`` (length ``N_rings − 2``)."""
    dv = abs(float(dv_chan_kms))
    if dv == 0.0:
        raise ValueError("dv_chan_kms must be nonzero")
    return np.abs(_second_differences(v_knots_kms)) / dv


def k_extra_rings(n_rings: int) -> int:
    """``k_extra = N_rings − 2`` (Stage B vs Stage A's two arctan parameters)."""
    return int(n_rings) - 2


@requires("DEC-066-OSCMETRIC")
def aic_keep_stage_a(
    aic_stage_a: float,
    aic_stage_b: float,
    n_rings: int,
) -> bool:
    """Keep Stage A unless Stage B beats it by more than ``ΔAIC = 2 k_extra``.

    AIC, not BIC. Lower AIC is better: retain A if
    ``AIC_A − AIC_B <= 2 (N_rings − 2)``.
    """
    threshold = 2.0 * k_extra_rings(n_rings)
    return (float(aic_stage_a) - float(aic_stage_b)) <= threshold


def _broadcast_sigma(sigma, shape: tuple[int, int]) -> np.ndarray:
    arr = _f64(sigma)
    if arr.ndim == 0:
        return np.full(shape, float(arr))
    if arr.shape == shape:
        return arr
    if arr.ndim == 1 and arr.size == shape[1]:
        return np.broadcast_to(arr, shape)
    raise ValueError(f"sigma shape {arr.shape} does not broadcast to {shape}")


@requires("DEC-066-OSCMETRIC")
def select_lambda_reg(
    lambdas,
    max_omega,
    v0_recovered,
    rt_recovered,
    v0_sigma,
    rt_sigma,
    v0_stage_a,
    *,
    v0_truth: float = CALIBRATION_V0_KM_S,
    rt_truth: float = CALIBRATION_RT_ARCSEC,
    omega_max: float = OMEGA_ACCEPT_MAX,
    omega_pass_fraction: float = OMEGA_PASS_FRACTION,
    recovery_fraction: float = RECOVERY_1SIGMA_FRACTION,
):
    """Smallest ``λ_reg`` that satisfies the three OSCMETRIC inequalities.

    Operates on mock summary arrays. Does not fit visibilities. ``max_omega``,
    ``v0_recovered``, and ``rt_recovered`` have shape ``(n_lambda, n_mock)``.
    """
    lam = _f64(lambdas)
    omega = _f64(max_omega)
    v0_b = _f64(v0_recovered)
    rt_b = _f64(rt_recovered)
    v0_a = _f64(v0_stage_a)
    if lam.ndim != 1:
        raise ValueError("lambdas must be 1-d")
    n_lam = lam.size
    if omega.shape[0] != n_lam or v0_b.shape != omega.shape or rt_b.shape != omega.shape:
        raise ValueError("lambda/mock array shapes disagree")
    n_mock = omega.shape[1]
    if n_mock < 2:
        raise ValueError("need >= 2 mocks to form a 1σ scatter")
    if v0_a.shape == (n_mock,):
        v0_a = np.broadcast_to(v0_a, omega.shape)
    elif v0_a.shape != omega.shape:
        raise ValueError("v0_stage_a shape must be (n_mock,) or (n_lambda, n_mock)")
    v0_sig = _broadcast_sigma(v0_sigma, omega.shape)
    rt_sig = _broadcast_sigma(rt_sigma, omega.shape)
    order = np.argsort(lam)
    for i in order:
        omega_ok = np.mean(omega[i] < omega_max) >= omega_pass_fraction
        within = (np.abs(v0_b[i] - v0_truth) <= v0_sig[i]) & (
            np.abs(rt_b[i] - rt_truth) <= rt_sig[i]
        )
        rec_ok = np.mean(within) >= recovery_fraction
        scatter = float(np.std(v0_b[i], ddof=1))
        mean_b = float(np.mean(v0_b[i]))
        mean_a = float(np.mean(v0_a[i]))
        bias_ok = mean_b >= mean_a - scatter
        if omega_ok and rec_ok and bias_ok:
            return float(lam[i])
    return None


def run_lambda_reg_campaign(*args, **kwargs):
    """20 mocks × ~5 ``λ_reg`` on real 066 uv (066-12). Delegates the loop."""
    from kinuv.infer.campaign import calibrate_lambda_reg

    return calibrate_lambda_reg(*args, **kwargs)
