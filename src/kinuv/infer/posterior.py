"""Stage A Laplace and Laplace-preconditioned MH. Not autodiff NUTS.

``ln L = -chi2/2 - prior/2``. ``chi2 = s * sum w |d-m|^2`` (XX empirical s).
A correct model has ``E[chi2] = 2 * n_vis``. Temperature ``T`` scales ``s``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kinuv.decisions import requires
from kinuv.likelihood.chi2 import chi2

SAMPLER_NAME = "laplace_mh"
FD_STEP = 1.0e-3
PARAM_NAMES = (
    "flux", "pa_deg", "vsys_kms", "gas_sigma_kms",
    "dx_arcsec", "dy_arcsec", "v0_kms", "r_t_arcsec",
)
Z68 = 0.994457883209753
Z95 = 1.959963984540054


def n_vis_of(data) -> int:
    return int(data.vis.shape[0] * data.vis.shape[1])


def t_dof(chi2_val: float, n_vis: int) -> float:
    """``chi2 / (2 n_vis)``. Product scale on real 066."""
    n = int(n_vis)
    if n < 1:
        raise ValueError("n_vis must be >= 1")
    return float(chi2_val) / (2.0 * n)


def t_nvis(chi2_val: float, n_vis: int) -> float:
    """``chi2 / n_vis``. Sensitivity if |z|^2 is treated as 1 dof."""
    n = int(n_vis)
    if n < 1:
        raise ValueError("n_vis must be >= 1")
    return float(chi2_val) / n


def params_to_vec(params: dict[str, float]) -> np.ndarray:
    return np.array([float(params[n]) for n in PARAM_NAMES], dtype=np.float64)


def vec_to_params(x, template: dict[str, float] | None = None) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    out = {} if template is None else dict(template)
    for i, name in enumerate(PARAM_NAMES):
        out[name] = float(x[i])
    return out


def gaussian_interval(mean: float, var: float, z: float) -> tuple[float, float]:
    sig = float(np.sqrt(max(float(var), 0.0)))
    m = float(mean)
    return m - z * sig, m + z * sig


def in_interval(truth: float, lo: float, hi: float) -> bool:
    return float(lo) <= float(truth) <= float(hi)


@requires("DEC-066-WEIGHT", "DEC-066-SHIFT")
def chi2_and_prior(data, params, template, grid, *, t: float = 1.0) -> float:
    """``chi2(s_eff) + shift_prior``. ``s_eff = s * T``."""
    from kinuv.infer.map import predict_binned, shift_prior

    model = predict_binned(data, params, template, grid)
    c = chi2(data.vis, model, data.weights, float(data.s) * float(t))
    return float(c) + shift_prior(params["dx_arcsec"], params["dy_arcsec"])


def log_prob(data, params, template, grid, *, t: float = 1.0) -> float:
    return -0.5 * chi2_and_prior(data, params, template, grid, t=t)


def fd_hessian(fun, x, *, step: float = FD_STEP) -> np.ndarray:
    """Central finite-difference Hessian of a scalar ``fun(x)``."""
    x = np.asarray(x, dtype=np.float64)
    n = int(x.size)
    h = float(step)
    f0 = float(fun(x))
    hess = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        xp = x.copy()
        xm = x.copy()
        xp[i] += h
        xm[i] -= h
        hess[i, i] = (float(fun(xp)) - 2.0 * f0 + float(fun(xm))) / (h * h)
        for j in range(i + 1, n):
            xpp = x.copy()
            xpm = x.copy()
            xmp = x.copy()
            xmm = x.copy()
            xpp[i] += h
            xpp[j] += h
            xpm[i] += h
            xpm[j] -= h
            xmp[i] -= h
            xmp[j] += h
            xmm[i] -= h
            xmm[j] -= h
            val = (
                float(fun(xpp)) - float(fun(xpm)) - float(fun(xmp)) + float(fun(xmm))
            ) / (4.0 * h * h)
            hess[i, j] = val
            hess[j, i] = val
    return hess


def laplace_cov(hess_chi2, *, t: float = 1.0) -> np.ndarray:
    """``cov = T * 2 * inv(H_chi2)`` because ``ln L = -chi2/2``."""
    h = np.asarray(hess_chi2, dtype=np.float64)
    try:
        cov = float(t) * 2.0 * np.linalg.inv(h)
    except np.linalg.LinAlgError:
        cov = float(t) * 2.0 * np.linalg.pinv(h)
    return cov


def _chol_psd(cov: np.ndarray) -> np.ndarray:
    cov = np.asarray(cov, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh(cov)
        w = np.maximum(w, 1.0e-18)
        return v @ np.diag(np.sqrt(w))


def _log_q_normal(x, mu, chol) -> float:
    """Unnormalized is fine for MH if cov is fixed; use exact log density."""
    d = x.size
    y = np.linalg.solve(chol, x - mu)
    logdet = 2.0 * float(np.sum(np.log(np.diag(chol))))
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + float(y @ y))


@dataclass
class MhResult:
    samples: np.ndarray  # (n_chain, n_draw, n_param)
    accept: float
    nfev: int
    eval_s: float
    sampler: str = SAMPLER_NAME


def mh_sample(
    logp_vec,
    x0,
    cov,
    *,
    n_chain: int = 4,
    n_warmup: int = 300,
    n_draw: int = 1200,
    rng: np.random.Generator | None = None,
) -> MhResult:
    """Independence MH with Laplace proposal ``N(x0, cov)``. ``sampler: laplace_mh``."""
    from time import perf_counter

    x0 = np.asarray(x0, dtype=np.float64)
    chol = _chol_psd(cov)
    rng = np.random.default_rng() if rng is None else rng
    n_tot = int(n_warmup) + int(n_draw)
    n_chain = int(n_chain)
    n_par = int(x0.size)
    draws = np.empty((n_chain, n_tot, n_par), dtype=np.float64)
    nfev = 0
    n_acc = 0
    n_prop = 0
    t0 = perf_counter()
    _ = logp_vec(x0)
    eval_s = perf_counter() - t0
    nfev += 1
    logp0 = float(_)
    logq0 = _log_q_normal(x0, x0, chol)

    for c in range(n_chain):
        x = x0 + 0.05 * (chol @ rng.standard_normal(n_par))
        lp = float(logp_vec(x))
        lq = _log_q_normal(x, x0, chol)
        nfev += 1
        for t in range(n_tot):
            z = rng.standard_normal(n_par)
            prop = x0 + chol @ z
            lp_p = float(logp_vec(prop))
            lq_p = _log_q_normal(prop, x0, chol)
            nfev += 1
            n_prop += 1
            log_a = (lp_p + lq) - (lp + lq_p)
            if np.log(rng.uniform()) < min(0.0, log_a):
                x, lp, lq = prop, lp_p, lq_p
                n_acc += 1
            draws[c, t] = x
    acc = n_acc / max(n_prop, 1)
    return MhResult(
        samples=draws[:, int(n_warmup) :, :],
        accept=float(acc),
        nfev=int(nfev),
        eval_s=float(eval_s),
        sampler=SAMPLER_NAME,
    )


def split_rhat(chains: np.ndarray) -> np.ndarray:
    """Split-R_hat (Vehtari et al. 2021). ``chains`` is (n_chain, n_draw, n_param)."""
    a = np.asarray(chains, dtype=np.float64)
    if a.ndim != 3:
        raise ValueError("chains must be (n_chain, n_draw, n_param)")
    n_c, n_d, n_p = a.shape
    if n_d < 4:
        return np.full(n_p, np.nan)
    half = n_d // 2
    split = np.concatenate([a[:, :half, :], a[:, n_d - half :, :]], axis=0)
    m, n = split.shape[0], split.shape[1]
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        mu_j = split.mean(axis=1)
        mu = mu_j.mean(axis=0)
        b = n * np.var(mu_j, axis=0, ddof=1)
        w = np.mean(np.var(split, axis=1, ddof=1), axis=0)
        var_hat = ((n - 1) / n) * w + b / n
        rhat = np.sqrt(var_hat / w)
    return np.asarray(rhat, dtype=np.float64)


def _acf_fft(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        x = x - x.mean()
        n = x.size
        f = np.fft.rfft(x, n=2 * n)
        ac = np.fft.irfft(f * np.conjugate(f), n=2 * n)[:n].real
        ac /= ac[0] if ac[0] != 0.0 else 1.0
    return ac


def ess_bulk(chains: np.ndarray) -> np.ndarray:
    """Bulk ESS from split chains (Geyer initial monotone truncation)."""
    a = np.asarray(chains, dtype=np.float64)
    if a.ndim != 3:
        raise ValueError("chains must be (n_chain, n_draw, n_param)")
    n_c, n_d, n_p = a.shape
    n_tot = n_c * n_d
    out = np.empty(n_p, dtype=np.float64)
    for p in range(n_p):
        rho = np.zeros(n_d, dtype=np.float64)
        for c in range(n_c):
            rho += _acf_fft(a[c, :, p])[:n_d]
        rho /= n_c
        tau = 1.0
        t = 1
        while t + 1 < n_d:
            pair = rho[t] + rho[t + 1]
            if pair < 0.0:
                break
            tau += 2.0 * pair
            t += 2
        out[p] = n_tot / max(tau, 1.0)
    return out


def ess_tail(chains: np.ndarray, *, prob: float = 0.05) -> np.ndarray:
    """Tail ESS: min ESS of ``I(x<=q)`` and ``I(x>=1-q)`` (Vehtari et al. 2021)."""
    a = np.asarray(chains, dtype=np.float64)
    if a.ndim != 3:
        raise ValueError("chains must be (n_chain, n_draw, n_param)")
    n_p = a.shape[2]
    out = np.empty(n_p, dtype=np.float64)
    for p in range(n_p):
        col = a[:, :, p]
        q_lo = np.quantile(col, prob)
        q_hi = np.quantile(col, 1.0 - prob)
        i_lo = (col <= q_lo).astype(np.float64)
        i_hi = (col >= q_hi).astype(np.float64)
        e_lo = float(ess_bulk(i_lo[:, :, None])[0])
        e_hi = float(ess_bulk(i_hi[:, :, None])[0])
        out[p] = min(e_lo, e_hi)
    return out


def interval_table(mean_vec, cov, names, *, z: float = Z68) -> dict[str, dict]:
    out = {}
    for i, name in enumerate(names):
        lo, hi = gaussian_interval(mean_vec[i], cov[i, i], z)
        out[name] = {"lo": lo, "hi": hi, "width": hi - lo}
    return out


def hessian_at(data, params, template, grid, *, t: float = 1.0) -> np.ndarray:
    """Hessian of ``chi2 + prior`` at ``params``."""
    x0 = params_to_vec(params)

    def fun(x):
        return chi2_and_prior(data, vec_to_params(x, params), template, grid, t=t)

    return fd_hessian(fun, x0)


def make_logp_vec(data, template, grid, base, *, t: float = 1.0):
    def logp_vec(x):
        return log_prob(data, vec_to_params(x, base), template, grid, t=t)

    return logp_vec


def mcmc_intervals(samples: np.ndarray, names, q68=(0.16, 0.84), q95=(0.025, 0.975)):
    """Percentile intervals from stacked MH draws. ``samples`` (n_chain, n_draw, n)."""
    flat = np.asarray(samples, dtype=np.float64).reshape(-1, len(names))
    out = {}
    for i, name in enumerate(names):
        col = flat[:, i]
        lo68, hi68 = np.quantile(col, q68)
        lo95, hi95 = np.quantile(col, q95)
        out[name] = {
            "p16": float(lo68),
            "p84": float(hi68),
            "p025": float(lo95),
            "p975": float(hi95),
            "mean": float(np.mean(col)),
            "median": float(np.median(col)),
        }
    return out
