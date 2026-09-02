#!/usr/bin/env python3
"""CUDA G1 identity + six-axis grad + eval/s. Fail-closed before GPU NUTS."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from kinuv.scratch import apply_scratch_env  # noqa: E402

apply_scratch_env()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from kinuv.infer.chart import PARAM_NAMES  # noqa: E402
from kinuv.infer.map import predict_binned  # noqa: E402
from kinuv.infer.nuts import make_potential, sampled_z_from_physical  # noqa: E402
from kinuv.infer.posterior import params_to_vec  # noqa: E402
from kinuv.likelihood.chi2 import chi2  # noqa: E402
from kinuv.runner.kind import ARTIFACT_GPU  # noqa: E402
from kinuv.transforms.nufft import BACKEND  # noqa: E402

MAP = Path(
    "/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS066/"
    "kinuv-KGAS066-uvsign-map/stage_a_map.json"
)
NPZ = Path("/arc/projects/KILOGAS/analysis/toby_sandbox/visibilities/KILOGAS066.npz")
ICO = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/"
    "KGAS66_Ico_K_kms-1.fits"
)
CUBE = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/"
    "KGAS66_clipped_cube.fits"
)
CPU_EVAL_S = 3.0119342791188477
CPU_GRAD_S = 0.4340723119676113
CHI2_REF = 168675.6


def main() -> int:
    from kinuv.forward.sb import load_sb_template
    from kinuv.infer.map import image_grid_for_vis
    from kinuv.io.vis import load_kgas066

    rec_map = json.loads(MAP.read_text())
    data = load_kgas066(NPZ, cube_path=CUBE if CUBE.is_file() else None)
    grid = image_grid_for_vis(data)
    tmpl = load_sb_template(grid, ico_path=ICO if ICO.is_file() else None)
    params = {n: rec_map[n] for n in PARAM_NAMES}
    devices = [str(d) for d in jax.devices()]
    has_cuda = any("cuda" in d.lower() for d in devices)
    vis = predict_binned(data, params, tmpl, grid, xla=True)
    vis_np = np.asarray(vis)
    c = float(chi2(data.vis, vis_np, data.weights, data.s))
    identity_ok = abs(c - CHI2_REF) < 1.0
    # warmup
    _ = vis
    t0 = time.perf_counter()
    vis2 = predict_binned(data, params, tmpl, grid, xla=True)
    jnp.asarray(vis2).block_until_ready()
    eval_s = 1.0 / max(time.perf_counter() - t0, 1e-9)
    dx, dy = params["dx_arcsec"], params["dy_arcsec"]
    U = make_potential(data, tmpl, grid, dx, dy)
    z6 = jnp.asarray(sampled_z_from_physical(params_to_vec(params)))
    g = jax.grad(U)(z6)
    g = jnp.asarray(g).block_until_ready()
    t1 = time.perf_counter()
    g2 = jax.grad(U)(z6)
    jnp.asarray(g2).block_until_ready()
    t_grad = time.perf_counter() - t1
    finite = bool(np.all(np.isfinite(np.asarray(g2))))
    out = {
        "backend": BACKEND,
        "JAX_PLATFORMS": str(__import__("os").environ.get("JAX_PLATFORMS")),
        "jax_version": jax.__version__,
        "devices": devices,
        "has_cuda_device": has_cuda,
        "chi2": c,
        "identity_ok": identity_ok,
        "s": float(data.s),
        "eval_per_s": eval_s,
        "cpu_eval_per_s": CPU_EVAL_S,
        "t_grad_s": t_grad,
        "cpu_t_grad_s": CPU_GRAD_S,
        "grad_finite_six": finite,
        "n_row": int(np.asarray(data.vis).shape[0]),
        "n_chan": int(np.asarray(data.vis).shape[1]),
        "ok": bool(
            identity_ok
            and has_cuda
            and jax.__version__ == "0.11.1"
            and BACKEND == "jax-finufft"
            and finite
        ),
    }
    dest = ARTIFACT_GPU
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "timing.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0 if out["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
