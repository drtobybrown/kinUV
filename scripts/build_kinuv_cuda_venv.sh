#!/usr/bin/env bash
# Build /arc/projects/KILOGAS/analysis/toby_sandbox/venvs/kinuv-cuda on a GPU node.
# Do not source or pip into kinuv-venv-recovery. PyPI jax-finufft wheels are CPU-only;
# CUDA support requires a source build with JAX_FINUFFT_USE_CUDA=ON.
set -euo pipefail

VENV="${KINUV_VENV:-/arc/projects/KILOGAS/analysis/toby_sandbox/venvs/kinuv-cuda}"
RUN_ID="${KINUV_RUN_ID:-KGAS066-gpu-venv}"
RUNS="${KINUV_RUNS:-/arc/projects/KILOGAS/analysis/toby_sandbox/kinuv_runs}"
REPO="${KINUV_REPO:-/arc/projects/KILOGAS/analysis/toby_sandbox/kinUV}"
OUT="${RUNS}/${RUN_ID}"
mkdir -p "${OUT}/logs" "$(dirname "${VENV}")"

if [[ "${VENV}" == *kinuv-venv-recovery* ]]; then
  echo "refuse: will not pip into recovery venv" >&2
  exit 2
fi

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export JAX_ENABLE_X64=1
export PYTHONUNBUFFERED=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

echo "=== kinuv cuda venv start host=$(hostname) venv=${VENV} reprobe=${KINUV_REPROBE:-0} ==="
command -v nvidia-smi && nvidia-smi -L || true
command -v nvcc && nvcc --version | head -5 || true
python3 --version

if [[ "${KINUV_REPROBE:-0}" == "1" ]]; then
  # Venv already built; install missing runtime deps and re-run probe/identity.
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  python -m pip install "pydantic>=2"
  python -c "import jax; assert jax.__version__=='0.11.1', jax.__version__"
else
  python3 -m venv --clear "${VENV}"
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  python -m pip install -U pip wheel setuptools
  python -m pip install "numpy>=2.1" "scipy>=1.15" pytest tqdm astropy matplotlib
  python -m pip install "scikit-build-core" cmake ninja
  python -m pip install "jax[cuda12]==0.11.1"
  python -c "import jax; assert jax.__version__=='0.11.1', jax.__version__"

  # CPU wheels on PyPI; force CUDA source build. Do not upgrade jax.
  export CMAKE_ARGS="${CMAKE_ARGS:-} -DJAX_FINUFFT_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native"
  python -m pip install \
    "jax-finufft==1.3.1" \
    --no-binary jax-finufft \
    --no-deps \
    -Ccmake.define.JAX_FINUFFT_USE_CUDA=ON \
    -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=native
  python -m pip install "pydantic>=2"
  python -m pip install --no-deps "numpyro==0.21.0"
  python -c "import jax; assert jax.__version__=='0.11.1', jax.__version__"

  python -m pip freeze > "${OUT}/pip-freeze.txt"
  python -c "import jax; print('jax', jax.__version__, 'devices', jax.devices())"
fi

export KINUV_RUNS="${RUNS}"
export KINUV_RUN_ID="${RUN_ID}"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
python - <<'PY'
import json, os, pathlib, subprocess
from datetime import datetime, timezone

def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return {"rc": p.returncode, "stdout": (p.stdout or "")[:2000], "stderr": (p.stderr or "")[:500]}

rec = {
    "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "venv": os.environ.get("VIRTUAL_ENV"),
    "hostname": os.uname().nodename,
    "nvidia_smi_L": run(["nvidia-smi", "-L"]),
}
import jax
rec["jax_version"] = jax.__version__
rec["devices"] = [str(d) for d in jax.devices()]
rec["has_cuda_device"] = any("cuda" in str(d).lower() for d in jax.devices())
try:
    from kinuv.transforms.nufft import BACKEND
except Exception as exc:
    rec["backend_import_error"] = repr(exc)
    BACKEND = None
rec["backend"] = BACKEND
try:
    import jax.numpy as jnp
    from jax_finufft import nufft2
    x = jnp.linspace(-1.0, 1.0, 32)
    src = jnp.ones((32, 32), dtype=jnp.complex128)
    vis = nufft2(src, x, x, iflag=-1, eps=1e-6)
    vis.block_until_ready()
    rec["cuda_nufft_ok"] = True
except Exception as exc:
    rec["cuda_nufft_ok"] = False
    rec["cuda_nufft_error"] = repr(exc)
rec["ok"] = (
    rec["jax_version"] == "0.11.1"
    and rec["has_cuda_device"]
    and BACKEND == "jax-finufft"
    and rec.get("cuda_nufft_ok") is True
)
out = pathlib.Path(os.environ["KINUV_RUNS"]) / os.environ["KINUV_RUN_ID"] / "gpu_probe.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rec, indent=2) + "\n")
print(json.dumps(rec, indent=2))
raise SystemExit(0 if rec["ok"] else 2)
PY

if [[ "${KINUV_RUN_IDENTITY:-0}" == "1" ]]; then
  echo "=== kinuv cuda identity ==="
  python "${REPO}/scripts/run_gpu_identity.py"
fi
echo "=== kinuv cuda venv done ==="
