#!/usr/bin/env bash
# Lightweight CANFAR GPU schedulability probe. Fixed CPU/RAM/GPU only (no flexible).
set -euo pipefail

RUN_ID="${KINUV_RUN_ID:-KGAS066-gpu-probe-local}"
RUNS_ROOT="${KINUV_RUNS:-/arc/projects/KILOGAS/analysis/toby_sandbox/kinuv_runs}"
RUN_DIR="${RUNS_ROOT}/${RUN_ID}"
VENV="${KINUV_VENV:-/arc/home/thbrown/kinuv-venv-recovery}"
IMAGE="${KINUV_GPU_PROBE_IMAGE:-unknown}"
mkdir -p "${RUN_DIR}/logs"

utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }

{
  echo "=== kinuv gpu probe utc=$(utc) host=$(hostname) run_id=${RUN_ID} ==="
  echo "image=${IMAGE}"
  env | grep -E '^(JAX_|KINUV_|CUDA|NVIDIA|SKAHA_)' | sort || true
  echo "--- nvidia-smi ---"
  nvidia-smi -L 2>&1 || echo "nvidia-smi missing"
  nvidia-smi 2>&1 | head -20 || true
  echo "--- venv jax ---"
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  python - <<'PY'
import json
import os
import pathlib
import subprocess
from datetime import datetime, timezone

def run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return {"rc": p.returncode, "stdout": (p.stdout or "")[:4000], "stderr": (p.stderr or "")[:1000]}
    except Exception as exc:
        return {"error": str(exc)}

rec = {
    "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "hostname": os.uname().nodename,
    "run_id": os.environ.get("KINUV_RUN_ID"),
    "image": os.environ.get("KINUV_GPU_PROBE_IMAGE"),
    "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
    "venv_python": os.environ.get("VIRTUAL_ENV"),
    "nvidia_smi_L": run(["nvidia-smi", "-L"]),
}
try:
    import jax
    rec["jax_version"] = jax.__version__
    rec["default_backend"] = jax.default_backend()
    rec["devices"] = [str(d) for d in jax.devices()]
    rec["device_count"] = len(jax.devices())
    rec["has_cuda_device"] = any("cuda" in str(d).lower() for d in jax.devices())
except Exception as exc:
    rec["jax_error"] = repr(exc)

try:
    import jax_finufft  # noqa: F401
    rec["jax_finufft"] = True
except Exception as exc:
    rec["jax_finufft"] = False
    rec["jax_finufft_error"] = repr(exc)

print(json.dumps(rec, indent=2))
out = pathlib.Path(os.environ["KINUV_RUNS"]) / os.environ["KINUV_RUN_ID"] / "gpu_probe.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out}", flush=True)
PY
} | tee "${RUN_DIR}/logs/gpu_probe.log"

echo "=== probe done utc=$(utc) ==="
