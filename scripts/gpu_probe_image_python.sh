#!/usr/bin/env bash
# GPU probe using the container's default Python (not kinuv-venv-recovery).
set -euo pipefail

RUN_ID="${KINUV_RUN_ID:-KGAS066-gpu-probe-imagepy}"
RUNS_ROOT="${KINUV_RUNS:-/arc/projects/KILOGAS/analysis/toby_sandbox/kinuv_runs}"
RUN_DIR="${RUNS_ROOT}/${RUN_ID}"
IMAGE="${KINUV_GPU_PROBE_IMAGE:-unknown}"
mkdir -p "${RUN_DIR}/logs"

python3 - <<'PY'
import json
import os
import subprocess
import sys
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
    "python": sys.executable,
    "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
    "nvidia_smi_L": run(["nvidia-smi", "-L"]),
}
try:
    import jax
    rec["jax_version"] = jax.__version__
    rec["default_backend"] = jax.default_backend()
    rec["devices"] = [str(d) for d in jax.devices()]
    rec["has_cuda_device"] = any("cuda" in str(d).lower() for d in jax.devices())
    x = jax.numpy.ones((1024, 1024))
    y = jax.numpy.dot(x, x)
    rec["matmul_ok"] = float(y[0, 0])
except Exception as exc:
    rec["jax_error"] = repr(exc)

for mod in ("jax_finufft", "jaxlib"):
    try:
        __import__(mod)
        rec[mod] = True
    except Exception as exc:
        rec[mod] = False
        rec[f"{mod}_error"] = repr(exc)

out = os.path.join(os.environ["KINUV_RUNS"], os.environ["KINUV_RUN_ID"], "gpu_probe.json")
with open(out, "w", encoding="utf-8") as fh:
    json.dump(rec, fh, indent=2)
    fh.write("\n")
print(json.dumps(rec, indent=2))
PY
