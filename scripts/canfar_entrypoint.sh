#!/usr/bin/env bash
# Headless entrypoint (DEC-067-RUNNER). Activate recovery venv; TMP on /scratch.
set -euo pipefail

RUN_ID="${1:-kgas066-nuts}"
USER_NAME="${USER:-thbrown}"
REPO="${KINUV_REPO:-/arc/projects/KILOGAS/analysis/toby_sandbox/kinUV}"
VENV="${KINUV_VENV:-/arc/home/thbrown/kinuv-venv-recovery}"
SCRATCH_BASE="/scratch/kinuv-${USER_NAME}"
mkdir -p "${SCRATCH_BASE}"
export TMPDIR="${SCRATCH_BASE}/tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export JAX_COMPILATION_CACHE_DIR="${SCRATCH_BASE}/jax-cache"
export XDG_CACHE_HOME="${SCRATCH_BASE}/xdg"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export JAX_ENABLE_X64=1
mkdir -p "${TMPDIR}" "${JAX_COMPILATION_CACHE_DIR}" "${XDG_CACHE_HOME}"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
cd "${REPO}"
export PYTHONPATH="${REPO}/src"
if [[ "${KINUV_SKIP_PULL:-0}" != "1" ]]; then
  git fetch origin dev >/dev/null 2>&1 || true
  git checkout dev >/dev/null 2>&1 || true
  git pull --ff-only origin dev >/dev/null 2>&1 || true
fi

exec python "${REPO}/scripts/run_kgas066_nuts_headless.py" --run-id "${RUN_ID}"
