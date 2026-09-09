#!/usr/bin/env bash
# One unified MAP start in one flexible CANFAR headless session.
set -euo pipefail

target="${1:?target required}"
start_id="${2:?start id required}"
durable_root="${3:?durable root required}"
code_commit="${4:?code commit required}"
maxiter="${5:-160}"
project="/arc/projects/KILOGAS/analysis/toby_sandbox"
source_repo="${project}/kinUV"
venv="/arc/home/thbrown/kinuv-venv-recovery"
session_id="${SKAHA_SESSION_ID:-${skaha_sessionid:-${HOSTNAME:-unknown}}}"
run_user="${USER:-$(id -un)}"
scratch="/scratch/kinuv-${run_user}/unified-map-${session_id}-${target}-s${start_id}"
code_root="${scratch}/repo"
run_root="${scratch}/run"
destination="${durable_root}/${target}/start-${start_id}"
staging="${durable_root}/${target}/.start-${start_id}-${session_id}.tmp"

mkdir -p "${code_root}" "${run_root}" "${scratch}/tmp" "${scratch}/jax-cache" "${durable_root}/${target}"
if [[ -e "${destination}" || -e "${staging}" ]]; then
  echo "refusing to replace existing result path: ${destination}" >&2
  exit 2
fi
git -C "${source_repo}" archive "${code_commit}" | tar -x -C "${code_root}"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1 TERM=dumb
export JAX_COMPILATION_CACHE_DIR="${scratch}/jax-cache" TMPDIR="${scratch}/tmp"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1 JAX_NUM_THREADS=4
export PYTHONPATH="${code_root}/src:${code_root}/experiments/unified_foundation"
export KINUV_WORKSPACE="${project}"

set +e
"${venv}/bin/python" "${code_root}/experiments/unified_foundation/unified_map_runner.py" \
  --target "${target}" --start-id "${start_id}" --code-commit "${code_commit}" \
  --output "${run_root}" \
  --heartbeat "${durable_root}/heartbeats/${target}-start-${start_id}.json" \
  --maxiter "${maxiter}" >"${scratch}/worker.log" 2>&1
code=$?
set -e
mkdir "${staging}"
cp -a "${run_root}/." "${staging}/"
cp "${scratch}/worker.log" "${staging}/worker.log"
printf '{"target":"%s","family":"%s","start_id":%s,"session_id":"%s","code_commit":"%s","exit_code":%d}\n' \
  "${target}" "unified" "${start_id}" "${session_id}" "${code_commit}" "${code}" \
  >"${staging}/headless_exit.json"
mv "${staging}" "${destination}"
exit "${code}"
