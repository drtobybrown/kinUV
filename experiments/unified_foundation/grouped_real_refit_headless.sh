#!/usr/bin/env bash
# One grouped real-visibility training refit in a flexible CANFAR session.
set -euo pipefail

target="${1:?target required}"
fold_id="${2:?fold id required}"
durable_root="${3:?durable root required}"
winner="${4:?winner result required}"
code_commit="${5:?code commit required}"
worker_source="${6:?worker source required}"
worker_sha256="${7:?worker sha256 required}"
maxiter="${8:-160}"
project="/arc/projects/KILOGAS/analysis/toby_sandbox"
source_repo="${project}/kinUV"
kinms_root="${project}/results/validation/crossdomain-recovery-s4-remediation-20260907-r2/grouped"
venv="/arc/home/thbrown/kinuv-venv-recovery"
session_id="${SKAHA_SESSION_ID:-${skaha_sessionid:-${HOSTNAME:-unknown}}}"
run_user="${USER:-$(id -un)}"
scratch="/scratch/kinuv-${run_user}/unified-real-refit-${session_id}-${target}-f${fold_id}"
code_root="${scratch}/repo"
run_root="${scratch}/run"
destination="${durable_root}/${target}/fold-${fold_id}"
staging="${durable_root}/${target}/.fold-${fold_id}-${session_id}.tmp"

mkdir -p "${code_root}" "${scratch}/tmp" "${scratch}/jax-cache" "${durable_root}/${target}"
if [[ -e "${destination}" || -e "${staging}" ]]; then
  echo "refusing to replace existing result path: ${destination}" >&2
  exit 2
fi
if [[ "$(sha256sum "${worker_source}" | awk '{print $1}')" != "${worker_sha256}" ]]; then
  echo "worker source checksum mismatch before scratch copy" >&2
  exit 3
fi
git -C "${source_repo}" archive "${code_commit}" | tar -x -C "${code_root}"
cp "${worker_source}" "${code_root}/experiments/unified_foundation/grouped_real_refit_worker.py"
if [[ "$(sha256sum "${code_root}/experiments/unified_foundation/grouped_real_refit_worker.py" | awk '{print $1}')" != "${worker_sha256}" ]]; then
  echo "worker source checksum mismatch after scratch copy" >&2
  exit 4
fi

export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1 TERM=dumb
export JAX_COMPILATION_CACHE_DIR="${scratch}/jax-cache" TMPDIR="${scratch}/tmp"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1 JAX_NUM_THREADS=4
export PYTHONPATH="${code_root}/src:${code_root}/experiments/unified_foundation"
export KINUV_WORKSPACE="${project}"

set +e
"${venv}/bin/python" "${code_root}/experiments/unified_foundation/grouped_real_refit_worker.py" \
  --target "${target}" --fold-id "${fold_id}" --winner "${winner}" \
  --kinms-root "${kinms_root}" --output "${run_root}" \
  --code-commit "${code_commit}" --source-sha256 "${worker_sha256}" \
  --maxiter "${maxiter}" >"${scratch}/worker.log" 2>&1
code=$?
set -e
mkdir "${staging}"
if [[ -d "${run_root}" ]]; then
  cp -a "${run_root}/." "${staging}/"
fi
cp "${scratch}/worker.log" "${staging}/worker.log"
printf '{"target":"%s","fold_id":%s,"session_id":"%s","code_commit":"%s","worker_source_sha256":"%s","exit_code":%d}\n' \
  "${target}" "${fold_id}" "${session_id}" "${code_commit}" "${worker_sha256}" "${code}" \
  >"${staging}/headless_exit.json"
mv "${staging}" "${destination}"
exit "${code}"
