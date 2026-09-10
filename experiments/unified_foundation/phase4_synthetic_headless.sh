#!/usr/bin/env bash
# Run one target's matched Phase-4 synthetic suite in a flexible session.
set -euo pipefail

target="${1:?target required}"
durable_root="${2:?durable root required}"
code_commit="${3:?production code commit required}"
kinms_archive="${4:?frozen KinMS venv archive required}"
maxiter="${5:-100}"
shift 5
seeds=("${@:-7401}")

project="/arc/projects/KILOGAS/analysis/toby_sandbox"
source_repo="${project}/kinUV"
kinuv_venv="/arc/home/thbrown/kinuv-venv-recovery"
session_id="${SKAHA_SESSION_ID:-${skaha_sessionid:-${HOSTNAME:-unknown}}}"
run_user="${USER:-$(id -un)}"
scratch="/scratch/kinuv-${run_user}/phase4-synthetic-${session_id}-${target}"
code_root="${scratch}/repo"
run_root="${scratch}/run"
destination="${durable_root}/targets/${target}"
staging="${durable_root}/targets/.${target}-${session_id}.tmp"

mkdir -p "${code_root}" "${run_root}" "${scratch}/tmp" "${scratch}/jax-cache" "${durable_root}/targets"
if [[ -e "${destination}" || -e "${staging}" ]]; then
  echo "refusing to replace existing target result: ${destination}" >&2
  exit 2
fi
git -C "${source_repo}" archive "${code_commit}" | tar -x -C "${code_root}"
mkdir -p /scratch/kinuv-thbrown
tar -xzf "${kinms_archive}" -C /scratch/kinuv-thbrown
test -x /scratch/kinuv-thbrown/s1-kinms/bin/python
/scratch/kinuv-thbrown/s1-kinms/bin/python -m pip freeze >"${scratch}/kinms-environment.freeze.txt"

export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1 TERM=dumb
export JAX_COMPILATION_CACHE_DIR="${scratch}/jax-cache" TMPDIR="${scratch}/tmp"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1 JAX_NUM_THREADS=4
export PYTHONPATH="${code_root}/src"
export KINUV_CODE_ROOT="${code_root}" KINUV_CODE_COMMIT="${code_commit}" KINUV_WORKSPACE="${project}"

set +e
"${kinuv_venv}/bin/python" "${durable_root}/source/phase4_synthetic_benchmark.py" \
  --target-config "${code_root}/configs/targets/${target}.json" \
  --covariance-metrics "${project}/results/validation/crossdomain-recovery-s2-20260907-r1/metrics.json" \
  --output "${run_root}" --kinms-python /scratch/kinuv-thbrown/s1-kinms/bin/python \
  --maxiter "${maxiter}" --seeds "${seeds[@]}" >"${scratch}/worker.log" 2>&1
code=$?
set -e

mkdir "${staging}"
if [[ -d "${run_root}/${target}" ]]; then
  cp -a "${run_root}/${target}/." "${staging}/"
fi
cp "${scratch}/worker.log" "${staging}/worker.log"
cp "${scratch}/kinms-environment.freeze.txt" "${staging}/kinms-environment.freeze.txt"
printf '{"target_id":"%s","session_id":"%s","code_commit":"%s","exit_code":%d}\n' \
  "${target}" "${session_id}" "${code_commit}" "${code}" >"${staging}/headless_exit.json"
mv "${staging}" "${destination}"
exit "${code}"
