#!/usr/bin/env bash
# Render one accepted posterior target on node-local scratch, then publish it.
set -euo pipefail

target="${1:?target required}"
production_root="${2:?production root required}"
dynesty_root="${3:?finalized dynesty root required}"
durable_root="${4:?durable staging root required}"
phase4_synthetic="${5:?phase-4 synthetic summary required}"
phase4_real="${6:?phase-4 real result required}"
stage_commit="${7:?staging code commit required}"
sampler_commit="${8:?sampler code commit required}"
session_id="${SKAHA_SESSION_ID:-${skaha_sessionid:-${HOSTNAME:-unknown}}}"
workspace="/arc/projects/KILOGAS/analysis/toby_sandbox"
scratch="/scratch/kinuv-${USER:-$(id -un)}/unified-posterior-stage-${session_id}"
repo="${scratch}/repo"
scoring_reference="${scratch}/scoring-reference"
scratch_output="${scratch}/staged"
durable_target="${durable_root}/${target}"
temporary_target="${durable_root}/.${target}.${session_id}.tmp"

if [[ -e "${durable_target}" || -e "${temporary_target}" ]]; then
  echo "immutable staging destination exists" >&2
  exit 2
fi
mkdir -p "${repo}" "${scoring_reference}" "${scratch}/tmp" "${scratch}/jax-cache" "${durable_root}"
git -C "${workspace}/kinUV" archive "${stage_commit}" | tar -x -C "${repo}"
git -C "${workspace}/kinUV" archive "${sampler_commit}" | tar -x -C "${scoring_reference}"

export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1 TERM=dumb
export JAX_COMPILATION_CACHE_DIR="${scratch}/jax-cache" TMPDIR="${scratch}/tmp"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1 JAX_NUM_THREADS=4
export PYTHONPATH="${repo}/src:${repo}/experiments/unified_foundation"
export KINUV_WORKSPACE="${workspace}"
export KINUV_SCORING_REFERENCE="${scoring_reference}"

log="${scratch}/worker.log"
set +e
/arc/home/thbrown/kinuv-venv-recovery/bin/python \
  "${repo}/experiments/unified_foundation/stage_unified_posterior.py" \
  --production-root "${production_root}" --dynesty-root "${dynesty_root}" \
  --output-root "${scratch_output}" --phase4-synthetic "${phase4_synthetic}" \
  --phase4-real "${phase4_real}" --targets "${target}" >"${log}" 2>&1
code=$?
set -e

if [[ "${code}" -eq 0 ]]; then
  cp -a "${scratch_output}/${target}" "${temporary_target}"
  mv "${temporary_target}" "${durable_target}"
fi
cp "${log}" "${durable_root}/stage-${target}.log.tmp"
mv "${durable_root}/stage-${target}.log.tmp" "${durable_root}/stage-${target}.log"
printf '{"target":"%s","session_id":"%s","stage_commit":"%s","sampler_commit":"%s","exit_code":%d}\n' \
  "${target}" "${session_id}" "${stage_commit}" "${sampler_commit}" "${code}" \
  >"${durable_root}/stage-${target}-exit.json.tmp"
mv "${durable_root}/stage-${target}-exit.json.tmp" \
  "${durable_root}/stage-${target}-exit.json"
exit "${code}"
