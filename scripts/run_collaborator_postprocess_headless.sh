#!/usr/bin/env bash
# Wait for both CANFAR target sessions, then build the posterior candidate.
set -euo pipefail

NUTS_ROOT="${1:?NUTS root required}"
CODE_COMMIT="${2:?code commit required}"
PROJECT="/arc/projects/KILOGAS/analysis/toby_sandbox"
SOURCE_REPO="${PROJECT}/kinUV"
CAMPAIGN_ROOT="${PROJECT}/results/incoming/collaborator-delivery-20260908"
VENV="/arc/home/thbrown/kinuv-venv-recovery"
SESSION_ID="${SKAHA_SESSION_ID:-${skaha_sessionid:-${HOSTNAME:-unknown}}}"
SCRATCH_ROOT="/scratch/kinuv-${USER:-thbrown}/postprocess-${SESSION_ID}"
CODE_ROOT="${SCRATCH_ROOT}/repo"

mkdir -p "${CODE_ROOT}" "${SCRATCH_ROOT}/tmp" "${SCRATCH_ROOT}/xdg"
git -C "${SOURCE_REPO}" archive "${CODE_COMMIT}" | tar -x -C "${CODE_ROOT}"
export TMPDIR="${SCRATCH_ROOT}/tmp"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/xdg"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export TERM=dumb
export PYTHONPATH="${CODE_ROOT}/src"

cd "${CODE_ROOT}"
"${VENV}/bin/python" scripts/run_collaborator_postprocess_controller.py \
  --map-root "${CAMPAIGN_ROOT}/map" \
  --nuts-root "${NUTS_ROOT}" \
  --targets KGAS066 KGAS007 \
  --map-candidate-root "${CAMPAIGN_ROOT}/early-candidate" \
  --posterior-root "${CAMPAIGN_ROOT}/posterior-headless" \
  --candidate-root "${CAMPAIGN_ROOT}/posterior-candidate-headless" \
  --python "${VENV}/bin/python" \
  --status-file "${NUTS_ROOT}/postprocess_status.json" \
  --log-file "${NUTS_ROOT}/postprocess.log"
