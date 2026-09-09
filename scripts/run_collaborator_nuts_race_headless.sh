#!/usr/bin/env bash
# Monitor fast and legacy campaigns and finalize the first accepted posterior.
set -euo pipefail

FAST_ROOT="${1:?fast root required}"
LEGACY_ROOT="${2:?legacy root required}"
RACE_ROOT="${3:?race root required}"
CODE_COMMIT="${4:?code commit required}"
PROJECT="/arc/projects/KILOGAS/analysis/toby_sandbox"
SOURCE_REPO="${PROJECT}/kinUV"
CAMPAIGN_ROOT="${PROJECT}/results/incoming/collaborator-delivery-20260908"
VENV="/arc/home/thbrown/kinuv-venv-recovery"
SESSION_ID="${SKAHA_SESSION_ID:-${skaha_sessionid:-${HOSTNAME:-unknown}}}"
SCRATCH_ROOT="/scratch/kinuv-${USER:-thbrown}/race-${SESSION_ID}"
CODE_ROOT="${SCRATCH_ROOT}/repo"

mkdir -p "${CODE_ROOT}" "${SCRATCH_ROOT}/tmp" "${SCRATCH_ROOT}/xdg" "${RACE_ROOT}"
git -C "${SOURCE_REPO}" archive "${CODE_COMMIT}" | tar -x -C "${CODE_ROOT}"
export TMPDIR="${SCRATCH_ROOT}/tmp"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/xdg"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export TERM=dumb
export NO_COLOR=1
export PYTHONPATH="${CODE_ROOT}/src"

cd "${CODE_ROOT}"
"${VENV}/bin/python" scripts/run_collaborator_nuts_race.py \
  --fast-root "${FAST_ROOT}" \
  --legacy-root "${LEGACY_ROOT}" \
  --race-root "${RACE_ROOT}" \
  --map-root "${CAMPAIGN_ROOT}/map" \
  --map-candidate-root "${CAMPAIGN_ROOT}/early-candidate" \
  --code-commit "${CODE_COMMIT}" \
  --python "${VENV}/bin/python"
