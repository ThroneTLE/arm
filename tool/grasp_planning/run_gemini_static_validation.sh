#!/usr/bin/env bash
# Offline Gemini static_frame -> YOLO instance mask -> AnyGrasp validation.
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_ROOT}/../.." && pwd -P)"
CONDA_ROOT="${CONDA_ROOT:-/home/throne/miniconda3}"
ENV_NAME="${ANYGRASP_ENV:-anygraspenv}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

cd "${PROJECT_ROOT}"
exec python -m tool.grasp_planning.gemini_static_validation "$@"
