#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export CAMERA_ROS_SETUP="${CAMERA_ROS_SETUP:-/home/throne/orbbec_ws/devel/setup.bash}"

exec "${SCRIPT_ROOT}/run_ui.sh" \
  --config "${SCRIPT_ROOT}/config/gemini_foundationpose_debug.yaml" \
  --stage capture \
  "$@"
