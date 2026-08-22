#!/usr/bin/env bash
set -euo pipefail

module_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$module_root/../.." && pwd -P)"
cd "$repo_root"

# OAK factory calibration is now the project default. Keep the old Astra
# ChArUco workbench available only through an explicit compatibility switch.
if [[ "${1:-}" == "--legacy-astra" ]]; then
    shift
    if [[ $# -eq 0 ]]; then
        set -- --auto-connect --stage intrinsics
    fi
    exec python3 -m tool.camera_calibration.calibration_ui "$@"
fi

exec "$repo_root/competition_pipeline/run_ui.sh" --stage rgbd --auto-connect "$@"
