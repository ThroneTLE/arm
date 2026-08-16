#!/usr/bin/env bash
set -euo pipefail

module_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$module_root/../.." && pwd -P)"
cd "$repo_root"

# No arguments is the normal one-click workflow after the targets are printed.
if [[ $# -eq 0 ]]; then
    set -- --auto-connect --stage intrinsics
fi

exec python3 -m tool.camera_calibration.calibration_ui "$@"
