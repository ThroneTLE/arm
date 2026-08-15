#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$root"

# No arguments is the normal one-click workflow after the targets are printed.
if [[ $# -eq 0 ]]; then
    set -- --auto-connect --stage intrinsics
fi

exec python3 "$root/calibration_ui.py" "$@"
