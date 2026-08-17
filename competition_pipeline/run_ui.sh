#!/usr/bin/env bash
set -euo pipefail

module_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$module_root/.." && pwd -P)"
cd "$repo_root"

# The RGB stream is UVC, while IR/depth are supplied by the Astra ROS driver.
# Source both workspaces so the same launcher works from a clean shell.
if [[ -f /opt/ros/noetic/setup.bash ]]; then
  source /opt/ros/noetic/setup.bash
fi
if [[ -f /home/throne/astra_ws/devel/setup.bash ]]; then
  source /home/throne/astra_ws/devel/setup.bash
fi
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

exec python3 -m competition_pipeline.ui "$@"
