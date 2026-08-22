#!/usr/bin/env bash
set -euo pipefail

module_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$module_root/.." && pwd -P)"
cd "$repo_root"

# OAK-D-PRO-FF is the default and uses DepthAI directly. Source the project ROS
# overlay first; the optional legacy overlays remain available when a profile
# is deliberately switched in the UI.
original_arguments=("$@")
set --
if [[ -f /opt/ros/noetic/setup.bash ]]; then
  source /opt/ros/noetic/setup.bash
fi
if [[ -f "$repo_root/ros_ws/devel/setup.bash" ]]; then
  source "$repo_root/ros_ws/devel/setup.bash"
fi
if [[ -f /home/throne/astra_ws/devel/setup.bash ]]; then
  source /home/throne/astra_ws/devel/setup.bash
fi
if [[ -f /home/throne/orbbec_ws/devel/setup.bash ]]; then
  source /home/throne/orbbec_ws/devel/setup.bash
fi
set -- "${original_arguments[@]}"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

exec python3 -m competition_pipeline.ui "$@"
