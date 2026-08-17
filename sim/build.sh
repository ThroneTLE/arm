#!/usr/bin/env bash
set -euo pipefail

SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/noetic/setup.bash
"${SIM_DIR}/bootstrap.sh"
cd "${SIM_DIR}/ws"
catkin_make -DCMAKE_BUILD_TYPE=Release -j"${BUILD_JOBS:-2}"
