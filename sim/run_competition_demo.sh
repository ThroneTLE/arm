#!/usr/bin/env bash
set -euo pipefail

SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${SIM_DIR}/.competition_demo.pid"
LOG_DIR="${SIM_DIR}/logs"
mkdir -p "${LOG_DIR}"
[[ ! -f "${PID_FILE}" ]] || {
  echo "A managed competition simulation is already recorded." >&2
  exit 1
}
source /opt/ros/noetic/setup.bash
source "${SIM_DIR}/ws/devel/setup.bash"

cleanup() {
  if [[ -n "${LAUNCH_PID:-}" ]] && kill -0 "${LAUNCH_PID}" 2>/dev/null; then
    kill -INT -- "-${LAUNCH_PID}" 2>/dev/null || true
    wait "${LAUNCH_PID}" 2>/dev/null || true
  fi
  rm -f "${PID_FILE}"
}
trap cleanup EXIT INT TERM

# Keep automated runs headless by default.  --gui is a wrapper-only option and
# may appear anywhere on the command line; all other options go to the Python
# competition runner unchanged.
GUI=false
RUNNER_ARGS=()
for ARG in "$@"; do
  if [[ "${ARG}" == "--gui" ]]; then
    GUI=true
  else
    RUNNER_ARGS+=("${ARG}")
  fi
done
if [[ "${GUI}" == false ]]; then
  unset DISPLAY
fi

setsid roslaunch competition_sim_bridge competition_world.launch start_rviz:="${GUI}" \
  >"${LOG_DIR}/competition_world.log" 2>&1 &
LAUNCH_PID=$!
START_TICKS="$(awk '{print $22}' "/proc/${LAUNCH_PID}/stat")"
echo "${LAUNCH_PID} ${START_TICKS}" > "${PID_FILE}"

for _ in $(seq 1 80); do
  if rostopic list 2>/dev/null | grep -qx /gazebo/link_states \
      && rosservice list 2>/dev/null | grep -qx /controller_manager/list_controllers; then
    break
  fi
  kill -0 "${LAUNCH_PID}" 2>/dev/null || {
    tail -100 "${LOG_DIR}/competition_world.log" >&2
    exit 1
  }
  sleep 0.5
done

python3 "${SIM_DIR}/run_competition_demo.py" "${RUNNER_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/competition_demo.log"
