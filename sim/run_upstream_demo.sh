#!/usr/bin/env bash
set -euo pipefail

SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${SIM_DIR}/.upstream_demo.pid"
LOG_DIR="${SIM_DIR}/logs"
mkdir -p "${LOG_DIR}"

if [[ -f "${PID_FILE}" ]]; then
  echo "A managed simulation PID file already exists; run sim/stop.sh first." >&2
  exit 1
fi
source /opt/ros/noetic/setup.bash
source "${SIM_DIR}/ws/devel/setup.bash"

cleanup() {
  if [[ -n "${DEMO_PID:-}" ]] && kill -0 "${DEMO_PID}" 2>/dev/null; then
    kill -INT -- "-${DEMO_PID}" 2>/dev/null || true
    wait "${DEMO_PID}" 2>/dev/null || true
  fi
  if [[ -n "${LAUNCH_PID:-}" ]] && kill -0 "${LAUNCH_PID}" 2>/dev/null; then
    kill -INT -- "-${LAUNCH_PID}" 2>/dev/null || true
    wait "${LAUNCH_PID}" 2>/dev/null || true
  fi
  rm -f "${PID_FILE}"
}
trap cleanup EXIT INT TERM

# A new session gives this wrapper ownership of the complete ROS/Gazebo process
# group. DISPLAY is removed by default so an accidental gzclient cannot leave a
# window behind; pass --gui to opt in.
GUI=false
if [[ "${1:-}" == "--gui" ]]; then
  GUI=true
fi
if [[ "${GUI}" == false ]]; then
  unset DISPLAY
fi

setsid roslaunch ur5e_robotiq_85_mtc_pkg setup.launch start_rviz:="${GUI}" \
  >"${LOG_DIR}/setup.log" 2>&1 &
LAUNCH_PID=$!
START_TICKS="$(awk '{print $22}' "/proc/${LAUNCH_PID}/stat")"
echo "${LAUNCH_PID} ${START_TICKS}" > "${PID_FILE}"

for _ in $(seq 1 60); do
  if rosservice list 2>/dev/null | grep -qx /controller_manager/list_controllers; then
    break
  fi
  kill -0 "${LAUNCH_PID}" 2>/dev/null || {
    tail -80 "${LOG_DIR}/setup.log" >&2
    exit 1
  }
  sleep 0.5
done
rosservice call /controller_manager/list_controllers > "${LOG_DIR}/controllers.yaml"
rostopic echo -n 1 /gazebo/model_states > "${LOG_DIR}/initial_model_states.yaml"

setsid roslaunch ur5e_robotiq_85_mtc_pkg pick_and_place.launch \
  >"${LOG_DIR}/pick_and_place.log" 2>&1 &
DEMO_PID=$!

for _ in $(seq 1 240); do
  if grep -q "Pick and Place Successful" "${LOG_DIR}/pick_and_place.log"; then
    rosservice call /gazebo/get_model_state \
      "{model_name: beer, relative_entity_name: world}" \
      > "${LOG_DIR}/final_beer_state.yaml"
    echo "Upstream MTC demo reported success; logs are in ${LOG_DIR}."
    exit 0
  fi
  kill -0 "${DEMO_PID}" 2>/dev/null || {
    tail -100 "${LOG_DIR}/pick_and_place.log" >&2
    exit 1
  }
  sleep 0.5
done
echo "Timed out waiting for the upstream demo." >&2
tail -100 "${LOG_DIR}/pick_and_place.log" >&2
exit 1
