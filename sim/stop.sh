#!/usr/bin/env bash
set -euo pipefail

SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOUND=false
for PID_FILE in "${SIM_DIR}/.upstream_demo.pid" "${SIM_DIR}/.competition_demo.pid"; do
  [[ -f "${PID_FILE}" ]] || continue
  FOUND=true
  read -r PID START_TICKS < "${PID_FILE}"
  if [[ -r "/proc/${PID}/stat" ]]; then
    CURRENT_TICKS="$(awk '{print $22}' "/proc/${PID}/stat")"
    if [[ "${CURRENT_TICKS}" == "${START_TICKS}" ]]; then
      kill -INT -- "-${PID}" 2>/dev/null || true
      for _ in $(seq 1 40); do
        [[ ! -r "/proc/${PID}/stat" ]] && break
        sleep 0.25
      done
      [[ ! -r "/proc/${PID}/stat" ]] || kill -TERM -- "-${PID}" 2>/dev/null || true
    fi
  fi
  rm -f "${PID_FILE}"
done
if [[ "${FOUND}" == true ]]; then
  echo "Managed simulation stopped."
else
  echo "No managed simulation is running."
fi
