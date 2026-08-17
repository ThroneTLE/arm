#!/usr/bin/env bash
set -euo pipefail

SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="${SIM_DIR}/ws"
SRC_DIR="${WS_DIR}/src"
UR_REPO="${SRC_DIR}/ur5e_robotiq_85_mtc"
MTC_REPO="${SRC_DIR}/moveit_task_constructor"

source /opt/ros/noetic/setup.bash
mkdir -p "${SRC_DIR}"
if [[ ! -e "${SRC_DIR}/CMakeLists.txt" ]]; then
  (cd "${SRC_DIR}" && catkin_init_workspace)
fi

fetch_pinned() {
  local destination="$1"
  local url="$2"
  local revision="$3"
  if [[ ! -d "${destination}/.git" ]]; then
    git clone "${url}" "${destination}"
  fi
  git -C "${destination}" fetch --tags origin
  git -C "${destination}" checkout --detach "${revision}"
}

fetch_pinned "${UR_REPO}" \
  https://github.com/blackcoffeerobotics/ur5e_robotiq_85_mtc.git \
  371f8a51ae297ec635dd10655e063a7e923fd36c
fetch_pinned "${MTC_REPO}" \
  https://github.com/ros-planning/moveit_task_constructor.git \
  4781ed563657cb7ecfcca857649bdf9f1c8d95b9

git -C "${MTC_REPO}" submodule update --init --recursive core/python/pybind11

apply_once() {
  local repository="$1"
  local patch_file="$2"
  if git -C "${repository}" apply --check "${patch_file}"; then
    git -C "${repository}" apply "${patch_file}"
  elif git -C "${repository}" apply --reverse --check "${patch_file}"; then
    : # already applied
  else
    echo "Patch does not apply cleanly: ${patch_file}" >&2
    exit 1
  fi
}
apply_once "${UR_REPO}" "${SIM_DIR}/patches/ur5e-package-deps.patch"
apply_once "${MTC_REPO}" "${SIM_DIR}/patches/mtc-noetic-compat.patch"

# The selected UR5e upstream ships these MTC extensions as source overlays.
cp "${UR_REPO}/third_party/scripts/generate_custom_pose.h" \
  "${MTC_REPO}/core/include/moveit/task_constructor/stages/generate_custom_pose.h"
cp "${UR_REPO}/third_party/scripts/generate_custom_pose.cpp" \
  "${MTC_REPO}/core/src/stages/generate_custom_pose.cpp"
cp "${UR_REPO}/third_party/scripts/CMakeLists.txt" \
  "${MTC_REPO}/core/src/stages/CMakeLists.txt"

# catkin_make configures dependent packages before this share directory exists.
mkdir -p "${WS_DIR}/devel/share/moveit_task_constructor_core"
ln -sfn "${MTC_REPO}/core/python/pybind11" \
  "${WS_DIR}/devel/share/moveit_task_constructor_core/pybind11"

echo "Pinned upstream sources and Noetic compatibility overlays are ready."
