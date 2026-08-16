#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_ROOT}/../.." && pwd -P)"
CONDA_ROOT="${CONDA_ROOT:-/home/throne/miniconda3}"
CAMERA_ROS_SETUP="${CAMERA_ROS_SETUP:-${ORBBEC_ROS_SETUP:-/home/throne/astra_ws/devel/setup.bash}}"

if [[ ! -r /opt/ros/noetic/setup.bash ]]; then
    echo "错误：未找到 ROS Noetic，无法启动瓶子定位与 RViz。" >&2
    exit 1
fi
original_arguments=("$@")
set --
source /opt/ros/noetic/setup.bash
if [[ ! -r "${CAMERA_ROS_SETUP}" ]]; then
    echo "错误：未找到相机 ROS 工作区：${CAMERA_ROS_SETUP}" >&2
    echo "请设置 CAMERA_ROS_SETUP 指向相机驱动的 devel/setup.bash。" >&2
    exit 1
fi
source "${CAMERA_ROS_SETUP}"
set -- "${original_arguments[@]}"

if [[ ! -r "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    echo "错误：未找到 Conda；请通过 CONDA_ROOT 指定安装目录。" >&2
    exit 1
fi
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate foundationpose

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${PROJECT_ROOT}"
exec python -m tool.bottle_localization.localization_node --rviz "$@"
