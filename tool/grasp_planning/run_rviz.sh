#!/usr/bin/env bash
# AnyGrasp 抓取规划节点 + RViz 启动脚本（独立于 bottle_localization 运行）。
#
# 前提：
#   1. ./setup_anygrasp_env.sh 已成功完成；
#   2. license 已解压到 anygrasp/grasp_detection/license；
#   3. checkpoint_detection.tar 已放到 anygrasp/grasp_detection/log/；
#   4. bottle_localization 节点已在另一个终端运行（提供 object_cloud 与 camera_pose）。
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_ROOT}/../.." && pwd -P)"
CONDA_ROOT="${CONDA_ROOT:-/home/throne/miniconda3}"
ENV_NAME="${ANYGRASP_ENV:-anygraspenv}"

if [[ ! -r /opt/ros/noetic/setup.bash ]]; then
    echo "错误：未找到 ROS Noetic，无法启动抓取规划与 RViz。" >&2
    exit 1
fi
original_arguments=("$@")
set --
source /opt/ros/noetic/setup.bash
set -- "${original_arguments[@]}"

if [[ ! -r "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    echo "错误：未找到 Conda；请通过 CONDA_ROOT 指定安装目录。" >&2
    exit 1
fi
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${PROJECT_ROOT}"
exec python -m tool.grasp_planning.anygrasp_node --rviz "$@"
