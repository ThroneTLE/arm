#!/usr/bin/env bash
# 无硬件跑真 UI：假相机 + 假控制器，界面和交互全是真的。
#
# 用来在没有机械臂、没有 OAK 相机的情况下把整个界面点一遍。
# 它**不改** oak_vision_node.py —— 只从外面替换"硬件从哪来"的两个构造入口。
#
# 用法::
#
#     ./tool/visual_grasp_pipeline/run_sim_ui.sh
#     ./tool/visual_grasp_pipeline/run_sim_ui.sh --enable-robot-motion
#     ./tool/visual_grasp_pipeline/run_sim_ui.sh --enable-robot-motion --fault servo-refuse
#
# --enable-robot-motion 在这里不会动任何真实硬件（控制器是假的），
# 它只是让你能点进"执行抓取"那条分支，看完整的十步序列跑起来。

set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_ROOT}/../.." && pwd -P)"
CONDA_ROOT="${CONDA_ROOT:-/home/throne/miniconda3}"

if [[ ! -r "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    echo "错误：未找到 Conda；请通过 CONDA_ROOT 指定安装目录。" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate foundationpose

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/ros_ws/src/arm_vision_framework/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

cd "${PROJECT_ROOT}"
exec python -m tool.visual_grasp_pipeline.simulate_node "$@"
