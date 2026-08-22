#!/usr/bin/env bash
# 无硬件全栈演练的启动脚本。
#
# 直接 `python -m competition_pipeline.scripts.offline_rehearsal` 会用系统
# python3.8，那里没有 trimesh/numpy 等依赖。这个脚本负责切到 foundationpose
# 环境并设好 PYTHONPATH —— 与 tool/visual_grasp_pipeline/run_oak_vision_node.sh
# 同一套做法。
#
# 用法（参数原样透传给演练脚本）::
#
#     ./competition_pipeline/scripts/run_offline_rehearsal.sh
#     ./competition_pipeline/scripts/run_offline_rehearsal.sh --object cola
#     ./competition_pipeline/scripts/run_offline_rehearsal.sh --rounds 5
#     ./competition_pipeline/scripts/run_offline_rehearsal.sh --fault servo-refuse
#     ./competition_pipeline/scripts/run_offline_rehearsal.sh --with-vision

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
# 演练不开窗口；即使某处间接引入 Qt 也不要弹出来。
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"

cd "${PROJECT_ROOT}"
exec python -m competition_pipeline.scripts.offline_rehearsal "$@"
