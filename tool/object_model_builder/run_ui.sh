#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_ROOT}/../.." && pwd -P)"
CONDA_ROOT="${CONDA_ROOT:-/home/throne/miniconda3}"
CAMERA_ROS_SETUP="${CAMERA_ROS_SETUP:-${PROJECT_ROOT}/ros_ws/devel/setup.bash}"
MIN_FREE_DISK_KB="${OBJECT_MODEL_BUILDER_MIN_FREE_DISK_KB:-2097152}"

available_disk_kb="$(df -Pk "${PROJECT_ROOT}" | awk 'NR == 2 {print $4}')"
if [[ ! "${available_disk_kb}" =~ ^[0-9]+$ ]]; then
    echo "错误：无法读取 ${PROJECT_ROOT} 所在磁盘的剩余空间。" >&2
    exit 1
fi

if (( available_disk_kb < MIN_FREE_DISK_KB )); then
    available_disk_gb="$(awk -v kb="${available_disk_kb}" 'BEGIN {printf "%.2f", kb / 1024 / 1024}')"
    required_disk_gb="$(awk -v kb="${MIN_FREE_DISK_KB}" 'BEGIN {printf "%.2f", kb / 1024 / 1024}')"
    echo "错误：磁盘剩余空间不足，无法启动物体模型构建工具。" >&2
    echo "当前可用 ${available_disk_gb} GiB，至少需要 ${required_disk_gb} GiB。请清理磁盘后重试。" >&2
    exit 28
fi

requires_ros=true
for argument in "$@"; do
    case "${argument}" in
        --pack-session|--reconstruct-zip)
            requires_ros=false
            ;;
    esac
done

if [[ "${requires_ros}" == true ]]; then
    if [[ ! -r /opt/ros/noetic/setup.bash ]]; then
        echo "错误：未找到 ROS Noetic。相机采集 UI 需要 ROS，离线 ZIP 重建不需要。" >&2
        exit 1
    fi
    original_arguments=("$@")
    set --
    source /opt/ros/noetic/setup.bash
    if [[ ! -r "${CAMERA_ROS_SETUP}" ]]; then
        echo "错误：未找到相机 ROS 工作区：${CAMERA_ROS_SETUP}" >&2
        echo "请设置 CAMERA_ROS_SETUP 指向相机驱动编译后的 devel/setup.bash。" >&2
        exit 1
    fi
    source "${CAMERA_ROS_SETUP}"
    set -- "${original_arguments[@]}"
fi

if [[ ! -r "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    echo "错误：未找到 Conda。可通过 CONDA_ROOT 指定服务器上的 Conda 安装目录。" >&2
    exit 1
fi
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate foundationpose

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${PROJECT_ROOT}"
exec python -m tool.object_model_builder.model_builder_ui "$@"
