#!/usr/bin/env bash
# AnyGrasp SDK 环境搭建脚本（CUDA 11.8 + PyTorch 2.5.1，按授权文章流程整理）。
#
# 用途：在拿到 license 与 checkpoint 之前先把环境准备好。
# 用法：
#   ./setup_anygrasp_env.sh [env_name] [log_file]
#   CONDA_ROOT=/path/to/miniconda3 THIRD_PARTY=/path/to/third_party ./setup_anygrasp_env.sh
#
# 说明：
#   - 本机 GPU 为 RTX 4060（sm_89），驱动 560.35 向下兼容 CUDA 11.8 运行时；
#   - 系统没有 /usr/local/cuda 工具链，因此 nvcc 11.8 通过 conda 安装到环境内；
#   - pytorch 用 pip 的 +cu118 wheel（官方 conda 频道缺 py39+cu118 构建，
#     会装成 CPU 版导致后续全部失败）；
#   - graspnetAPI 按文章修改 setup.py（numpy 不锁版本、transforms3d==0.4.1、
#     sklearn 改名 scikit-learn）后本地安装；
#   - MinkowskiEngine 从 NVIDIA 官方 master 源码编译，链接 conda 的 openblas；
#     自带的 3rdparty pybind11 太老，需移除后使用 torch 自带的 pybind11 头文件。
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-/home/throne/miniconda3}"
ENV_NAME="${1:-anygraspenv}"
LOG_FILE="${2:-/home/throne/anygrasp_env_setup.log}"
THIRD_PARTY="${THIRD_PARTY:-/home/throne/workspaces/arm_data/third_party}"
SDK_DIR="${THIRD_PARTY}/anygrasp_sdk"

TUNA_MAIN="https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main"
ANACONDA_OFFICIAL="https://conda.anaconda.org/anaconda"
NVIDIA_OFFICIAL="https://conda.anaconda.org/nvidia"
PYPI_TUNA="https://pypi.tuna.tsinghua.edu.cn/simple"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
exec >> "${LOG_FILE}" 2>&1

echo "== [1/8] 创建环境 ${ENV_NAME} (python 3.9)"
if [[ -x "${CONDA_ROOT}/envs/${ENV_NAME}/bin/python" ]]; then
  echo "环境已存在，保留已安装包并继续补全"
else
  conda create -y -n "${ENV_NAME}" python=3.9 -c "${TUNA_MAIN}"
fi

conda activate "${ENV_NAME}"
export PIP_NO_CACHE_DIR=1

echo "== [2/8] openblas-devel"
conda install -y -n "${ENV_NAME}" openblas-devel -c "${ANACONDA_OFFICIAL}"

echo "== [3/8] pytorch 2.5.1 (pip +cu118 wheel)"
# 官方 pytorch conda 频道对 2.5.1 没有 py39+cu118 构建（solver 会装成 CPU 版），
# 改用 download.pytorch.org 的 +cu118 wheel（本机可达），运行时与 conda 版一致。
# AnyGrasp SDK 只依赖 torch + MinkowskiEngine，不装 torchvision/torchaudio（省磁盘）。
python -m pip install "torch==2.5.1+cu118" \
  --extra-index-url "https://download.pytorch.org/whl/cu118" -i "${PYPI_TUNA}"

echo "== [4/8] conda nvcc 11.8 工具链（编译 MinkowskiEngine / pointnet2）"
conda install -y -n "${ENV_NAME}" -c "${NVIDIA_OFFICIAL}/label/cuda-11.8.0" \
  cuda-nvcc cuda-cudart-dev cuda-cccl

echo "== [4.5/8] cusparse/cublas 头文件（NVIDIA 官方 Ubuntu 仓库的 .deb 解包提取）"
# torch 的 CUDA 头文件 CUDAContextLight.h 需要 <cusparse.h> 和 <cublas_v2.h>，
# 而 conda 各频道都没有 CUDA 11.8 的 dev 头文件；从 NVIDIA Ubuntu 20.04 仓库
# 下载 dev deb 并只把 include 拷进环境（无需 root）。
mkdir -p /tmp/cuda_dev_headers
for PAIR in \
  "libcusparse-dev-11-8_11.7.5.86-1_amd64.deb|/tmp/libcusparse-dev-11-8.deb" \
  "libcublas-dev-11-8_11.11.3.6-1_amd64.deb|/tmp/libcublas-dev-11-8.deb" \
  "libcusolver-dev-11-8_11.4.1.48-1_amd64.deb|/tmp/libcusolver-dev-11-8.deb" \
  "libcurand-dev-11-8_10.3.0.86-1_amd64.deb|/tmp/libcurand-dev-11-8.deb" \
  "cuda-nvtx-11-8_11.8.86-1_amd64.deb|/tmp/cuda-nvtx-11-8.deb"; do
  DEB_NAME="${PAIR%%|*}"
  DEB_PATH="${PAIR##*|}"
  # ``dpkg-deb --info`` only validates the small control member and can accept
  # a truncated data member.  Stream the full archive index before deciding a
  # previous interrupted download is complete, then resume it in place.
  if ! (dpkg-deb --fsys-tarfile "${DEB_PATH}" 2>/dev/null | tar -tf - >/dev/null 2>&1); then
    curl -fL --retry 12 --retry-delay 3 --continue-at - \
      -o "${DEB_PATH}" \
      "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/${DEB_NAME}"
  fi
  if ! (dpkg-deb --fsys-tarfile "${DEB_PATH}" 2>/dev/null | tar -tf - >/dev/null 2>&1); then
    echo "CUDA 开发包下载不完整：${DEB_PATH}" >&2
    exit 1
  fi
  dpkg -x "${DEB_PATH}" /tmp/cuda_dev_headers
done
cp -r /tmp/cuda_dev_headers/usr/local/cuda-11.8/include/. "${CONDA_PREFIX}/include/" 2>/dev/null || true
# ME 链接需要 -lcusparse：把 pip wheel 的运行时库拷进环境 lib，并保留 dev deb 的
# libcusparse.so 符号链接（指向 .so.11）。
if [[ ! -e "${CONDA_PREFIX}/lib/libcusparse.so.11" ]]; then
  cp -a "$(python -c 'import site; print(site.getsitepackages()[0])')/nvidia/cusparse/lib/libcusparse.so.11" \
    "${CONDA_PREFIX}/lib/" 2>/dev/null || true
fi
cp -a /tmp/cuda_dev_headers/usr/local/cuda-11.8/lib64/libcusparse.so "${CONDA_PREFIX}/lib/" 2>/dev/null || true
rm -rf /tmp/cuda_dev_headers

export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"

echo "== [5/8] MinkowskiEngine（源码编译，openblas）"
ME_DIR="${THIRD_PARTY}/MinkowskiEngine"
if [[ ! -f "${ME_DIR}/setup.py" ]]; then
  curl -sL --retry 3 -o /tmp/minkowskiengine.tar.gz \
    "https://codeload.github.com/NVIDIA/MinkowskiEngine/tar.gz/refs/heads/master"
  tar xzf /tmp/minkowskiengine.tar.gz -C "${THIRD_PARTY}"
  mv "${THIRD_PARTY}/MinkowskiEngine-master" "${ME_DIR}"
  rm -f /tmp/minkowskiengine.tar.gz
fi
# ME 自带的 3rdparty pybind11 太老（缺少 torch 2.5 头文件需要的 py::module_），
# 移除后编译会使用 torch wheel 自带的 pybind11 头文件。
if [[ -d "${ME_DIR}/src/3rdparty/pybind11" ]] && [[ ! -d "${ME_DIR}/src/3rdparty/pybind11.bak" ]]; then
  mv "${ME_DIR}/src/3rdparty/pybind11" "${ME_DIR}/src/3rdparty/pybind11.bak"
fi
cd "${ME_DIR}"
MAX_JOBS=4 python setup.py install \
  --blas_include_dirs="${CONDA_PREFIX}/include" --blas=openblas

echo "== [6/8] graspnetAPI（修改 setup.py 后本地安装）"
GA_DIR="${THIRD_PARTY}/graspnetAPI"
if [[ ! -f "${GA_DIR}/setup.py" ]]; then
  curl -sL --retry 3 -o /tmp/graspnetapi.tar.gz \
    "https://codeload.github.com/graspnet/graspnetAPI/tar.gz/refs/heads/master"
  tar xzf /tmp/graspnetapi.tar.gz -C "${THIRD_PARTY}"
  mv "${THIRD_PARTY}/graspnetAPI-master" "${GA_DIR}"
  rm -f /tmp/graspnetapi.tar.gz
fi
python - "${GA_DIR}/setup.py" <<'PY'
import pathlib
import sys

# 上游已修复大部分依赖（numpy 不锁、scikit-learn 改名等），这里按官方经验
# 对旧版本 setup.py 做兼容修正；片段不存在则跳过。
path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
replacements = {
    "'numpy==1.23.4'": "'numpy'",
    "'transforms3d==0.3.1'": "'transforms3d>=0.4.1'",
    "'sklearn'": "'scikit-learn'",
}
changed = False
for old, new in replacements.items():
    if old in text:
        text = text.replace(old, new)
        changed = True
path.write_text(text, encoding="utf-8")
print("graspnetAPI setup.py patched" if changed else "graspnetAPI setup.py already compatible")
PY
cd "${GA_DIR}"
python -m pip install . -i "${PYPI_TUNA}"
# gsnet 预编译扩展按 numpy 1.x ABI 构建；graspnetAPI 不锁 numpy，必须钉回 <2.0。
python -m pip install "numpy<2" -i "${PYPI_TUNA}"
# ROS Noetic's Python modules import rospkg, which is not included in a fresh
# Conda environment even after sourcing /opt/ros/noetic/setup.bash.
python -m pip install rospkg -i "${PYPI_TUNA}"

echo "== [6.5/8] Gemini 静态帧验证依赖（YOLO segmentation）"
# Pin torchvision to the ABI matching torch 2.5.1+cu118.  Letting pip choose
# the newest torchvision upgrades torch to CUDA 12.x and breaks gsnet/ME.
python -m pip install "torchvision==0.20.1+cu118" \
  --extra-index-url "https://download.pytorch.org/whl/cu118" -i "${PYPI_TUNA}"
python -m pip install "ultralytics-thop==2.0.14" --no-deps -i "${PYPI_TUNA}"
python -m pip install "ultralytics==8.3.170" --no-deps -i "${PYPI_TUNA}"
# graspnetAPI currently resolves to an OpenCV 5 prerelease which requires
# NumPy 2.x, while the vendor gsnet binary requires the NumPy 1.x ABI.
python -m pip install "opencv-python==4.10.0.84" psutil py-cpuinfo -i "${PYPI_TUNA}"

echo "== [7/8] pointnet2（AnyGrasp SDK 内）"
cd "${SDK_DIR}/pointnet2"
python setup.py install

echo "== [7.5/8] gsnet 预编译扩展（cp39）"
# SDK 的 gsnet 是预编译 .so：按 python 版本拷贝对应文件到 grasp_detection。
# 当前官方 SDK（anygrasp_sdk）不需要 lib_cxx；机器码应通过
#   python -c "from gsnet import get_feature_id; print(get_feature_id())"
# 获取（无需 license）。
cd "${SDK_DIR}/grasp_detection"
cp -f gsnet_versions/gsnet.cpython-39-x86_64-linux-gnu.so gsnet.so

echo "== [8/8] 导入校验"
python - <<PY
import numpy
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(),
      "numpy", numpy.__version__)
import MinkowskiEngine
print("MinkowskiEngine", MinkowskiEngine.__version__)
from graspnetAPI import GraspGroup
print("graspnetAPI import ok")
import pointnet2
print("pointnet2 import ok")
PY

echo "== 完成：环境 ${ENV_NAME} 就绪（license 与 checkpoint 到位后即可运行 SDK demo）"
