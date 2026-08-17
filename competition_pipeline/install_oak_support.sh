#!/usr/bin/env bash
set -euo pipefail

# Competition UI runtime (Ubuntu 20.04 / system Python 3.8).
python3 -m pip install --user --upgrade 'pip<25'
python3 -m pip install --user --only-binary=:all: 'depthai==2.30.0.0'

# Official Luxonis calibration tool in an isolated OpenCV 4.5.5 environment.
conda_bin="/home/throne/miniconda3/bin/conda"
oak_env="/home/throne/miniconda3/envs/oak-calibration"
tool_root="/home/throne/workspaces/arm_data/third_party/oak_calibration_tool"
if [[ ! -x "$conda_bin" ]]; then
  echo "Conda not found: $conda_bin" >&2
  exit 1
fi
if [[ ! -x "$oak_env/bin/python" ]]; then
  "$conda_bin" create -y -p "$oak_env" python=3.9 pip
fi
"$oak_env/bin/python" -m pip install \
  'numpy==1.26.4' 'depthai==2.32.0.0' \
  'opencv-python==4.5.5.62' 'opencv-contrib-python==4.5.5.62' \
  scipy matplotlib packaging Qt.py
mkdir -p "$tool_root/depthai_calibration" "$tool_root/resources/depthai_boards/boards"
curl -L --fail --retry 4 \
  'https://raw.githubusercontent.com/luxonis/depthai/5d9b9c7bd57b2dc184ba2a6087e63cc2f596038e/calibrate.py' \
  -o "$tool_root/calibrate.py"
curl -L --fail --retry 4 \
  'https://raw.githubusercontent.com/luxonis/depthai-calibration/bedcf34b50151692b85adf71cf9425ce265216a8/calibration_utils.py' \
  -o "$tool_root/depthai_calibration/calibration_utils.py"
touch "$tool_root/depthai_calibration/__init__.py"
curl -L --fail --retry 4 \
  'https://raw.githubusercontent.com/luxonis/depthai-boards/03553aee5052824394efda91127437d08ccb2dcf/boards/OAK-D-PRO.json' \
  -o "$tool_root/resources/depthai_boards/boards/OAK-D-PRO.json"
"$oak_env/bin/python" "$tool_root/calibrate.py" --help >/dev/null

# Luxonis USB bootloader/device access. This step asks for sudo once.
rules_file="$(mktemp)"
trap 'rm -f "$rules_file"' EXIT
printf '%s\n' 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' > "$rules_file"
sudo install -m 0644 "$rules_file" /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules
sudo udevadm trigger

python3 -c 'import depthai; print("DepthAI runtime:", depthai.__version__)'
"$oak_env/bin/python" -c 'import cv2, depthai; print("Calibration env:", cv2.__version__, depthai.__version__)'
echo 'OAK support installed. Replug the camera if it was already connected.'
