#!/usr/bin/env bash
# 一键启动 Gemini 深度相机（深度注册模式）

echo "=== 检查 USB 设备 ==="
if lsusb | grep -q "2bc5"; then
  echo "OK: 相机已连接到 WSL"
else
  echo "错误: 未检测到 Orbbec 相机 (2bc5)"
  echo "请先在 Windows 双击 attach_camera.bat 连接 USB"
  exit 1
fi

conda deactivate 2>/dev/null || true

echo "=== 启动相机 ==="
if tmux new-session -d -s cam \
  "ros2 launch astra_camera gemini.launch.xml depth_registration:=true" 2>/dev/null; then
  sleep 5
  if tmux has-session -t cam 2>/dev/null; then
    echo "相机已在后台启动 (tmux 会话: cam)"
    echo "查看日志: tmux attach -t cam   (退出: Ctrl+B 然后按 D)"
    exit 0
  fi
  echo "tmux 会话启动后立即退出，改为前台启动..."
else
  echo "tmux 不可用，直接前台启动（请保持此终端开着）..."
fi

exec ros2 launch astra_camera gemini.launch.xml depth_registration:=true
