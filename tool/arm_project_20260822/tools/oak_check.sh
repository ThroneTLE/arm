#!/usr/bin/env bash
# OAK-D Pro 一键检查(两段式) —— 处理"固件引导时 USB 重枚举导致 usbipd 掉线"问题
# 用法: bash ~/tools/oak_check.sh
set -u

USBIPD="/mnt/c/Program Files/usbipd-win/usbipd.exe"
REPO="/home/huyk/arm"

ensure_attached() {
  local busid="$1"
  "$USBIPD" attach --wsl --auto-attach --busid "$busid" >/dev/null 2>&1 \
    || "$USBIPD" attach --wsl --busid "$busid" >/dev/null 2>&1
  sleep 3
  lsusb | grep -q 03e7
}

echo "==== OAK-D Pro 检查(两段式) ===="
echo "[1/5] 查找设备..."
"$USBIPD" list > /tmp/usbipd_list.txt 2>&1
BUSID=$(grep -iE "03e7|movidius" /tmp/usbipd_list.txt | awk '/^[0-9]/{print $1}' | head -1)
if [ -z "$BUSID" ]; then
  echo "❌ Windows 侧未发现 03e7 设备。请确认:"
  echo "   ① 相机插紧(换直连 USB 口, 别用 HUB/延长线)  ② USB 线是否完好  ③ usbipd list 是否出现设备"
  exit 1
fi
echo "    设备 BUSID=$BUSID ✓"

echo "[2/5] 绑定..."
"$USBIPD" list | grep -qE "^$BUSID .*Shared|^$BUSID .*Attached" || "$USBIPD" bind --busid "$BUSID" >/dev/null 2>&1

echo "[3/5] 第 1 段: 挂载 + 完成固件引导(会自动重挂, 若掉线属预期)..."
if ! ensure_attached "$BUSID"; then
  echo "    ⚠️ 首次挂载失败, 再试一次"; sleep 2; ensure_attached "$BUSID" || true
fi

echo "[4/5] 自检(失败会自动重试: 引导完成后的设备不再重枚举)..."
for attempt in 1 2 3; do
  echo "    --- 第 $attempt 次自检 ---"
  OUT=$(cd "$REPO" && CONDA_ROOT=/home/huyk/miniconda3 \
    ./tool/visual_grasp_pipeline/run_oak_vision_node.sh --camera-check 2>&1)
  if echo "$OUT" | grep -q '"status": "ok"'; then
    echo "$OUT" | grep -E '"status"|"mxid"|rgb_shape|depth_shape|sync_delta|"name"|confidence' | head -12
    echo ""
    echo "🎉 相机驱动 + 算法链路全部打通! 可运行:"
    echo "   cd ~/arm && CONDA_ROOT=/home/huyk/miniconda3 ./tool/visual_grasp_pipeline/run_oak_vision_node.sh"
    exit 0
  fi
  echo "    失败(掉线), 重新挂载后重试..."
  sleep 2
  ensure_attached "$BUSID" || sleep 3
done

echo ""
echo "❌ 3 次自检均失败。常见原因与处理:"
echo "   1. USB 口协商成了 USB2 / 供电不足 → 换直连 USB3 口(蓝口)或换线"
echo "   2. Windows USB 选择性暂停 → 设备管理器取消『允许计算机关闭此设备以节约电源』,"
echo "      或管理员 PowerShell: powercfg /setacvalueindex SCHEME_CURRENT 2a737441-1930-4402-8d77-b2bebba308a3 48e6b7a6-50f5-4782-a5d4-53bb8f07e226 0 && powercfg /setactive SCHEME_CURRENT"
echo "   3. 拔插一次让 Windows 重新枚举, 再重跑本脚本"
exit 1
