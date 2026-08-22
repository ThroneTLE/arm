#!/usr/bin/env bash
# OAK 设备自动挂载守护进程(解决"引导后以新 BUSID 重连"问题)
# 用法: nohup bash ~/tools/oak_attach_daemon.sh >/tmp/oak_daemon.log 2>&1 &
set -u
USBIPD="/mnt/c/Program Files/usbipd-win/usbipd.exe"

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "OAK 自动挂载守护进程启动 (每 2 秒扫描一次)"
while true; do
  # 找出所有 OAK(03e7/Movidius)设备及其状态
  "$USBIPD" list 2>/dev/null | grep -iE "03e7|movidius|myriad" | while read -r line; do
    busid=$(echo "$line" | awk '{print $1}')
    state=$(echo "$line" | grep -oiE "not shared|shared|attached" | head -1)
    if [ -z "$busid" ]; then continue; fi
    case "$state" in
      "Not shared")
        log "发现未绑定设备 $busid -> bind"
        "$USBIPD" bind --busid "$busid" >/dev/null 2>&1
        ;;
      "Shared")
        if lsusb 2>/dev/null | grep -q 03e7; then
          : # WSL 内设备已可见, 不重复挂载(防止僵尸实例)
        else
          log "发现已绑定未挂载设备 $busid -> attach"
          "$USBIPD" attach --wsl --busid "$busid" >/dev/null 2>&1
        fi
        ;;
      "Attached")
        : # 已挂载, 不动
        ;;
    esac
  done
  sleep 2
done
