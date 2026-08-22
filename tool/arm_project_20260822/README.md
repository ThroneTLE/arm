# arm_project_20260822 — U 盘备份说明(2026-08-22)

## 内容
| 目录 | 内容 | 说明 |
|---|---|---|
| `arm/` | ThroneTLE/arm 工程(含 git 历史, 最新提交 5e0267f) | 完整源码+配置, 含本次全部改动(2D/3D 视觉、新模型接入) |
| `fp_release_20260821_155930/` | 模型库: YOLO 权重(22.042次)+ CAD 模型(can/banana/apple/orange/lemon)+ 静态帧 | 视觉系统运行必需数据 |
| `tools/` | 辅助脚本: oak_preview/oak_depth_analysis/oak_usb_stability/oak_check/oak_attach_daemon/user_frame_3d 等 | |
| `foundationpose_utils_compat.patch` | FoundationPose Utils.py 兼容补丁 | 见下 |
| `FoundationPose_Utils_original.py` | Utils.py 原版函数(备份对照) | |

## 目标机器部署步骤
1. `arm/` 放任意位置(建议 ~/arm), `fp_release_...` 放到相同相对位置或改配置路径
2. FoundationPose 权重(585MB)+ demo_data(1.4GB) **未拷**——目标机已有则直接用;
   如需完整, 原机在 `/home/huyk/FoundationPose`(权重在 `FoundationPose/weights/`)
3. **必须打补丁**(苹果/橙子/柠檬 OBJ/GLB 需要):
   ```bash
   cd <FoundationPose 根目录>
   patch -p1 < foundationpose_utils_compat.patch   # 或把 Utils.py 里 make_mesh_tensors
   # 用 /mnt/e/arm_project_20260822/foundationpose_utils_compat.patch 的 hunk 手工替换
   ```
4. conda foundationpose 环境包版本(numpy 1.26.4 / trimesh 4.2.2 / opencv-contrib 4.9 /
   open3d 0.18 / ultralytics 8.3.170 / depthai 2.32 / PyQt5 5.15.10)需与仓库 requirements 对齐
5. 相机 USB 透传(WSL2): usbipd bind/attach + `tools/oak_attach_daemon.sh`(守护) 
