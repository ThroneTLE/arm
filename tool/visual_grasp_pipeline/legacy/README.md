# 视觉抓取流水线 复现包

## 包内容
- scripts/           核心脚本（放目标机器 ~/ 下）
- scripts/tools/     开发工具脚本（可选）
- models/            YOLO 模型 + CAD 模型
- weights/           FoundationPose 预训练权重（默认不打包，见下方说明）
- static_frame/      静态照片（离线模式用）

## 目标机器上的放置位置
1. scripts/ 里所有文件 -> 放到 Linux 主目录 ~/
2. models/yolo_model.pt -> ~/yolo_model.pt
3. models/<name>/mesh/* -> ~/FoundationPose/demo_data/<name>/mesh/
   （前提：目标机器已装好 FoundationPose 仓库）
4. 权重：默认不在包里。两个 model_best.pth 需放到
   ~/FoundationPose/weights/2023-10-28-18-33-37/ 和
   ~/FoundationPose/weights/2024-01-11-20-02-45/
   （从 FoundationPose 仓库 readme 的 Google Drive 链接下载，
    或让发给你的人单独发这两个文件）
5. static_frame/* -> ~/fp_capture/

## 环境要求
- Windows 10 + WSL2（Ubuntu 20.04）
- ROS2 Foxy + Orbbec Gemini 相机驱动 astra_camera
- conda 环境 foundationpose（Python 3.11），FoundationPose 仓库 commit a1b694b
- 关键 Python 包：torch(CUDA)、warp、ultralytics、lap、pyzmq、trimesh、
  scipy、matplotlib、opencv-python、pillow、numpy
- NVIDIA GPU（RTX 40 系列实测）

## 有相机运行流程
1. Windows：双击 attach_camera.bat（把相机接入 WSL）
2. WSL 终端1：bash ~/start_camera.sh（启动相机驱动）
3. WSL 终端2（系统 python，不要用 conda）：/usr/bin/python3 ~/fp_bridge.py
4. WSL 终端3（conda）：python ~/arm_node.py
5. WSL 终端4（conda）：python ~/vision_node.py
   界面里：拍照识别 -> 选目标点＋加入序列 -> 开始抓取

## 无相机离线运行
前提：~/fp_capture/ 里已有 rgb.png、depth.png、cam_K.txt
直接跑 arm_node.py + vision_node.py，拍照识别会自动等 300ms 后改用静态照片。
注意：静态照片里的深度和相机内参是原机器的，换相机后建议重新拍一张。

## 换机器后可能要改的地方（都在 scripts/fp_pipeline.py 配置区）
- TARGET_LABEL：默认抓的目标标签
- OBJECT_MODELS / YOLO_TO_OBJECT / GRASP_RULES：模型与抓取规则映射
- OFFSET_X_MM / OFFSET_Y_MM / CENTER_OFFSET_MM / FLIP_X / FLIP_Y：坐标补偿
- VIZ2D_FILE / VIZ3D_FILE：桌面输出路径（原机器是 /mnt/c/Users/Administrator/Desktop/）
- ui_font.py 的 UI_FONT_FAMILY：字体偏好
- vision_node.py 的 STATIC_FRAME_DIR：静态照片目录
