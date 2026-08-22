# AGENTS.md · 接手本项目前必读

**先读这两份，顺序不要换：**

1. [`调试方法论-所有AI必读.md`](调试方法论-所有AI必读.md) —— 九条用真实代价换来的调试纪律
2. [`CLAUDE.md`](CLAUDE.md) —— 设备结论、环境设置、坐标约定、文档索引

（`CLAUDE.md` 是 Claude Code 自动载入的；其它工具请手动读。两份都不长。）

---

## 如果你只读得进这一页，至少记住这四条

2026-08-22 现场：摔了一次机械臂，被赶出场地，隔壁组压爆了一个瓶子。
下面每一条都有日志证据。

1. **`0x2314` 不是急停，是直接下电。** 实测 5/5 次映射到 `Deadan_End -> PowerOff`，
   伺服失力，伸展着的手臂会**靠自重坠落**。真正的急停只能是示教器上的物理按钮。
   运动还没下发时不要"保险起见"发它。

2. **远程模式下每条运动都会被拒，拒绝会顺手把带电的伺服下掉。**
   出厂配置 `RemoteIO[0].posReset` 是 `safeEnable=true` 但 `deviation=null`，
   复位点安全闸门恒判"不在安全位置"。全量归档复核：08-22 被拒 **18** 次，
   全部走 `Deadan_End` 停止链，其中 2 次伺服带电 → 真下电（其余 16 次本就没电）。
   闸门只咬 `0x3002/0x3003/0x3007`；示教下跑得通的是 **`0x4502` MOVL**，
   `0x4501` MOVJ 参数错误率极高，**执行链只用 MOVL**。
   症状是"机械臂完全不动但夹爪照常开合"。**对策：全程留在示教模式。**

3. **A/B/C 回读是"度"，运动指令收"弧度"。** 跨边界不转换会摔臂（已发生）。
   旋转约定是内旋 `Rx(A)·Ry(B)·Rz(C)`，不是固定系 `Rz·Ry·Rx`。

4. **不要从 `tool/arm_project_20260822/` 启动任何东西。** 那是分叉的旧副本，
   缺全部修复；启动脚本按自身位置推 `PROJECT_ROOT`，从那里启动会整个绕过主仓。

---

## 环境

```bash
cd /home/throne/workspaces/arm
export ARMPY=/home/throne/miniconda3/envs/foundationpose/bin/python
export PYTHONPATH=$PWD:$PWD/ros_ws/src/arm_vision_framework/src
```

系统 `python` 是 3.8，缺 trimesh/torch，会让自检**静默跳过**关键检查。

## 改动前后都要能离线验证

```bash
./competition_pipeline/scripts/run_offline_rehearsal.sh     # 无硬件全栈演练
$ARMPY -m unittest discover -s competition_pipeline/tests -t . -p 'test_*.py'
$ARMPY -m unittest discover -s tool/visual_grasp_pipeline/tests -t . -p 'test_*.py'
```

## 写结论必须标证据等级

【实测】/【配置】/【推断】/【存疑】——**不标就等于骗人**。
这个项目已经因为"把推断当结论"付过代价，详见方法论第 9 条下方。

## 遇到通信问题

**不要查手册和网站**（公开文档已被实测推翻多处）。
看 [`docs/通信问题现场定位手册.md`](docs/通信问题现场定位手册.md)。
