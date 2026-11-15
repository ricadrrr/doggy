# Go2 Mimic Canter 补充说明

本文说明本仓库在上游 `unitree_rl_lab` 基础上新增的 Go2 动作模仿与 MuJoCo 部署链路。

- **动作模仿（Mimic）**：策略跟踪一段参考动作中的姿态、关节和速度。
- **sim2sim（仿真到仿真）**：在 Isaac Lab 中训练，再把同一策略放入 MuJoCo 闭环验证。

当前结果：`Velocity → Mimic_Canter → Velocity` 连续测试 5 次均稳定通过，返回 Velocity 后仍可响应速度命令。尚未在 Go2 真机上验证，因此不能称为 sim2real 成功。

## 新增内容

```text
scripts/mimic/
  kine2go_to_csv.py                 Kine2Go NPY → Go2 CSV
  csv_to_npz.py                     CSV → Isaac Lab NPZ，新增 Go2 支持
  replay_npz.py                     NPZ 回放，新增 Go2 支持

source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/go2/canter/
  __init__.py                       注册 Unitree-Go2-Mimic-Canter
  tracking_env_cfg.py               场景、观测、奖励、终止和 Motion 配置

deploy/robots/go2/
  include/State_Mimic.h
  src/State_Mimic.cpp               Mimic 状态和双策略平滑交接
  config/config.yaml                Go2 状态机与策略路径

datasets/kine2go/ai4_dog_run_04/
  ai4_dog_run_04_clean_464_734_60hz.csv
  ai4_dog_run_04_clean_464_734_50hz.npz
```

训练日志、checkpoint 和导出的 ONNX 模型属于生成产物，不纳入 Git。

## 1. 环境准备

先按 [完整部署指南](README_DEPLOYMENT.md) 安装训练与部署依赖；主项目简介见
[README_zh.md](README_zh.md)。

机器人模型默认从 `~/unitree_ros` 读取，也可用环境变量（运行程序前设置的外部配置值）覆盖：

```bash
export UNITREE_ROS_DIR=/path/to/unitree_ros
```

训练 Motion 默认读取仓库中的 50 Hz NPZ。如需替换：

```bash
export UNITREE_GO2_CANTER_MOTION_FILE=/path/to/motion.npz
```

## 2. 回放参考动作

```bash
python scripts/mimic/replay_npz.py \
  --robot go2 \
  --file datasets/kine2go/ai4_dog_run_04/ai4_dog_run_04_clean_464_734_50hz.npz
```

验收重点：

- 关节顺序正确，无腿部交叉；
- 动作包含进入奔跑和减速退出；
- 首尾不进入原始数据中的坐姿穿模区。

## 3. 训练与导出

训练任务：

```bash
python scripts/rsl_rl/train.py \
  --headless \
  --task Unitree-Go2-Mimic-Canter
```

回放并导出策略：

```bash
python scripts/rsl_rl/play.py \
  --task Unitree-Go2-Mimic-Canter \
  --checkpoint /path/to/model.pt
```

部署侧需要每个策略目录至少包含：

```text
params/deploy.yaml
exported/policy.onnx
```

`deploy/robots/go2/config/config.yaml` 默认从以下目录中自动选择最新且含 `exported/` 的运行：

```text
logs/rsl_rl/unitree_go2_mimic_canter/
logs/rsl_rl/unitree_go2_velocity/
```

## 4. MuJoCo sim2sim

安装并启动 [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)，确保 DDS 域为 `0`。DDS（数据分发服务）是控制器与仿真器交换状态和命令的通信层。

编译 Go2 控制器：

```bash
cmake -S deploy/robots/go2 -B deploy/robots/go2/build
cmake --build deploy/robots/go2/build -j
```

启动控制器：

```bash
./deploy/robots/go2/build/go2_ctrl --network lo
```

状态切换：

1. `Passive → FixStand`：`LT + A`
2. `FixStand → Velocity`：`Start`
3. `Velocity → Mimic_Canter`：按住 `LT` 2 秒并按 `A`
4. Motion 结束后自动返回 `Velocity`

如需排查交接问题，把 `deploy/robots/go2/config/config.yaml` 中的 `diagnostics` 改为 `true`；正常运行保持 `false`。

## 5. 交接设计

控制器分四个阶段：

1. `EntryVelocity`：Velocity 策略逐步加速；
2. `HandoffToMimic`：按比例把控制权交给 Mimic；
3. `Mimic`：跟踪参考动作；
4. `HandoffToVelocity`：按参考动作末端速度平滑返回 Velocity。

内部的“影子速度策略”只用于计算过渡目标，不直接控制电机。它只在参考速度回到 Velocity 策略训练范围后工作，并在每次推理前重置历史动作，避免 `last_action` 在没有真实物理反馈时递归放大，产生饱和输出。

## 6. 数据许可和限制

动作来自 Kine2Go 转换的 AI4Animation 数据，适用 **CC BY-NC 4.0**，不是本仓库的 Apache-2.0 许可；详情见 [Motion 数据说明](datasets/kine2go/ai4_dog_run_04/README.md)。

当前限制：

- 只验证一次 4.5 秒参考动作，不支持无限循环；
- Mimic 策略不接收实时速度和转向命令；
- 训练产物需要本地重新生成；
- 尚未完成真机 sim2real。

## 7. 调试记录

- [Go2 Mimic Canter 进度](.learn/GO2_MIMIC_CANTER_PROGRESS.md)
- [Go2 Motion 数据复核](.learn/GO2_MOTION_DATA_REVIEW.md)
