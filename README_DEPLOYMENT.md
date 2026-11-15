# 从训练到 sim2sim / sim2real 的部署指南

本文面向第一次使用本仓库的人，说明从 Isaac Lab 训练、导出 ONNX，到 MuJoCo sim2sim，再到 Go2 真机 sim2real 所需的仓库、软件包和操作顺序。

- **ONNX**：跨框架的神经网络模型格式，C++ 控制器用它执行策略。
- **sim2sim（仿真到仿真）**：在 Isaac Lab 训练，在 MuJoCo 中闭环验证。
- **sim2real（仿真到真机）**：把仿真中验证过的控制器连接到真实机器人。
- **DDS（数据分发服务）**：Unitree SDK、MuJoCo 和真机之间传输状态与控制指令的通信层。

当前 Go2 Mimic Canter 已完成 Isaac Lab → MuJoCo sim2sim，连续 5 次闭环测试通过；尚未在 Go2 真机上测试，不能视为 sim2real 已完成。

## 1. 依赖总览

### 必需仓库

| 阶段 | 仓库 | 用途 |
|---|---|---|
| 训练 | [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html) | GPU 物理仿真 |
| 训练 | [Isaac Lab](https://github.com/isaac-sim/IsaacLab) | 训练环境、RSL-RL 接口和策略导出 |
| 训练 | [unitree_ros](https://github.com/unitreerobotics/unitree_ros) | Go2 URDF 和网格模型；本仓库默认使用它 |
| 训练/部署 | 本仓库 | Go2 任务、Motion 工具、C++ 控制器 |
| sim2sim/sim2real | [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2) | Go2 低层状态和命令通信 |
| sim2sim | [MuJoCo](https://github.com/google-deepmind/mujoco) | 第二套物理仿真器 |
| sim2sim | [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) | 用 Unitree SDK2 接口驱动 MuJoCo 中的 Go2 |

### 可选仓库

| 仓库 | 什么时候需要 |
|---|---|
| [unitree_model](https://huggingface.co/datasets/unitreerobotics/unitree_model) | 改用 USD 模型时；当前 Go2 配置不需要 |
| [MIMUW-Robotics/kine2go](https://huggingface.co/datasets/MIMUW-Robotics/kine2go) | 重新生成或更换参考 Motion 时 |
| [kine2go-pipeline](https://github.com/nomagiclab/kine2go-pipeline) | 重新做动作重定向和数据筛选时 |
| [AI4Animation](https://github.com/sebastianstarke/AI4Animation) | 追溯原始犬类动作数据时 |

### 版本和平台

本仓库当前对应：

- Linux x86_64，推荐 Ubuntu 22.04；
- NVIDIA GPU 和可运行 Isaac Sim 的驱动；
- Isaac Sim `5.1.0`；
- Isaac Lab `2.3.x`；
- Python `3.11`（由 Isaac Sim / Isaac Lab 环境提供）；
- C++17；
- ONNX Runtime `1.22.0`（已放在 `deploy/thirdparty/`，无需另装）。

Isaac Lab 与 Isaac Sim 必须使用兼容版本；不要单独升级其中一个。

## 2. 系统软件包

训练机器先安装基础工具：

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake g++ git git-lfs wget unzip \
  libyaml-cpp-dev libboost-all-dev libeigen3-dev \
  libspdlog-dev libfmt-dev
```

MuJoCo 图形和离屏渲染再需要：

```bash
sudo apt install -y \
  libglfw3-dev libgl1-mesa-dev libglew-dev \
  libosmesa6-dev libxi-dev
```

Python 侧的 `torch`、`numpy`、`gymnasium`、`rsl_rl`、`isaaclab` 和 `isaaclab_rl` 应由 Isaac Lab 环境统一提供，不要在系统 Python 中另装一套。项目自身直接依赖 `psutil`，安装脚本会自动处理。

## 3. 安装训练环境

### 3.1 安装 Isaac Sim 与 Isaac Lab

按 [Isaac Lab 官方安装文档](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)完成安装，并创建对应的 Conda 环境。以下用 `env_isaaclab` 和 `~/IsaacLab` 举例：

```bash
conda activate env_isaaclab
export ISAACLAB_PATH="$HOME/IsaacLab"
```

先确认 Isaac Lab 自带示例能运行，再安装本仓库；否则无法区分是 Isaac Lab 问题还是本项目问题。

### 3.2 下载 Go2 模型

```bash
git clone https://github.com/unitreerobotics/unitree_ros.git "$HOME/unitree_ros"
export UNITREE_ROS_DIR="$HOME/unitree_ros"
```

未设置 `UNITREE_ROS_DIR` 时，本仓库默认读取 `~/unitree_ros`。

### 3.3 克隆并安装本仓库

```bash
git clone https://gitee.com/Article_ac/rl-lab.git unitree_rl_lab
cd unitree_rl_lab
git lfs pull

conda activate env_isaaclab
export ISAACLAB_PATH="$HOME/IsaacLab"
./unitree_rl_lab.sh -i
```

重新打开终端并激活环境，然后检查任务：

```bash
conda activate env_isaaclab
cd /path/to/unitree_rl_lab
./unitree_rl_lab.sh -l
```

列表中至少应包含：

```text
Unitree-Go2-Velocity
Unitree-Go2-Mimic-Canter
```

如果使用 `uv` 或普通虚拟环境，脚本仍检查 `CONDA_PREFIX`，可在激活后临时设置：

```bash
export CONDA_PREFIX="$VIRTUAL_ENV"
```

## 4. Motion 数据

仓库已包含两份最小运行资产：

```text
datasets/kine2go/ai4_dog_run_04/
  ai4_dog_run_04_clean_464_734_60hz.csv  # C++ 部署
  ai4_dog_run_04_clean_464_734_50hz.npz  # Isaac Lab 训练
```

先回放检查：

```bash
python scripts/mimic/replay_npz.py \
  --robot go2 \
  --file datasets/kine2go/ai4_dog_run_04/ai4_dog_run_04_clean_464_734_50hz.npz
```

如需从 Kine2Go 原始 `motion.npy` 重建：

```bash
python scripts/mimic/kine2go_to_csv.py \
  datasets/kine2go/ai4_dog_run_04/raw/motion.npy \
  datasets/kine2go/ai4_dog_run_04/ai4_dog_run_04_clean_464_734_60hz.csv \
  --frame-start 464 \
  --frame-end 734

python scripts/mimic/csv_to_npz.py \
  --robot go2 \
  --input_file datasets/kine2go/ai4_dog_run_04/ai4_dog_run_04_clean_464_734_60hz.csv \
  --input_fps 60 \
  --output_fps 50 \
  --output_name datasets/kine2go/ai4_dog_run_04/ai4_dog_run_04_clean_464_734_50hz.npz
```

该 Motion 来自 AI4Animation，经 Kine2Go 重定向，适用 CC BY-NC 4.0；不能按本仓库 Apache-2.0 许可用于商业用途。详见 [数据说明](datasets/kine2go/ai4_dog_run_04/README.md)。

## 5. 训练两个策略

当前平滑交接需要两个独立策略：

1. `Unitree-Go2-Velocity`：正常速度控制；
2. `Unitree-Go2-Mimic-Canter`：跟踪参考动作。

### 5.1 Velocity 策略

```bash
python scripts/rsl_rl/train.py \
  --headless \
  --task Unitree-Go2-Velocity
```

### 5.2 Mimic Canter 策略

```bash
python scripts/rsl_rl/train.py \
  --headless \
  --task Unitree-Go2-Mimic-Canter
```

训练结果默认写入：

```text
logs/rsl_rl/unitree_go2_velocity/<run>/
logs/rsl_rl/unitree_go2_mimic_canter/<run>/
```

其中 `<run>` 是带时间的运行目录。训练日志和 checkpoint 不进入 Git。

## 6. 回放并导出 ONNX

分别对两个策略执行一次 `play.py`：

```bash
python scripts/rsl_rl/play.py \
  --task Unitree-Go2-Velocity \
  --checkpoint logs/rsl_rl/unitree_go2_velocity/<run>/model_<iteration>.pt

python scripts/rsl_rl/play.py \
  --task Unitree-Go2-Mimic-Canter \
  --checkpoint logs/rsl_rl/unitree_go2_mimic_canter/<run>/model_<iteration>.pt
```

部署所需目录结构：

```text
<run>/
  params/
    deploy.yaml
  exported/
    policy.onnx
```

`deploy/robots/go2/config/config.yaml` 指向实验根目录，运行时会自动选择最新且含 `exported/` 的 `<run>`：

```yaml
policy_dir: ../../../logs/rsl_rl/unitree_go2_mimic_canter
velocity_policy_dir: ../../../logs/rsl_rl/unitree_go2_velocity
```

如果加载了错误实验，直接把这两个字段改成明确的 `<run>` 路径。

## 7. 安装 C++ 部署依赖

### 7.1 安装 Unitree SDK2

本仓库的 CMake 当前按默认 `/usr/local` 路径查找 DDS 头文件和库，因此使用默认安装位置：

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2.git
cd unitree_sdk2

cmake -S . -B build -DBUILD_EXAMPLES=OFF
cmake --build build -j
sudo cmake --install build
```

确认以下内容存在：

```text
/usr/local/include/unitree/
/usr/local/include/ddscxx/
/usr/local/lib/              # unitree_sdk2、ddsc、ddscxx
```

如选择 `/opt/unitree_robotics`，必须同步修改 CMake 搜索路径或配置 `CMAKE_PREFIX_PATH`，不能只改安装目录。

### 7.2 编译 Go2 控制器

```bash
cd /path/to/unitree_rl_lab
cmake -S deploy/robots/go2 -B deploy/robots/go2/build
cmake --build deploy/robots/go2/build -j
```

成功后得到：

```text
deploy/robots/go2/build/go2_ctrl
```

## 8. 安装并运行 MuJoCo sim2sim

### 8.1 安装 MuJoCo 与 unitree_mujoco

`unitree_mujoco` 官方当前示例使用 MuJoCo `3.3.6`。若上游更新，请以其 README 中的匹配版本为准：

```bash
mkdir -p "$HOME/.mujoco"
cd "$HOME/.mujoco"
wget https://github.com/google-deepmind/mujoco/releases/download/3.3.6/mujoco-3.3.6-linux-x86_64.tar.gz
tar -xzf mujoco-3.3.6-linux-x86_64.tar.gz

git clone https://github.com/unitreerobotics/unitree_mujoco.git "$HOME/unitree_mujoco"
cd "$HOME/unitree_mujoco/simulate"
ln -s "$HOME/.mujoco/mujoco-3.3.6" mujoco

cmake -S . -B build
cmake --build build -j
```

### 8.2 配置通信

在 `unitree_mujoco/simulate/config.yaml` 中：

- 机器人设为 `go2`；
- `domain_id` 设为 `0`；
- 本机测试使用回环网卡 `lo`；
- 确保没有第二个程序占用 `lowcmd` 通道。

本仓库 Go2 控制器的 DDS 域固定为 `0`，两边必须一致。

### 8.3 启动顺序

终端 1：

```bash
cd "$HOME/unitree_mujoco/simulate/build"
./unitree_mujoco
```

终端 2：

```bash
cd /path/to/unitree_rl_lab
./deploy/robots/go2/build/go2_ctrl --network lo
```

控制顺序：

1. `Passive → FixStand`：`LT + A`
2. `FixStand → Velocity`：`Start`
3. `Velocity → Mimic_Canter`：按住 `LT` 2 秒并按 `A`
4. Motion 完成后自动返回 `Velocity`

`diagnostics: false` 是正常配置。交接异常时，可在 `deploy/robots/go2/config/config.yaml` 中临时设为 `true`，输出两个策略的动作差异和机身倾角。

## 9. sim2real：连接真实 Go2

真机不需要 MuJoCo 和 `unitree_mujoco`，但仍需要：

- 编译好的 `go2_ctrl`；
- `unitree_sdk2`；
- 两个策略的 `deploy.yaml` 和 `policy.onnx`；
- 与 Go2 同网段的有线网卡；
- 机器人端开放的低层控制权限和匹配固件。

首次真机测试前：

1. 按 Unitree 官方文档关闭占用低层命令通道的机载高层控制服务；
2. 确认只有一个进程发布 `lowcmd`；
3. 使用吊架或安全绳，清空机器人周围区域；
4. 遥控器和急停始终由一人控制；
5. 先只验证 `Passive` 和低增益站立，再逐步进入策略；
6. 不要直接在人员附近测试 Mimic 动作。

查找真实网卡名称：

```bash
ip link
```

假设连接 Go2 的网卡为 `eth0`：

```bash
./deploy/robots/go2/build/go2_ctrl --network eth0
```

出现以下提示时必须停止，不要绕过：

```text
The other process is using the lowcmd channel
```

它表示已有其他控制程序正在发送低层命令。同时控制同一机器人会造成冲突。

> 本仓库的 Go2 Mimic Canter 尚未完成真机测试。sim2sim 通过只是 sim2real 的前置条件，不是安全保证。

## 10. 常见问题

### `No conda environment activated`

先激活 Isaac Lab 环境。使用 `uv` 时设置：

```bash
export CONDA_PREFIX="$VIRTUAL_ENV"
```

### 找不到 Go2 URDF

```bash
export UNITREE_ROS_DIR=/absolute/path/to/unitree_ros
```

目录下应存在：

```text
robots/go2_description/urdf/go2_description.urdf
```

### 找不到 Motion NPZ

默认路径是：

```text
datasets/kine2go/ai4_dog_run_04/ai4_dog_run_04_clean_464_734_50hz.npz
```

也可覆盖：

```bash
export UNITREE_GO2_CANTER_MOTION_FILE=/absolute/path/to/motion.npz
```

### 找不到策略或 `policy.onnx`

确认 Velocity 与 Mimic 两个运行目录都同时包含：

```text
params/deploy.yaml
exported/policy.onnx
```

### `Waiting for connection to robot...`

检查：

- MuJoCo 是否已启动；
- 两边 DDS 域是否都是 `0`；
- `--network` 是否选择正确；
- sim2sim 使用 `lo`，真机使用连接 Go2 的实际网卡。

### 找不到 `libonnxruntime`

本仓库直接链接：

```text
deploy/thirdparty/onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so.1
```

先执行 `git lfs pull`，再重新运行 CMake 和编译。

### MuJoCo 无法初始化 GLFW

GLFW 是创建图形窗口的库。确认已安装 `libglfw3-dev`，并在有图形桌面的终端运行；无显示器环境需要正确配置离屏渲染或虚拟显示。

## 11. 最终验收

### 训练侧

- Motion 回放关节顺序正确；
- `play.py` 能稳定完成完整 Motion；
- 导出目录存在 `deploy.yaml` 和 `policy.onnx`。

### sim2sim

- Velocity 能持续响应速度命令；
- 切入 Mimic 无明显过冲；
- Motion 全程不摔倒；
- 自动切回 Velocity 后仍可正常控制；
- 至少连续测试 5 次。

### sim2real

- 只有真机完成同样闭环并满足安全要求后，才能称为 sim2real 成功。

Go2 Mimic 的设计和调试细节见 [README_GO2_MIMIC.md](README_GO2_MIMIC.md)。
