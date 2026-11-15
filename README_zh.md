# Unitree RL Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?style=flat&logo=Discord&logoColor=white)](https://discord.gg/ZwcVwxv5rq)


## 概述

本项目基于 [IsaacLab](https://github.com/isaac-sim/IsaacLab)，为宇树（Unitree）机器人提供了一套强化学习训练环境。

目前支持宇树 **Go2**、**H1** 和 **G1-29dof** 机器人。

本仓库还加入了 Go2 Mimic Canter 训练与 MuJoCo sim2sim 链路。新增代码、Motion
数据、使用方法和限制见 [Go2 Mimic Canter 补充说明](README_GO2_MIMIC.md)。
完整的训练 → sim2sim → sim2real 依赖与部署流程见
[部署指南](README_DEPLOYMENT.md)。

<div align="center">

| <div align="center"> Isaac Lab </div> | <div align="center">  Mujoco </div> |  <div align="center"> 实机 </div> |
|--- | --- | --- |
| [<img src="https://oss-global-cdn.unitree.com/static/d879adac250648c587d3681e90658b49_480x397.gif" width="240px">](https://oss-global-cdn.unitree.com/static/d879adac250648c587d3681e90658b49_480x397.gif) | [<img src="https://oss-global-cdn.unitree.com/static/3c88e045ab124c3ab9c761a99cb5e71f_480x397.gif" width="240px">](https://oss-global-cdn.unitree.com/static/3c88e045ab124c3ab9c761a99cb5e71f_480x397.gif) | [<img src="https://oss-global-cdn.unitree.com/static/6c17c6cf52ec4e26bbfab1fbf591adb2_480x270.gif" width="240px">](https://oss-global-cdn.unitree.com/static/6c17c6cf52ec4e26bbfab1fbf591adb2_480x270.gif) |

</div>

## 安装

- 按照[安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)安装 Isaac Lab。
- 安装 Unitree RL IsaacLab 独立环境。

  - 将本仓库单独克隆或复制到 Isaac Lab 安装目录**之外**（即不要放在 `IsaacLab` 目录内）：

    ```bash
    git clone https://github.com/unitreerobotics/unitree_rl_lab.git
    ```
  - 使用已安装 Isaac Lab 的 python 解释器，以可编辑模式安装本库：

    ```bash
    conda activate env_isaaclab
    ./unitree_rl_lab.sh -i
    # 重启你的 shell 以使环境变更生效。
    ```
- 下载宇树机器人的模型描述文件

  *方式一：使用 USD 文件*
  - 从 [unitree_model](https://huggingface.co/datasets/unitreerobotics/unitree_model/tree/main) 下载宇树的 usd 文件，注意保持文件夹结构
    ```bash
    git clone https://huggingface.co/datasets/unitreerobotics/unitree_model
    ```
  - 设置环境变量 `UNITREE_MODEL_DIR`；未设置时默认使用 `~/unitree_model`。

    ```bash
    export UNITREE_MODEL_DIR="/home/user/projects/unitree_model"
    ```

  *方式二：使用 URDF 文件【推荐】* 仅适用于 Isaac Sim >= 5.0
  -  从 [unitree_ros](https://github.com/unitreerobotics/unitree_ros) 下载宇树机器人的 urdf 文件
      ```
      git clone https://github.com/unitreerobotics/unitree_ros.git
      ```
  - 设置环境变量 `UNITREE_ROS_DIR`；未设置时默认使用 `~/unitree_ros`。
    ```bash
    export UNITREE_ROS_DIR="/home/user/projects/unitree_ros"
    ```
  - 【可选】：如果你想使用 urdf 文件，可以修改 *robot_cfg.spawn*



- 通过以下方式验证环境是否安装正确：

  - 列出可用任务：

    ```bash
    ./unitree_rl_lab.sh -l # 比 isaaclab 自带的方式更快
    ```
  - 运行一个任务：

    ```bash
    ./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity # 支持任务名自动补全
    # 等价于
    python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity
    ```
  - 用训练好的智能体进行推理：

    ```bash
    ./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity # 支持任务名自动补全
    # 等价于
    python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity
    ```

## 部署

模型训练完成后，我们需要先在 Mujoco 中对训练好的策略做 sim2sim（仿真到仿真）测试，验证模型的效果。
然后再部署 sim2real（仿真到真机）。

### 环境搭建

```bash
# 安装依赖
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
# 安装 unitree_sdk2
git clone git@github.com:unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF # 安装到 /usr/local 目录
sudo make install
# 编译 robot_controller
cd unitree_rl_lab/deploy/robots/g1_29dof # 或其他机器人型号
mkdir build && cd build
cmake .. && make
```

### Sim2Sim（仿真到仿真）

安装 [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco?tab=readme-ov-file#installation)。

- 将 `/simulate/config.yaml` 中的 `robot` 设置为 g1
- 将 `domain_id` 设置为 0
- 将 `enable_elastic_hand` 设置为 1
- 将 `use_joystck` 设置为 1

```bash
# 启动仿真
cd unitree_mujoco/simulate/build
./unitree_mujoco
# ./unitree_mujoco -i 0 -n eth0 -r g1 -s scene_29dof.xml # 备选方式
```

```bash
cd unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl
# 1. 按下 [L2 + Up]，让机器人站立
# 2. 点击 mujoco 窗口，然后按 8，让机器人双脚接触地面
# 3. 按下 [R1 + X]，运行策略
# 4. 点击 mujoco 窗口，然后按 9，关闭吊带（elastic band）
```

### Sim2Real（仿真到真机）

你可以用这个程序直接控制真机，但请确保机器人自带的机载控制程序已经关闭。

```bash
./g1_ctrl --network eth0 # eth0 是网卡接口名称
```

## 致谢

本仓库的实现得益于以下开源项目的支持和贡献，特此感谢：

- [IsaacLab](https://github.com/isaac-sim/IsaacLab)：训练和运行代码的基础框架。
- [mujoco](https://github.com/google-deepmind/mujoco.git)：提供了强大的仿真功能。
- [robot_lab](https://github.com/fan-ziqi/robot_lab)：项目结构和部分实现参考了该项目。
- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)：功能多样的人形机器人动作跟踪控制框架。
