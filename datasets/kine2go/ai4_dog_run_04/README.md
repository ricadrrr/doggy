# Go2 Run-04 Motion

本目录只保留 Go2 Mimic Canter 任务复现所需的最小动作数据：

- `ai4_dog_run_04_clean_464_734_60hz.csv`：270 帧、60 Hz，供 C++ 部署侧读取。
- `ai4_dog_run_04_clean_464_734_50hz.npz`：225 帧、50 Hz，供 Isaac Lab 训练和回放。

原始 `motion.npy`、官方视频和其他试验切片不纳入 Git。

## 数据来源与许可

- 上游数据集：[MIMUW-Robotics/kine2go](https://huggingface.co/datasets/MIMUW-Robotics/kine2go)
- 原始动作：[AI4Animation SIGGRAPH 2018](https://github.com/sebastianstarke/AI4Animation/tree/master/AI4Animation/SIGGRAPH_2018)
- 动作：`dog_run04_joint_pos.txt`
- 论文：H. Zhang et al., *Mode-Adaptive Neural Networks for Quadruped Motion Control*, SIGGRAPH 2018
- 许可：[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

这些动作数据不是 Apache-2.0 许可。其衍生数据仅可按 CC BY-NC 4.0 使用，并需保留署名；不可用于商业用途。

## 处理过程

最终采用原始 60 Hz 序列的 `[464:734)` 帧：

- 入口速度约 `0.96 m/s`；
- 出口速度约 `0.03 m/s`；
- 避开首尾坐姿穿模区；
- 覆盖加速、奔跑和减速阶段。

重新生成：

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

详细的数据筛选依据见 [Go2 Motion 数据复核](../../../.learn/GO2_MOTION_DATA_REVIEW.md)。
