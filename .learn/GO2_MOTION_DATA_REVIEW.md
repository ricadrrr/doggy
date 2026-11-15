# Go2 Canter Motion 数据复核（2026-08-12）

## 结论

`ai4_dog_canter/raw/motion.npy` 可以作为“静止 → 加速 → 稳定 Canter”的完整
固定表演 Motion，但**不能**作为“静止 → Canter → 减速 → 静止”的完整 Motion。

原因是它的结尾仍处于约 `2.3–3.5 m/s` 的高速 Canter，数据中不存在减速和停止过程。
因此，单靠这一段数据重新训练，无法解决 Mimic 结束后可靠切回当前 `≤1 m/s`
Velocity 策略的问题。

## 输入与方法

- 原始文件：`datasets/kine2go/ai4_dog_canter/raw/motion.npy`
- 实测形状：`(567, 61)`，`float32`，60 Hz，时长 9.45 s。
- 使用原始字段 `base_lin_vel=[55:58]` 计算平面速度。
- 使用已确认的 Genesis → Unitree SDK 关节重排，比较关节角和关节速度。

## 实测阶段

| 阶段 | 原始帧范围 | 时间范围 | 平面速度 | 说明 |
|---|---:|---:|---:|---|
| 静止准备 | `[0:342)` | 0.00–5.70 s | `<0.1 m/s` | 姿态稳定、关节速度低，但不是仓库默认站立姿态。 |
| 加速过渡 | `[342:437)` | 5.70–7.28 s | `0.1→2.75 m/s` | 包含真实的重心转移、触地和关节速度变化。 |
| 稳态 Canter | `[437:567)` | 7.28–9.45 s | 约 `2.5–3.5 m/s` | 与当前循环 Canter 的用途一致。 |
| 减速/停止 | 不存在 | — | — | 最后一帧仍约 `2.45 m/s`。 |

关键采样点：

```text
frame   0 (0.00 s): 0.000 m/s
frame 342 (5.70 s): 首次超过 0.1 m/s
frame 370 (6.17 s): 0.500 m/s
frame 381 (6.35 s): 1.000 m/s
frame 409 (6.82 s): 2.000 m/s
frame 437 (7.28 s): 2.75 m/s，进入稳态 Canter
frame 566 (9.43 s): 2.45 m/s，未减速
```

## 已有可用文件

之前已正确转换并保留：

- `ai4_dog_full_60hz.csv`：567 帧、19 列；用于部署侧 `MotionLoader_`。
- `ai4_dog_full_50hz.npz`：472 帧、50 Hz、9.44 s；包含 `fps`、关节数据和 Isaac Lab
  Go2 前向运动学产生的全身刚体数据，可用于训练和运动回放。
- `ai4_dog_canter_cycle_483_510_repeat10_50hz.npz`：当前训练使用的纯稳态循环 Canter。

CSV 与原始 NPY 的逐字段复核结果：

```text
base position 最大绝对误差：4.98e-10
base quaternion 最大绝对误差：5.00e-11
重排后 joint position 最大绝对误差：5.00e-10
```

因此，60 Hz 部署 CSV 的关节顺序和字段没有转换误差；50 Hz NPZ 的字段为
`fps`、`joint_pos`、`joint_vel`、`body_pos_w`、`body_quat_w`、
`body_lin_vel_w`、`body_ang_vel_w`，且长度为 472 帧，和 9.44 s 一致。

## 候选与决策

1. **完整表演候选：`ai4_dog_full_50hz.npz`**
   - 能验证 Mimic 是否可从静止自然进入 Canter。
   - 不能验证自然退出；结束时必须进入安全状态，或另有减速数据/过渡策略。
   - 原始项目的 `reference_rollout.mp4` 在 0.0 s、6.2 s、8.5 s 的画面分别对应
     静态准备、加速和稳态 Canter，和数值阶段划分一致。

2. **当前循环候选：`ai4_dog_canter_cycle_483_510_repeat10_50hz.npz`**
   - 适合研究稳态 Canter 跟踪。
   - 不适合从静止进入或退出。

3. **最终数据需求**
   - 补充一段 `2.5–3 m/s → ≤1 m/s → 静止` 的真实减速 Motion；
   - 或取得覆盖站立、慢速、快速、Canter 的连续速度数据，训练统一的指令条件策略。

## 对部署原型的影响

`deploy/robots/go2/State_Mimic.cpp` 中的双策略交接实验不能弥补这个数据缺口：
当前 Velocity 策略部署配置的线速度范围是 `[-1, 1] m/s`，而 Canter 数据约为
`3 m/s`，两套策略熟悉的状态分布没有充分重叠。后续不应继续以动作线性插值作为
正式方案。

## 今天的路线决策

**不立即重训，也不继续改双策略交接原型。**

下一轮工作先寻找或构造包含减速的候选数据。`ai4_dog_full_50hz.npz` 保留为独立的
“静止 → Canter”验证基线：若后续需要验证它，必须新建一份全程 Motion 的任务配置，
将 Motion 文件替换为该 NPZ，并将 episode 时长设为约 `9.44 s`；不能覆盖当前的
循环 Canter 训练配置。

可视化回放参数为：

```text
scripts/mimic/replay_npz.py
--robot go2
--file datasets/kine2go/ai4_dog_canter/ai4_dog_full_50hz.npz
```

只有在回放确认静态起始姿态可被当前 Go2 模型稳定复现，并获得减速 Motion 后，才开始
新的端到端训练。

## 2026-08-12 补充：`ai4_dog_run_04` 候选与坐姿污染问题

### 发现

在 Kine2Go 数据集（HuggingFace: `MIMUW-Robotics/kine2go`）里核对了全部 40 个片段的
`config.json` / `clip.json`，确认：

- `ai4_dog_canter` 与 `ai4_dog_run_00` 共享同一份原始文件 `dog_run00_joint_pos.txt`
  （`motion.npy` md5 完全一致），因此这条源数据本身不含减速段，不可能通过换切片解决。
- `ai4_dog_run_04`（源文件 `dog_run04_joint_pos.txt`）是独立的一次动捕录制，
  形状 `(1263, 61)`，时长 21.05 s，用 `base_lin_vel=[55:58]` 复核后确认包含完整的
  `静止 → 加速 → 稳态奔跑(≈1.7-2.0 m/s) → 减速 → 静止` 闭环（frame 0-440 静止，
  440-500 加速，500-720 稳态/减速，740+ 回落）。

### 新问题：首尾“静止”段实际是坐姿，非站立

用 `base_z`（第50列，机身离地高度）复核发现：

```text
Go2 默认站立高度参考 ≈ 0.42 m
frame 0-440   (标记为“静止”) base_z ≈ 0.147 m  -- 实际是坐姿，膝关节角度被压到极限
frame 460-600 (稳态奔跑)     base_z ≈ 0.30-0.33 m
frame 760+    (标记为“静止”) base_z ≈ 0.15 m   -- 同样跌回坐姿
```

坐姿下 thigh≈1.33 rad、calf≈-2.47 rad（膝关节几乎折死），retarget 到 Go2 12-DoF
腿部结构后出现明显穿模，用户回放截图确认了这个现象。

### 结论与裁切方案（未执行）

- 结论：`ai4_dog_run_04` 的首尾“静止”段不可直接使用；中段（加速-稳态-减速）姿态干净。
- 建议裁切范围：`frame [442:730]`（t = 7.37s-12.17s，共 288 帧 / 4.8s），起点为
  刚从坐姿抬起进入加速的瞬间，终点为减速到 ~0.1 m/s 但机身仍保持抬高
  （`base_z≈0.31 m`）、尚未塌回坐姿之前。
- 该裁切范围仍不包含严格的“速度=0 完全站定”边界，只是姿态干净的低速状态。
- **决策：暂不裁切这条数据。** 用户要求继续寻找姿态干净、真正以站立开始/结束的
  candidate motion，再决定是否需要退回到裁切方案。

### 已保存的中间产物（未提交 git，`datasets/` 整体在 `.gitignore` 中）

```text
datasets/kine2go/ai4_dog_run_04/raw/motion.npy           原始1263帧数据
datasets/kine2go/ai4_dog_run_04/raw/config.json
datasets/kine2go/ai4_dog_run_04/raw/clip.json
datasets/kine2go/ai4_dog_run_04/raw/reference_rollout.mp4  官方渲染片段(仅覆盖训练用裁剪窗口)
datasets/kine2go/ai4_dog_run_04/ai4_dog_run_04_full_60hz.csv  完整19列x1263帧, 未裁切
```

### 下一步

继续在 Kine2Go 之外/AI4Animation 原始数据集中寻找“真正以站立姿态开始和结束”的
干净候选（而非坐姿）。若最终找不到，再回头执行上面的裁切方案作为退让选项。

## 2026-08-12 最终决定：`ai4_dog_run_04_clean_464_734`

### 追加发现：官方标注窗口起点速度仍偏高

在 nomagiclab/kine2go-pipeline 仓库中找到原始动捕的标注文件
`motion_retargeting/data/AI4Animation/dog_clips_info.txt`，确认 `dog_run04` 的
官方裁剪窗口为 `trim_start=8.15s(frame 487), trim_end=11.95s(frame 716)`，与我们
独立复核的 `[442:730]` 基本吻合，交叉验证了坐姿污染的结论。

但官方窗口起点速度约 `1.54 m/s`，仍高于当前 Velocity 部署策略的 `≤1 m/s` 上限，
与 M1（状态分布不重叠）是同一类问题。进一步查看边界外的渐变区后，扩展到
`[464:734]`：

```text
             官方窗口[487:716]      扩展窗口[464:734]
起点速度      1.54 m/s              0.96 m/s
起点高度      0.339 m               0.224 m（渐变区，非坐姿）
终点速度      0.32 m/s              0.03 m/s
终点高度      0.321 m               0.300 m（未跌入坐姿 0.147 m）
```

用户在 MuJoCo/Isaac Lab 实际回放确认：明显优于旧版本，机身高度仍有轻微瑕疵但
可接受。**已采纳为最终训练 Motion。**

### 已执行的变更

- 生成 `datasets/kine2go/ai4_dog_run_04/ai4_dog_run_04_clean_464_734_60hz.csv`
  （270帧, 60Hz）及对应 `ai4_dog_run_04_clean_464_734_50hz.npz`
  （225帧, 50Hz, 4.5s）。
- 替换 `source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/go2/canter/tracking_env_cfg.py`
  中的 `motion_file`，由 `ai4_dog_canter_cycle_483_510_repeat10_50hz.npz` 改为
  `ai4_dog_run_04_clean_464_734_50hz.npz`；`episode_length_s=4.5` 与新motion时长
  刚好匹配，未改动。

### 待办

用新 Motion 重新训练，训练完成后重复 sim2sim S6/S7（MuJoCo闭环 + 与Isaac Lab对比），
重点验证进入/退出 Mimic 时跟 Velocity 策略的过渡是否比旧版本平滑。
