# TensorBoard 标量清单

勾选 = 已讲完且确认明白（对照代码 6 点见 `.learn/CONTRACT.md`）。

## A. 页面与用法

- [x] 01 TB 整体 + URL / timeseries
- [x] 02 左侧 Runs、对比实验
- [x] 03 Smooth / Step / Value / Time / Relative

## B. Curriculum

- [x] Curriculum/lin_vel_cmd_levels（官方路径；staged 任务里此项为关闭，改用 soft_penalty_schedule）
- [x] Curriculum/terrain_levels

## C. Metrics 跟踪误差

- [ ] Metrics/base_velocity/error_vel_xy（讲过，待再确认）
- [ ] Metrics/base_velocity/error_vel_yaw

## D. Train

- [ ] Train/mean_reward
- [ ] Train/mean_episode_length
- [ ] Train/mean_reward/time
- [ ] Train/mean_episode_length/time

## E. Episode_Reward

- [ ] track_lin_vel_xy
- [ ] track_ang_vel_z
- [ ] flat_orientation_l2
- [ ] action_rate（半懂；含 cmd 归一化）
- [ ] base_linear_velocity
- [ ] base_angular_velocity
- [ ] joint_vel / joint_acc / joint_torques / joint_pos
- [ ] dof_pos_limits / energy（energy 亦有 cmd 归一化）
- [x] air_time_variance（6 点已打通）
- [ ] feet_air_time
- [ ] feet_slide（进行中）
- [ ] undesired_contacts

## F. Episode_Termination

- [ ] time_out
- [ ] base_contact
- [ ] bad_orientation

## G. Loss / Policy

- [ ] Loss/surrogate
- [ ] Loss/value_function
- [ ] Loss/entropy
- [ ] Loss/learning_rate
- [ ] Policy/mean_noise_std

## H. Perf

- [ ] Perf/collection time
- [ ] Perf/learning_time
- [ ] Perf/total_fps
