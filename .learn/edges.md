# 依赖边 — 当前大目标：脚部三件套 / air_time_variance

| 从 | 到 | 关系 | 状态 |
|----|----|------|------|
| ContactSensor `track_air_time` | `last_air_time` / `last_contact_time` | 传感器记录每脚上次腾空/着地时长 | 未验证 |
| `last_*` 四脚时长 | `torch.var`（+clip） | 罚四脚之间的差异，不是总腾空量 | 未验证 |
| `air_time_variance_penalty` 返回值 | `RewTerm weight=-1` | 方差≥0，乘负权重后进总分/TB | 未验证 |
| `feet_air_time`（正） | `air_time_variance`（负） | 推抬脚 vs 管齐步 | 未验证 |
| 步态不齐 / 跛行 | `air_time_variance` 更负、常连带 `feet_slide`/`error`/`track` | 工程观感与 TB 联动 | 未验证 |

状态：`未验证` | `已验证`
