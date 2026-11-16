from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def lin_vel_cmd_levels_gated(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    expand_ratio: float = 0.85,
    shrink_ratio: float = 0.65,
    delta: float = 0.1,
    min_forward: float = 2.6,
    high_band: float = 0.3,
    expand_needed: int = 1,
) -> torch.Tensor:
    """Expand/shrink forward lin_vel_x max with a high-band tracking gate.

    Unlike ``lin_vel_cmd_levels``, this:
    - only moves the forward ceiling ``ranges.lin_vel_x[1]`` toward ``limit_ranges``
    - expands when high-band (near current max) tracking is strong enough; falls back
      to overall tracking if too few high-band envs
    - shrinks toward ``min_forward`` when tracking is weak (official has no shrink)
    - ``expand_needed`` consecutive good windows required before each +delta (slow gate)

    Sampling stays uniform inside current ``ranges``. Returns current forward max for TB.
    """
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    weight = float(reward_term.weight)
    episode_rate = env.reward_manager._episode_sums[reward_term_name][env_ids] / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0 and len(env_ids) > 0:
        cmd_x = command_term.command[env_ids, 0]
        hi = float(ranges.lin_vel_x[1])
        high_mask = cmd_x > (hi - high_band)
        if torch.count_nonzero(high_mask) >= max(1, len(env_ids) // 10):
            gate_reward = torch.mean(episode_rate[high_mask])
        else:
            gate_reward = torch.mean(episode_rate)

        lo_x = float(ranges.lin_vel_x[0])
        cur_hi = float(ranges.lin_vel_x[1])
        lim_lo = float(limit_ranges.lin_vel_x[0])
        lim_hi = float(limit_ranges.lin_vel_x[1])
        floor_hi = max(float(min_forward), lim_lo)
        need = max(1, int(expand_needed))
        streak = int(getattr(env, "_gated_expand_streak", 0))

        if gate_reward > weight * expand_ratio:
            streak += 1
            if streak >= need:
                cur_hi = min(cur_hi + delta, lim_hi)
                streak = 0
        elif gate_reward < weight * shrink_ratio:
            cur_hi = max(cur_hi - delta, floor_hi)
            streak = 0
        else:
            streak = 0

        env._gated_expand_streak = streak
        ranges.lin_vel_x = [lo_x, cur_hi]

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def lin_vel_cmd_levels_gated_bins(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    high_expand_ratio: float = 0.90,
    worst_expand_ratio: float = 0.80,
    shrink_ratio: float = 0.70,
    delta: float = 0.05,
    min_forward: float = 2.6,
    bin_width: float = 0.5,
    expand_needed: int = 3,
    min_bin_count: int = 4,
) -> torch.Tensor:
    """Slow gate: expand only if high-bin AND worst-bin tracking both pass.

    Bins are width ``bin_width`` on cmd_x in [0, current_hi]. High bin is the top
    slice near the ceiling. Worst bin is the minimum mean track rate among bins
    with enough samples. Sampling stays uniform in current ranges.
    """
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    weight = float(reward_term.weight)
    episode_rate = env.reward_manager._episode_sums[reward_term_name][env_ids] / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0 and len(env_ids) > 0:
        cmd_x = command_term.command[env_ids, 0]
        lo_x = float(ranges.lin_vel_x[0])
        cur_hi = float(ranges.lin_vel_x[1])
        lim_lo = float(limit_ranges.lin_vel_x[0])
        lim_hi = float(limit_ranges.lin_vel_x[1])
        floor_hi = max(float(min_forward), lim_lo)
        width = max(float(bin_width), 1e-3)

        high_mask = cmd_x > (cur_hi - width)
        if torch.count_nonzero(high_mask) >= max(1, min_bin_count):
            high_rate = torch.mean(episode_rate[high_mask])
        else:
            high_rate = torch.mean(episode_rate)

        # forward bins only (cmd_x >= 0); ignore sparse bins
        worst_rate = high_rate
        bin_rates: list[torch.Tensor] = []
        start = 0.0
        while start < cur_hi - 1e-6:
            end = min(start + width, cur_hi)
            if end >= cur_hi - 1e-6:
                mask = (cmd_x >= start) & (cmd_x <= cur_hi + 1e-6)
            else:
                mask = (cmd_x >= start) & (cmd_x < end)
            if int(torch.count_nonzero(mask)) >= max(1, min_bin_count):
                bin_rates.append(torch.mean(episode_rate[mask]))
            start = end

        if bin_rates:
            worst_rate = torch.stack(bin_rates).min()

        need = max(1, int(expand_needed))
        streak = int(getattr(env, "_gated_bin_expand_streak", 0))

        high_ok = high_rate > weight * high_expand_ratio
        worst_ok = worst_rate > weight * worst_expand_ratio
        high_bad = high_rate < weight * shrink_ratio
        worst_bad = worst_rate < weight * shrink_ratio

        if high_ok and worst_ok:
            streak += 1
            if streak >= need:
                cur_hi = min(cur_hi + delta, lim_hi)
                streak = 0
        elif high_bad or worst_bad:
            cur_hi = max(cur_hi - delta, floor_hi)
            streak = 0
        else:
            streak = 0

        env._gated_bin_expand_streak = streak
        # stash for optional TB helpers
        env._gated_bin_high_rate = float(high_rate.detach())
        env._gated_bin_worst_rate = float(worst_rate.detach())
        ranges.lin_vel_x = [lo_x, cur_hi]

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def log_lin_vel_x_bin_track_rate(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    bin_lo: float,
    bin_hi: float,
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    """TB helper: mean episode track rate for cmd_x in [bin_lo, bin_hi). 0 if empty."""
    if len(env_ids) == 0:
        return torch.tensor(0.0, device=env.device)
    command_term = env.command_manager.get_term("base_velocity")
    cmd_x = command_term.command[env_ids, 0]
    rate = env.reward_manager._episode_sums[reward_term_name][env_ids] / env.max_episode_length_s
    mask = (cmd_x >= float(bin_lo)) & (cmd_x < float(bin_hi))
    if torch.count_nonzero(mask) == 0:
        return torch.tensor(0.0, device=env.device)
    return torch.mean(rate[mask])


def log_lin_vel_x_bin_error_xy(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    bin_lo: float,
    bin_hi: float,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """TB helper: mean command-metric error_vel_xy for cmd_x in [bin_lo, bin_hi). 0 if empty."""
    if len(env_ids) == 0:
        return torch.tensor(0.0, device=env.device)
    command_term = env.command_manager.get_term(command_name)
    cmd_x = command_term.command[env_ids, 0]
    err = command_term.metrics["error_vel_xy"][env_ids]
    mask = (cmd_x >= float(bin_lo)) & (cmd_x < float(bin_hi))
    if torch.count_nonzero(mask) == 0:
        return torch.tensor(0.0, device=env.device)
    return torch.mean(err[mask])


def log_gated_bin_worst_track(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
) -> torch.Tensor:
    """TB: last computed worst-bin track rate from gated_bins curriculum (else 0)."""
    return torch.tensor(float(getattr(env, "_gated_bin_worst_rate", 0.0)), device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def soft_penalty_schedule(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    ramp_steps: int = 60000,
    action_rate: tuple[float, float] = (-0.03, -0.08),
    flat_orientation_l2: tuple[float, float] = (-1.5, -2.5),
    joint_torques: tuple[float, float] = (-5.0e-5, -1.5e-4),
    feet_slide: tuple[float, float] = (-0.03, -0.08),
    undesired_contacts: tuple[float, float] = (-0.3, -0.8),
    energy: tuple[float, float] | None = None,
    joint_acc: tuple[float, float] | None = None,
    track_lin_vel_xy: float = 1.5,
    track_ang_vel_z: float = 0.75,
) -> torch.Tensor:
    """Ramp soft-penalty weights; keep tracking rewards fixed.

    Progress uses common_step_counter / ramp_steps (≈24 steps/iter → 60000 ≈ 2500 iters).
    Optional energy / joint_acc pairs are applied only when provided (not None).
    Returns progress in [0, 1] for TensorBoard (Curriculum/soft_penalty_schedule).
    """
    progress = min(1.0, float(env.common_step_counter) / float(max(ramp_steps, 1)))

    def _lerp(pair: tuple[float, float]) -> float:
        return pair[0] + (pair[1] - pair[0]) * progress

    # Soft penalties: ramp listed terms (skip optional ones left as None).
    terms: list[tuple[str, tuple[float, float] | None]] = [
        ("action_rate", action_rate),
        ("flat_orientation_l2", flat_orientation_l2),
        ("joint_torques", joint_torques),
        ("feet_slide", feet_slide),
        ("undesired_contacts", undesired_contacts),
        ("energy", energy),
        ("joint_acc", joint_acc),
    ]
    for name, pair in terms:
        if pair is None:
            continue
        term_cfg = env.reward_manager.get_term_cfg(name)
        term_cfg.weight = _lerp(pair)

    # Task rewards: never shrink during the stage.
    env.reward_manager.get_term_cfg("track_lin_vel_xy").weight = track_lin_vel_xy
    env.reward_manager.get_term_cfg("track_ang_vel_z").weight = track_ang_vel_z

    return torch.tensor(progress, device=env.device)
