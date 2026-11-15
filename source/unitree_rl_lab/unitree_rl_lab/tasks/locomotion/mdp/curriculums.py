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
