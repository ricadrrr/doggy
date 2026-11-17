from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def track_lin_vel_xy_raw_meter(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Accumulate unweighted ``track_lin_vel_xy_exp`` for TB; return 0 (no learning effect).

    Pair with curriculum ``log_track_lin_vel_xy_raw``. Use a non-zero RewTerm weight so
    Isaac Lab does not skip this term; returned zeros keep the total reward unchanged.
    """
    from isaaclab.envs.mdp.rewards import track_lin_vel_xy_exp

    raw = track_lin_vel_xy_exp(env, std=std, command_name=command_name, asset_cfg=asset_cfg)
    if not hasattr(env, "_raw_track_xy_ep_sum"):
        env._raw_track_xy_ep_sum = torch.zeros(env.num_envs, device=env.device)
    env._raw_track_xy_ep_sum += raw * env.step_dt
    return torch.zeros_like(raw)


def track_lin_vel_xy_exp_speed_weighted(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    speed_ref: float = 1.5,
    speed_hi: float = 3.0,
    boost: float = 0.35,
) -> torch.Tensor:
    """Exponential xy tracking, scaled up toward higher |cmd_x| (scheme-3 high-speed weight).

    scale = 1 + boost * clip((|vx|-speed_ref)/(speed_hi-speed_ref), 0, 1).
    Low-speed tracking stays ~unchanged; near speed_hi the same tracking error pays more.
    """
    from isaaclab.envs.mdp.rewards import track_lin_vel_xy_exp

    base = track_lin_vel_xy_exp(env, std=std, command_name=command_name, asset_cfg=asset_cfg)
    cmd_x = torch.abs(env.command_manager.get_command(command_name)[:, 0])
    span = max(float(speed_hi) - float(speed_ref), 1e-6)
    t = torch.clamp((cmd_x - float(speed_ref)) / span, 0.0, 1.0)
    return base * (1.0 + float(boost) * t)


def track_lin_vel_xy_exp_bin_weighted(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    bin_edges: list[float] | None = None,
    bin_weights: list[float] | None = None,
) -> torch.Tensor:
    """Exponential xy tracking × discrete |cmd_x| bin weights (scheme-C).

    Default bins (m/s): [0,1)→1.0, [1,2)→1.15, [2,2.5)→1.25, [2.5,+)→1.35.
    Negative cmd_x uses weight 1.0. ``bin_edges`` length must be len(weights)+1.
    """
    from isaaclab.envs.mdp.rewards import track_lin_vel_xy_exp

    if bin_edges is None:
        bin_edges = [0.0, 1.0, 2.0, 2.5, 10.0]
    if bin_weights is None:
        bin_weights = [1.0, 1.15, 1.25, 1.35]
    if len(bin_edges) != len(bin_weights) + 1:
        raise ValueError("bin_edges must have length len(bin_weights)+1")

    base = track_lin_vel_xy_exp(env, std=std, command_name=command_name, asset_cfg=asset_cfg)
    cmd_x = torch.abs(env.command_manager.get_command(command_name)[:, 0])
    scale = torch.ones_like(cmd_x)
    for i, w in enumerate(bin_weights):
        lo = float(bin_edges[i])
        hi = float(bin_edges[i + 1])
        mask = (cmd_x >= lo) & (cmd_x < hi)
        scale = torch.where(mask, torch.full_like(scale, float(w)), scale)
    # |vx| >= last edge: use last weight
    scale = torch.where(cmd_x >= float(bin_edges[-1]), torch.full_like(scale, float(bin_weights[-1])), scale)
    return base * scale


def action_rate_l2_cmd_normalized(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity"
) -> torch.Tensor:
    """Action-rate L2 scaled by 1/(1+||cmd||) so faster commands are not auto-heavier."""
    raw = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    return raw / (1.0 + cmd)


def energy_cmd_normalized(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint energy scaled by 1/(1+||cmd||) for fairer penalties across speed."""
    return energy(env, asset_cfg) / (1.0 + torch.linalg.norm(env.command_manager.get_command(command_name), dim=1))


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize left-right feet closer than ``threshold`` (stance collapse).

    - 2 bodies: compare that pair.
    - 4 bodies (``.*_foot``): pair FL-FR and RL-RR by name; falls back to
      index pairs (0,1)/(2,3) if names are unavailable.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    n_feet = feet_pos.shape[1]
    if n_feet == 2:
        distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
        return (threshold - distance).clamp(min=0)

    # Resolve left-right pairs from body names when possible.
    body_ids = list(asset_cfg.body_ids)
    names = [asset.body_names[int(i)] for i in body_ids]
    name_to_local = {n: i for i, n in enumerate(names)}

    def _pair_idx(left_key: str, right_key: str) -> tuple[int, int] | None:
        left = next((name_to_local[n] for n in names if left_key in n), None)
        right = next((name_to_local[n] for n in names if right_key in n), None)
        if left is None or right is None:
            return None
        return left, right

    pairs = [_pair_idx("FL", "FR"), _pair_idx("RL", "RR")]
    pairs = [p for p in pairs if p is not None]
    if len(pairs) == 0 and n_feet >= 4:
        pairs = [(0, 1), (2, 3)]
    if len(pairs) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    penalty = torch.zeros(env.num_envs, device=env.device)
    for i, j in pairs:
        distance = torch.norm(feet_pos[:, i] - feet_pos[:, j], dim=-1)
        penalty += (threshold - distance).clamp(min=0)
    return penalty


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward
