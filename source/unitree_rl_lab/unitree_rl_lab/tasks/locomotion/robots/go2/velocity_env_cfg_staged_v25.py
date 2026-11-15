# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Go2 staged velocity: cmd-norm + soft-penalty path (kept tasks only).

Active lineage (resume previous stage ckpt into the next task):
  2p2-ns15 → 2p6-normsoft → 2p6-track → 3p0-normsoft → 3p0-track
A/B at 3.0: Official-3p0 (in velocity_env_cfg.py) vs staged 3p0 here.

Official velocity_env_cfg.py base task untouched.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg import CurriculumCfg, RobotEnvCfg


def _pin_velocity_x(cfg: RobotEnvCfg, lin_vel_x: tuple[float, float]):
    """Set command ceiling and sample from it immediately (no speed curriculum)."""
    cfg.commands.base_velocity.limit_ranges.lin_vel_x = lin_vel_x
    cfg.commands.base_velocity.ranges = cfg.commands.base_velocity.limit_ranges


def _init_cmd_norm_soft_penalties(cfg: RobotEnvCfg):
    """Cmd-normalized action_rate/energy + start weights for soft ramp."""
    cfg.rewards.action_rate.func = mdp.action_rate_l2_cmd_normalized
    cfg.rewards.action_rate.weight = -0.06
    cfg.rewards.action_rate.params = {"command_name": "base_velocity"}
    cfg.rewards.energy.func = mdp.energy_cmd_normalized
    cfg.rewards.energy.weight = -1.5e-5
    cfg.rewards.energy.params = {"command_name": "base_velocity"}
    cfg.rewards.joint_acc.weight = -2.5e-7
    cfg.rewards.flat_orientation_l2.weight = -2.0
    cfg.rewards.joint_torques.weight = -1.0e-4
    cfg.rewards.feet_slide.weight = -0.06
    cfg.rewards.undesired_contacts.weight = -0.6
    cfg.rewards.track_lin_vel_xy.weight = 1.5
    cfg.rewards.track_ang_vel_z.weight = 0.75


@configclass
class NormSoftFrom15CurriculumCfg(CurriculumCfg):
    """Cmd-norm soft ramp (~3000 iters) — used by ns15 and 2p6-normsoft."""

    lin_vel_cmd_levels = None
    soft_penalty_schedule = CurrTerm(
        func=mdp.soft_penalty_schedule,
        params={
            # ≈24 env-steps/iter → 72000 ≈ 3000 iters
            "ramp_steps": 72000,
            "action_rate": (-0.06, -0.10),
            "flat_orientation_l2": (-2.0, -2.5),
            "joint_torques": (-1.0e-4, -2.0e-4),
            "feet_slide": (-0.06, -0.10),
            "undesired_contacts": (-0.6, -1.0),
            "energy": (-1.5e-5, -3.0e-5),
            "joint_acc": (-2.5e-7, -4.0e-7),
            "track_lin_vel_xy": 1.5,
            "track_ang_vel_z": 0.75,
        },
    )


@configclass
class RobotEnvCfgStagedV25_2p2_NormSoftFrom15(RobotEnvCfg):
    """1.5 → 2.2 with cmd-norm + longer soft-penalty ramp."""

    curriculum: NormSoftFrom15CurriculumCfg = NormSoftFrom15CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        _init_cmd_norm_soft_penalties(self)
        _pin_velocity_x(self, (-1.5, 2.2))


@configclass
class RobotEnvCfgStagedV25_2p6_NormSoft(RobotEnvCfg):
    """2.2 → 2.6: same cmd-norm + soft ramp as ns15."""

    curriculum: NormSoftFrom15CurriculumCfg = NormSoftFrom15CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        _init_cmd_norm_soft_penalties(self)
        _pin_velocity_x(self, (-1.5, 2.6))


@configclass
class NormSoft3p0CurriculumCfg(CurriculumCfg):
    """Cmd-norm soft ramp ~5000 iters for 2.6 → 3.0."""

    lin_vel_cmd_levels = None
    soft_penalty_schedule = CurrTerm(
        func=mdp.soft_penalty_schedule,
        params={
            # ≈24 env-steps/iter → 120000 ≈ 5000 iters
            "ramp_steps": 120000,
            "action_rate": (-0.06, -0.10),
            "flat_orientation_l2": (-2.0, -2.5),
            "joint_torques": (-1.0e-4, -2.0e-4),
            "feet_slide": (-0.06, -0.10),
            "undesired_contacts": (-0.6, -1.0),
            "energy": (-1.5e-5, -3.0e-5),
            "joint_acc": (-2.5e-7, -4.0e-7),
            "track_lin_vel_xy": 1.5,
            "track_ang_vel_z": 0.75,
        },
    )


@configclass
class RobotEnvCfgStagedV25_3p0_NormSoft(RobotEnvCfg):
    """Continue norm+soft at forward max 3.0 (A/B vs Official-3p0)."""

    curriculum: NormSoft3p0CurriculumCfg = NormSoft3p0CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        _init_cmd_norm_soft_penalties(self)
        _pin_velocity_x(self, (-1.5, 3.0))


@configclass
class TrackTuneCurriculumCfg(CurriculumCfg):
    """Fixed mid soft penalties + higher track weights; no time ramp."""

    lin_vel_cmd_levels = None
    soft_penalty_schedule = None


def _init_track_tune_penalties(cfg: RobotEnvCfg):
    """Cmd-norm + mid soft weights; boost tracking (same recipe as 2.6-track)."""
    cfg.rewards.action_rate.func = mdp.action_rate_l2_cmd_normalized
    cfg.rewards.action_rate.weight = -0.08
    cfg.rewards.action_rate.params = {"command_name": "base_velocity"}
    cfg.rewards.energy.func = mdp.energy_cmd_normalized
    cfg.rewards.energy.weight = -2.0e-5
    cfg.rewards.energy.params = {"command_name": "base_velocity"}
    cfg.rewards.joint_acc.weight = -3.0e-7
    cfg.rewards.flat_orientation_l2.weight = -2.2
    cfg.rewards.joint_torques.weight = -1.5e-4
    cfg.rewards.feet_slide.weight = -0.08
    cfg.rewards.undesired_contacts.weight = -0.8
    cfg.rewards.track_lin_vel_xy.weight = 1.65
    cfg.rewards.track_ang_vel_z.weight = 0.80


@configclass
class RobotEnvCfgStagedV25_2p6_Track(RobotEnvCfg):
    """2.6 track tune: cmd-norm, fixed mid penalties, higher track weights."""

    curriculum: TrackTuneCurriculumCfg = TrackTuneCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        _init_track_tune_penalties(self)
        _pin_velocity_x(self, (-1.5, 2.6))


@configclass
class RobotEnvCfgStagedV25_3p0_Track(RobotEnvCfg):
    """3.0 track tune: same mid-penalty + track boost as 2.6-track; pin 3.0."""

    curriculum: TrackTuneCurriculumCfg = TrackTuneCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        _init_track_tune_penalties(self)
        _pin_velocity_x(self, (-1.5, 3.0))


def _play_post_init(cfg: RobotEnvCfg):
    cfg.scene.num_envs = 8
    cfg.scene.terrain.terrain_generator.num_rows = 2
    cfg.scene.terrain.terrain_generator.num_cols = 1


@configclass
class RobotPlayEnvCfgStagedV25_2p2_NormSoftFrom15(RobotEnvCfgStagedV25_2p2_NormSoftFrom15):
    def __post_init__(self):
        super().__post_init__()
        _play_post_init(self)


@configclass
class RobotPlayEnvCfgStagedV25_2p6_NormSoft(RobotEnvCfgStagedV25_2p6_NormSoft):
    def __post_init__(self):
        super().__post_init__()
        _play_post_init(self)


@configclass
class RobotPlayEnvCfgStagedV25_2p6_Track(RobotEnvCfgStagedV25_2p6_Track):
    def __post_init__(self):
        super().__post_init__()
        _play_post_init(self)


@configclass
class RobotPlayEnvCfgStagedV25_3p0_NormSoft(RobotEnvCfgStagedV25_3p0_NormSoft):
    def __post_init__(self):
        super().__post_init__()
        _play_post_init(self)


@configclass
class RobotPlayEnvCfgStagedV25_3p0_Track(RobotEnvCfgStagedV25_3p0_Track):
    def __post_init__(self):
        super().__post_init__()
        _play_post_init(self)
