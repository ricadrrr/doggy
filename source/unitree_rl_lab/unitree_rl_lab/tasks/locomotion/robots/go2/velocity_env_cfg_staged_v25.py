# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Go2 staged velocity: cmd-norm + soft-penalty path (kept tasks only).

Active lineage (resume previous stage ckpt into the next task):
  2p2-ns15 → 2p6-normsoft → 2p6-track → 3p0-normsoft → 3p0-track
  preferred to 3.0: 2p6-track ckpt → 3p0-gate (expand ranges to 3.0 with high-band gate)
  scheme-3: gate mid ckpt → 3p0-gatew (slower gate + |cmd_x|-weighted track)
  scheme-C: gatew ckpt → 3p0-gatewb (bin gate + discrete bin track weights + bin TB)

Official velocity_env_cfg.py base task untouched.
"""

import math

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


@configclass
class Gate3p0CurriculumCfg(CurriculumCfg):
    """Expand forward max 2.6→3.0 with high-band gate + shrink; no soft ramp."""

    soft_penalty_schedule = None
    lin_vel_cmd_levels = CurrTerm(
        func=mdp.lin_vel_cmd_levels_gated,
        params={
            "reward_term_name": "track_lin_vel_xy",
            "expand_ratio": 0.85,
            "shrink_ratio": 0.65,
            "delta": 0.1,
            "min_forward": 2.6,
            "high_band": 0.3,
        },
    )


def _init_gate_velocity_x(cfg: RobotEnvCfg, limit_x: tuple[float, float], start_x: tuple[float, float]):
    """Ceiling at limit; sample from start ranges (curriculum grows start→limit)."""
    cfg.commands.base_velocity.limit_ranges.lin_vel_x = limit_x
    cfg.commands.base_velocity.ranges.lin_vel_x = start_x


@configclass
class RobotEnvCfgStagedV25_3p0_Gate(RobotEnvCfg):
    """One-run path: track-tune rewards, start at 2.6 sampling, gate-expand to 3.0."""

    curriculum: Gate3p0CurriculumCfg = Gate3p0CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        _init_track_tune_penalties(self)
        _init_gate_velocity_x(self, limit_x=(-1.5, 3.0), start_x=(-1.5, 2.6))


@configclass
class GateW3p0CurriculumCfg(CurriculumCfg):
    """Scheme-3 slow gate: stricter expand, smaller delta, 3 consecutive windows."""

    soft_penalty_schedule = None
    lin_vel_cmd_levels = CurrTerm(
        func=mdp.lin_vel_cmd_levels_gated,
        params={
            "reward_term_name": "track_lin_vel_xy",
            "expand_ratio": 0.90,
            "shrink_ratio": 0.70,
            "delta": 0.05,
            "min_forward": 2.6,
            "high_band": 0.3,
            "expand_needed": 3,
        },
    )


def _init_gatew_track_rewards(cfg: RobotEnvCfg):
    """Track-tune penalties + |cmd_x|-weighted linear tracking (boost toward 3.0)."""
    _init_track_tune_penalties(cfg)
    cfg.rewards.track_lin_vel_xy.func = mdp.track_lin_vel_xy_exp_speed_weighted
    cfg.rewards.track_lin_vel_xy.weight = 1.65
    cfg.rewards.track_lin_vel_xy.params = {
        "command_name": "base_velocity",
        "std": math.sqrt(0.25),
        "speed_ref": 1.5,
        "speed_hi": 3.0,
        "boost": 0.35,
    }


@configclass
class RobotEnvCfgStagedV25_3p0_GateW(RobotEnvCfg):
    """Scheme-3: slow high-band gate + high-|cmd_x| track weighting; sample 2.6→3.0."""

    curriculum: GateW3p0CurriculumCfg = GateW3p0CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        _init_gatew_track_rewards(self)
        _init_gate_velocity_x(self, limit_x=(-1.5, 3.0), start_x=(-1.5, 2.6))


def _bin_track_term(lo: float, hi: float) -> CurrTerm:
    return CurrTerm(
        func=mdp.log_lin_vel_x_bin_track_rate,
        params={"bin_lo": lo, "bin_hi": hi, "reward_term_name": "track_lin_vel_xy"},
    )


def _bin_err_term(lo: float, hi: float) -> CurrTerm:
    return CurrTerm(
        func=mdp.log_lin_vel_x_bin_error_xy,
        params={"bin_lo": lo, "bin_hi": hi, "command_name": "base_velocity"},
    )


@configclass
class GateWB3p0CurriculumCfg(CurriculumCfg):
    """Scheme-C: high+worst bin gate, per-bin TB track/error curves."""

    soft_penalty_schedule = None
    lin_vel_cmd_levels = CurrTerm(
        func=mdp.lin_vel_cmd_levels_gated_bins,
        params={
            "reward_term_name": "track_lin_vel_xy",
            "high_expand_ratio": 0.90,
            "worst_expand_ratio": 0.80,
            "shrink_ratio": 0.70,
            "delta": 0.05,
            "min_forward": 2.6,
            "bin_width": 0.5,
            "expand_needed": 3,
            "min_bin_count": 4,
        },
    )
    gated_bin_worst_track = CurrTerm(func=mdp.log_gated_bin_worst_track)
    # Fixed 0.5 m/s bins up to 3.0 for TensorBoard (Curriculum/*).
    bin_trk_0_0p5 = _bin_track_term(0.0, 0.5)
    bin_trk_0p5_1 = _bin_track_term(0.5, 1.0)
    bin_trk_1_1p5 = _bin_track_term(1.0, 1.5)
    bin_trk_1p5_2 = _bin_track_term(1.5, 2.0)
    bin_trk_2_2p5 = _bin_track_term(2.0, 2.5)
    bin_trk_2p5_3 = _bin_track_term(2.5, 3.0)
    bin_err_0_0p5 = _bin_err_term(0.0, 0.5)
    bin_err_0p5_1 = _bin_err_term(0.5, 1.0)
    bin_err_1_1p5 = _bin_err_term(1.0, 1.5)
    bin_err_1p5_2 = _bin_err_term(1.5, 2.0)
    bin_err_2_2p5 = _bin_err_term(2.0, 2.5)
    bin_err_2p5_3 = _bin_err_term(2.5, 3.0)


def _init_gatewb_track_rewards(cfg: RobotEnvCfg):
    """Track-tune penalties + discrete |cmd_x| bin track weights (replaces continuous)."""
    _init_track_tune_penalties(cfg)
    cfg.rewards.track_lin_vel_xy.func = mdp.track_lin_vel_xy_exp_bin_weighted
    cfg.rewards.track_lin_vel_xy.weight = 1.65
    cfg.rewards.track_lin_vel_xy.params = {
        "command_name": "base_velocity",
        "std": math.sqrt(0.25),
        "bin_edges": [0.0, 1.0, 2.0, 2.5, 10.0],
        "bin_weights": [1.0, 1.15, 1.25, 1.35],
    }


@configclass
class RobotEnvCfgStagedV25_3p0_GateWB(RobotEnvCfg):
    """Scheme-C: bin dual-gate + discrete bin track weights + bin TB; sample 2.6→3.0."""

    curriculum: GateWB3p0CurriculumCfg = GateWB3p0CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        _init_gatewb_track_rewards(self)
        _init_gate_velocity_x(self, limit_x=(-1.5, 3.0), start_x=(-1.5, 2.6))


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


@configclass
class RobotPlayEnvCfgStagedV25_3p0_Gate(RobotEnvCfgStagedV25_3p0_Gate):
    def __post_init__(self):
        super().__post_init__()
        _play_post_init(self)
        # play: exercise full ceiling
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotPlayEnvCfgStagedV25_3p0_GateW(RobotEnvCfgStagedV25_3p0_GateW):
    def __post_init__(self):
        super().__post_init__()
        _play_post_init(self)
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotPlayEnvCfgStagedV25_3p0_GateWB(RobotEnvCfgStagedV25_3p0_GateWB):
    def __post_init__(self):
        super().__post_init__()
        _play_post_init(self)
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
