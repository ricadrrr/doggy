import gymnasium as gym

gym.register(
    id="Unitree-Go2-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# A/B: official curriculum/rewards, only raise lin_vel_x ceiling to 3.0
gym.register(
    id="Unitree-Go2-Velocity-Official-3p0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfgOfficial3p0",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfgOfficial3p0",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# Active staged path: ns15 → 2p6-normsoft → 2p6-track → 2p8-track → 2p9-track → (3p0-* optional)
_STAGED_V25_RL = "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg10k"
for _suffix, _cfg, _play in (
    ("2p2-ns15", "RobotEnvCfgStagedV25_2p2_NormSoftFrom15", "RobotPlayEnvCfgStagedV25_2p2_NormSoftFrom15"),
    ("2p6-normsoft", "RobotEnvCfgStagedV25_2p6_NormSoft", "RobotPlayEnvCfgStagedV25_2p6_NormSoft"),
    ("2p6-track", "RobotEnvCfgStagedV25_2p6_Track", "RobotPlayEnvCfgStagedV25_2p6_Track"),
    ("2p8-track", "RobotEnvCfgStagedV25_2p8_Track", "RobotPlayEnvCfgStagedV25_2p8_Track"),
    ("2p9-track", "RobotEnvCfgStagedV25_2p9_Track", "RobotPlayEnvCfgStagedV25_2p9_Track"),
    ("3p0-normsoft", "RobotEnvCfgStagedV25_3p0_NormSoft", "RobotPlayEnvCfgStagedV25_3p0_NormSoft"),
    ("3p0-track", "RobotEnvCfgStagedV25_3p0_Track", "RobotPlayEnvCfgStagedV25_3p0_Track"),
    ("3p0-gate", "RobotEnvCfgStagedV25_3p0_Gate", "RobotPlayEnvCfgStagedV25_3p0_Gate"),
    ("3p0-gatew", "RobotEnvCfgStagedV25_3p0_GateW", "RobotPlayEnvCfgStagedV25_3p0_GateW"),
    ("3p0-gatewb", "RobotEnvCfgStagedV25_3p0_GateWB", "RobotPlayEnvCfgStagedV25_3p0_GateWB"),
):
    gym.register(
        id=f"Unitree-Go2-Velocity-Staged-{_suffix}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_staged_v25:{_cfg}",
            "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_staged_v25:{_play}",
            "rsl_rl_cfg_entry_point": _STAGED_V25_RL,
        },
    )
