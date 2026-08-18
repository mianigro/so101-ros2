"""SO-101 visual pick-and-place task registrations."""

import gymnasium as gym

gym.register(
    id="SO101-Object-In-Cup-Vision-Fixed-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.object_in_cup_env_cfg:SO101ObjectInCupVisionFixedEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.agents.rsl_rl_vision_ppo_cfg:"
            "SO101ObjectInCupVisionFixedPPORunnerCfg"
        ),
    },
)

gym.register(
    id="SO101-Object-In-Cup-Vision-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.object_in_cup_env_cfg:SO101ObjectInCupVisionEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.agents.rsl_rl_vision_ppo_cfg:"
            "SO101ObjectInCupVisionPPORunnerCfg"
        ),
    },
)

gym.register(
    id="SO101-Object-In-Cup-Vision-Transformer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.object_in_cup_env_cfg:"
            "SO101ObjectInCupVisionTemporalEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.agents.rsl_rl_vision_ppo_cfg:"
            "SO101ObjectInCupVisionTransformerPPORunnerCfg"
        ),
    },
)

gym.register(
    id="SO101-Object-In-Cup-Vision-Transformer-Fixed-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.object_in_cup_env_cfg:"
            "SO101ObjectInCupVisionTemporalFixedEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.agents.rsl_rl_vision_ppo_cfg:"
            "SO101ObjectInCupVisionTransformerFixedPPORunnerCfg"
        ),
    },
)

gym.register(
    id="SO101-Object-In-Cup-Vision-Mamba-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.object_in_cup_env_cfg:SO101ObjectInCupVisionEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.agents.rsl_rl_vision_ppo_cfg:"
            "SO101ObjectInCupVisionMambaPPORunnerCfg"
        ),
    },
)

gym.register(
    id="SO101-Object-In-Cup-Vision-Mamba-Fixed-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.object_in_cup_env_cfg:SO101ObjectInCupVisionFixedEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.agents.rsl_rl_vision_ppo_cfg:"
            "SO101ObjectInCupVisionMambaFixedPPORunnerCfg"
        ),
    },
)
