"""SO-101 pick-and-place task registration."""

import gymnasium as gym

gym.register(
    id="SO101-Object-In-Cup-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.object_in_cup_env_cfg:SO101ObjectInCupEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:SO101ObjectInCupPPORunnerCfg",
    },
)

gym.register(
    id="SO101-Object-In-Cup-Vision-Fixed-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.vision_env_cfg:SO101ObjectInCupVisionFixedEnvCfg"
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
        "env_cfg_entry_point": f"{__name__}.vision_env_cfg:SO101ObjectInCupVisionEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.agents.rsl_rl_vision_ppo_cfg:"
            "SO101ObjectInCupVisionPPORunnerCfg"
        ),
    },
)
