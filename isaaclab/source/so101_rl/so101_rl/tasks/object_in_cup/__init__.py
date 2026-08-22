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
    },
)
