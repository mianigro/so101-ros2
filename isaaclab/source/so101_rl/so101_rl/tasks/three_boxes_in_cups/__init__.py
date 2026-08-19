"""SO-101 three-box/three-cup visual task registrations."""

import gymnasium as gym

gym.register(
    id="SO101-Three-Boxes-In-Cups-Vision-Fixed-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.three_boxes_in_cups_env_cfg:"
            "SO101ThreeBoxesInCupsVisionFixedEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.agents.rsl_rl_vision_ppo_cfg:"
            "SO101ThreeBoxesInCupsVisionFixedPPORunnerCfg"
        ),
    },
)

gym.register(
    id="SO101-Three-Boxes-In-Cups-Vision-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.three_boxes_in_cups_env_cfg:"
            "SO101ThreeBoxesInCupsVisionEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.agents.rsl_rl_vision_ppo_cfg:"
            "SO101ThreeBoxesInCupsVisionPPORunnerCfg"
        ),
    },
)
