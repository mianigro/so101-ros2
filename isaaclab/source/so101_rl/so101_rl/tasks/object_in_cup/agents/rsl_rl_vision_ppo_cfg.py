"""Named PPO runner presets for the object-in-cup scenario."""

from isaaclab.utils.configclass import configclass
from isaaclab_tasks.utils import PresetCfg

from so101_rl.tasks.common.agents.rsl_rl_ppo_cfg import SO101VisualPPOCfg


@configclass
class SO101ObjectInCupVisionPPOCfg(SO101VisualPPOCfg):
    experiment_name = "so101_object_in_cup_vision"


@configclass
class SO101ObjectInCupVisionPPORunnerCfg(PresetCfg):
    default = SO101ObjectInCupVisionPPOCfg()


@configclass
class SO101ObjectInCupVisionFixedPPORunnerCfg(PresetCfg):
    default = SO101ObjectInCupVisionPPOCfg().replace(
        experiment_name="so101_object_in_cup_vision_fixed"
    )
