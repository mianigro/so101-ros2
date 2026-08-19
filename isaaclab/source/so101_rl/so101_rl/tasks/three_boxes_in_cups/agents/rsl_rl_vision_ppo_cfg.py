"""Named PPO runner presets for the three-box/three-cup scenario."""

from isaaclab.utils.configclass import configclass
from isaaclab_tasks.utils import PresetCfg

from so101_rl.tasks.common.agents.rsl_rl_ppo_cfg import SO101VisualPPOCfg


@configclass
class SO101ThreeBoxesInCupsVisionPPOCfg(SO101VisualPPOCfg):
    experiment_name = "so101_three_boxes_in_cups_vision"


@configclass
class SO101ThreeBoxesInCupsVisionPPORunnerCfg(PresetCfg):
    default = SO101ThreeBoxesInCupsVisionPPOCfg()


@configclass
class SO101ThreeBoxesInCupsVisionFixedPPORunnerCfg(PresetCfg):
    default = SO101ThreeBoxesInCupsVisionPPOCfg().replace(
        experiment_name="so101_three_boxes_in_cups_vision_fixed"
    )
