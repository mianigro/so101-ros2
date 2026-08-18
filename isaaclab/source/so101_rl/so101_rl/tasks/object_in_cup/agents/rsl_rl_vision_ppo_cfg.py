"""Named PPO runner presets for the object-in-cup scenario."""

from isaaclab.utils.configclass import configclass
from isaaclab_tasks.utils import PresetCfg

from models.mamba_ppo.ppo_cfg import RslRlMambaActorCfg
from models.transformer_ppo.ppo_cfg import RslRlTransformerActorCfg
from so101_rl.tasks.common.agents.rsl_rl_ppo_cfg import SO101VisualPPOCfg
from so101_rl.visual_contract import SO101_TEMPORAL_LOOKBACK_FRAMES


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


@configclass
class SO101ObjectInCupVisionTransformerPPOCfg(SO101ObjectInCupVisionPPOCfg):
    """Temporal transformer actor over the SO-101 camera history window."""

    experiment_name = "so101_object_in_cup_vision_transformer"
    actor = RslRlTransformerActorCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlTransformerActorCfg.GaussianDistributionCfg(
            init_std=0.7
        ),
        lookback_frames=SO101_TEMPORAL_LOOKBACK_FRAMES,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        dropout=0.0,
        frame_diff=True,
        causal_mask=True,
    )


@configclass
class SO101ObjectInCupVisionTransformerPPORunnerCfg(PresetCfg):
    default = SO101ObjectInCupVisionTransformerPPOCfg()


@configclass
class SO101ObjectInCupVisionTransformerFixedPPORunnerCfg(PresetCfg):
    default = SO101ObjectInCupVisionTransformerPPOCfg().replace(
        experiment_name="so101_object_in_cup_vision_transformer_fixed"
    )


@configclass
class SO101ObjectInCupVisionMambaPPOCfg(SO101ObjectInCupVisionPPOCfg):
    """Recurrent state-space actor over the SO-101 camera frame streams.

    Consumes single frames; temporal memory lives in the actor's recurrent
    state, so the environment needs no camera history.
    """

    experiment_name = "so101_object_in_cup_vision_mamba"
    actor = RslRlMambaActorCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMambaActorCfg.GaussianDistributionCfg(init_std=0.7),
        d_model=64,
        num_layers=2,
    )


@configclass
class SO101ObjectInCupVisionMambaPPORunnerCfg(PresetCfg):
    default = SO101ObjectInCupVisionMambaPPOCfg()


@configclass
class SO101ObjectInCupVisionMambaFixedPPORunnerCfg(PresetCfg):
    default = SO101ObjectInCupVisionMambaPPOCfg().replace(
        experiment_name="so101_object_in_cup_vision_mamba_fixed"
    )
