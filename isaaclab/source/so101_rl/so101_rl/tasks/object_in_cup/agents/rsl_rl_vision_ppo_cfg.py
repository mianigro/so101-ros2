"""Named PPO runner presets for the object-in-cup scenario."""

from isaaclab.utils.configclass import configclass
from isaaclab_tasks.utils import PresetCfg

from models.mamba_ppo.ppo_cfg import RslRlMambaActorCfg
from models.ppo.ppo_cfg import TensorBroadcastPPOCfg
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
    # KL-adaptive LR from a higher starting point than the shared visual
    # default: the deeper temporal trunk needs the KL feedback to stay stable,
    # while the CNN baseline keeps the shared fixed 7e-5 for comparability.
    algorithm = TensorBroadcastPPOCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=2.0e-4,
        schedule="adaptive",
        gamma=0.9933,
        lam=0.9664,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    # 96-step rollouts track gamma's ~150-step effective horizon; 48-step
    # rollouts truncate value bootstrapping mid-episode.
    num_steps_per_env = 96
    actor = RslRlTransformerActorCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlTransformerActorCfg.GaussianDistributionCfg(
            init_std=0.5
        ),
        lookback_frames=SO101_TEMPORAL_LOOKBACK_FRAMES,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        dropout=0.0,
        # Last-frame readout: the causally-masked newest token already
        # aggregates the whole window, so pooling over all four tokens only
        # dilutes it with less-informed ones and adds a noisy learned softmax.
        final_layer_pooling=False,
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
    # Longer rollouts keep recurrent BPTT chunks at 12 steps (96 / 8
    # minibatches) so temporal credit spans more than a handful of frames.
    num_steps_per_env = 96
    # Same KL-adaptive rationale as the transformer preset.
    algorithm = TensorBroadcastPPOCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.9933,
        lam=0.9664,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
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
