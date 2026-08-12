"""Asymmetric visual PPO configuration for the deployable SO-101 actor."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
)
from isaaclab_tasks.utils import PresetCfg

from .distributed_ppo import TensorBroadcastPPOCfg


@configclass
class RslRlSpatialSoftmaxCNNModelCfg(RslRlCNNModelCfg):
    class_name = "so101_rl.tasks.object_in_cup.agents.models:SpatialSoftmaxCNNModel"
    init_temperature: float = 1.0


_ACTOR = RslRlSpatialSoftmaxCNNModelCfg(
    hidden_dims=[512, 256, 128],
    activation="elu",
    obs_normalization=True,
    distribution_cfg=RslRlSpatialSoftmaxCNNModelCfg.GaussianDistributionCfg(
        init_std=0.7
    ),
    cnn_cfg=RslRlSpatialSoftmaxCNNModelCfg.CNNCfg(
        output_channels=[16, 32, 32],
        kernel_size=[8, 4, 3],
        stride=[4, 2, 1],
        activation="elu",
    ),
)

_CRITIC = RslRlMLPModelCfg(
    hidden_dims=[256, 256, 128],
    activation="elu",
    obs_normalization=True,
)

_ALGORITHM = TensorBroadcastPPOCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.005,
    num_learning_epochs=5,
    num_mini_batches=8,
    learning_rate=7.0e-5,
    schedule="fixed",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
)


@configclass
class SO101ObjectInCupVisionPPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 15_000
    save_interval = 250
    experiment_name = "so101_object_in_cup_vision"
    obs_groups = {
        "actor": ["joint_state", "wrist", "overhead_1", "overhead_2"],
        "critic": ["critic"],
    }
    clip_actions = 1.0
    actor = _ACTOR
    critic = _CRITIC
    algorithm = _ALGORITHM


@configclass
class SO101ObjectInCupVisionPPORunnerCfg(PresetCfg):
    default = SO101ObjectInCupVisionPPOCfg()


@configclass
class SO101ObjectInCupVisionFixedPPORunnerCfg(PresetCfg):
    default = SO101ObjectInCupVisionPPOCfg().replace(
        experiment_name="so101_object_in_cup_vision_fixed"
    )
