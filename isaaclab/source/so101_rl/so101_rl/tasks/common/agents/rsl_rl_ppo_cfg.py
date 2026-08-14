"""Reusable asymmetric PPO defaults for SO-101 visual manipulation."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
)

from so101_rl.visual_contract import SO101_ACTOR_OBSERVATION_GROUPS

from .distributed_ppo import TensorBroadcastPPOCfg


@configclass
class RslRlSpatialSoftmaxCNNModelCfg(RslRlCNNModelCfg):
    class_name = "so101_rl.tasks.common.agents.models:SpatialSoftmaxCNNModel"
    init_temperature: float = 1.0


@configclass
class SO101VisualPPOCfg(RslRlOnPolicyRunnerCfg):
    """Shared network, observation, rollout, and PPO contract for visual tasks."""

    num_steps_per_env = 48
    max_iterations = 15_000
    save_interval = 250
    experiment_name = "so101_visual"
    obs_groups = {
        "actor": list(SO101_ACTOR_OBSERVATION_GROUPS),
        "critic": ["critic_state"],
    }
    clip_actions = 1.0
    actor = RslRlSpatialSoftmaxCNNModelCfg(
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
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = TensorBroadcastPPOCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=7.0e-5,
        schedule="fixed",
        gamma=0.9933,
        lam=0.9664,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
