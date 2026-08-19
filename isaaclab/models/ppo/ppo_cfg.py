"""Task-agnostic PPO runner defaults shared by every visual model family."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
)

from .distributed_ppo import TensorBroadcastPPOCfg


@configclass
class RslRlSpatialSoftmaxCNNModelCfg(RslRlCNNModelCfg):
    class_name = "models.ppo.models:SpatialSoftmaxCNNModel"
    init_temperature: float = 1.0


@configclass
class VisualPPOCfg(RslRlOnPolicyRunnerCfg):
    """Shared rollout, critic, and PPO contract for visual tasks.

    Model families and tasks derive from this class, supplying their own
    ``obs_groups`` and ``actor``. The privileged critic is an MLP over the
    task's critic observation group and the algorithm is the repository-owned
    distributed-safe PPO.
    """

    num_steps_per_env = 48
    max_iterations = 15_000
    save_interval = 250
    experiment_name = "visual_ppo"
    clip_actions = 1.0
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
