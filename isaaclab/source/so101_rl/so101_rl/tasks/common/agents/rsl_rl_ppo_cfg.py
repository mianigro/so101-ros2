"""SO-101 asymmetric PPO contract built on the shared models library."""

from isaaclab.utils.configclass import configclass

from models.ppo.ppo_cfg import RslRlSpatialSoftmaxCNNModelCfg, VisualPPOCfg

from so101_rl.visual_contract import SO101_ACTOR_OBSERVATION_GROUPS


@configclass
class SO101VisualPPOCfg(VisualPPOCfg):
    """SO-101 observation groups with the spatial-softmax visual actor."""

    experiment_name = "so101_visual"
    obs_groups = {
        "actor": list(SO101_ACTOR_OBSERVATION_GROUPS),
        "critic": ["critic_state"],
    }
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
