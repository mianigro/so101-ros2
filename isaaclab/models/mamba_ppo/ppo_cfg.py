"""RSL-RL training configuration for the recurrent state-space actor."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg


@configclass
class RslRlMambaActorCfg(RslRlMLPModelCfg):
    """Recurrent selective state-space actor over per-camera frame streams."""

    class_name = "models.mamba_ppo.models:MambaActorCritic"
    d_model: int = 64
    num_layers: int = 2
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
