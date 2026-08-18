"""RSL-RL training configuration for the temporal transformer actor."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg


@configclass
class RslRlTransformerActorCfg(RslRlMLPModelCfg):
    """Temporal transformer actor over per-camera frame-history windows."""

    class_name = "models.transformer_ppo.models:TransformerActorCritic"
    lookback_frames: int = 4
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.0
    final_layer_pooling: bool = False
    final_pool_skip: bool = False
    frame_diff: bool = True
    causal_mask: bool = True
