"""RSL-RL training configurations for the temporal actor family."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg


@configclass
class RslRlTransformerActorCfg(RslRlMLPModelCfg):
    """Temporal transformer actor over per-camera frame-history windows."""

    class_name = "models.transformers_ppo.models:TransformerActorCritic"
    lookback_frames: int = 4
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.1
    final_layer_pooling: bool = False
    final_pool_skip: bool = False


@configclass
class RslRlMambaActorCfg(RslRlMLPModelCfg):
    """Temporal mamba actor over per-camera frame-history windows.

    Requires the optional ``mamba_ssm`` package (separate CUDA build).
    """

    class_name = "models.transformers_ppo.models:MambaActorCritic"
    lookback_frames: int = 4
    d_model: int = 256
    num_layers: int = 2
    final_layer_pooling: bool = False
    final_pool_skip: bool = False
