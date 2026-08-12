"""RSL-RL PPO variant with Kit-safe distributed parameter synchronization."""

from __future__ import annotations

import torch
from rsl_rl.algorithms import PPO

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg


class TensorBroadcastPPO(PPO):
    """Synchronize model state with tensor collectives instead of object pickling.

    RSL-RL's default implementation sends CUDA state dictionaries through
    ``broadcast_object_list``. That path crashes the source-built Isaac Sim CUDA
    interop plugin on this system. Direct tensor broadcasts cover parameters and
    buffers while retaining RSL-RL's normal tensor all-reduce for gradients.
    """

    @torch.no_grad()
    def broadcast_parameters(self) -> None:
        modules = [self._raw_actor, self._raw_critic]
        if self.rnd:
            modules.append(self.rnd.predictor)
        for module in modules:
            for tensor in module.state_dict().values():
                if tensor.is_contiguous():
                    torch.distributed.broadcast(tensor, src=0)
                else:
                    contiguous = tensor.contiguous()
                    torch.distributed.broadcast(contiguous, src=0)
                    tensor.copy_(contiguous)


@configclass
class TensorBroadcastPPOCfg(RslRlPpoAlgorithmCfg):
    """Select the repository-owned distributed-safe PPO implementation."""

    class_name: str = "so101_rl.tasks.common.agents.distributed_ppo:TensorBroadcastPPO"
