"""Deployment-matched SO-101 joint actions with bounded policy-step latency."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions import RelativeJointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
from isaaclab.utils import DelayBuffer
from isaaclab.utils.configclass import configclass


class DelayedRelativeJointPositionAction(RelativeJointPositionAction):
    """Apply normalized joint deltas after a per-environment zero/one-step delay."""

    cfg: DelayedRelativeJointPositionActionCfg

    def __init__(self, cfg: DelayedRelativeJointPositionActionCfg, env) -> None:
        super().__init__(cfg, env)
        self._delay = DelayBuffer(cfg.max_delay_steps, self.num_envs, self.device)
        self._sample_delays(slice(None))

    def _sample_delays(self, env_ids: Sequence[int] | slice) -> None:
        if self.cfg.max_delay_steps <= 0:
            self._delay.set_time_lag(0, env_ids)
            return
        count = self.num_envs if isinstance(env_ids, slice) else len(env_ids)
        lags = torch.randint(
            0,
            self.cfg.max_delay_steps + 1,
            (count,),
            device=self.device,
            dtype=self._delay.time_lags.dtype,
        )
        self._delay.set_time_lag(lags, env_ids)

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(self._delay.compute(actions))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        resolved_ids = slice(None) if env_ids is None else env_ids
        super().reset(resolved_ids)
        # ActionManager passes slice(None) for a full reset, while DelayBuffer
        # requires None to mean all batches.
        self._delay.reset(None if isinstance(resolved_ids, slice) else resolved_ids)
        self._sample_delays(resolved_ids)


@configclass
class DelayedRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Configuration for policy-step action latency."""

    class_type: type[DelayedRelativeJointPositionAction] = (
        DelayedRelativeJointPositionAction
    )
    max_delay_steps: int = 1

    def __post_init__(self) -> None:
        if self.max_delay_steps not in (0, 1):
            raise ValueError("SO-101 visual action latency must be zero or one policy step")
