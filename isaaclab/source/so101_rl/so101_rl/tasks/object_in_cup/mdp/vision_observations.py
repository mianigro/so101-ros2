"""Actor RGB observations with deployment-equivalent preprocessing."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import DelayBuffer


class camera_rgb(ManagerTermBase):
    """Return CHW RGB as ``RGB / 255 - 0.5`` with optional visual perturbations."""

    def __init__(self, cfg: ObservationTermCfg, env) -> None:
        super().__init__(cfg, env)
        self._randomize = bool(cfg.params.get("randomize", False))
        self._delay_steps = int(cfg.params.get("max_delay_steps", 0))
        if self._delay_steps not in (0, 1):
            raise ValueError("camera latency must be zero or one policy step")
        self._delay = DelayBuffer(self._delay_steps, self.num_envs, self.device)
        self._exposure = torch.ones((self.num_envs, 1, 1, 1), device=self.device)
        self._contrast = torch.ones_like(self._exposure)
        self._rgb_gain = torch.ones((self.num_envs, 3, 1, 1), device=self.device)
        self._noise_std = torch.zeros_like(self._exposure)
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # Tensor indexing uses a full slice, but DelayBuffer.reset expects None
        # for all environments (its CircularBuffer calls len() on non-None ids).
        ids = slice(None) if env_ids is None else env_ids
        delay_ids = None if isinstance(ids, slice) else ids
        self._delay.reset(delay_ids)
        count = self.num_envs if delay_ids is None else len(delay_ids)
        if self._delay_steps:
            self._delay.set_time_lag(
                torch.randint(
                    0,
                    2,
                    (count,),
                    device=self.device,
                    dtype=self._delay.time_lags.dtype,
                ),
                delay_ids,
            )
        else:
            self._delay.set_time_lag(0, delay_ids)
        if not self._randomize:
            self._exposure[ids] = 1.0
            self._contrast[ids] = 1.0
            self._rgb_gain[ids] = 1.0
            self._noise_std[ids] = 0.0
            return
        self._exposure[ids] = torch.pow(
            2.0,
            torch.empty((count, 1, 1, 1), device=self.device).uniform_(-0.25, 0.25),
        )
        self._contrast[ids] = torch.empty(
            (count, 1, 1, 1), device=self.device
        ).uniform_(0.80, 1.20)
        self._rgb_gain[ids] = torch.empty(
            (count, 3, 1, 1), device=self.device
        ).uniform_(0.90, 1.10)
        self._noise_std[ids] = torch.empty(
            (count, 1, 1, 1), device=self.device
        ).uniform_(0.0, 0.018)

    def __call__(
        self,
        env,
        sensor_cfg: SceneEntityCfg,
        randomize: bool = False,
        max_delay_steps: int = 0,
    ) -> torch.Tensor:
        del randomize, max_delay_steps
        rgb = env.scene[sensor_cfg.name].data.output["rgb"].torch[..., :3]
        image = rgb.to(dtype=torch.float32).permute(0, 3, 1, 2).contiguous() / 255.0
        image = self._delay.compute(image)
        if self._randomize:
            image = (image - 0.5) * self._contrast + 0.5
            image = image * self._exposure * self._rgb_gain
            image = image + torch.randn_like(image) * self._noise_std
            image = image.clamp_(0.0, 1.0)
        return image - 0.5
