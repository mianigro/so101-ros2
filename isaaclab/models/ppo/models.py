"""Reusable RSL-RL multi-camera spatial-softmax actor."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from rsl_rl.models.cnn_model import CNNModel
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import CNN, HiddenState
from tensordict import TensorDict


class SpatialSoftmax(nn.Module):
    """Reduce every feature channel to its expected normalized XY location."""

    def __init__(self, channels: int, height: int, width: int, temperature: float = 1.0):
        super().__init__()
        self.channels = channels
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height),
            torch.linspace(-1.0, 1.0, width),
            indexing="ij",
        )
        self.register_buffer("x", x.reshape(1, -1))
        self.register_buffer("y", y.reshape(1, -1))
        self.log_temperature = nn.Parameter(
            torch.full((channels,), float(torch.log(torch.tensor(temperature))))
        )

    @property
    def output_dim(self) -> int:
        return self.channels * 2

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch = features.shape[0]
        temperature = self.log_temperature.exp().clamp(min=1e-3).view(
            1, self.channels, 1
        )
        weights = torch.softmax(
            features.reshape(batch, self.channels, -1) / temperature, dim=-1
        )
        expected_x = (weights * self.x).sum(dim=-1)
        expected_y = (weights * self.y).sum(dim=-1)
        return torch.cat((expected_x, expected_y), dim=-1)


class SpatialSoftmaxCNNModel(MLPModel):
    """Encode each image independently and fuse keypoints with joint positions."""

    is_recurrent = False
    _get_obs_dim = CNNModel._get_obs_dim

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims=(512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
        cnn_cfg: dict | None = None,
        init_temperature: float = 1.0,
    ) -> None:
        if cnn_cfg is None:
            raise ValueError("CNN configuration is required")
        self._get_obs_dim(obs, obs_groups, obs_set)
        cnn_kwargs = dict(cnn_cfg)
        if cnn_kwargs.get("global_pool", "none") != "none":
            raise ValueError("spatial softmax requires global_pool='none'")
        cnn_kwargs.update(flatten=False, global_pool="none")
        encoders: dict[str, CNN] = {}
        reducers: dict[str, SpatialSoftmax] = {}
        self.keypoint_dim = 0
        for index, group in enumerate(self.obs_groups_2d):
            encoder = CNN(
                input_dim=self.obs_dims_2d[index],
                input_channels=self.obs_channels_2d[index],
                **cnn_kwargs,
            )
            if encoder.output_channels is None:
                raise ValueError("CNN encoder unexpectedly flattened its feature map")
            height, width = encoder.output_dim
            reducer = SpatialSoftmax(
                int(encoder.output_channels),
                int(height),
                int(width),
                init_temperature,
            )
            encoders[group] = encoder
            reducers[group] = reducer
            self.keypoint_dim += reducer.output_dim
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )
        self.encoders = nn.ModuleDict(encoders)
        self.reducers = nn.ModuleDict(reducers)

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        del masks, hidden_state
        images = torch.cat(
            [
                self.reducers[group](self.encoders[group](obs[group]))
                for group in self.obs_groups_2d
            ],
            dim=-1,
        )
        if not self.obs_groups:
            return images
        return torch.cat((MLPModel.get_latent(self, obs), images), dim=-1)

    def _get_latent_dim(self) -> int:
        return self.obs_dim + self.keypoint_dim

    def as_jit(self) -> nn.Module:
        return _TorchSpatialSoftmaxActor(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return _OnnxSpatialSoftmaxActor(self, verbose)


class _TorchSpatialSoftmaxActor(nn.Module):
    """TorchScript actor signature: joints plus ordered list of three images."""

    def __init__(self, model: SpatialSoftmaxCNNModel) -> None:
        super().__init__()
        self.joint_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    copy.deepcopy(model.encoders[group]),
                    copy.deepcopy(model.reducers[group]),
                )
                for group in model.obs_groups_2d
            ]
        )
        self.mlp = copy.deepcopy(model.mlp)
        self.output = (
            model.distribution.as_deterministic_output_module()
            if model.distribution is not None
            else nn.Identity()
        )

    def forward(
        self, joint_positions: torch.Tensor, images: list[torch.Tensor]
    ) -> torch.Tensor:
        latents = [self.joint_normalizer(joint_positions)]
        for index, encoder in enumerate(self.encoders):
            latents.append(encoder(images[index]))
        return self.output(self.mlp(torch.cat(latents, dim=-1)))

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxSpatialSoftmaxActor(_TorchSpatialSoftmaxActor):
    def __init__(self, model: SpatialSoftmaxCNNModel, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.image_groups = model.obs_groups_2d
        self.image_shapes = model.obs_dims_2d
        self.image_channels = model.obs_channels_2d
        self.joint_count = model.obs_dim

    def forward(
        self, joint_positions: torch.Tensor, *images: torch.Tensor
    ) -> torch.Tensor:
        return super().forward(joint_positions, list(images))

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        inputs = [torch.zeros(1, self.joint_count)]
        for index in range(len(self.image_groups)):
            height, width = self.image_shapes[index]
            inputs.append(
                torch.zeros(1, self.image_channels[index], height, width)
            )
        return tuple(inputs)

    @property
    def input_names(self) -> list[str]:
        return [
            "observation.state",
            *[f"observation.images.{name}" for name in self.image_groups],
        ]

    @property
    def output_names(self) -> list[str]:
        return ["normalized_joint_deltas"]
