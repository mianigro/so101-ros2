"""Temporal transformer and mamba actor-critics for RSL-RL visual PPO.

Ported from the Atari/gymnasium proof of concept. The temporal cores
(positional encoding, pre-LN encoder layers, mamba layers with a global
skip) and the ResNet-style frame encoder keep the original architecture and
initialization. The RSL-RL ``MLPModel`` base replaces the proof of concept's
policy/value heads: the privileged value head moves to the separate critic
network and action distributions come from ``distribution_cfg``.

Each 2D observation group must carry a frame-history window of shape
``(batch, lookback_frames, channels, height, width)`` ordered oldest to
newest, which Isaac Lab produces with ``history_length`` set on the camera
observation terms and ``flatten_history_dim = False``.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState
from tensordict import TensorDict


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.norm1 = nn.GroupNorm(8, out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.GroupNorm(8, out_channels)

        # Residual connection with down sampling if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.GroupNorm(8, out_channels),
            )

        # Initialise layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class CNNEncoder(nn.Module):
    """ResNet-style per-frame encoder projecting one image to ``output_dim``."""

    def __init__(self, input_shape, output_dim, in_channels=1):
        super().__init__()
        self.input_shape = input_shape
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3)
        self.norm1 = nn.GroupNorm(8, 32)

        # ResNet blocks process each frame
        self.layer1 = self._make_layer(32, 64, stride=2)
        self.layer2 = self._make_layer(64, 128, stride=2)

        # Calculate final spatial dimensions
        h, w = input_shape
        h = math.ceil(h / 8)
        w = math.ceil(w / 8)

        self.flat_dim = 128 * h * w
        self.fc = nn.Linear(self.flat_dim, output_dim)

        # Initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResBlock(in_channels, out_channels, stride),
            ResBlock(out_channels, out_channels, 1),
        )

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_frames: int):
        super().__init__()
        position = torch.arange(max_frames).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(1, max_frames, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.scale = 1.0 / math.sqrt(self.d_k)

        # Store attention weights for analysis
        self.last_attention_weights: torch.Tensor | None = None

    def forward(self, q, k, v):
        batch_size = q.size(0)

        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        attention = F.softmax(scores, dim=-1)

        # Store attention weights for analysis
        self.last_attention_weights = attention.detach()

        output = torch.matmul(attention, v)

        output = output.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        return self.w_o(output)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # Pre-LayerNorm architecture
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-LN: Apply normalization before attention
        normed = self.norm1(x)
        attn_output = self.self_attn(normed, normed, normed)
        x = x + self.dropout(attn_output)

        # Pre-LN: Apply normalization before feedforward
        normed = self.norm2(x)
        ff_output = self.feed_forward(normed)
        x = x + self.dropout(ff_output)

        return x


class _LastFramePooling(nn.Module):
    """Reduce the sequence to its newest frame."""

    def forward(self, x):
        return x[:, -1]


class _AttentionPooling(nn.Module):
    """Learned attention pooling with an optional additive last-frame skip."""

    def __init__(self, d_model, final_pool_skip):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)
        self.final_pool_skip = final_pool_skip

    def forward(self, x):
        attention_weights = torch.softmax(self.proj(x).squeeze(-1), dim=-1)
        pooled = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)
        if self.final_pool_skip:
            pooled = pooled + x[:, -1]
        return pooled


class _CameraPipeline(nn.Module):
    """Per-camera frame-history encoding: CNN, temporal core, pooling.

    Shared by training and by the TorchScript/ONNX export wrappers so the
    deployed policy runs the exact same operations as the trained actor.
    """

    def __init__(self, encoder, sequence_model, d_model, final_layer_pooling, final_pool_skip):
        super().__init__()
        self.encoder = encoder
        self.sequence_model = sequence_model
        if final_layer_pooling:
            self.pooling = _AttentionPooling(d_model, final_pool_skip)
        else:
            self.pooling = _LastFramePooling()

    def forward(self, images):
        # images: (batch, lookback_frames, channels, height, width),
        # oldest frame first so index -1 is the newest observation.
        batch = images.shape[0]
        lookback = images.shape[1]
        frames = images.reshape(
            batch * lookback, images.shape[2], images.shape[3], images.shape[4]
        )
        features = self.encoder(frames)
        x = features.reshape(batch, lookback, features.shape[-1])
        x = self.sequence_model(x)
        return self.pooling(x)


class TemporalActorCritic(MLPModel):
    """Base for actors that mix per-camera frame-history encodings.

    Subclasses provide the temporal core through ``_build_sequence_models``.
    Every 2D observation group is processed independently and the pooled
    per-camera embeddings are concatenated with the (normalized) 1D groups
    before the shared MLP head.
    """

    is_recurrent = False

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
        lookback_frames: int = 4,
        d_model: int = 256,
        num_layers: int = 2,
        final_layer_pooling: bool = False,
        final_pool_skip: bool = False,
    ) -> None:
        self.lookback_frames = lookback_frames
        self.d_model = d_model
        self.num_layers = num_layers
        self.final_layer_pooling = final_layer_pooling
        self.final_pool_skip = final_pool_skip
        self._get_obs_dim(obs, obs_groups, obs_set)
        self._temporal_latent_dim = d_model * len(self.obs_groups_2d)
        sequence_models = self._build_sequence_models()
        pipelines = {
            group: _CameraPipeline(
                CNNEncoder(
                    tuple(self.obs_dims_2d[index]),
                    d_model,
                    in_channels=int(self.obs_channels_2d[index]),
                ),
                sequence_models[group],
                d_model,
                final_layer_pooling,
                final_pool_skip,
            )
            for index, group in enumerate(self.obs_groups_2d)
        }
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
        self.pipelines = nn.ModuleDict(pipelines)

    def _build_sequence_models(self) -> nn.ModuleDict:
        raise NotImplementedError

    def _get_obs_dim(
        self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str
    ) -> tuple[list[str], int]:
        """Split observation groups into 1D inputs and frame-history 2D inputs."""
        active_obs_groups = obs_groups[obs_set]
        obs_dim_1d = 0
        obs_groups_1d = []
        obs_dims_2d = []
        obs_channels_2d = []
        obs_groups_2d = []
        obs_lookbacks = []

        for obs_group in active_obs_groups:
            shape = obs[obs_group].shape
            if len(shape) == 5:  # B, T, C, H, W
                obs_groups_2d.append(obs_group)
                obs_lookbacks.append(int(shape[1]))
                obs_dims_2d.append(shape[3:5])
                obs_channels_2d.append(shape[2])
            elif len(shape) == 2:  # B, C
                obs_groups_1d.append(obs_group)
                obs_dim_1d += shape[-1]
            else:
                raise ValueError(
                    f"The temporal actor accepts 1D observations and frame-history "
                    f"images of shape (batch, lookback, channels, height, width), "
                    f"got shape {tuple(shape)} for '{obs_group}'."
                )

        if not obs_groups_2d:
            raise ValueError(
                "No frame-history observations are provided. Use the MLP model "
                "for purely 1D observation sets."
            )
        if any(lookback != obs_lookbacks[0] for lookback in obs_lookbacks):
            raise ValueError(
                f"All camera observation groups must share one history length, "
                f"got {obs_lookbacks}."
            )
        if obs_lookbacks[0] != self.lookback_frames:
            raise ValueError(
                f"Camera observations carry a history of {obs_lookbacks[0]} frames "
                f"but the actor was configured for lookback_frames="
                f"{self.lookback_frames}."
            )

        self.obs_groups_2d = obs_groups_2d
        self.obs_dims_2d = obs_dims_2d
        self.obs_channels_2d = obs_channels_2d
        self.obs_lookbacks = obs_lookbacks
        return obs_groups_1d, obs_dim_1d

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        del masks, hidden_state
        temporal = torch.cat(
            [self.pipelines[group](obs[group]) for group in self.obs_groups_2d],
            dim=-1,
        )
        if not self.obs_groups:
            return temporal
        return torch.cat((MLPModel.get_latent(self, obs), temporal), dim=-1)

    def _get_latent_dim(self) -> int:
        return self.obs_dim + self._temporal_latent_dim

    def as_jit(self) -> nn.Module:
        return _TorchTemporalActor(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return _OnnxTemporalActor(self, verbose)


class _TransformerTemporalCore(nn.Module):
    def __init__(self, lookback_frames, d_model, num_heads, num_layers, d_ff, dropout):
        super().__init__()
        self.pos_encoding = TemporalPositionalEncoding(d_model, lookback_frames)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pos_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return self.final_norm(x)


class TransformerActorCritic(TemporalActorCritic):
    """Bidirectional transformer over per-camera frame-history embeddings."""

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
        lookback_frames: int = 4,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
        final_layer_pooling: bool = False,
        final_pool_skip: bool = False,
    ) -> None:
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )
        if d_model % 2 != 0:
            raise ValueError(
                f"d_model ({d_model}) must be even for the temporal positional "
                f"encoding."
            )
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
            lookback_frames,
            d_model,
            num_layers,
            final_layer_pooling,
            final_pool_skip,
        )

    def _build_sequence_models(self) -> nn.ModuleDict:
        return nn.ModuleDict(
            {
                group: _TransformerTemporalCore(
                    self.lookback_frames,
                    self.d_model,
                    self.num_heads,
                    self.num_layers,
                    self.d_ff,
                    self.dropout,
                )
                for group in self.obs_groups_2d
            }
        )


class _MambaTemporalCore(nn.Module):
    def __init__(self, d_model, num_layers):
        super().__init__()
        from mamba_ssm import Mamba

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2),
                    nn.LayerNorm(d_model),
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        skip_features = x
        for mamba_layer in self.layers:
            x = mamba_layer(x)
        return x + skip_features


class MambaActorCritic(TemporalActorCritic):
    """State-space sequence model over per-camera frame-history embeddings.

    Requires the optional ``mamba_ssm`` package (separate CUDA build). The
    transformer family works without it.
    """

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
        lookback_frames: int = 4,
        d_model: int = 256,
        num_layers: int = 2,
        final_layer_pooling: bool = False,
        final_pool_skip: bool = False,
    ) -> None:
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
            lookback_frames,
            d_model,
            num_layers,
            final_layer_pooling,
            final_pool_skip,
        )

    def _build_sequence_models(self) -> nn.ModuleDict:
        try:
            import mamba_ssm  # noqa: F401
        except ImportError as error:
            raise ImportError(
                "MambaActorCritic requires the 'mamba_ssm' package, which needs a "
                "separate CUDA build. Install it or use TransformerActorCritic."
            ) from error
        return nn.ModuleDict(
            {
                group: _MambaTemporalCore(self.d_model, self.num_layers)
                for group in self.obs_groups_2d
            }
        )


class _TorchTemporalActor(nn.Module):
    """TorchScript actor signature: joints plus ordered frame-history images."""

    def __init__(self, model: TemporalActorCritic) -> None:
        super().__init__()
        self.joint_normalizer = copy.deepcopy(model.obs_normalizer)
        self.pipelines = nn.ModuleList(
            [copy.deepcopy(model.pipelines[group]) for group in model.obs_groups_2d]
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
        for index, pipeline in enumerate(self.pipelines):
            latents.append(pipeline(images[index]))
        return self.output(self.mlp(torch.cat(latents, dim=-1)))

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxTemporalActor(_TorchTemporalActor):
    def __init__(self, model: TemporalActorCritic, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.image_groups = model.obs_groups_2d
        self.image_lookbacks = model.obs_lookbacks
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
            lookback = self.image_lookbacks[index]
            height, width = self.image_shapes[index]
            channels = self.image_channels[index]
            inputs.append(torch.zeros(1, lookback, channels, height, width))
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
