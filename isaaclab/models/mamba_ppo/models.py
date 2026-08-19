"""Recurrent state-space actor-critic for RSL-RL visual PPO.

A streaming policy: every camera group consumes single frames of shape
``(batch, channels, height, width)`` and a selective state-space stack
(Mamba/S6 math implemented in plain PyTorch — no ``mamba_ssm`` dependency)
integrates them over time through a constant-size hidden state. The recurrent
state is threaded through RSL-RL's recurrent rollout storage, reset on
terminations, and carried across rollout boundaries, so the model itself
supplies the temporal memory instead of an env-side frame window.

The per-frame encoder mirrors the transformer family: a ResNet-style CNN
with spatial-softmax reduction followed by a linear projection.

The hidden state is packed into a single ``[1, num_envs, state_dim]`` tensor
(one flat block per (camera, layer) holding the causal-conv window and the
SSM state) to satisfy RSL-RL's rank-3 recurrent storage contract. A zero
state is a valid sequence start for both components.
"""

from __future__ import annotations

import copy
import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.utils import unpad_trajectories
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
    """ResNet-style frame encoder projecting one image to ``output_dim``.

    The final feature map is reduced with a spatial softmax (expected XY per
    channel) before a linear projection, keeping the output a smooth function
    of feature positions instead of binding spatial coordinates to weights
    through a flattened linear layer.
    """

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
        channels = 128

        self.reducer = SpatialSoftmax(channels, int(h), int(w))
        self.proj = nn.Linear(channels * 2, output_dim)

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
        x = self.reducer(x)
        return self.proj(x)


class SelectiveSSM(nn.Module):
    """Selective state-space block (Mamba/S6) in plain PyTorch.

    One code path serves single steps and whole sequences: a sequence call is
    a scan whose recurrence is identical to stepping frame by frame with the
    returned states fed back. Conv/state inputs and outputs are functional, so
    the scan stays differentiable through the whole rollout window.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if d_conv < 2:
            raise ValueError(f"d_conv must be at least 2, got {d_conv}")
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        # Depthwise causal conv; the (d_conv - 1) history frames are supplied
        # by the caller through conv_state, so no padding here.
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            bias=True,
        )
        self.activation = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        # S4D-real init: A = -(1..d_state) replicated across channels.
        A = torch.arange(1, d_state + 1, dtype=torch.float).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Unit-gain orthogonal init: the gated SSM should not start with
        # inflated projections; the dt/A S4D init below follows the Mamba paper.
        for module in (self.in_proj, self.x_proj, self.out_proj):
            nn.init.orthogonal_(module.weight)
        # Mamba's dt init: softplus(bias) sampled log-uniformly in
        # [dt_min, dt_max] and a small uniform weight so selectivity is
        # present but the initial step sizes stay bounded.
        dt_min, dt_max = 1.0e-3, 1.0e-1
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        inv_softplus = dt + torch.log(-torch.expm1(-dt))
        dt_scale = 1.0 / (self.dt_rank**0.5)
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_softplus)
            self.dt_proj.weight.uniform_(-dt_scale, dt_scale)

    def initial_conv_state(self, batch: int, device, dtype) -> torch.Tensor:
        return torch.zeros(batch, self.d_inner, self.d_conv - 1, device=device, dtype=dtype)

    def initial_ssm_state(self, batch: int, device, dtype) -> torch.Tensor:
        return torch.zeros(batch, self.d_inner, self.d_state, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scan ``x`` of shape (batch, steps, d_model) from the given states.

        Returns ``(output, new_conv_state, new_ssm_state)`` with output shaped
        ``(batch, steps, d_model)``.
        """
        steps = x.shape[1]
        xz = self.in_proj(x)
        x_in = xz[..., : self.d_inner]
        z = xz[..., self.d_inner :]
        seq = torch.cat((conv_state, x_in.transpose(1, 2)), dim=-1)
        act = self.activation(self.conv1d(seq)).transpose(1, 2)

        x_dbl = self.x_proj(act)
        dt_low = x_dbl[..., : self.dt_rank]
        B = x_dbl[..., self.dt_rank : self.dt_rank + self.d_state]
        C = x_dbl[..., self.dt_rank + self.d_state :]
        delta = F.softplus(self.dt_proj(dt_low))
        A = -torch.exp(self.A_log)

        state = ssm_state
        outputs: List[torch.Tensor] = []
        for t in range(steps):
            deltaA = torch.exp(delta[:, t].unsqueeze(-1) * A)
            deltaB_u = (
                delta[:, t].unsqueeze(-1)
                * B[:, t].unsqueeze(1)
                * act[:, t].unsqueeze(-1)
            )
            state = deltaA * state + deltaB_u
            y = torch.einsum("bdn,bn->bd", state, C[:, t]) + self.D * act[:, t]
            outputs.append(y)
        y = torch.stack(outputs, dim=1)
        y = y * self.activation(z)
        out = self.out_proj(y)
        new_conv_state = seq[..., -(self.d_conv - 1) :]
        return out, new_conv_state, state


class _SSMStreamLayer(nn.Module):
    """One selective-SSM layer with post-LayerNorm, step-compatible.

    Pairing the norm with the SSM in a single module keeps the scripted
    pipeline a flat ``ModuleList`` (TorchScript cannot variable-index nested
    ModuleLists).
    """

    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.ssm = SelectiveSSM(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = nn.LayerNorm(d_model)

    def step(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out, conv_state, ssm_state = self.ssm(x.unsqueeze(1), conv_state, ssm_state)
        return self.norm(out.squeeze(1)), conv_state, ssm_state


class _RecurrentCameraPipeline(nn.Module):
    """Per-camera streaming pipeline: frame encoder + state-space stack.

    The whole-stack residual keeps the encoder embedding on the output path so
    the stack starts near identity behaviour.
    """

    def __init__(self, encoder, d_model, num_layers, d_state, d_conv, expand):
        super().__init__()
        self.num_layers = num_layers
        self.encoder = encoder
        self.layers = nn.ModuleList(
            [
                _SSMStreamLayer(d_model, d_state, d_conv, expand)
                for _ in range(num_layers)
            ]
        )

    def initial_states(
        self, batch: int, device, dtype
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        conv_states: List[torch.Tensor] = []
        ssm_states: List[torch.Tensor] = []
        for layer in self.layers:
            conv_states.append(layer.ssm.initial_conv_state(batch, device, dtype))
            ssm_states.append(layer.ssm.initial_ssm_state(batch, device, dtype))
        return conv_states, ssm_states

    def step(
        self,
        frame: torch.Tensor,
        conv_states: List[torch.Tensor],
        ssm_states: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        emb = self.encoder(frame)
        out = emb
        new_conv_states: List[torch.Tensor] = []
        new_ssm_states: List[torch.Tensor] = []
        for index, layer in enumerate(self.layers):
            out, conv_state, ssm_state = layer.step(
                out, conv_states[index], ssm_states[index]
            )
            new_conv_states.append(conv_state)
            new_ssm_states.append(ssm_state)
        return out + emb, new_conv_states, new_ssm_states


class MambaActorCritic(MLPModel):
    """Streaming selective state-space actor over per-camera frame streams.

    Every 2D observation group is processed independently (single-frame ResNet
    encoder with spatial-softmax reduction, recurrent state-space stack) and
    the per-camera embeddings are concatenated with the (normalized) 1D groups
    before the shared MLP head. Temporal memory lives entirely in the
    recurrent state: no frame stacking, no lookback windows.
    """

    is_recurrent = True

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
        d_model: int = 64,
        num_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        self.d_model = d_model
        self.num_layers = num_layers
        self._get_obs_dim(obs, obs_groups, obs_set)
        self._temporal_latent_dim = d_model * len(self.obs_groups_2d)
        pipelines = {
            group: _RecurrentCameraPipeline(
                CNNEncoder(
                    tuple(self.obs_dims_2d[index]),
                    d_model,
                    in_channels=int(self.obs_channels_2d[index]),
                ),
                d_model,
                num_layers,
                d_state,
                d_conv,
                expand,
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
        # Hidden-state layout: per camera, per layer, a flat conv window
        # followed by a flat SSM state — packed into one [1, num_envs, D]
        # tensor for RSL-RL's recurrent rollout storage.
        example_block = pipelines[self.obs_groups_2d[0]].layers[0].ssm
        self._d_inner = example_block.d_inner
        self._d_state = d_state
        self._d_conv = d_conv
        conv_size = self._d_inner * (d_conv - 1)
        ssm_size = self._d_inner * d_state
        self._state_sizes: list[int] = []
        for _group in self.obs_groups_2d:
            for _layer in range(num_layers):
                self._state_sizes.extend((conv_size, ssm_size))
        self._state_dim = sum(self._state_sizes)
        self._conv_size = conv_size
        self._ssm_size = ssm_size
        self._state: torch.Tensor | None = None

    def _get_obs_dim(
        self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str
    ) -> tuple[list[str], int]:
        """Split observation groups into 1D inputs and single-frame 2D inputs."""
        active_obs_groups = obs_groups[obs_set]
        obs_dim_1d = 0
        obs_groups_1d = []
        obs_dims_2d = []
        obs_channels_2d = []
        obs_groups_2d = []

        for obs_group in active_obs_groups:
            shape = obs[obs_group].shape
            if len(shape) == 4:  # B, C, H, W
                obs_groups_2d.append(obs_group)
                obs_dims_2d.append(shape[2:4])
                obs_channels_2d.append(shape[1])
            elif len(shape) == 2:  # B, C
                obs_groups_1d.append(obs_group)
                obs_dim_1d += shape[-1]
            else:
                raise ValueError(
                    f"The recurrent actor accepts 1D observations and single-frame "
                    f"images of shape (batch, channels, height, width), got shape "
                    f"{tuple(shape)} for '{obs_group}'. Frame-history windows belong "
                    f"to the transformer_ppo family."
                )

        if not obs_groups_2d:
            raise ValueError(
                "No camera observations are provided. Use the MLP model "
                "for purely 1D observation sets."
            )

        self.obs_groups_2d = obs_groups_2d
        self.obs_dims_2d = obs_dims_2d
        self.obs_channels_2d = obs_channels_2d
        return obs_groups_1d, obs_dim_1d

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        if masks is not None:
            return self._sequence_latent(obs, masks, hidden_state)
        return self._step_latent(obs)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset the recurrent state: all envs, or only the done envs."""
        if dones is None:
            self._state = None if hidden_state is None else hidden_state
        elif self._state is not None and hidden_state is None:
            # Out-of-place so reset works both inside and outside inference
            # mode (rollout state tensors are created under inference mode).
            done_mask = (dones == 1).view(1, -1, 1)
            self._state = torch.where(done_mask, torch.zeros_like(self._state), self._state)

    def get_hidden_state(self) -> HiddenState:
        return self._state

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        if self._state is not None:
            self._state = self._state.detach()

    def _unpack_state(self, packed: torch.Tensor):
        """Split a packed [1, B, D] state into per-(camera, layer) states."""
        chunks = packed[0].split(self._state_sizes, dim=-1)
        states = []
        index = 0
        for _group in self.obs_groups_2d:
            conv_states = []
            ssm_states = []
            for _layer in range(self.num_layers):
                conv_states.append(
                    chunks[index].reshape(
                        chunks[index].shape[0], self._d_inner, self._d_conv - 1
                    )
                )
                index += 1
                ssm_states.append(
                    chunks[index].reshape(
                        chunks[index].shape[0], self._d_inner, self._d_state
                    )
                )
                index += 1
            states.append((conv_states, ssm_states))
        return states

    def _step_latent(self, obs: TensorDict) -> torch.Tensor:
        frame = obs[self.obs_groups_2d[0]]
        batch = frame.shape[0]
        if self._state is None or self._state.shape[1] != batch:
            device, dtype = frame.device, frame.dtype
            states = [
                self.pipelines[group].initial_states(batch, device, dtype)
                for group in self.obs_groups_2d
            ]
        else:
            states = self._unpack_state(self._state)

        outputs = []
        packed_chunks = []
        for group_index, group in enumerate(self.obs_groups_2d):
            conv_states, ssm_states = states[group_index]
            output, conv_states, ssm_states = self.pipelines[group].step(
                obs[group], conv_states, ssm_states
            )
            outputs.append(output)
            for layer in range(self.num_layers):
                packed_chunks.append(
                    conv_states[layer].reshape(batch, -1)
                )
                packed_chunks.append(
                    ssm_states[layer].reshape(batch, -1)
                )
        self._state = torch.cat(packed_chunks, dim=-1).unsqueeze(0).detach()
        temporal = torch.cat(outputs, dim=-1)
        if not self.obs_groups:
            return temporal
        return torch.cat((MLPModel.get_latent(self, obs), temporal), dim=-1)

    def _sequence_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor,
        hidden_state: HiddenState,
    ) -> torch.Tensor:
        cameras = [obs[group] for group in self.obs_groups_2d]
        steps, batch = cameras[0].shape[0], cameras[0].shape[1]
        device, dtype = cameras[0].device, cameras[0].dtype
        if hidden_state is None:
            packed = torch.zeros(1, batch, self._state_dim, device=device, dtype=dtype)
        else:
            packed = hidden_state
        states = self._unpack_state(packed)

        outputs = [[] for _ in self.obs_groups_2d]
        for t in range(steps):
            for group_index, group in enumerate(self.obs_groups_2d):
                conv_states, ssm_states = states[group_index]
                output, conv_states, ssm_states = self.pipelines[group].step(
                    cameras[group_index][t], conv_states, ssm_states
                )
                outputs[group_index].append(output)
                states[group_index] = (conv_states, ssm_states)

        temporal = torch.cat(
            [torch.stack(group_outputs, dim=0) for group_outputs in outputs], dim=-1
        )
        joints = MLPModel.get_latent(self, obs)
        latent = torch.cat((joints, temporal), dim=-1)
        return unpad_trajectories(latent, masks)

    def _get_latent_dim(self) -> int:
        return self.obs_dim + self._temporal_latent_dim

    def as_jit(self) -> nn.Module:
        return _TorchRecurrentActor(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return _OnnxRecurrentActor(self, verbose)


class _TorchRecurrentActor(nn.Module):
    """TorchScript actor: joints plus ordered single-frame images.

    The packed recurrent state lives in a buffer; ``reset()`` zeroes it. The
    pipeline modules are shared with training so the deployed policy runs the
    exact same operations as the trained actor.
    """

    def __init__(self, model: MambaActorCritic) -> None:
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
        self.num_layers = model.num_layers
        self.state_sizes: List[int] = list(model._state_sizes)
        self.conv_window = (
            model.pipelines[model.obs_groups_2d[0]].layers[0].ssm.d_conv - 1
        )
        self.ssm_states = model.pipelines[model.obs_groups_2d[0]].layers[0].ssm.d_state
        self.register_buffer("state", torch.zeros(1, 1, model._state_dim))

    def forward(
        self, joint_positions: torch.Tensor, images: List[torch.Tensor]
    ) -> torch.Tensor:
        batch = joint_positions.shape[0]
        if self.state.shape[1] != batch:
            # Fresh zero state whenever the served batch size changes, so a
            # zeroed state always matches the streamed batch.
            self.state = torch.zeros(
                1, batch, self.state.shape[2], device=self.state.device, dtype=self.state.dtype
            )
        latents: List[torch.Tensor] = [self.joint_normalizer(joint_positions)]
        chunks = torch.split(self.state[0], self.state_sizes, dim=-1)
        packed: List[torch.Tensor] = []
        for index, pipeline in enumerate(self.pipelines):
            conv_states: List[torch.Tensor] = []
            ssm_states: List[torch.Tensor] = []
            base = index * self.num_layers * 2
            for layer in range(self.num_layers):
                conv = chunks[base + 2 * layer]
                conv_states.append(conv.reshape(conv.shape[0], -1, self.conv_window))
                ssm = chunks[base + 2 * layer + 1]
                ssm_states.append(ssm.reshape(ssm.shape[0], -1, self.ssm_states))
            output, conv_states, ssm_states = pipeline.step(
                images[index], conv_states, ssm_states
            )
            latents.append(output)
            for layer in range(self.num_layers):
                packed.append(conv_states[layer].reshape(conv_states[layer].shape[0], -1))
                packed.append(ssm_states[layer].reshape(ssm_states[layer].shape[0], -1))
        self.state[:] = torch.cat(packed, dim=-1).unsqueeze(0)
        return self.output(self.mlp(torch.cat(latents, dim=-1)))

    @torch.jit.export
    def reset(self) -> None:
        self.state[:] = torch.zeros_like(self.state)


class _OnnxRecurrentActor(nn.Module):
    """ONNX actor with the packed recurrent state as explicit input/output."""

    is_recurrent: bool = True

    def __init__(self, model: MambaActorCritic, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
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
        self.image_groups = model.obs_groups_2d
        self.image_shapes = model.obs_dims_2d
        self.image_channels = model.obs_channels_2d
        self.joint_count = model.obs_dim
        self.num_layers = model.num_layers
        self.state_sizes = list(model._state_sizes)
        self.conv_window = (
            model.pipelines[model.obs_groups_2d[0]].layers[0].ssm.d_conv - 1
        )
        self.ssm_states = model.pipelines[model.obs_groups_2d[0]].layers[0].ssm.d_state
        self.state_dim = model._state_dim

    def forward(
        self, joint_positions: torch.Tensor, state: torch.Tensor, *images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        latents: List[torch.Tensor] = [self.joint_normalizer(joint_positions)]
        chunks = torch.split(state[0], self.state_sizes, dim=-1)
        packed: List[torch.Tensor] = []
        for index, pipeline in enumerate(self.pipelines):
            conv_states: List[torch.Tensor] = []
            ssm_states: List[torch.Tensor] = []
            base = index * self.num_layers * 2
            for layer in range(self.num_layers):
                conv = chunks[base + 2 * layer]
                conv_states.append(conv.reshape(conv.shape[0], -1, self.conv_window))
                ssm = chunks[base + 2 * layer + 1]
                ssm_states.append(ssm.reshape(ssm.shape[0], -1, self.ssm_states))
            output, conv_states, ssm_states = pipeline.step(
                images[index], conv_states, ssm_states
            )
            latents.append(output)
            for layer in range(self.num_layers):
                packed.append(conv_states[layer].reshape(conv_states[layer].shape[0], -1))
                packed.append(ssm_states[layer].reshape(ssm_states[layer].shape[0], -1))
        new_state = torch.cat(packed, dim=-1).unsqueeze(0)
        deltas = self.output(self.mlp(torch.cat(latents, dim=-1)))
        return deltas, new_state

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        inputs = [
            torch.zeros(1, self.joint_count),
            torch.zeros(1, 1, self.state_dim),
        ]
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
            "recurrent_state_in",
            *[f"observation.images.{name}" for name in self.image_groups],
        ]

    @property
    def output_names(self) -> list[str]:
        return ["normalized_joint_deltas", "recurrent_state_out"]
