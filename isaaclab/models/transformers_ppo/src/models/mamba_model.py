import numpy as np
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from ..models import CNNEncoder


class MambaCVModel(nn.Module):
    def __init__(
        self,
        lookback_frames,
        input_shape,
        d_model,
        d_ff,
        num_layers,
        num_actions,
        discrete,
        final_layer_pooling,
        final_pool_skip,
        log_clamp_lower,
        log_clamp_upper,
        dropout,
    ):
        super().__init__()
        self.discrete = discrete
        self.d_model = d_model
        self.lookback_frames = lookback_frames
        self.final_layer_pooling = final_layer_pooling
        self.final_pool_skip = final_pool_skip
        self.num_layers = num_layers
        self.input_shape = input_shape
        self.log_clamp_lower = log_clamp_lower
        self.log_clamp_upper = log_clamp_upper

        # input dim: (lookback_frames, 1, width, height)

        # CNN encoder
        self.cnn_encoder = CNNEncoder(input_shape, d_model)

        # current dim: (lookback_frames, 128 * h/8 * w/8)

        # Mamba sequence model parameters
        self.mamba = nn.ModuleList(
            [
                nn.Sequential(
                    Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2),
                    nn.LayerNorm(d_model),
                )
                for _ in range(num_layers)
            ]
        )

        # current dim: (lookback_frames, d_model)

        # 1 frame selected either with [-1] or mean

        # current dim: (1, d_model)

        # Final frame aggregation
        if final_layer_pooling:
            self.pooling = nn.Linear(d_model, 1)

        # Policy head
        self.policy_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, num_actions if discrete else num_actions * 2),
        )

        # Policy head output dim: (1, num_actions)

        # Value head
        self.value_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1),
        )

        # Value head output dim: (1, 1)

        # Handle mean and log_std in continuous
        if not discrete:
            # Mean
            nn.init.orthogonal_(self.policy_head[-1].weight[:num_actions], gain=0.01)
            # log_std
            nn.init.constant_(self.policy_head[-1].weight[num_actions:], 0.0)

    def forward(self, state):
        batch_size = state.size(0)

        # Process frames through CNN encoder
        frames = state.view(batch_size * self.lookback_frames, 1, *state.shape[2:])
        frame_features = self.cnn_encoder(frames)
        x = frame_features.view(batch_size, self.lookback_frames, -1)

        # Store skip connection
        skip_features = x

        # Process through all mamba layers
        for mamba_layer in self.mamba:
            x = mamba_layer(x)

        # Add skip connection
        x = x + skip_features

        # Use the final sequence representation for prediction
        #   Either selects the final frame or does pooling
        if not self.final_layer_pooling:
            x = x[:, -1]
        else:
            if self.final_pool_skip:
                skip = x[:, -1]
                attention_weights = torch.softmax(self.pooling(x).squeeze(-1), dim=-1)
                x = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)
                x = x + skip

            else:
                attention_weights = torch.softmax(self.pooling(x).squeeze(-1), dim=-1)
                x = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)

        # Policy head
        action_logits = self.policy_head(x)

        # Action selection
        if self.discrete:
            action_dist = action_logits
        else:
            mean, log_std = action_logits.chunk(2, dim=-1)
            log_std = torch.clamp(log_std, self.log_clamp_lower, self.log_clamp_upper)

            # Cat mean and log_std, gets chunk in ppo_agent to be split again
            action_dist = torch.cat([mean, log_std], dim=-1)

        # Value head
        state_value = self.value_head(x)

        return action_dist, state_value
