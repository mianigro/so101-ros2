import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..models import CNNEncoder


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

        # Store attention weights for analysis
        self.last_attention_weights = None

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.scalar_tensor(self.d_k, dtype=torch.float32)
        )

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)

        # Store attention weights for analysis
        self.last_attention_weights = attention.detach()

        output = torch.matmul(attention, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
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

    def forward(self, x, mask=None):
        # Pre-LN: Apply normalization before attention
        normed = self.norm1(x)
        attn_output = self.self_attn(normed, normed, normed, mask)
        x = x + self.dropout(attn_output)

        # Pre-LN: Apply normalization before feedforward
        normed = self.norm2(x)
        ff_output = self.feed_forward(normed)
        x = x + self.dropout(ff_output)

        return x


class TransformerModel(nn.Module):
    def __init__(
        self,
        lookback_frames,
        input_shape,
        d_model,
        num_heads,
        num_layers,
        d_ff,
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
        self.lookback_frames = lookback_frames
        self.final_layer_pooling = final_layer_pooling
        self.final_pool_skip = final_pool_skip
        self.num_layers = num_layers
        self.input_shape = input_shape
        self.log_clamp_lower = log_clamp_lower
        self.log_clamp_upper = log_clamp_upper

        # CNN encoder
        # input dim: (lookback_frames, 1, width, height)
        self.cnn_encoder = CNNEncoder(input_shape, d_model)
        self.pos_encoding = TemporalPositionalEncoding(d_model, lookback_frames)

        # current dim: (lookback_frames, 128 * h/8 * w/8)

        # Pre-LN encoder layers
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # current dim: (lookback_frames, d_model)

        # 1 frame selected either with [-1] or mean

        # current dim: (1, d_model)

        # Final input aggregation
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

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.orthogonal_(m.in_proj_weight, gain=np.sqrt(2))
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
                nn.init.orthogonal_(m.out_proj.weight, gain=np.sqrt(2))
                if m.out_proj.bias is not None:
                    nn.init.zeros_(m.out_proj.bias)

        # Handle mean and log_std in continuous
        if not discrete:
            # Mean
            nn.init.orthogonal_(self.policy_head[-1].weight[:num_actions], gain=0.01)
            # log_std
            nn.init.constant_(self.policy_head[-1].weight[num_actions:], 0.0)

    def forward(self, state):
        batch_size = state.size(0)

        # Split input into separate frames
        frames = state.view(batch_size * self.lookback_frames, 1, *state.shape[2:])

        # Process each frame through CNN
        frame_features = self.cnn_encoder(frames)

        # Reshape back to [batch_size, lookback_frames, d_model]
        x = frame_features.view(batch_size, self.lookback_frames, -1)

        # Apply positional encoding to the sequence of frame features
        x = self.pos_encoding(x)

        # Process through transformer layers
        for layer in self.encoder_layers:
            x = layer(x)

        x = self.final_norm(x)

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
            # Logits out
            action_dist = action_logits
        else:
            mean, log_std = action_logits.chunk(2, dim=-1)

            log_std = torch.clamp(log_std, self.log_clamp_lower, self.log_clamp_upper)

            # Cat mean and log_std, gets chunk in ppo_agent to be split again
            action_dist = torch.cat([mean, log_std], dim=-1)

        # Value head
        state_value = self.value_head(x)

        return action_dist, state_value
