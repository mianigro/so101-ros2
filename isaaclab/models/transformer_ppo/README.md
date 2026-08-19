# transformer_ppo

Temporal transformer actor for visual PPO, ported from the Atari/gymnasium
proof of concept: pre-LN transformer encoder over per-camera frame embeddings
(sinusoidal temporal positional encoding, configurable depth/width, last-frame
or learned attention pooling).

The family follows the shared pattern: `models.py` holds the actor class and
`ppo_cfg.py` holds the `@configclass` training configuration
(`RslRlTransformerActorCfg`) whose `class_name` dotted path selects it. The
PPO algorithm, rollout settings, and privileged MLP critic come from the
shared `models/ppo` stack (`VisualPPOCfg` + `TensorBroadcastPPO`).

## Observation contract

Each camera observation group must carry a frame-history window shaped
`(num_envs, lookback_frames, channels, height, width)`, ordered oldest to
newest. In an Isaac Lab environment config this is one setting per camera
term:

```python
term.history_length = actor_cfg.lookback_frames
term.flatten_history_dim = False
```

Every camera group must use the same history length, and it must equal the
actor's `lookback_frames`. 1D groups (e.g. joint positions) pass through the
standard normalized MLP path and need no history.

Rollout-storage memory scales with `lookback_frames x cameras x resolution`,
so prefer short windows (3-4 frames) at full resolution or longer windows
with smaller images.

## Architecture notes (relative to the proof of concept)

- The per-frame `CNNEncoder` reduces its 128-channel feature map with a
  **spatial softmax** (expected XY per channel → 256 keypoints) followed by a
  `Linear(256, d_model)` projection. This replaces the proof of concept's
  flattened `Linear(128*15*20, d_model)` head (~2.46M parameters per camera),
  cutting the actor from ~9.9M to ~2.6M parameters and keeping the embedding
  a smooth function of feature positions instead of binding spatial
  coordinates to weights.
- **Frame-difference features** (`frame_diff`, default on): the per-frame
  embeddings are fused with their one-step deltas (zero for the oldest frame)
  through a learned `Linear(2*d_model, d_model)`, so velocity information
  reaches the temporal core explicitly instead of being relearned from raw
  positions.
- Attention is **causal by default** (`causal_mask`): each frame's
  representation depends only on the past, matching the policy's information
  structure. Set `causal_mask=False` for the proof of concept's
  bidirectional attention.
- **Dropout defaults to 0.0.** Rollout and update passes sample from
  different distributions when an actor drops units, breaking PPO's
  on-policy assumption; the parameter remains for ablation.
- `CNNEncoder` takes `in_channels` (RGB cameras use 3; the POC was
  grayscale-only) and one encoder is built per camera group.
- The POC's fused policy/value heads and its custom PPO agent are replaced
  by the RSL-RL `MLPModel` head + `distribution_cfg` and the repository PPO
  stack (asymmetric PPO: the critic is a separate MLP on privileged state).
- Sequence position 0 is the oldest frame; "last frame" pooling uses index
  `-1`, matching Isaac Lab's oldest-to-newest history buffers.
