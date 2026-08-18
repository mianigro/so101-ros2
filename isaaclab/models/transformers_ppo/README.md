# transformers_ppo

Temporal sequence-model actors for visual PPO, ported from the Atari/
gymnasium proof of concept. Two architectures share one temporal contract:

- `TransformerActorCritic` -- bidirectional pre-LN transformer encoder over
  per-camera frame embeddings (sinusoidal temporal positional encoding,
  configurable depth/width, last-frame or learned attention pooling).
- `MambaActorCritic` -- selective state-space (Mamba) layers with a global
  skip connection over the same frame embeddings. Requires the optional
  `mamba_ssm` package (separate CUDA build); the transformer works without
  it.

Both follow the family pattern: `models.py` holds the actor classes and
`ppo_cfg.py` holds the `@configclass` training configurations
(`RslRlTransformerActorCfg`, `RslRlMambaActorCfg`) whose `class_name` dotted
paths select them. The PPO algorithm, rollout settings, and privileged MLP
critic come from the shared `models/ppo` stack (`VisualPPOCfg` +
`TensorBroadcastPPO`).

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

- The ResNet-style `CNNEncoder`, transformer encoder layers, mamba layers +
  skip, pooling variants (`final_layer_pooling`, `final_pool_skip`), and
  orthogonal initialization are kept from the proof of concept.
- `CNNEncoder` now takes `in_channels` (RGB cameras use 3; the POC was
  grayscale-only) and one encoder is built per camera group.
- The POC's fused policy/value heads and its custom PPO agent are replaced
  by the RSL-RL `MLPModel` head + `distribution_cfg` and the repository PPO
  stack (asymmetric PPO: the critic is a separate MLP on privileged state).
- The POC's activation-visualization forward hook and TCP streaming client
  were dropped.
- Sequence position 0 is the oldest frame; "last frame" pooling uses index
  `-1`, matching Isaac Lab's oldest-to-newest history buffers.
