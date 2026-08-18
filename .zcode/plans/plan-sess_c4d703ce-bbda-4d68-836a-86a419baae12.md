# Plan: `isaaclab/models/` — shared RL model + training-class library

Turn `isaaclab/models/` into the repo's model library where each family provides **model classes + training classes (algorithm + runner configs)** following one pattern. Task package `so101_rl` keeps only SO-101 glue (obs groups, task presets, gym registrations) and imports from `models`.

## Target layout

```
isaaclab/models/
├── __init__.py
├── README.md                 # the pattern: how to add/pick a model family
├── ppo/                      # Family 1: current stack, abstracted from so101_rl
│   ├── models.py             # SpatialSoftmax, SpatialSoftmaxCNNModel, jit/onnx wrappers (moved verbatim)
│   ├── distributed_ppo.py    # TensorBroadcastPPO + cfg (moved; shared by ALL families)
│   └── ppo_cfg.py            # RslRlSpatialSoftmaxCNNModelCfg + task-agnostic VisualPPOCfg base
└── transformers_ppo/         # Family 2: temporal transformer/mamba actors (POC, restructured)
    ├── models.py             # ported architectures as rsl_rl actor-critics
    ├── ppo_cfg.py            # RslRlTransformerActorCfg, RslRlMambaActorCfg (+ runner base)
    └── README.md
```

Import mechanics: `isaaclab/models/` becomes a top-level package `models` by adding `sys.path.insert(0, PROJECT_ROOT)` to the entry scripts (`train`, `play`, `export`, `live`, `test`; `train_multigpu` spawns `train`). `class_name` dotted paths become e.g. `"models.ppo.models:SpatialSoftmaxCNNModel"`.

## Step 1 — Abstract current PPO into `models/ppo/`

1. Move `so101_rl/tasks/common/agents/models.py` → `models/ppo/models.py` and `distributed_ppo.py` → `models/ppo/distributed_ppo.py` **verbatim** (code unchanged, only module path).
2. `models/ppo/ppo_cfg.py`:
   - `RslRlSpatialSoftmaxCNNModelCfg` with `class_name = "models.ppo.models:SpatialSoftmaxCNNModel"`.
   - New task-agnostic `VisualPPOCfg(RslRlOnPolicyRunnerCfg)` holding the current shared rollout/PPO defaults (`num_steps_per_env=48`, `save_interval`, `clip_actions`, critic MLP cfg, `algorithm=TensorBroadcastPPOCfg(...)` with today's hyperparams). `TensorBroadcastPPOCfg.class_name` updated to the new path.
3. `so101_rl/tasks/common/agents/rsl_rl_ppo_cfg.py` shrinks to `SO101VisualPPOCfg(VisualPPOCfg)` adding only SO-101 glue: `obs_groups` (SO-101 actor groups + critic state) and the spatial-softmax actor cfg. Delete the old `models.py`/`distributed_ppo.py` there; `agents/__init__.py` keeps exporting `SO101VisualPPOCfg`. Task presets in `object_in_cup`/`three_boxes_in_cups` keep importing `SO101VisualPPOCfg` — unchanged.
4. Update `tests/test_visual_policy_contract.py` imports (`models.ppo.models:SpatialSoftmaxCNNModel`).
5. Checkpoint-safe: parameter names are unchanged and `class_name` resolves from current configs, so existing checkpoints still `--resume`/load.

## Step 2 — Restructure `models/transformers_ppo/` (delete Atari scripts)

**Delete** (Atari/gym-specific, recoverable from git history): `train.py`, `run_eval.py`, `requirements.txt`, `config/` (main yaml + all 16 per-game yamls), `src/agent/` (PPOAgent, reward_shaping), `src/training/`, `src/utils/` (memories, frame_preprocessing, streaming_client).

**Port** to `models/transformers_ppo/models.py` as rsl_rl-native actor-critics (rsl_rl `OnPolicyRunner` + `TensorBroadcastPPO` replace PPOAgent; distribution/value heads come from the rsl_rl `MLPModel` base + asymmetric MLP critic, matching the existing SO-101 pattern):
- Keep verbatim: `TemporalPositionalEncoding`, `MultiHeadAttention`, `EncoderLayer`, orthogonal init.
- Port `CNNEncoder` (ResNet-style) with configurable `in_channels` (RGB 3-channel, per-camera instances).
- `TransformerActorCritic(MLPModel)` mirroring `SpatialSoftmaxCNNModel`: per-camera encode of the history window `(B, T·C, H, W) → (B, T, C, H, W)` → CNN → positional encoding → encoder layers → `final_layer_pooling`/`final_pool_skip` reduction → concat with MLP-encoded `joint_state` → `mlp` head → distribution. Implements `get_latent(obs, masks, hidden_state)`, `_get_latent_dim()`, `as_jit()`, `as_onnx()` (same ONNX input-name contract, images now carry the history window).
- `MambaActorCritic` — same contract; `mamba_ssm` imported lazily so the transformer path works without it (clear error only if the mamba model is instantiated).
- `ppo_cfg.py`: `RslRlTransformerActorCfg` / `RslRlMambaActorCfg` with `class_name` into `models.transformers_ppo.models` and the POC's architecture params (`lookback_frames`, `d_model`, `num_heads`, `num_layers`, `d_ff`, `dropout`, pooling flags, `cnn_cfg`).

## Step 3 — SO-101 example presets (pick-a-model wiring)

In `so101_rl/tasks/object_in_cup/`:
- Env variant `SO101ObjectInCupVision*TransformerEnvCfg` subclassing the existing env cfgs; only change: set `history_length=lookback_frames` on the three camera observation terms (verify exact `ObservationTermCfg` history fields against the installed Isaac Lab version during implementation).
- Runner presets `SO101ObjectInCupVisionTransformerPPOCfg` / `...MambaPPOCfg` (`SO101VisualPPOCfg` with the actor swapped).
- Register task IDs `SO101-Object-In-Cup-Vision-Transformer-v0` (+ `-Fixed-`) and `SO101-Object-In-Cup-Vision-Mamba-v0` (+ `-Fixed-`), so `./train --task <id>` trains end-to-end and swapping architectures is purely a task-ID/config choice.

## Step 4 — Docs + tests

- `isaaclab/models/README.md`: the family pattern (models.py + ppo_cfg.py), how tasks compose presets, how to add a new family; update `isaaclab/README.md` usage with new task IDs.
- New `tests/test_transformer_models.py` (pure torch, mirrors the contract-test style): forward/action shapes with history-shaped dummy obs, TorchScript `as_jit` parity, cfg instantiation; mamba test `skipUnless(mamba_ssm available)`.

## Verification

- `./test` (repo's own suite; bootstraps the source-built Isaac Sim) after Step 1 and again at the end.
- Import/smoke checks that all `class_name` dotted paths resolve.
- Full GPU training run left as a manual user step (long-running); export-script parity for history obs is covered by the as_jit/as_onnx wrapper test.

Implementation order: Step 1 → run tests → Steps 2–4 → run tests. All work stays on `dev/adding-trans-ppo`; no commits unless requested.