# models — RL model families for Isaac Lab training

This directory is the repository's model library. Every family follows the
same two-file pattern so Isaac Lab users can pick a model to train and
develop new training scripts against one contract:

| Family | Actor | Notes |
|---|---|---|
| `ppo` | `SpatialSoftmaxCNNModel` | Spatial-softmax keypoints per camera fused with joint positions; the default SO-101 actor. Also owns the shared PPO training stack. |
| `transformer_ppo` | `TransformerActorCritic` | Temporal transformer over per-camera frame-history windows; causal attention and explicit frame-difference features by default. |
| `mamba_ppo` | `MambaActorCritic` | Recurrent selective state-space actor over single-frame camera streams; pure-PyTorch SSM core, no `mamba_ssm` dependency. |

## The pattern

Each family subpackage provides:

- `models.py` — actor-critic classes subclassing RSL-RL's `MLPModel`, so
  they plug into `OnPolicyRunner` and support asymmetric PPO (separate
  privileged critic), observation groups, normalization, and
  TorchScript/ONNX export via `as_jit()`/`as_onnx()`.
- `ppo_cfg.py` — `@configclass` training configurations whose `class_name`
  dotted paths (e.g. `"models.ppo.models:SpatialSoftmaxCNNModel"`) select
  the classes above. Everything else in the cfg passes to the model
  constructor as keyword arguments.

`models/ppo` additionally owns the shared training classes:
`TensorBroadcastPPO` (distributed-safe PPO algorithm) and `VisualPPOCfg`
(shared rollout/critic/algorithm defaults). Every family derives its runner
configuration from `VisualPPOCfg`.

## Picking a model

Models are selected through task IDs. The SO-101 object-in-cup scenario is
registered once per actor family:

```bash
# Spatial-softmax CNN (default)
"$ISAACLAB_PYTHON" isaaclab/train --task SO101-Object-In-Cup-Vision-Fixed-v0

# Temporal transformer over the camera history window
"$ISAACLAB_PYTHON" isaaclab/train --task SO101-Object-In-Cup-Vision-Transformer-Fixed-v0

# Recurrent state-space mamba over single-frame camera streams
"$ISAACLAB_PYTHON" isaaclab/train --task SO101-Object-In-Cup-Vision-Mamba-Fixed-v0
```

All entry points (`train`, `train_multigpu`, `play`, `export`) work
unchanged for every family.

## Adding a new family

1. Create `models/<name>/` with `models.py` (subclass `MLPModel`; implement
   `get_latent` and `_get_latent_dim`, and `as_jit`/`as_onnx` if you need
   deployment exports) and `ppo_cfg.py` (model cfg deriving from
   `RslRlMLPModelCfg` with your architecture fields). For frame-history
   inputs, copy an existing temporal family (`transformer_ppo` or
   `mamba_ppo`) as the starting point — each family is fully self-contained
   by design.
2. Derive the task's runner cfg from `VisualPPOCfg`, swapping in your actor
   cfg, and register the task ID (see
   `so101_rl/tasks/object_in_cup/agents/rsl_rl_vision_ppo_cfg.py` for a
   worked example).
3. If your actor consumes camera history, set `history_length` on the
   camera observation terms (and `flatten_history_dim = False`) in the env
   cfg variant you register alongside it. Recurrent actors instead use the
   plain single-frame env cfgs and keep their state in the model (set
   `is_recurrent = True`, implement `reset`/`get_hidden_state`/
   `detach_hidden_state`, and pack the state as rank-3 tensors so RSL-RL's
   recurrent rollout storage can carry it — see `models/mamba_ppo`).

## Import path

The entry scripts put this directory's parent on `sys.path`, so families
import as the top-level `models` package (`from models.ppo import ...`).
`so101_rl.runtime` does the same for the re-exec'd Isaac Sim process.
