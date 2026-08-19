# mamba_ppo

Recurrent selective state-space actor for visual PPO: a streaming policy
that consumes single camera frames and integrates them over time through a
constant-size hidden state. The selective SSM core (Mamba/S6 math) is
implemented in plain PyTorch — **no `mamba_ssm` dependency** — which also
makes TorchScript/ONNX export work for this family.

The family follows the shared pattern: `models.py` holds the actor class and
`ppo_cfg.py` holds the `@configclass` training configuration
(`RslRlMambaActorCfg`) whose `class_name` dotted path selects it. The PPO
algorithm, rollout settings, and privileged MLP critic come from the shared
`models/ppo` stack (`VisualPPOCfg` + `TensorBroadcastPPO`).

## How the recurrence works

- Cameras are observed as single frames `(num_envs, channels, height, width)`
  — the env needs no camera history. The per-frame encoder (ResNet-style CNN
  with spatial-softmax reduction, shared design with `transformer_ppo`)
  embeds each frame per camera.
- Each camera runs a stack of selective SSM layers (default
  `d_model=64, num_layers=2, d_state=16, d_conv=4, expand=2`) with
  post-LayerNorm and a whole-stack residual. The recurrent state is the SSM
  state (plus the causal-conv window) — temporal memory is the model's, not
  the observation's.
- The hidden state is packed into one `[1, num_envs, state_dim]` tensor so
  RSL-RL's recurrent rollout storage carries it: captured before each step,
  reset to zeros on terminations, threaded across rollout boundaries, and
  replayed per trajectory during PPO updates (`is_recurrent = True` selects
  the recurrent minibatch generator). Gradients flow through the whole
  rollout window (BPTT over `num_steps_per_env`).
- One code path serves stepping and training: a sequence scan is literally
  the same recurrence as stepping frame by frame, and the tests assert their
  outputs agree. Zero state is a valid sequence start for both the conv
  window (zero padding) and the SSM state (scan initialisation).

## Export

`as_jit()` returns a TorchScript actor holding the packed state in a buffer
(`reset()` zeroes it; the state resizes if the served batch changes).
`as_onnx()` exposes the packed state as explicit `recurrent_state_in` /
`recurrent_state_out` tensors alongside the usual observation inputs, so a
deployment loop feeds the state output back in each step.

## Performance note

The SSM scan runs as a Python loop over time (vectorised over the batch).
Rollout stepping is one iteration per env step; PPO updates loop over
`num_steps_per_env` per minibatch forward. This favours correctness and
portability over throughput — swapping in fused scan kernels later only
touches `SelectiveSSM.forward`.
