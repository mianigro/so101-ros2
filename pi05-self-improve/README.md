# pi05-self-improve — π0.5-style self-improvement loop for SO-101

This project implements the π0.5 recipe — *knowledge-insulated self-improvement
through autonomous data* — on the SO-101 stack: a flow-matching VLA baseline
trained by behavioral cloning, which then **generates its own training data**
(autonomous rollouts), **evaluates** it (scripted oracle in sim, VLM judge on
the real robot), **filters out failures**, and **retrains on its own
successes co-trained with human data** — round after round.

```
            ┌──────────────────────────────────────────────────┐
            │                                                  │
  human     ▼   generate            evaluate        refine    │
 teleop ─────► π (BC baseline) ──► rollouts ──► success filter ─┘
  data          round 0            (sim/real)   + teleop co-train
                                   judge            │
                                                    ▼
                                              π round N (BC retrain)
```

Nothing here replaces the existing repo pieces — it orchestrates them:

| Loop stage | Reuses | Adds |
|---|---|---|
| Baseline BC | `lerobot-train` (pixi `lerobot` env), `lerobot/pi05_base` | round config + wrapper |
| Sim rollouts | `isaaclab/` SO-101 task (`SO101-Object-In-Cup-Vision-Fixed-v0`), `so101_rl` runtime bootstrap, calibrated 3-camera rig | VLA rollout client, chunked-action execution mirroring the async real-robot client, npz episode recording |
| Success judging | the RL env's scripted `stable_placement` oracle | Qwen3-VL judge for real episodes |
| Real rollouts | `follower_recording.launch.py`, `async_infer.launch.py`, `policy_server`, `teleop_episode_keyboard`, `rosbag_to_lerobot` | supervised-assist session wrapper + VLM judging |
| Dataset | LeRobot v3.0 writer (as in `rosbag_to_lerobot`) | success filtering + teleop mixing |
| Retraining | `lerobot-train` | round-N fine-tune from the selected base or previous checkpoint |

## Phase 1 — the baseline (mimicry and its limits)

π0.5 is a ~3B VLA (PaliGemma backbone + Gemma action expert) trained with
flow matching on human teleoperation: it predicts the action trajectory a
human would take. Behavioral cloning alone suffers from covariate shift —
one small error puts the robot into a state the human data never covered,
and the policy has never learned to recover. The baseline is therefore only
the *starting point* of the loop, not the deliverable.

Round 0 fine-tunes `lerobot/pi05_base` on the human teleop dataset:

```bash
pixi shell -e lerobot
python pi05-self-improve/train_bc.py --config pi05-self-improve/configs/round_0.yaml --dry-run  # inspect
python pi05-self-improve/run_round.py --config pi05-self-improve/configs/round_0.yaml --stages train,eval
```

This is plain `lerobot-train --policy.path=lerobot/pi05_base` under the hood;
round checkpoints stay local by default (`--policy.push_to_hub=false`).
The round's checkpoint lands in `rounds/round_0/checkpoint/run/checkpoints/<step>/`
and its sim success rate in `rounds/round_0/metrics_eval.json`.

**VRAM note:** fine-tuning the 3B π0.5 needs a large GPU (≥40 GB with
`freeze_vision_encoder: true`, the default). On smaller cards the entire loop
runs unchanged with a lighter VLA — set `train.policy_type: smolvla` and
`train.base_repo_id: lerobot/smolvla_base` in the round config.

## Phase 2 — the self-improvement loop

Each round is **generate → evaluate → refine**:

1. **Generate** — the checkpoint drives the sim robot autonomously
   (`rollout_sim.py`), executing 50-step flow-matching chunks with the same
   temporal aggregation (`weighted_average`) as the real async inference
   nodes, so sim data matches real deployment semantics. Everything is
   recorded: 3 cameras at dataset resolution (480×640 @ 30 Hz), joint
   states, executed actions, task prompt.
2. **Evaluate** — in sim, ground truth comes from the RL task's scripted
   oracle (cube settled in the cup, gripper released, 15 consecutive steps
   → termination term `success`). On the real robot, Qwen3-VL-2B judges the
   final frames against the task text, and a human confirms the verdicts.
3. **Refine** — successes are encoded into a LeRobot v3.0 dataset mixed with
   the original teleop data (co-training anchor), and the policy is
   BC-retrained from the round's selected input policy. Because the robot now
   learns from *its own successful executions and recoveries*, it acquires
   the recovery skills pure mimicry never showed it.

Run a full sim round (the wrapper starts/stops the GPU policy server around
each stage so rollouts and training never compete for memory):

```bash
pixi shell -e lerobot
python pi05-self-improve/run_round.py --config pi05-self-improve/configs/round_1.yaml
```

`configs/round_1.yaml` needs `checkpoint_in` pointing at round 0's
checkpoint directory.

### Start Phase 2 directly from the base model

Phase 1 is optional. To generate the first autonomous round directly with
`lerobot/pi05_base`, use the standalone config:

```bash
pixi shell -e lerobot
python pi05-self-improve/train_bc.py \
    --config pi05-self-improve/configs/round_1_from_base.yaml --dry-run
python pi05-self-improve/run_round.py \
    --config pi05-self-improve/configs/round_1_from_base.yaml
```

Add `--visualizer kit` to watch rollout and evaluation episodes in the native
Isaac Sim window. Without it, simulation remains headless for faster batch data
generation:

```bash
python pi05-self-improve/run_round.py \
    --config pi05-self-improve/configs/round_1_from_base.yaml \
    --stages rollout \
    --visualizer kit \
    --policy-device cuda:1 \
    --sim-device cuda:0
```

On a two-GPU machine this keeps the π0.5 policy server on GPU 1 and Isaac Sim
on GPU 0, so their VRAM allocations do not compete.

This skips the Phase 1 checkpoint, not the human-data anchor: the refine stage
still trains on the successful autonomous rollouts mixed with the datasets in
`dataset.teleop_repo_ids`. The config also supplies the SO-101-to-OpenPI camera
mapping required by the raw base checkpoint. The same mapping is passed to
`lerobot-train` as `rename_map` and remains valid for evaluation of the new
checkpoint. If the base policy produces fewer than `judge.min_successes`
successful episodes, dataset construction stops instead of training on an
undersized autonomous sample.

Every stage can also run alone:

```bash
# Serve a checkpoint for rollouts/eval (pixi lerobot env, GPU)
python pi05-self-improve/serve_policy.py --repo-id rounds/round_0/checkpoint/run/checkpoints/20000

# Autonomous rollouts (boots Isaac Sim like the isaaclab/ entry points)
pi05-self-improve/rollout_sim.py --config pi05-self-improve/configs/round_1.yaml \
    --rounds-root pi05-self-improve/rounds

# Success-rate measurement only, no recording
python pi05-self-improve/eval_policy.py --repo-id <checkpoint> --num-episodes 24

# Filter successes + mix teleop + write the round dataset
python pi05-self-improve/build_round_dataset.py --config pi05-self-improve/configs/round_1.yaml

# BC retrain from the selected input policy on the mixed dataset
python pi05-self-improve/train_bc.py --config pi05-self-improve/configs/round_1.yaml
```

Round artifacts (all under `rounds/round_N/`, gitignored):

```
rounds/round_1/
├── config.yaml           # effective round configuration
├── rollouts_raw/         # every episode as compressed npz (kept, failures too)
├── rollouts.jsonl        # per-episode oracle verdicts {success, reason, ...}
├── verdicts.jsonl        # VLM verdicts (cross-check in sim, authoritative on real)
├── dataset_summary.json  # what went into the mixed dataset
├── checkpoint/           # lerobot-train run + checkpoints/<step>/
├── metrics_eval.json     # success rate + failure breakdown after this round
└── round_meta.json       # accumulated stage records
```

### Prerequisites

- Isaac Sim assets built exactly as for the RL project (`isaaclab/README.md`):
  `build/isaacsim_so101/…` robot/camera-rig USDs and `isaaclab/prepare_assets`
  cube/cup USDs — `rollout_sim.py` reuses the same `so101_rl.runtime` bootstrap.
- The pixi `lerobot` env (lerobot 0.6.1 with `pi` extras) — already in `pixi.toml`.
- The teleop dataset(s) listed under `dataset.teleop_repo_ids` reachable
  through the LeRobot cache or the Hub.

### Sim-to-real details worth knowing

- **Action space**: datasets record *absolute* calibrated joint positions
  (radians, `shoulder_pan … gripper`). The rollout loop replaces the RL
  env's delta-action term with absolute `JointPositionAction` targets in the
  same joint order, so the policy's output chunks are applied 1:1.
- **Gripper mapping**: the calibrated gripper range (≈[-0.56, 0.55] rad in
  dataset units) differs from the URDF/sim range ([0, 1.5] rad nominal).
  `JointMap` in `pi05_selfimprove/contract.py` carries this affine map
  (identity for the five arm joints) and is configurable per round via
  `rollout.joint_map`. Commands are clamped to sim joint limits.
- **Cameras**: the sim rig renders at dataset resolution (480×640) with the
  same calibrated intrinsics/aspect as the RL cameras, so the VLA sees the
  same image distribution it was fine-tuned on.
- **Disk**: each recorded episode is ~1.2 GiB raw (3×480×640 @ 30 Hz × 15 s);
  frames stream through memmaps and land as compressed npz (~10–50% of raw).
  A 48-episode round is on the order of tens of GiB under `rounds/`.

## Real-robot rounds (supervised-assist)

The real phase uses the same generate→evaluate→refine stages with the
*existing* robot stack; a human supervisor stays in the loop for safety.

```bash
pixi shell -e lerobot   # on the robot host

# 1) Supervised session: wrapper spawns follower_recording + async pi05
#    inference; you run teleop_episode_keyboard in another terminal and
#    press r/s/d around each autonomous episode.
python pi05-self-improve/real_rollout.py session \
    --config pi05-self-improve/configs/round_2.yaml --experiment pi05_selfimprove

# 2) Post-process: convert kept MCAP episodes (--dataset-source autonomous),
#    VLM-judge final frames, write verdicts.jsonl; --interactive to confirm
#    each verdict yourself.
python pi05-self-improve/real_rollout.py post \
    --config pi05-self-improve/configs/round_2.yaml --interactive

# 3) Same refine stages as sim rounds:
python pi05-self-improve/build_round_dataset.py --config pi05-self-improve/configs/round_2.yaml \
    --real-dataset-repo-id local/so101_pi05_round2_real
python pi05-self-improve/train_bc.py --config pi05-self-improve/configs/round_2.yaml
```

The GPU side serves the checkpoint to the robot with the existing
`policy_server` (see `policy_server/README.md`); point
`real_rollout session --policy-server-address` at it.

## VLM judge

`judge_rollouts.py` loads Qwen3-VL-2B locally (already in the HF cache) and
asks for a strict-JSON verdict over the final overhead+wrist frames:

```bash
# Cross-check the sim oracle (agreement rate lands in round_meta.json)
python pi05-self-improve/judge_rollouts.py --round-dir pi05-self-improve/rounds/round_1

# Authoritative judging of converted real episodes
python pi05-self-improve/judge_rollouts.py --round-dir pi05-self-improve/rounds/round_2 \
    --dataset-repo-id local/so101_pi05_round2_real --interactive
```

Sim training data always stays gated by the scripted oracle (ground truth);
the VLM verdicts are recorded alongside for analysis. Real training data is
gated by VLM + human confirmation.

## Tests

```bash
cd pi05-self-improve
python -m unittest discover -s tests            # fast, no lerobot needed
../.pixi/envs/lerobot/bin/python -m unittest discover -s tests   # full
```

The suite covers the wire protocol round-trip, the joint map math (including
a cross-check against `so101_rl.visual_contract`), the chunk-aggregation
buffer, npz episode recording, verdict parsing, the training command
assembly, and an end-to-end dataset build (npz + teleop → mixed LeRobot
v3.0 dataset with real video encoding).

## Layout

```
pi05-self-improve/
├── serve_policy.py      # GPU rollout policy server (pixi lerobot env)
├── rollout_sim.py       # autonomous sim rollouts (boots Isaac Sim)
├── run_round.py         # one-click round: rollout→build→train→eval
├── eval_policy.py       # success-rate measurement for a checkpoint
├── judge_rollouts.py    # Qwen3-VL judge (sim cross-check / real authoritative)
├── build_round_dataset.py  # filter successes + mix teleop → LeRobot dataset
├── train_bc.py          # round-0 baseline / round-N retrain wrapper
├── real_rollout.py      # supervised-assist real-robot session + post
├── configs/              # baseline, continuation, and base-start round YAMLs
├── pi05_selfimprove/
│   ├── contract.py       # joints, cameras, JointMap (dataset ↔ sim units)
│   ├── config.py         # RoundConfig
│   ├── wire.py           # stdlib-only client/server protocol
│   ├── rollout_server.py # policy serving (batched predict_action_chunk)
│   ├── sim_rollout.py    # rollout loop, recorder, oracle classification
│   ├── judges/           # scripted oracle + Qwen3-VL judge
│   ├── dataset_tools/    # success filtering + mixing
│   ├── train_bc.py       # lerobot-train wrapper
│   ├── real_rollout.py   # real-phase orchestration
│   └── loop.py           # run_round orchestration (server lifecycle)
└── tests/
```
