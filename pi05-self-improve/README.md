# π0.5 self-improvement for SO-101

This project implements the π0.5 self-improvement recipe on the SO-101 stack.
A policy first learns from human teleoperation, then collects autonomous
rollouts, keeps successful episodes, mixes them with the original human data,
and retrains. Repeating that loop teaches the policy from states and recoveries
that are missing from pure behavioral-cloning data.

The workflow has two phases:

```text
Phase 1 — human data and baseline
    human teleop data → behavioral-cloning baseline (round 0)

Phase 2 — self-improvement rounds
    checkpoint → generate → evaluate → refine → next checkpoint
                       │          │
                       ├─ A: Isaac Sim + scripted oracle
                       └─ B: real robot + VLM and human confirmation
```

Each Phase 2 round uses **one** rollout option: Isaac Sim or the real robot.
The two options generate and judge data differently, then rejoin at the same
dataset-build and BC-retraining steps.

---

## Requirements

### Common

- Complete the main repository [installation](../README.md#installation) and
  build the ROS 2 workspace.
- Install the Pixi `lerobot` environment. It provides LeRobot 0.6.1 with the
  π0.5 extras and is already defined in `pixi.toml`.
- Make the human teleoperation dataset listed in
  `dataset.teleop_repo_ids` available through the LeRobot cache or Hub.
- Use a GPU with enough memory for the selected policy.

Fine-tuning the 3B π0.5 model generally needs at least 40 GB of VRAM with
`freeze_vision_encoder: true`, the default. On a smaller GPU, the workflow can
use SmolVLA by setting `train.policy_type: smolvla` and
`train.base_repo_id: lerobot/smolvla_base` in each round config.

### Option A — Isaac Sim

- Build the Isaac Sim assets described in
  [`isaaclab/README.md`](../isaaclab/README.md): the SO-101 robot and camera-rig
  USDs under `build/isaacsim_so101/`, plus the cube and cup assets produced by
  `isaaclab/prepare_assets`.
- Allow sufficient disk space for rollouts. One 15-second episode with three
  480×640 cameras is about 1.2 GiB before compression; a 48-episode round can
  require tens of GiB.

### Option B — real robot

- Complete the [hardware setup](../docs/hardware.md), including motor
  calibration and udev rules, before commanding the SO-101.
- Use a calibrated follower and the cameras defined by the selected setup.
- Keep a human supervisor beside the robot throughout every autonomous
  episode.
- Run a reachable policy server on the GPU host.

## Initial setup

Run commands from the repository root unless a section says otherwise:

```bash
cd /home/anon/Documents/so101-ros2
source /opt/ros/jazzy/setup.bash
source install/setup.bash
pixi shell -e lerobot
```

---

## Phase 1 — human data and baseline

Round 0 fine-tunes `lerobot/pi05_base` on human demonstrations. This baseline
is the starting point of self-improvement, not the final policy: behavioral
cloning alone does not show the policy how to recover after it leaves the
states covered by the demonstrations.

### Step 1 — collect and convert human demonstrations

Use the main README to [collect teleoperated episodes](../README.md#data-collection)
on either the physical follower or Isaac Sim, then
[convert them to a LeRobot dataset](../README.md#lerobot-dataset-conversion).
These are human-controlled demonstrations; the sim/real choice for autonomous
self-improvement comes later in Phase 2.

If an existing LeRobot dataset will be used, make sure it is available locally
or from the Hub.

### Step 2 — configure round 0

Edit `pi05-self-improve/configs/round_0.yaml` and set:

- `dataset.teleop_repo_ids` to the human dataset used for the baseline and as
  the co-training anchor in later rounds.
- `train.base_repo_id` and `train.policy_type` if π0.5 is not being used.
- The rollout task, camera mapping, and evaluation settings when the baseline
  will be evaluated in Isaac Sim.

`round_index: 0`, `train.init_from: base`, and `checkpoint_in: null` identify
this as baseline training from the published base model.

### Step 3 — inspect, train, and evaluate

First inspect the generated `lerobot-train` command without starting a job:

```bash
python pi05-self-improve/train_bc.py \
  --config pi05-self-improve/configs/round_0.yaml \
  --dry-run
```

Train the baseline and evaluate it in Isaac Sim:

```bash
python pi05-self-improve/run_round.py \
  --config pi05-self-improve/configs/round_0.yaml \
  --stages train,eval
```

If Isaac Sim is not available because Phase 2 will use only the real robot,
train without the sim-only evaluation stage:

```bash
python pi05-self-improve/run_round.py \
  --config pi05-self-improve/configs/round_0.yaml \
  --stages train
```

The checkpoint is written under
`pi05-self-improve/rounds/round_0/checkpoint/run/checkpoints/<step>/`. Sim
evaluation metrics are written to
`pi05-self-improve/rounds/round_0/metrics_eval.json`.

---

## Phase 2 — self-improvement rounds

Each self-improvement round performs three operations:

1. **Generate** autonomous rollouts in Isaac Sim or on the supervised real
   robot.
2. **Evaluate** each rollout with the sim oracle or the real-robot VLM judge
   plus human confirmation.
3. **Refine** by filtering successes, mixing them with the human teleop data,
   and BC-retraining from the selected input checkpoint.

### Step 1 — configure the next round

Use `configs/round_1.yaml` as the Isaac Sim example or
`configs/round_2_real.yaml` as the real-robot example. Before running either
one, check these fields:

| Field | Required value |
|---|---|
| `round_index` | A new round number; do not reuse an existing artifact directory |
| `phase` | `sim` for Option A or `real` for Option B |
| `checkpoint_in` | The previous checkpoint, for example `pi05-self-improve/rounds/round_0/checkpoint/run/checkpoints/<step>` |
| `train.init_from` | `previous` when continuing from `checkpoint_in` |
| `dataset.teleop_repo_ids` | The human-data anchor mixed into this round |
| `judge.name` | `scripted` for sim or `vlm` for real rollouts |

If the first self-improvement round runs on the real robot, adapt the real
example to `round_index: 1` and point `checkpoint_in` at the round-0 baseline.

### Step 2 — collect autonomous rollouts

Choose one of the following options for this round.

#### Option A — Isaac Sim rollouts

Run only the autonomous data-collection stage:

```bash
python pi05-self-improve/run_round.py \
  --config pi05-self-improve/configs/round_1.yaml \
  --stages rollout
```

`run_round.py` starts and stops the rollout policy server automatically. The
checkpoint drives the sim robot with chunked actions, while every episode's
three camera streams, joint states, executed actions, task prompt, and scripted
oracle verdict are recorded.

To watch the rollout in the native Isaac Sim window:

```bash
python pi05-self-improve/run_round.py \
  --config pi05-self-improve/configs/round_1.yaml \
  --stages rollout \
  --visualizer kit
```

On a two-GPU machine, keep the policy and simulation allocations separate:

```bash
python pi05-self-improve/run_round.py \
  --config pi05-self-improve/configs/round_1.yaml \
  --stages rollout \
  --policy-device cuda:1 \
  --sim-device cuda:0
```

To run rollout, refinement, and sim evaluation in one command instead, omit
`--stages`:

```bash
python pi05-self-improve/run_round.py \
  --config pi05-self-improve/configs/round_1.yaml
```

#### Option B — real-robot rollouts

Real rounds use the existing ROS 2 recording and async-inference stack. They
do **not** support the `run_round.py` `rollout` or `eval` stages. Collection is
a supervised session followed by conversion and judging.

This workflow uses three terminals.

**Terminal 1 — policy server on the GPU host:**

```bash
cd /home/anon/Documents/so101-ros2
pixi run -e lerobot python policy_server/zmq_server.py \
  --host 0.0.0.0 \
  --port 8090
```

The async client loads the checkpoint named by the real-round config. Port
`8090` is specified explicitly because the server's default port is `5555`.

**Terminal 2 — follower, cameras, recording, and inference on the robot
host:**

```bash
cd /home/anon/Documents/so101-ros2
source /opt/ros/jazzy/setup.bash
source install/setup.bash
pixi shell -e lerobot

export POLICY_GPU_HOST=192.168.1.100

python pi05-self-improve/real_rollout.py session \
  --config pi05-self-improve/configs/round_2_real.yaml \
  --experiment pi05_selfimprove \
  --policy-server-address "${POLICY_GPU_HOST}:8090"
```

Set `POLICY_GPU_HOST` to the GPU machine's hostname or IP address. Use
`127.0.0.1` when the policy server and robot stack run on the same machine.

Wait for the launch processes to become ready before starting the keyboard
controller.

**Terminal 3 — supervised episode controls:**

```bash
source /opt/ros/jazzy/setup.bash
source /home/anon/Documents/so101-ros2/install/setup.bash
ros2 run episode_recorder teleop_episode_keyboard
```

The supervisor controls each autonomous episode:

- **r** or **Right Arrow**: start recording.
- **s** or **Left Arrow**: stop and keep the episode.
- **d** or **Backspace**: discard the episode.
- **q**: quit the keyboard controller.

Watch the arm continuously and stop the session immediately if the motion is
unsafe. End Terminal 2 with `Ctrl-C` when collection is complete.

Post-process the kept episodes on the robot host. This converts the MCAP
recordings to a LeRobot dataset, asks Qwen3-VL for verdicts, and lets the human
confirm or correct every verdict:

```bash
python pi05-self-improve/real_rollout.py post \
  --config pi05-self-improve/configs/round_2_real.yaml \
  --repo-id local/so101_pi05_round2_real \
  --interactive
```

The default input directory is
`~/.ros/so101_episodes/pi05_selfimprove/`. Pass `--input-dir` when a different
experiment directory was used.

### Step 3 — evaluate and filter the rollouts

For **Option A**, the Isaac Lab task writes its scripted `stable_placement`
verdict during rollout: the cube must settle in the cup, the gripper must be
released, and the condition must hold for 15 consecutive steps. These verdicts
in `rollouts.jsonl` remain authoritative for the sim training dataset. The VLM
can optionally cross-check them:

```bash
python pi05-self-improve/judge_rollouts.py \
  --round-dir pi05-self-improve/rounds/round_1
```

For **Option B**, the `real_rollout.py post --interactive` command writes the
VLM verdicts and human approvals to `verdicts.jsonl`. Real episodes enter the
training dataset only after approval.

Both paths enforce `judge.min_successes`. Dataset construction stops instead
of training on an undersized autonomous sample when too few rollouts succeed.

### Step 4 — build the mixed dataset and retrain

For an Isaac Sim round:

```bash
python pi05-self-improve/run_round.py \
  --config pi05-self-improve/configs/round_1.yaml \
  --stages build,train
```

For a real-robot round, pass the converted autonomous dataset from the post
step:

```bash
python pi05-self-improve/run_round.py \
  --config pi05-self-improve/configs/round_2_real.yaml \
  --stages build,train \
  --real-dataset-repo-id local/so101_pi05_round2_real
```

Both commands filter approved autonomous successes, mix them with the datasets
in `dataset.teleop_repo_ids`, and fine-tune from the round's selected input
policy.

### Step 5 — evaluate and repeat

Evaluate a newly trained sim round with the scripted oracle:

```bash
python pi05-self-improve/run_round.py \
  --config pi05-self-improve/configs/round_1.yaml \
  --stages eval
```

Real-robot evaluation remains supervised and uses the real session workflow;
`run_round.py --stages eval` is sim-only.

For the next round, increment `round_index`, point `checkpoint_in` at the new
checkpoint, select `phase: sim` or `phase: real`, and repeat Phase 2. A later
round may switch from simulation to the real robot, as long as its config and
judge match the selected rollout option.

---

## Reference

### Run individual stages

The wrappers above are the normal workflow. These commands expose the
individual sim and refinement stages for debugging or reruns:

```bash
CHECKPOINT=pi05-self-improve/rounds/round_0/checkpoint/run/checkpoints/20000

# Serve a checkpoint for sim rollout clients
python pi05-self-improve/serve_policy.py \
  --repo-id "$CHECKPOINT"

# Collect autonomous Isaac Sim rollouts; serve_policy.py must already be running
python pi05-self-improve/rollout_sim.py \
  --config pi05-self-improve/configs/round_1.yaml \
  --rounds-root pi05-self-improve/rounds

# Measure sim success rate without recording episodes
python pi05-self-improve/eval_policy.py \
  --repo-id "$CHECKPOINT" \
  --num-episodes 24

# Filter successes, mix teleop data, and write the round dataset
python pi05-self-improve/build_round_dataset.py \
  --config pi05-self-improve/configs/round_1.yaml

# Retrain from the input policy on the mixed dataset
python pi05-self-improve/train_bc.py \
  --config pi05-self-improve/configs/round_1.yaml
```

### Round artifacts

All generated artifacts are gitignored under
`pi05-self-improve/rounds/round_N/`:

```text
rounds/round_1/
├── config.yaml           # effective round configuration
├── rollouts_raw/         # every sim episode as compressed npz, including failures
├── rollouts.jsonl        # per-episode sim oracle verdicts
├── verdicts.jsonl        # VLM verdicts; authoritative with human approval for real data
├── dataset_summary.json  # inputs to the mixed dataset
├── checkpoint/           # lerobot-train run and checkpoints/<step>/
├── metrics_eval.json     # sim success rate and failure breakdown
└── round_meta.json       # accumulated stage records
```

### Sim-to-real contract

- **Action space:** datasets record absolute calibrated joint positions in
  `shoulder_pan … gripper` order. The sim rollout replaces the RL environment's
  delta action term with absolute `JointPositionAction` targets, so policy
  chunks are applied directly.
- **Temporal aggregation:** sim uses the same chunked-action execution and
  `weighted_average` aggregation as the real async-inference client.
- **Gripper mapping:** the calibrated dataset range, approximately
  `[-0.56, 0.55]` radians, differs from the nominal sim range `[0, 1.5]`.
  `JointMap` in `pi05_selfimprove/contract.py` applies the configurable affine
  mapping and clamps commands to sim joint limits.
- **Cameras:** the sim rig renders three cameras at the dataset resolution of
  480×640 and 30 Hz with calibrated intrinsics and aspect ratio.

### VLM judge

`judge_rollouts.py` loads Qwen3-VL-2B locally and requests a strict JSON
verdict from the final overhead and wrist frames:

```bash
# Cross-check the sim oracle; agreement is stored in round_meta.json
python pi05-self-improve/judge_rollouts.py \
  --round-dir pi05-self-improve/rounds/round_1

# Judge converted real episodes and confirm each verdict
python pi05-self-improve/judge_rollouts.py \
  --round-dir pi05-self-improve/rounds/round_2 \
  --dataset-repo-id local/so101_pi05_round2_real \
  --interactive
```

Sim training data always stays gated by the scripted oracle. Real training
data is gated by the VLM verdict plus human confirmation.

### Implementation map

| Loop stage | Reuses | Adds |
|---|---|---|
| Baseline BC | `lerobot-train`, `lerobot/pi05_base` | Round config and training wrapper |
| Sim rollouts | Isaac Lab SO-101 task, `so101_rl` bootstrap, calibrated three-camera rig | VLA rollout client, chunked execution, NPZ episode recording |
| Real rollouts | ROS 2 follower recording, async inference, policy server, episode keyboard, LeRobot conversion | Supervised-assist session and VLM judging |
| Success judging | Isaac Lab `stable_placement` oracle | Qwen3-VL judge and human confirmation for real episodes |
| Dataset | LeRobot v3.0 writer | Success filtering and teleop mixing |
| Retraining | `lerobot-train` | Round-N fine-tuning from the selected checkpoint |

### Tests

```bash
cd pi05-self-improve
python -m unittest discover -s tests
../.pixi/envs/lerobot/bin/python -m unittest discover -s tests
```

The suite covers the wire protocol, joint mapping, action-chunk aggregation,
NPZ episode recording, verdict parsing, training-command assembly, and the
end-to-end mixed-dataset build.

### Layout

```text
pi05-self-improve/
├── serve_policy.py          # GPU policy server for Isaac Sim rollouts
├── rollout_sim.py           # autonomous Isaac Sim rollouts
├── run_round.py             # sim round orchestration; shared build/train stages
├── eval_policy.py           # sim success-rate measurement
├── judge_rollouts.py        # Qwen3-VL judge
├── build_round_dataset.py   # filter successes and mix teleop data
├── train_bc.py              # baseline and round-N BC training wrapper
├── real_rollout.py          # supervised real session and post-processing
├── configs/                 # baseline, sim, base-start, and real round YAMLs
├── pi05_selfimprove/
│   ├── contract.py          # joint, camera, and sim/dataset mappings
│   ├── config.py            # round configuration
│   ├── wire.py              # sim rollout client/server protocol
│   ├── rollout_server.py    # batched policy serving
│   ├── sim_rollout.py       # rollout loop, recorder, and oracle classification
│   ├── judges/              # scripted and Qwen3-VL judges
│   ├── dataset_tools/       # filtering and dataset mixing
│   ├── train_bc.py          # lerobot-train integration
│   ├── real_rollout.py      # real-phase orchestration
│   └── loop.py              # run_round orchestration and server lifecycle
└── tests/
```
