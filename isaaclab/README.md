# SO-101 three-camera PPO

This directory is a repository-owned Isaac Lab project for training a visual
SO-101 object-in-cup policy with RSL-RL PPO. The deployable actor receives three
RGB images and six measured joint positions. A separate training-only critic
receives exact simulator task state and is never exported to ROS 2.

Isaac Lab supplies simulation, rendering, managers, and the RSL-RL runtime.
This project owns the task, assets, camera contract, models, PPO configuration,
export contract, and entry points. For reward formulas, network details, PPO
parameters, design rationale, and extension instructions, see
[`METHODOLOGY.md`](METHODOLOGY.md).


## Actor and critic boundary

The deployable actor receives exactly these observation groups:

```text
joint_state  [N, 6]            absolute joint positions in radians
wrist       [N, 3, 120, 160]  RGB / 255 - 0.5
overhead_1  [N, 3, 120, 160]  RGB / 255 - 0.5
overhead_2  [N, 3, 120, 160]  RGB / 255 - 0.5
```

The actor does not receive object pose, cup pose, object velocity, previous
action, reward state, or success flags. The training-only critic receives one
noiseless 34-value `critic_state` group:

| Component | Width |
|---|---:|
| Absolute joint positions | 6 |
| Joint velocities | 6 |
| Last requested action | 6 |
| Gripper-to-object delta | 3 |
| Object-to-cup delta | 3 |
| Object quaternion | 4 |
| Object linear and angular velocity | 6 |
| **Total** | **34** |

This is asymmetric PPO: exact simulator information improves the value estimate
and actor learning, but cannot enter the exported actor. Simulator state also
drives rewards, terminations, resets, and physics.

## Prerequisites

The commands below assume:

- repository: `/home/anon/Documents/so101-ros2`;
- Isaac Lab: `/home/anon/Documents/IsaacLab`;
- source-built Isaac Sim: `/home/anon/Documents/isaacsim`;
- camera profile: `dual_overhead`;
- a CUDA GPU for visual training and deployment;
- `cube.stl` is manipulated and `cup.stl` remains fixed during an episode;
- the generated repository SO-101 USD is authoritative.

Start from the repository root:

```bash
cd /home/anon/Documents/so101-ros2
export ISAACLAB_PYTHON=/home/anon/Documents/IsaacLab/.venv/bin/python
```

The entry points automatically switch to the configured source-built Isaac Sim
runtime when required.

## Registered tasks

| Task | Purpose |
|---|---|
| `SO101-Object-In-Cup-Vision-Fixed-v0` | Nominal visual task for smoke checks and initial overfitting |
| `SO101-Object-In-Cup-Vision-v0` | Randomized visual task for robust training and deployment |

The intended lifecycle is:

```text
prepare assets
  -> inspect the fixed task
  -> train and accept a fixed checkpoint
  -> resume on the randomized task
  -> evaluate held-out randomization
  -> export
  -> run real inputs in shadow mode
  -> arm only after every gate passes
```

## 1. Prepare and validate assets

The visual task uses these teleop-generated assets:

```text
build/isaacsim_so101/so101_follower/so101_follower.usda
build/isaacsim_so101/camera_rig/cam_mount_bottom.usd
build/isaacsim_so101/camera_rig/cam_mount_top.usd
```

If they are missing, build and source the ROS workspace, then generate them:

```bash
source /opt/ros/jazzy/setup.bash
source /home/anon/Documents/so101-ros2/install/setup.bash

/home/anon/Documents/isaacsim/_build/linux-x86_64/release/python.sh \
  /home/anon/Documents/so101-ros2/scripts/isaac_sim_teleop.py \
  --camera-profile dual_overhead \
  --headless --max-frames 1
```

Use `--rebuild-asset` only after changing robot Xacro/meshes, camera mount
meshes, or camera rig configuration.

Prepare and validate the cube and cup:

```bash
"$ISAACLAB_PYTHON" isaaclab/prepare_assets
"$ISAACLAB_PYTHON" isaaclab/prepare_assets --validate-only
```

Expected outputs:

```text
build/isaaclab_assets/cube.usda
build/isaaclab_assets/cup.usda
build/isaaclab_assets/manifest.json
```

Repeat preparation after changing either STL or regenerating the robot USD.

## 2. Inspect the fixed task

From a graphical desktop terminal, run:

```bash
"$ISAACLAB_PYTHON" isaaclab/live --num_envs 4
```

`live` defaults to `SO101-Object-In-Cup-Vision-Fixed-v0` and opens the native
Isaac Sim visualizer. Before training, verify:

- robot, table, cube, and open cup geometry;
- the cube fits through the cup opening;
- wrist camera attachment and both overhead camera poses;
- camera optical axes, FOV, image content, and support geometry;
- joint ordering, limits, initial pose, gripper direction, actions, and resets.

Exercise actions and resets with:

```bash
"$ISAACLAB_PYTHON" isaaclab/live --num_envs 4 --policy random
```

Do not train until this visual check is correct.

## 3. Train and accept the fixed task

Choose one launch mode.

Visible debugging run:

```bash
"$ISAACLAB_PYTHON" isaaclab/train \
  --task SO101-Object-In-Cup-Vision-Fixed-v0 \
  --rl_library rsl_rl \
  --visualizer kit --num_envs 4 \
  physics=isaacsim_physx
```

Headless single-GPU run:

```bash
"$ISAACLAB_PYTHON" isaaclab/train \
  --task SO101-Object-In-Cup-Vision-Fixed-v0 \
  --rl_library rsl_rl \
  --num_envs 64 --headless \
  physics=isaacsim_physx
```

Headless two-GPU run:

```bash
"$ISAACLAB_PYTHON" isaaclab/train_multigpu \
  --num_gpus 2 \
  --task SO101-Object-In-Cup-Vision-Fixed-v0 \
  --rl_library rsl_rl \
  --num_envs 64 --headless \
  physics=isaacsim_physx
```

`64` environments per process is a starting point, not a guaranteed capacity.
The actor owns three camera encoders and the critic is a compact MLP, while every
environment still renders three views. Reduce `--num_envs` if startup or
training exceeds GPU memory.

Two-GPU training is replicated data parallel: each GPU owns a complete model,
optimizer, and simulation batch, and RSL-RL averages gradients. Do not remap
the workers with `CUDA_VISIBLE_DEVICES`; Isaac Sim's Vulkan selection uses
physical device indices. Add `--log_all_ranks` immediately after
`--num_gpus 2` only when diagnosing distributed startup.

Fixed checkpoints are written under:

```text
logs/rsl_rl/so101_object_in_cup_vision_fixed/<run>/model_<iteration>.pt
```

Inspect a checkpoint visibly:

```bash
FIXED_CHECKPOINT=/absolute/path/to/model_<iteration>.pt

"$ISAACLAB_PYTHON" isaaclab/play \
  --task SO101-Object-In-Cup-Vision-Fixed-v0 \
  --rl_library rsl_rl \
  --checkpoint "$FIXED_CHECKPOINT" \
  --visualizer kit --num_envs 4 \
  physics=isaacsim_physx
```

Require at least 95% fixed-task success and inspect contacts, release behavior,
and camera inputs before continuing. This is an acceptance target, not a result
already achieved by the repository.

## 4. Resume on the randomized task

Start randomized training from the accepted fixed checkpoint:

```bash
"$ISAACLAB_PYTHON" isaaclab/train \
  --task SO101-Object-In-Cup-Vision-v0 \
  --rl_library rsl_rl \
  --resume --checkpoint "$FIXED_CHECKPOINT" \
  --num_envs 64 --headless \
  physics=isaacsim_physx
```

For two GPUs, use the same task and checkpoint with the multi-GPU launcher:

```bash
"$ISAACLAB_PYTHON" isaaclab/train_multigpu \
  --num_gpus 2 \
  --task SO101-Object-In-Cup-Vision-v0 \
  --rl_library rsl_rl \
  --resume --checkpoint "$FIXED_CHECKPOINT" \
  --num_envs 64 --headless \
  physics=isaacsim_physx
```

Randomized checkpoints are written under:

```text
logs/rsl_rl/so101_object_in_cup_vision/<run>/model_<iteration>.pt
```

Fixed and randomized tasks use the same model contract, so a new fixed
checkpoint can be resumed directly. Checkpoints using either the former
35-value critic or the temporary camera critic are incompatible and are not
migrated.

## 5. Evaluate the randomized policy

```bash
RANDOMIZED_CHECKPOINT=/absolute/path/to/model_<iteration>.pt

"$ISAACLAB_PYTHON" isaaclab/play \
  --task SO101-Object-In-Cup-Vision-v0 \
  --rl_library rsl_rl \
  --checkpoint "$RANDOMIZED_CHECKPOINT" \
  --visualizer kit --num_envs 4 \
  physics=isaacsim_physx
```

Before export or real command publication, require:

- at least 90% success across 1,000 held-out randomized episodes;
- held-out camera, lighting, appearance, mass, friction, and actuator samples;
- no NaNs, persistent penetrations, invalid actions, or limit violations;
- confirmation that the exported actor contains only camera and joint inputs.

Visible playback is for qualitative inspection. Record the quantitative held-out
result separately; this repository does not claim that it has been completed.

## 6. Export the accepted checkpoint

```bash
ARTIFACT_DIR=/absolute/path/to/policy-artifacts

"$ISAACLAB_PYTHON" isaaclab/export \
  --task SO101-Object-In-Cup-Vision-v0 \
  --checkpoint "$RANDOMIZED_CHECKPOINT" \
  --output-dir "$ARTIFACT_DIR" \
  physics=isaacsim_physx --headless
```

The output directory contains:

```text
policy.pt
policy.onnx
policy_manifest.json
```

Export includes only the actor. It checks checkpoint/TorchScript and
checkpoint/ONNX action parity on the same observation. The manifest records
camera preprocessing/order, joint order, action scales, limits, frequency,
task ID, calibration hash, and artifact checksums.

## 7. Validate real inputs, then arm

Build and source the existing inference package:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select so101_inference
source install/setup.bash

ros2 launch so101_inference rsl_rl_infer.launch.py \
  model_dir:="$ARTIFACT_DIR" camera_profile:=dual_overhead
```

The node starts in shadow mode and requires fresh, synchronized data from:

```text
/follower/image_raw
/static_camera_1/image_raw
/static_camera_2/image_raw
/follower/joint_states
```

Validate preprocessing, 20 Hz inference, action bounds, timestamp rejection,
hold-on-failure behavior, and manual arming using recorded real inputs before
publishing commands.

Arm only after simulation and shadow-mode gates pass:

```bash
ros2 service call /so101_rl/set_enabled \
  std_srvs/srv/SetBool "{data: true}"
```

Disable with:

```bash
ros2 service call /so101_rl/set_enabled \
  std_srvs/srv/SetBool "{data: false}"
```

Manual disable, stale/skewed data, inference failure, invalid output, or
non-finite output sends one measured-position hold target, disarms, and requires
explicit rearming.

## Task reference

Canonical joint and action order:

```text
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
```

The policy runs at 20 Hz over 100 Hz physics. Normalized outputs are clipped to
`[-1, 1]` and converted to measured-position deltas:

```text
arm target delta      0.05 rad * action
gripper target delta  0.15 rad * action
```

The fixed task uses nominal geometry, appearance, physics, deterministic resets,
and no observation/action latency. The randomized task adds layout, yaw, mass,
friction, actuator, joint noise, camera calibration, appearance, lighting,
image corruption, and zero/one-step camera/action latency.

## Files and outputs

| Purpose | Location |
|---|---|
| Source cube/cup meshes | `isaaclab/assets/source/` |
| Task and model package | `isaaclab/source/so101_rl/` |
| Focused tests | `isaaclab/tests/` |
| Generated cube/cup assets | `build/isaaclab_assets/` |
| Generated robot and camera supports | `build/isaacsim_so101/` |
| Fixed checkpoints | `logs/rsl_rl/so101_object_in_cup_vision_fixed/` |
| Randomized checkpoints | `logs/rsl_rl/so101_object_in_cup_vision/` |
| Exported actor | User-selected `--output-dir` |

Generated assets, logs, exports, and local environments are ignored by Git.

## Tests

Run the focused Isaac Lab suite through the source-runtime bootstrap:

```bash
"$ISAACLAB_PYTHON" isaaclab/test
```

The suite checks asset geometry, action ordering/bounds, task timing, success
boundaries, visual task registration, the exact 34-value critic layout, actor
input isolation, camera calibration, latency reset behavior, actor export
signatures, and the deployment manifest. It does not prove PPO convergence,
rendered-camera correctness, held-out success, or sim-to-real transfer.

## Troubleshooting

### No Isaac Sim window

Use `isaaclab/live`, or pass `--visualizer kit` to `train`/`play`. Do not pass
`--headless`. Run from a desktop session with `DISPLAY`; remote sessions require
an Isaac Lab livestream visualizer.

### Missing assets

Repeat asset generation and [asset preparation](#1-prepare-and-validate-assets).
Regenerate after robot, camera mount, cube, or cup geometry changes.

### Wrong camera views

Stop training. Compare the fixed task with sim teleop at the same joint pose and
inspect `so101_bringup/config/cameras/isaac_dual_overhead.yaml`. Nominal
calibration is not proof of pixel-perfect real calibration.

### CUDA out of memory

Reduce `--num_envs` from 64 to 32, 16, or 4. Each environment renders three
views; only the actor encodes images, while the critic uses an MLP.

### Multi-GPU startup stalls

Use `isaaclab/train_multigpu` without `CUDA_VISIBLE_DEVICES`. Add
`--log_all_ranks` after `--num_gpus 2` and confirm both workers initialize NCCL,
build their environments, and reach parameter synchronization. Initial shader
compilation can take time.

### Checkpoint fails to load

Use an absolute path and the matching fixed/randomized task ID. Pre-cleanup
35-value-critic checkpoints and temporary camera-critic checkpoints are
incompatible.

### ROS node refuses to arm

Check all four input topics, source timestamp skew, manifest/artifact checksums,
and CUDA availability. Any failure requires explicit rearming.
