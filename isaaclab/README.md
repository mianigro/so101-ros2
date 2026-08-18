# SO-101 three-camera PPO

This directory is a repository-owned Isaac Lab project for training visual
SO-101 manipulation policies with RSL-RL PPO. It includes single-box and
three-box placement scenarios built on one reusable workcell configuration.
The deployable actor receives three RGB images and six measured joint positions.
A separate training-only critic receives exact scenario state and is never
exported to ROS 2.

Isaac Lab supplies simulation, rendering, managers, and the RSL-RL runtime.
This project owns the tasks, assets, camera contract, models, PPO configuration,
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

The actor does not receive box/cup poses, velocities, previous actions, reward
state, or success flags. The single-box task adds this noiseless training-only
34-value `critic_state` group:

| Component | Width |
|---|---:|
| Absolute joint positions | 6 |
| Joint velocities | 6 |
| Last requested action | 6 |
| Gripper-to-object delta | 3 |
| Object-to-cup delta | 3 |
| Object quaternion | 4 |
| Object linear and angular velocity | 6 |
| **Single-box total** | **34** |

The three-box task instead uses an 84-value state: the same 18 robot/action
values, three gripper-to-box deltas, all nine box-to-cup deltas in box-major
order, and pose/velocity values for all three boxes. See
[`METHODOLOGY.md`](METHODOLOGY.md#three-box-scenario) for the exact layout.

This is asymmetric PPO: exact simulator information improves the value estimate
and actor learning, but cannot enter the exported actor. Simulator state also
drives rewards, terminations, resets, and physics.

## Prerequisites

The commands below assume:

- repository: cloned anywhere (examples use `/path/to/so101-ros2`);
- Isaac Lab: `~/Documents/IsaacLab` — override with `SO101_ISAACLAB_ROOT`;
- source-built Isaac Sim: `~/Documents/isaacsim` — override with
  `SO101_ISAACSIM_PYTHON`, pointing at
  `_build/linux-x86_64/release/python.sh` inside the source build;
- setup: `monomanual_dual_overhead` (visual policies are locked to it);
- a CUDA GPU for visual training and deployment;
- `cube.stl` is manipulated and `cup.stl` remains fixed during an episode;
- the generated repository SO-101 USD is authoritative.

Start from the repository root:

```bash
cd /path/to/so101-ros2
export ISAACLAB_PYTHON=~/Documents/IsaacLab/.venv/bin/python
```

The entry points automatically switch to the configured source-built Isaac Sim
runtime when required.

## Scenarios

Choose the scenario first, then choose its training stage:

| Scenario | Goal | Episode | Fixed task | Randomized task |
|---|---|---:|---|---|
| Single box | Put one box into one cup and release it stably | 15 s | `SO101-Object-In-Cup-Vision-Fixed-v0` | `SO101-Object-In-Cup-Vision-v0` |
| Three boxes | Put all three interchangeable boxes into three distinct cups, in any assignment | 45 s | `SO101-Three-Boxes-In-Cups-Vision-Fixed-v0` | `SO101-Three-Boxes-In-Cups-Vision-v0` |

Use the fixed task to inspect the scene and prove that PPO can learn the
scenario without randomization. Resume that scenario's accepted fixed
checkpoint into its randomized task for robustness and export. Do not resume a
single-box checkpoint into the three-box task or vice versa: the deployable
actors have the same interface, but the training-only critics have different
input widths.

### Model families

The actor model is chosen through the task ID. The single-box scenario is
additionally registered for the temporal model families
[`transformer_ppo`](models/transformer_ppo/README.md) and
[`mamba_ppo`](models/mamba_ppo/README.md):

| Actor family | Fixed task | Randomized task |
|---|---|---|
| Spatial-softmax CNN (default) | `SO101-Object-In-Cup-Vision-Fixed-v0` | `SO101-Object-In-Cup-Vision-v0` |
| Temporal transformer (four-frame window) | `SO101-Object-In-Cup-Vision-Transformer-Fixed-v0` | `SO101-Object-In-Cup-Vision-Transformer-v0` |
| Recurrent state-space mamba (single frames) | `SO101-Object-In-Cup-Vision-Mamba-Fixed-v0` | `SO101-Object-In-Cup-Vision-Mamba-v0` |

The transformer consumes a four-frame camera history window; the mamba is a
streaming policy that consumes single frames and keeps its own recurrent
state (reset on terminations, carried across rollout boundaries by RSL-RL's
recurrent storage). Both temporal families share the spatial-softmax frame
encoder; the mamba's selective state-space core is pure PyTorch, so no
optional packages are needed and TorchScript/ONNX export works for all
families.

Do not resume a checkpoint across actor families: the deployable actor
interfaces differ (single frame vs. history window vs. recurrent state) and
so do the stored network weights.

Pass the selected ID with `--task` to `live`, `train`, `train_multigpu`, `play`,
or `export`. For example:

```bash
# Inspect the deterministic single-box scenario (also the `live` default).
"$ISAACLAB_PYTHON" isaaclab/live \
  --task SO101-Object-In-Cup-Vision-Fixed-v0 --num_envs 4

# Inspect the deterministic three-box scenario.
"$ISAACLAB_PYTHON" isaaclab/live \
  --task SO101-Three-Boxes-In-Cups-Vision-Fixed-v0 --num_envs 4
```

The remaining commands use the single-box IDs as the concise baseline. Replace
them with both three-box IDs at the corresponding fixed and randomized stages
to run the three-box workflow; a complete command pair is included below.

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
SO101_REPO=/home/anon/Documents/so101-ros2
source "$SO101_REPO/install/setup.bash"

/home/anon/Documents/isaacsim/_build/linux-x86_64/release/python.sh \
  "$SO101_REPO/scripts/isaac_sim_teleop.py" \
  --setup monomanual_dual_overhead \
  --headless --max-frames 1 \
  --rebuild-asset
```

Use `--rebuild-asset` only after changing robot Xacro/meshes, camera mount
meshes, or camera rig configuration.

The Isaac expansion enables `simulation_contact_pads:=true`. It imports the
two invisible inner-jaw pad links without fixed-joint merging so PhysX can
address them independently; the normal ROS description keeps the option false.
The generated articulation must still expose only the canonical six movable
joints. Rebuilding after the pad or grasp-geometry change is mandatory.

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
- For the single-box task: the reward ladder metrics are `approach` (episode-best
  proximity), `lift_progress`, then `transport`/`insertion`/`release`/`stable`.
  The task is grasp-agnostic — there are no contact-sensor metrics to inspect.
- For the three-box task: no pad/box contact at reset, bilateral pad contact
  during a centred pinch, and stable contact forces without pad/jaw overlap;
  pickup metrics firing in order `approach_progress`, `closure_progress`,
  `grasp_held`, `grasp_acquired`, then `lift_progress`, with no continuing
  approach reward while camping. `grasp_held` (dense, geometry-based) is the
  bridge that should rise as the gripper clamps the box, ahead of the
  contact-validated `grasp_acquired` and `lift_progress`.

Exercise actions and resets with:

```bash
"$ISAACLAB_PYTHON" isaaclab/live --num_envs 4 --policy random
```

Do not train until this visual check is correct.

Inspect the three-box fixed layout explicitly with:

```bash
"$ISAACLAB_PYTHON" isaaclab/live \
  --task SO101-Three-Boxes-In-Cups-Vision-Fixed-v0 \
  --num_envs 4
```

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

Viewable two-GPU run:

```bash
"$ISAACLAB_PYTHON" isaaclab/train_multigpu \
  --num_gpus 2 \
  --task SO101-Object-In-Cup-Vision-Fixed-v0 \
  --rl_library rsl_rl \
  --num_envs 8 --visualizer kit \
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

Reward or checkpoint return alone is insufficient.

For the single-box task, reject a run whose `approach` metric rises while
`lift_progress` stays zero: the policy is reaching the cube but not getting it
off the table. Remember the ladder is grasp-agnostic — any lift counts, so a
flat `lift_progress` with rising `approach` indicates a visual or control
failure, not a wrong grasp style.

Reject a run whose `transport` metric plateaus while insertion/release/stable
stay zero: that indicates the policy is hovering the cube near the cup to farm
the dense transport term. Transport decays to 20% over ~2.5 s of sustained
aloft holding and re-arms on a drop, so a healthy run shows `transport` rising
and falling as the cube is carried and inserted, not a flat plateau.

For the three-box task, reject a run whose `approach_progress` or
`closure_progress` rises while grasp and lift remain zero. The dense
`grasp_held` term is the intended remedy for that stall — it should rise as
the gripper clamps the box (geometry-based, gated on a near-closed gripper)
before the contact-validated `grasp_acquired` and `lift_progress`.
`grasp_acquired` is retryable per grasp attempt, so re-pinch after a drop is
rewarded too. If `grasp_held` rises but `grasp_acquired`/`lift_progress` stay
flat, the gripper is closing near the box without achieving a real pinch —
tighten the grasp (contact tuning) before resuming.

Because the jaw collision geometry and pickup semantics changed together,
start fixed-task training from scratch; older checkpoints remain structurally
loadable but are not valid continuation points for this experiment.

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

### Three-box workflow

The three-box task uses the same commands and actor contract. Train its fixed
variant first:

```bash
"$ISAACLAB_PYTHON" isaaclab/train \
  --task SO101-Three-Boxes-In-Cups-Vision-Fixed-v0 \
  --rl_library rsl_rl \
  --num_envs 64 --headless \
  physics=isaacsim_physx
```

Then resume that checkpoint on the randomized variant:

```bash
THREE_BOX_FIXED_CHECKPOINT=/absolute/path/to/model_<iteration>.pt

"$ISAACLAB_PYTHON" isaaclab/train \
  --task SO101-Three-Boxes-In-Cups-Vision-v0 \
  --rl_library rsl_rl \
  --resume --checkpoint "$THREE_BOX_FIXED_CHECKPOINT" \
  --num_envs 64 --headless \
  physics=isaacsim_physx
```

Its 45-second episode succeeds only after all three interchangeable boxes have
occupied three distinct cups for fifteen consecutive policy steps. The randomized
variant grows separated pickup and placement zones over 45 million environment
steps. `64` environments remains only a starting example.

Fixed and randomized checkpoints are compatible within this scenario. A full
three-box PPO checkpoint is not compatible with the single-box task because
their critics consume 84 and 34 values respectively; the exported actor
interface is identical.

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
  model_dir:="$ARTIFACT_DIR" setup:=monomanual_dual_overhead
```

The node starts in shadow mode and requires fresh, synchronized data from:

```text
/follower/image_raw
/static_camera_1/image_raw
/static_camera_2/image_raw
/follower/joint_states
```

Validate preprocessing, 30 Hz inference, action bounds, timestamp rejection,
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

The policy runs at 30 Hz over 120 Hz physics. Normalized outputs are clipped to
`[-1, 1]` and converted to measured-position deltas:

```text
arm target delta      0.033333 rad * action
gripper target delta  0.10 rad * action
```

Each fixed task uses nominal geometry, appearance, physics, deterministic
resets, and no observation/action latency. Each randomized task adds its own
layout curriculum plus yaw, mass, friction, actuator, joint noise, camera
calibration, appearance, lighting, image corruption, and zero/one-step
camera/action latency.

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
| Three-box fixed checkpoints | `logs/rsl_rl/so101_three_boxes_in_cups_vision_fixed/` |
| Three-box randomized checkpoints | `logs/rsl_rl/so101_three_boxes_in_cups_vision/` |
| Exported actor | User-selected `--output-dir` |

Generated assets, logs, exports, and local environments are ignored by Git.

## Tests

Run the focused Isaac Lab suite through the source-runtime bootstrap:

```bash
"$ISAACLAB_PYTHON" isaaclab/test
```

The suite checks asset geometry, projected grasp support, bounded approach and
retryable closure after 1 mm of edge insertion, synchronous same-box bilateral
contacts, the shared actor/action contract, both task families, the exact 34- and 84-value
critics, permutation-invariant three-box success, reset spacing, camera
calibration, latency resets, actor export signatures, and deployment
validation. It does not prove PPO convergence, rendered-camera correctness,
held-out success, or sim-to-real transfer.

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
inspect the `sim:` section of `so101_bringup/config/setups/monomanual_dual_overhead.yaml`. Nominal
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

Use an absolute path and the matching scenario and fixed/randomized task ID.
Single-box and three-box full checkpoints are mutually incompatible because the
critic widths differ. Pre-cleanup 35-value-critic and temporary camera-critic
checkpoints are also incompatible.

### ROS node refuses to arm

Check all four input topics, source timestamp skew, manifest/artifact checksums,
and CUDA availability. Any failure requires explicit rearming.
