# SO-101 deployable three-camera visual RL

This is a repository-owned external Isaac Lab project. Isaac Lab supplies the
simulation and RSL-RL runtime; this repository owns the SO-101 robot asset,
object/cup assets, environments, visual model, PPO configuration, checkpoints,
export contract, and ROS 2 deployment node. Isaac Lab's source checkout is not
modified, and ROS 2 is not part of the training loop.

For the design rationale, exact reward formulas, PPO details, neural-network
architecture, supported algorithm boundary, and instructions for creating a new
reward, model, observation, action, or task, read
[`METHODOLOGY.md`](METHODOLOGY.md).

## Current status — read this first

Implemented:

- fixed-pose and randomized visual Isaac Lab tasks;
- visible and headless vectorized simulation;
- one-, multi-environment, single-GPU, and two-GPU launch paths;
- an actor that receives only three RGB images and six absolute joint positions;
- an asymmetric 35-value training-only critic;
- visual and dynamics randomization, including one-step camera/action latency;
- TorchScript/ONNX export, manifest generation, and numerical parity checks;
- a separate safety-gated ROS 2 inference node that starts in shadow mode.

Not yet claimed:

- no PPO checkpoint has been trained by this implementation work;
- the 95% fixed-pose and 90% randomized success gates have not been achieved or
  measured;
- no real robot has been armed with an exported visual policy;
- matching teleop inputs establishes interface compatibility, not guaranteed
  visual sim-to-real transfer.

## Assumptions and prerequisites

The commands below assume:

- repository: `/home/anon/Documents/so101-ros2`;
- Isaac Lab: `/home/anon/Documents/IsaacLab`;
- source-built Isaac Sim: `/home/anon/Documents/isaacsim`;
- camera profile: always `dual_overhead`;
- a CUDA GPU for visual training and deployment;
- a graphical desktop terminal with `DISPLAY` available for native Isaac Sim
  windows;
- `cube.stl` is the manipulated object and `cup.stl` is fixed during each
  episode;
- the supplied STLs use millimetres and are converted at `0.001 m/unit`;
- the generated repository SO-101 USD is authoritative.

Every command in the ordered workflow is run from the repository root:

```bash
cd /home/anon/Documents/so101-ros2
export ISAACLAB_PYTHON=/home/anon/Documents/IsaacLab/.venv/bin/python
```

## Choose what you want to do

| Goal | Go to |
|---|---|
| Understand or modify rewards, PPO, networks, or tasks | [Methodology](METHODOLOGY.md) |
| Run the complete training/deployment lifecycle | [Required workflow](#required-workflow--run-these-steps-in-order) |
| Open the fixed task in a visible Isaac Sim window | [Step 3](#step-3--open-the-fixed-task-visibly-required-smoke-check) |
| Train while watching Isaac Sim | [Step 4, option A](#step-4--train-the-fixed-task-choose-one-option) |
| Train faster on one GPU | [Step 4, option B](#step-4--train-the-fixed-task-choose-one-option) |
| Train on two GPUs | [Step 4, option C](#step-4--train-the-fixed-task-choose-one-option) |
| Watch a trained checkpoint | [Step 5](#step-5--play-and-accept-the-fixed-checkpoint-visibly) |
| Export a policy for ROS 2 | [Step 9](#step-9--export-the-accepted-randomized-checkpoint) |
| Run on real observations without commanding the arm | [Step 10](#step-10--start-real-robot-inference-in-shadow-mode) |
| Arm or disarm real command publication | [Step 11](#step-11--arm-the-real-controller-only-after-all-gates-pass) |
| Copy one standalone command | [Standalone command recipes](#standalone-command-recipes) |

## Required workflow — run these steps in order

Do not start randomized training first. The intended progression is:

```text
prepare assets
  -> inspect fixed task visibly
  -> train fixed task
  -> accept fixed checkpoint
  -> train randomized task from fixed checkpoint
  -> accept randomized checkpoint
  -> export
  -> real-input shadow mode
  -> supervised real arming
```

### Step 1 — configure the shell

```bash
cd /home/anon/Documents/so101-ros2
export ISAACLAB_PYTHON=/home/anon/Documents/IsaacLab/.venv/bin/python
```

Keep this shell open, or repeat these two commands in every new terminal.

### Step 2 — prepare and validate all assets

The visual task requires these existing teleop-generated assets:

```text
build/isaacsim_so101/so101_follower/so101_follower.usda
build/isaacsim_so101/camera_rig/cam_mount_bottom.usd
build/isaacsim_so101/camera_rig/cam_mount_top.usd
```

If any are missing, generate them once through the existing sim-teleop asset
path. This command also needs the built ROS workspace:

```bash
source /opt/ros/jazzy/setup.bash
source /home/anon/Documents/so101-ros2/install/setup.bash

/home/anon/Documents/isaacsim/_build/linux-x86_64/release/python.sh \
  /home/anon/Documents/so101-ros2/scripts/isaac_sim_teleop.py \
  --camera-profile dual_overhead \
  --headless --max-frames 1
```

Add `--rebuild-asset` only after changing the robot Xacro, robot meshes, camera
mount meshes, or camera rig configuration.

Now prepare the supplied cube and open cup:

```bash
"$ISAACLAB_PYTHON" isaaclab/prepare_assets
"$ISAACLAB_PYTHON" isaaclab/prepare_assets --validate-only
```

Expected generated files:

```text
build/isaaclab_assets/cube.usda
build/isaaclab_assets/cup.usda
build/isaaclab_assets/manifest.json
```

Re-run preparation after changing either STL or regenerating the robot USD.

### Step 3 — open the fixed task visibly (required smoke check)

Run this from a graphical desktop terminal:

```bash
"$ISAACLAB_PYTHON" isaaclab/live \
  --task SO101-Object-In-Cup-Vision-Fixed-v0 \
  --num_envs 4
```

This is **not headless**. A native Isaac Sim window should open with four
concurrent cloned environments. `--num_envs 1`, `4`, `16`, and so on select the
number of simulations stepped concurrently in one Isaac Sim process.

Before training, verify all of the following visually:

- the robot, table, object, and open cup are present in every environment;
- the cube can physically pass through the cup opening;
- the wrist camera is attached to the gripper;
- both overhead cameras and support geometry match sim teleop;
- camera views, optical axes, FOV, robot initial pose, joint order, and gripper
  direction are correct.

Do not continue if this smoke check is wrong.

To exercise actions and resets visibly instead of holding zero action:

```bash
"$ISAACLAB_PYTHON" isaaclab/live \
  --task SO101-Object-In-Cup-Vision-Fixed-v0 \
  --num_envs 4 --policy random
```

### Step 4 — train the fixed task (choose one option)

Choose exactly one launch option. They train the same fixed-pose task and write
to the same experiment family.

#### Option A — visible training while watching

Use this for debugging, not maximum throughput:

```bash
"$ISAACLAB_PYTHON" isaaclab/train \
  --task SO101-Object-In-Cup-Vision-Fixed-v0 \
  --rl_library rsl_rl \
  --visualizer kit --num_envs 4 \
  physics=isaacsim_physx
```

This is **visible**, not headless.

#### Option B — faster single-GPU training

```bash
"$ISAACLAB_PYTHON" isaaclab/train \
  --task SO101-Object-In-Cup-Vision-Fixed-v0 \
  --rl_library rsl_rl \
  --num_envs 64 --headless \
  physics=isaacsim_physx
```

This is **headless** and uses 64 rendered environments on one GPU.

#### Option C — two-GPU headless training

```bash
"$ISAACLAB_PYTHON" isaaclab/train_multigpu \
  --num_gpus 2 \
  --task SO101-Object-In-Cup-Vision-Fixed-v0 \
  --rl_library rsl_rl \
  --visualizer kit --num_envs 4 \
  physics=isaacsim_physx
```

This is **headless** and launches 64 environments per process/GPU, 128 total.
It is data-parallel training: each GPU owns one complete actor, critic, optimizer,
and 64-environment Isaac Sim batch. RSL-RL averages gradients across the two
complete model replicas after every PPO minibatch; the neural network is not
split across GPUs.

The local source-built Isaac Sim runtime creates the NCCL training group before
Kit starts on each rank and keeps that group for the entire training run. This
avoids a CUDA-interop crash when a new NCCL communicator is otherwise created
after the rendered scene and PhysX CUDA contexts are active. RSL-RL's matching
initialization request reuses the validated group.
The repository PPO variant also broadcasts actor/critic state as tensors rather
than using RSL-RL's CUDA `broadcast_object_list` path; PPO losses, optimizers,
rollouts, and gradient averaging are otherwise unchanged.

Do not set `CUDA_VISIBLE_DEVICES` separately for these workers. Isaac Sim's
Vulkan device enumeration does not follow CUDA's remapped indices; use
`--num_gpus 2` and let the launcher assign physical GPU 0 and GPU 1.

To show both worker logs while diagnosing distributed startup, add
`--log_all_ranks` immediately after `--num_gpus 2`. Normal training filters the
duplicated rank-1 startup output.

Fixed checkpoints are written under:

```text
logs/rsl_rl/so101_object_in_cup_vision_fixed/<run>/model_<iteration>.pt
```

### Step 5 — play and accept the fixed checkpoint visibly

Replace the checkpoint placeholder with an absolute path:

```bash
FIXED_CHECKPOINT=/absolute/path/to/logs/rsl_rl/so101_object_in_cup_vision_fixed/<run>/model_<iteration>.pt

"$ISAACLAB_PYTHON" isaaclab/play \
  --task SO101-Object-In-Cup-Vision-Fixed-v0 \
  --rl_library rsl_rl \
  --checkpoint "$FIXED_CHECKPOINT" \
  --visualizer kit --num_envs 4 \
  physics=isaacsim_physx
```

This is **visible**. Do not start randomized training until the fixed task has
at least 95% simulation success and the contacts, release behavior, and camera
inputs are correct. The 95% value is an acceptance requirement, not a result
already achieved by this repository.

### Step 6 — train the randomized task from the fixed checkpoint

Keep `FIXED_CHECKPOINT` set to the accepted fixed checkpoint. Choose exactly one
option below. Every option resumes from the fixed actor.

#### Option A — visible randomized training

```bash
"$ISAACLAB_PYTHON" isaaclab/train \
  --task SO101-Object-In-Cup-Vision-v0 \
  --rl_library rsl_rl \
  --resume --checkpoint "$FIXED_CHECKPOINT" \
  --visualizer kit --num_envs 4 \
  physics=isaacsim_physx
```

#### Option B — single-GPU headless randomized training

```bash
"$ISAACLAB_PYTHON" isaaclab/train \
  --task SO101-Object-In-Cup-Vision-v0 \
  --rl_library rsl_rl \
  --resume --checkpoint "$FIXED_CHECKPOINT" \
  --num_envs 64 --headless \
  physics=isaacsim_physx
```

#### Option C — two-GPU headless randomized training

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

### Step 7 — play the randomized checkpoint visibly

```bash
RANDOMIZED_CHECKPOINT=/absolute/path/to/logs/rsl_rl/so101_object_in_cup_vision/<run>/model_<iteration>.pt

"$ISAACLAB_PYTHON" isaaclab/play \
  --task SO101-Object-In-Cup-Vision-v0 \
  --rl_library rsl_rl \
  --checkpoint "$RANDOMIZED_CHECKPOINT" \
  --visualizer kit --num_envs 4 \
  physics=isaacsim_physx
```

Increase `--num_envs` to inspect more concurrent randomized environments when
GPU memory permits.

### Step 8 — pass simulation acceptance

Before export or real command publication, require:

- at least 90% success across 1,000 held-out randomized vision episodes;
- held-out camera, lighting, material, mass, friction, and actuator variations;
- no use of training-only reset states;
- no NaNs, persistent penetrations, invalid actions, or joint-limit violations;
- correct actor observations: only three images and six absolute joints.

Visible `play` is for inspection. This repository does not currently claim a
completed 1,000-episode acceptance report; record that result separately before
deployment.

### Step 9 — export the accepted randomized checkpoint

```bash
ARTIFACT_DIR=/absolute/path/to/policy-artifacts

"$ISAACLAB_PYTHON" isaaclab/export \
  --task SO101-Object-In-Cup-Vision-v0 \
  --checkpoint "$RANDOMIZED_CHECKPOINT" \
  --output-dir "$ARTIFACT_DIR" \
  physics=isaacsim_physx --headless
```

The output directory must contain:

```text
policy.pt
policy.onnx
policy_manifest.json
```

Export checks checkpoint/TorchScript and checkpoint/ONNX action parity on the
same environment observation. The manifest records camera order, image shape
and preprocessing, joint order, frequency, delta scales, hard joint limits,
0.98 safety margin, task ID, camera-calibration hash, and artifact checksums.

### Step 10 — start real-robot inference in shadow mode

Build and source the ROS package:

```bash
cd /home/anon/Documents/so101-ros2
source /opt/ros/jazzy/setup.bash
colcon build --packages-select so101_inference
source install/setup.bash
```

Launch the exported policy:

```bash
ros2 launch so101_inference rsl_rl_infer.launch.py \
  model_dir:="$ARTIFACT_DIR" camera_profile:=dual_overhead
```

The node starts in **shadow mode**. It runs inference but publishes no controller
commands. It requires:

- `/follower/image_raw`;
- `/static_camera_1/image_raw`;
- `/static_camera_2/image_raw`;
- `/follower/joint_states`;
- every stream fresh and all source timestamps within 50 ms;
- a local CUDA workstation GPU.

Feed recorded real camera/joint observations through shadow mode and verify
preprocessing, 20 Hz inference, action bounds, freshness rejection, manual
arming, and hold-on-failure behavior before enabling publication.

### Step 11 — arm the real controller only after all gates pass

Do not arm until simulation acceptance and recorded-real-input shadow checks
have passed. To arm:

```bash
ros2 service call /so101_rl/set_enabled \
  std_srvs/srv/SetBool "{data: true}"
```

To disable:

```bash
ros2 service call /so101_rl/set_enabled \
  std_srvs/srv/SetBool "{data: false}"
```

Manual disable, stale or skewed data, inference failure, invalid output, or
NaN/Inf publishes the current measured positions once as a hold target,
disarms, and requires explicit rearming. Initial real validation is ten
supervised episodes with no freshness, invalid-output, or joint-limit
violations.

## Standalone command recipes

These recipes are shortcuts. The complete lifecycle and gates above still
apply.

| Action | Visible? | Task/checkpoint |
|---|---:|---|
| Inspect fixed task | Yes | Fixed task, no checkpoint |
| Inspect randomized task | Yes | Randomized task, no checkpoint |
| Train fixed with four environments | Yes | Fixed task |
| Train fixed with 64 environments | No | Fixed task |
| Train randomized | Either | Must resume fixed checkpoint |
| Play a checkpoint | Yes | Use the matching task ID |
| Export | No | Accepted randomized checkpoint |

### Inspect the fixed task now

```bash
"$ISAACLAB_PYTHON" isaaclab/live \
  --task SO101-Object-In-Cup-Vision-Fixed-v0 --num_envs 4
```

### Inspect the randomized task now

```bash
"$ISAACLAB_PYTHON" isaaclab/live \
  --task SO101-Object-In-Cup-Vision-v0 --num_envs 4
```

### Inspect randomized actions and resets

```bash
"$ISAACLAB_PYTHON" isaaclab/live \
  --task SO101-Object-In-Cup-Vision-v0 \
  --num_envs 16 --policy random
```

### Train or play visibly

Use `--visualizer kit` and a small `--num_envs` value. Do not include
`--headless`:

```bash
"$ISAACLAB_PYTHON" isaaclab/train \
  --task SO101-Object-In-Cup-Vision-Fixed-v0 \
  --rl_library rsl_rl --visualizer kit --num_envs 4 \
  physics=isaacsim_physx
```

```bash
"$ISAACLAB_PYTHON" isaaclab/play \
  --task SO101-Object-In-Cup-Vision-v0 \
  --rl_library rsl_rl --checkpoint "$RANDOMIZED_CHECKPOINT" \
  --visualizer kit --num_envs 4 physics=isaacsim_physx
```

### Train headless

Use `--headless`; 64 environments is the intended starting point per GPU:

```bash
"$ISAACLAB_PYTHON" isaaclab/train \
  --task SO101-Object-In-Cup-Vision-Fixed-v0 \
  --rl_library rsl_rl --num_envs 64 --headless \
  physics=isaacsim_physx
```

## Task and model reference

### Registered tasks

| Task ID | Purpose | Actor input |
|---|---|---|
| `SO101-Object-In-Cup-v0` | State-based physics/debug baseline | Simulator state |
| `SO101-Object-In-Cup-Vision-Fixed-v0` | Nominal fixed-pose visual overfit | Three images + six joints |
| `SO101-Object-In-Cup-Vision-v0` | Randomized visual training/deployment | Three images + six joints |

### Deployable actor observations

```text
observation.images.wrist       [N, 3, 120, 160] RGB / 255 - 0.5
observation.images.overhead_1  [N, 3, 120, 160] RGB / 255 - 0.5
observation.images.overhead_2  [N, 3, 120, 160] RGB / 255 - 0.5
observation.state              [N, 6] absolute joint positions in radians
```

Canonical joint and action order:

```text
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
```

The actor receives no joint velocity, previous action, object pose, cup pose, or
object velocity. Those values are not exported.

### Training-only critic

The asymmetric critic receives one 35-value group containing joint position,
joint velocity, previous action, gripper/object delta, object/cup delta, object
orientation, object velocity, and gripper position. Rewards and termination use
simulator truth.

### Model and PPO

- independent `[16, 32, 32]` CNN per camera;
- kernels `[8, 4, 3]`, strides `[4, 2, 1]`, ELU;
- spatial-softmax feature-location reduction;
- actor head `[512, 256, 128]`;
- critic MLP `[256, 256, 128]`;
- 32 steps per environment;
- fixed `7e-5` learning rate;
- eight minibatches;
- initial Gaussian action standard deviation `0.7`;
- 15,000 maximum iterations;
- checkpoint every 250 iterations.

### Control and simulation

- physics: 100 Hz;
- policy and camera refresh: 20 Hz;
- normalized policy output clipped to `[-1, 1]`;
- arm delta scale: `0.05 rad/action`;
- gripper delta scale: `0.15 rad/action`.

### Fixed versus randomized task

The fixed task uses nominal camera geometry, nominal physics/materials,
deterministic robot/object/cup resets, and no observation or latency corruption.

The randomized task adds object/cup pose, object orientation, mass, friction,
actuator response, and observation noise plus:

- overhead translation jitter up to 10 mm;
- wrist translation jitter up to 5 mm;
- camera rotation and FOV jitter up to 3 degrees;
- lighting intensity/color, material color/roughness, exposure, contrast, RGB
  gain, and Gaussian sensor noise;
- sampled zero/one-policy-step camera and action latency.

## Files and generated outputs

| Purpose | Location |
|---|---|
| Source cube/cup STLs | `isaaclab/assets/source/` |
| Generated cube/cup USD and manifest | `build/isaaclab_assets/` |
| Generated SO-101 robot USD | `build/isaacsim_so101/so101_follower/` |
| Generated camera support USDs | `build/isaacsim_so101/camera_rig/` |
| Nominal camera calibration | `so101_bringup/config/cameras/isaac_dual_overhead.yaml` |
| Fixed checkpoints | `logs/rsl_rl/so101_object_in_cup_vision_fixed/` |
| Randomized checkpoints | `logs/rsl_rl/so101_object_in_cup_vision/` |
| Exported deployment artifacts | User-selected `--output-dir` |
| ROS 2 deployment launch file | `so101_inference/launch/rsl_rl_infer.launch.py` |

Generated assets, logs, and local environments are ignored by Git.

## Verification and tests

Run the focused test suite from the repository root:

```bash
PYTHONPATH=isaaclab/source/so101_rl \
  "$ISAACLAB_PYTHON" -m unittest discover -s isaaclab/tests -v

PYTHONPATH=so101_inference \
  "$ISAACLAB_PYTHON" so101_inference/test/test_rsl_rl_policy.py
```

These tests cover task-critical contracts only: open cup collision geometry,
object fit, action bounds and ordering, insertion/settling boundaries, exact
actor groups, critic width, camera calibration conversion, model export
signature, preprocessing, timestamp skew, manifest contents, and safe absolute
target conversion.

They do not prove PPO convergence, rendered-camera correctness inside a running
Isaac Sim process, the 1,000-episode randomized acceptance target, or sim-to-real
transfer. Those are runtime acceptance experiments.

## Troubleshooting

### No Isaac Sim window appears

- confirm the command does not contain `--headless`;
- add `--visualizer kit` for `train` or `play`;
- run from a graphical desktop terminal and check `echo "$DISPLAY"` is nonempty;
- use `isaaclab/live`, not the headless training recipe, for the first smoke test;
- remote shells need an Isaac Lab livestream visualizer instead of a native
  window.

### Assets are reported missing

Run [Step 2](#step-2--prepare-and-validate-all-assets). If the robot or camera
support USDs are missing, run the existing sim-teleop generation command first;
then run `isaaclab/prepare_assets`.

### Camera views are wrong

Stop training. Compare the fixed task with sim teleop at the same joint pose and
inspect `so101_bringup/config/cameras/isaac_dual_overhead.yaml`. That YAML is
nominal calibration, not proof of pixel-perfect real calibration.

### CUDA out of memory

Reduce `--num_envs` from 64 to 32, 16, or 4. Three RGB cameras are rendered for
every environment. For live visualization, start with four.

### Multi-GPU appears to stop at parameter synchronization

Use the repository's `isaaclab/train_multigpu` command from Option C. It starts
the persistent NCCL group before Isaac Sim and performs replicated data-parallel
PPO: one complete model and one simulation process on each GPU. Do not add
`CUDA_VISIBLE_DEVICES`.

Add `--log_all_ranks` immediately after `--num_gpus 2` for diagnosis. Both ranks
must print the persistent NCCL message, build their environments, and reach
`Synchronizing parameters`. The first source-built RTX launch may spend time
compiling shaders, so distinguish that startup work from a crashed worker by
checking both rank logs.

### Checkpoint not found

Use an absolute path. Fixed and randomized tasks use different log directories.
Use the fixed task ID to play a fixed checkpoint and the randomized task ID to
play a randomized checkpoint.

### ROS node refuses to arm

Check that all three raw RGB topics and `/follower/joint_states` are present,
fresh, and within 50 ms source-timestamp skew. Confirm the manifest and
TorchScript checksums match and CUDA is available. Any failure requires explicit
rearming.

### Local Isaac Sim source build aborts before creating the environment

Use the documented `$ISAACLAB_PYTHON` value and repository entry points; they
re-execute with the local source-built Isaac Sim runtime before importing task
USD modules. If the state baseline also fails before environment creation, the
problem is in the local Isaac Sim/Isaac Lab runtime rather than the vision task.

## Geometry and task details

The supplied cube is 25 mm per side. The cup is 50 mm tall with an inner radius
of approximately 22.50 mm. The cube's worst-case horizontal radius is about
17.68 mm, leaving roughly 4.83 mm radial clearance. Asset preparation derives a
conservative 3.86 mm success tolerance.

Preparation verifies watertight meshes, millimetre scale, a convex cube
collision, a compound-convex open cup collision, unobstructed entry through the
cup opening, mass, inertia, friction, restitution, contact offsets, and source
asset hashes. A collision approximation that seals the cup is rejected.

Task rewards cover reach, grasp, lift, transport, insertion, release, and stable
placement, with action-rate and joint-velocity penalties. Success requires the
released object to remain inside the cup below its rim at low velocity for ten
consecutive policy steps (0.5 seconds). Episodes terminate on stable success,
dropped object, non-finite state, or the 15-second timeout.

This is a tight-contact manipulation task. PPO is not the main architectural
risk: grasp geometry, contact behavior, calibration, reward progression, and
matching the real actuator interface are the deployment-critical risks.
