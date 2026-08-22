# SO-101 Physical AI Stack


## Physical AI Stack
### Stack
A ROS2 combined with VLA stack for the SO-101 robot arm in a leader/follower configuration using `ros2_control` and LeRobot. This covers leader-follower [teleop](#teleop), [episode collecting](#data-collection), [train](#training), [LeRobot dataset conversion](#lerobot-dataset-conversion) and [policy inference](#inference).

### Isaac Lab and Isaac Sim
Isaac Sim and Isaac Lab provide the simulated SO-101 workcell used by both
Isaac Sim [teleop](#teleop) data collection and
[VLA self-improvement](self-improve/README.md). The workflow supports the
configured VLA, with π0.5 as the default.

### Self-Improvement
This provides a method to do [self improvement](#policy-self-improvement) by
allowing inference to create datasets both in real life and in Isaac Sim,
including data labelling using a small VLM. See the
[VLA self-improvement guide](self-improve/README.md).

---

## Requirements

- **Ubuntu 24.04** + **ROS 2 Jazzy**
- One SO-101 leader/follower pair (`monomanual`, `monomanual_dual_overhead`) or two pairs (`bimanual`)
- Isaac Sim 6.0.1 for simulated-follower teleop
- Two to four cameras, matching the selected setup's `cameras.profile`
- `rosdep`, `colcon`, and [Pixi](https://pixi.sh/)

> **Before launching ROS**, complete LeRobot motor setup + calibration for every arm and create the udev rules for your setup. ROS uses the calibration stored by LeRobot — do not command the arms until this is done.
> **[→ Full hardware setup guide (docs/hardware.md)](docs/hardware.md)**

---

## Installation

### Setup
```bash
# Clone
cd ~/Documents
git clone --recurse-submodules https://github.com/legalaspro/so101-ros-physical-ai.git so101-ros2
cd ~/Documents/so101-ros2

# Install deps and build
sudo apt update
rosdep update
rosdep install --from-paths so101_bringup so101_description so101_teleop episode_recorder rosbag_to_lerobot so101_inference policy_server feetech_ros2_driver --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash

# Install packages
pixi install --all
```

### Configuration

`setup` selects the canonical rig contract used by bringup, recording, Rerun,
LeRobot conversion, and inference. Use the same setup throughout a workflow:

- `monomanual`: one leader/follower pair, one wrist camera, and one overhead camera
- `monomanual_dual_overhead`: one leader/follower pair, one wrist camera, and two overhead cameras
- `bimanual`: two leader/follower pairs, two wrist cameras, and one shared overhead camera

The checked-in setup files work directly when their `/dev` names match your
udev rules. For machine-specific changes, copy the selected setup and pass the
absolute copy with `setup_config_file`:

```bash
export SO101_REPO="$PWD"
export SO101_SETUP=monomanual_dual_overhead
export SO101_CONFIG_DIR="$HOME/.config/so101/setups"

mkdir -p "$SO101_CONFIG_DIR"
cp "$SO101_REPO/so101_bringup/config/setups/$SO101_SETUP.yaml" \
  "$SO101_CONFIG_DIR/$SO101_SETUP.yaml"

# Edit the copied YAML, then launch it
ros2 launch so101_bringup teleop.launch.py \
  setup:=$SO101_SETUP \
  setup_config_file:="$SO101_CONFIG_DIR/$SO101_SETUP.yaml"
```

Keep `setup` equal to the file's canonical setup name and keep
`schema_version: 1`. The usual machine-specific fields are arm `usb_port`,
`tf_xyz`, and `tf_yaw_deg`; camera `device` or `gscam_config`; and, for the
single-pair setups, `sim` camera geometry. `bimanual` has no `sim` section and
is not supported by the Isaac Sim workflow.

Arm namespaces and camera IDs, features, and topics form the dataset and policy
contract. Leave them unchanged unless you intend to change that contract across
the complete stack. If you do change it, also point conversion, inference, and
the standalone visualization scripts at the directory containing the custom
setup:

```bash
export SO101_SETUPS_DIR="$SO101_CONFIG_DIR"
```

#### Configuration files

| File | Purpose | How it is selected |
|------|---------|--------------------|
| `so101_bringup/config/setups/<setup>.yaml` | Arm namespaces, USB ports, TF placement, logical camera contract, physical devices, and optional Isaac Sim geometry | `setup:=...`; use `setup_config_file:=/absolute/path.yaml` for an edited copy |
| `so101_bringup/config/ros2_control/{leader,follower}_controllers.yaml` | Reusable controller types, six-joint order, position command interface, and 100 Hz controller-manager rate | Loaded automatically for every arm namespace |
| `so101_teleop/config/teleop.yaml` | Leader/follower topic defaults, stale timeout, and joint order; command publication is fixed at 30 Hz | `teleop_params_file:=/absolute/path.yaml` on top-level teleop and recording launches |
| `episode_recorder/config/recorder.yaml` | MCAP storage and episode freshness/timing settings | `params_file:=/absolute/path.yaml` on `episode_recorder/recorder.launch.py`; common output and task values are top-level launch arguments |
| `rosbag_to_lerobot/config/so101_30hz.yaml` | Single-arm 30 Hz state/action conversion schema | `--config .../so101_30hz.yaml` with either single-arm `--setup` |
| `rosbag_to_lerobot/config/bimanual_30hz.yaml` | Bimanual 30 Hz state/action conversion schema | `--config .../bimanual_30hz.yaml --setup bimanual` |

Camera features in converted datasets always come from `--setup`, not from
camera entries in the conversion YAML. See [Extra Configuration](#extra-configuration)
for the complete launch-argument reference and common combinations.

#### ROS 2 controls

Real arms use `feetech_ros2_driver/FeetechHardwareInterface` over the setup's
USB port. `hardware_type:=mock` replaces it with
`mock_components/GenericSystem` for wiring tests without connected hardware.
The setup YAML supplies arm namespaces and ports; controller parameters are
static wildcard YAMLs reused in each namespace rather than generated from the
setup.

| Arm role | Controllers | Interfaces |
|----------|-------------|------------|
| Leader | `joint_state_broadcaster` | Position and velocity state only |
| Follower | `joint_state_broadcaster`, `forward_controller` | Position and velocity state; absolute position command |

The forward command is a `Float64MultiArray` containing six absolute joint
positions in this order: `shoulder_pan`, `shoulder_lift`, `elbow_flex`,
`wrist_flex`, `wrist_roll`, `gripper`. Teleop publishes it at the canonical
30 Hz; each controller manager runs at 100 Hz.

#### ROS 2 topics

The setup expands the topic patterns below. `<arm>` means any leader or
follower namespace in the selected setup, while `<camera>` means a logical
camera namespace.

| Setup | Arm namespaces | Camera namespaces |
|-------|----------------|-------------------|
| `monomanual` | `leader`, `follower` | wrist: `follower`; overhead 1: `static_camera_1` |
| `monomanual_dual_overhead` | `leader`, `follower` | wrist: `follower`; overheads: `static_camera_1`, `static_camera_2` |
| `bimanual` | `leader_left`, `leader_right`, `follower_left`, `follower_right` | wrists: `follower_left`, `follower_right`; overhead: `static_camera_1` |

| Topic | Type | Purpose |
|-------|------|---------|
| `/<leader>/joint_states` | `sensor_msgs/msg/JointState` | Leader position and velocity state consumed by teleop |
| `/<follower>/joint_states` | `sensor_msgs/msg/JointState` | Physical or mock follower state used by recording, visualization, and inference |
| `/follower_sim/joint_states` | `sensor_msgs/msg/JointState` | Isaac Sim follower state for either single-arm setup |
| `/<follower>/forward_controller/commands` | `std_msgs/msg/Float64MultiArray` | Absolute six-joint targets from teleop or inference; Isaac Sim consumes the single-arm `/follower/...` topic too |
| `/<camera>/image_raw` | `sensor_msgs/msg/Image` | Raw physical or Isaac Sim image |
| `/<camera>/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` | JPEG stream recorded to MCAP and used by Rerun and optional compressed inference |
| `/<camera>/camera/camera_info` | `sensor_msgs/msg/CameraInfo` | Physical `gscam` calibration metadata |
| `/<camera>/camera_info` | `sensor_msgs/msg/CameraInfo` | Isaac Sim calibration metadata |
| `/<arm>/robot_description` | `std_msgs/msg/String` | Namespaced URDF published for `ros2_control` and 3D visualization |
| `/<arm>/dynamic_joint_states` | `control_msgs/msg/DynamicJointState` | Complete state-interface values from the joint-state broadcaster |
| `/<arm>/controller_manager/activity` | `controller_manager_msgs/msg/ControllerManagerActivity` | Controller and hardware lifecycle changes |
| `/tf` | `tf2_msgs/msg/TFMessage` | Dynamic arm transforms |
| `/tf_static` | `tf2_msgs/msg/TFMessage` | Arm placement and other static transforms |
| `/parameter_events` | `rcl_interfaces/msg/ParameterEvent` | Standard ROS 2 parameter changes |
| `/rosout` | `rcl_interfaces/msg/Log` | Standard ROS 2 logs |


---

## Quick start

### Initial setup
Initial setup should be run before every usage, or add this to `.bashrc`.

```bash
# Setup env
source /opt/ros/jazzy/setup.bash

# Init
export SO101_REPO=/home/anon/Documents/so101-ros2
source $SO101_REPO/install/setup.bash

# Rerun bridges run via Pixi
export SO101_RERUN_ENV_DIR=$SO101_REPO

# Lets the Pixi bridge tasks source this workspace
export ROS_WS=$SO101_REPO
```

---

## Teleop

### Physical leader and physical follower

Mirror the physical leader arm to the physical follower through the `/follower/forward_controller/commands` topic at 30 Hz.

**Camera**
```bash
# Setup
source /opt/ros/jazzy/setup.bash
source /home/anon/Documents/so101-ros2/install/setup.bash

export SO101_SETUP=monomanual_dual_overhead
export SO101_REPO=/home/anon/Documents/so101-ros2

# Launch
ros2 launch so101_bringup teleop.launch.py \
  setup:=$SO101_SETUP \
  use_teleop_rviz:=true
```

**No Camera**
```bash
# Setup
source /opt/ros/jazzy/setup.bash
source /home/anon/Documents/so101-ros2/install/setup.bash

# Launch
ros2 launch so101_bringup teleop.launch.py \
  use_cameras:=false
```

To use different visualisation, `use_rerun:=true` or `use_rerun_3d:=true`, See [visualization](#visualization) for more information.

### Physical leader and Isaac Sim follower

Isaac Sim can replace the follower while the physical leader and 30 Hz relay remain unchanged, using the ROS 2 Bridge Action Graph directly. This requires two terminals.

**Terminal 1 — Isaac Sim follower:**

```bash
# Setup
source /opt/ros/jazzy/setup.bash
ISAACSIM_BUILD=/home/anon/Documents/isaacsim/_build/linux-x86_64/release
SO101_REPO=/home/anon/Documents/so101-ros2
source "$SO101_REPO/install/setup.bash"

# Run
"$ISAACSIM_BUILD/python.sh" "$SO101_REPO/scripts/isaac_sim_teleop.py" \
  --setup monomanual_dual_overhead

# Use `--rebuild-asset` after changing the Xacro or meshes
```

**Terminal 2 — physical leader and command relay:**

```bash
# Setup
source /opt/ros/jazzy/setup.bash
SO101_REPO=/home/anon/Documents/so101-ros2
source "$SO101_REPO/install/setup.bash"

# Launch
ros2 launch so101_bringup teleop.launch.py \
  use_follower:=false \
  use_cameras:=false \
  use_teleop_rviz:=false
```

### Physical leader with both followers

The physical follower and the Isaac Sim follower can mirror the leader at the same time. Both subscribe to the same command topic.

**Terminal 1 — physical leader, physical follower, and command relay:**

```bash
# Init
source /opt/ros/jazzy/setup.bash
SO101_REPO=/home/anon/Documents/so101-ros2
source "$SO101_REPO/install/setup.bash"

# Launch
ros2 launch so101_bringup teleop.launch.py use_cameras:=false
```

**Terminal 2 — Isaac Sim follower mirror:**

```bash
# Init
source /opt/ros/jazzy/setup.bash
ISAACSIM_BUILD=/home/anon/Documents/isaacsim/_build/linux-x86_64/release
SO101_REPO=/home/anon/Documents/so101-ros2
source "$SO101_REPO/install/setup.bash"

# Launch
"$ISAACSIM_BUILD/python.sh" "$SO101_REPO/scripts/isaac_sim_teleop.py" \
  --setup none \
  --no-publish-clock
```

---

## Data collection

Record teleoperated episodes (joint states + camera frames + commands) to timestamped MCAP episodes.

### Physical dataset recording

This workflow uses the physical leader, physical follower, and physical cameras. This requires two terminals.

**Terminal 1 — physical recording session:**

```bash
# Setup
source /opt/ros/jazzy/setup.bash
source /home/anon/Documents/so101-ros2/install/setup.bash

export SO101_SETUP=monomanual_dual_overhead
export SO101_REPO=/home/anon/Documents/so101-ros2

# Launch
ros2 launch so101_bringup recording_session.launch.py \
  setup:=$SO101_SETUP \
  experiment_name:=pick_and_place \
  task:="Pick up the cube and place it in the container." \
  use_rerun:=true
```

Wait until the launch reports that all required topics are ready. Then start the keyboard controller.

**Terminal 2 — episode controls:**

```bash
# Setup
source /opt/ros/jazzy/setup.bash
source /home/anon/Documents/so101-ros2/install/setup.bash

# Launch
ros2 run episode_recorder teleop_episode_keyboard
```

The keyboard controls the recorder:
- **r** or **Right Arrow**: start recording an episode.
- **s** or **Left Arrow**: stop and save the current episode.
- **d** or **Backspace**: discard the current episode.
- **t**: change the recorder's task description.
- **h**: show the key bindings.
- **q**: quit the keyboard controller.

Episodes are stored in `~/.ros/so101_episodes/$experiment_name/`.

### Isaac Sim dataset recording

This workflow keeps the physical leader but replaces the follower and all three cameras with Isaac Sim. This requires two or three terminals.

**Terminal 1 — Isaac Sim follower and cameras:**

```bash
# Setup
source /opt/ros/jazzy/setup.bash
ISAACSIM_BUILD=/home/anon/Documents/isaacsim/_build/linux-x86_64/release
SO101_REPO=/home/anon/Documents/so101-ros2
source "$SO101_REPO/install/setup.bash"

# Launch
"$ISAACSIM_BUILD/python.sh" "$SO101_REPO/scripts/isaac_sim_teleop.py" \
  --setup monomanual_dual_overhead
```

Wait for `Isaac Sim is ready`, then press **Play**. Isaac Sim publishes the
simulated follower joint states on `/follower_sim/joint_states`.

**Terminal 2 — physical leader, relay, JPEG republishers, watchdog, and recorder:**

```bash
# Setup
source /opt/ros/jazzy/setup.bash
SO101_REPO=/home/anon/Documents/so101-ros2
source "$SO101_REPO/install/setup.bash"

# Launch
ros2 launch so101_bringup recording_session.launch.py \
  setup:=monomanual_dual_overhead \
  use_follower:=false \
  use_cameras:=false \
  use_sim_cameras:=true \
  experiment_name:=pick_and_place \
  task:="Pick up the cube and place it in the container."
```

This republishes Isaac's three raw images as the existing `image_raw/compressed` JPEG topics. Wait until all five required topics are reported ready before starting an episode.

**Terminal 3 — episode controls:**

```bash
# Setup
source /opt/ros/jazzy/setup.bash
source /home/anon/Documents/so101-ros2/install/setup.bash

# Launch
ros2 run episode_recorder teleop_episode_keyboard
```

The keyboard controls the recorder:

- **r** or **Right Arrow**: start recording an episode.
- **s** or **Left Arrow**: stop and save the current episode.
- **d** or **Backspace**: discard the current episode.
- **t**: change the recorder's task description.
- **h**: show the key bindings.
- **q**: quit the keyboard controller.


The recorder records `/follower_sim/joint_states`. Convert the result with `so101_30hz.yaml` and `--setup monomanual_dual_overhead` as shown
in [LeRobot dataset conversion](#lerobot-dataset-conversion), adding `--joint-states-topic /follower_sim/joint_states` because the simulated follower publishes its state on that topic.


### Review episodes
Review episodes in the browser, this replays MCAP through ROS2 so images/joints/actions decode with the live stack.

```bash
# Launch
pixi run replay -- \
  --episodes_root ~/.ros/so101_episodes/pick_and_place \
  --setup $SO101_SETUP
```

See the [episode_recorder README](episode_recorder/README.md) for details.

---

## Policy Self-Improvement

### Physical Robot Self Improvement


### Issac Sim Self Improvement

---

## LeRobot dataset conversion

Convert recorded MCAP episodes into LeRobot v3.0 datasets (local or on the Hub) using the `lerobot` Pixi env:

```bash
# Convert to local dataset
pixi run -e lerobot convert -- \
  --input-dir ~/.ros/so101_episodes/pick_and_place \
  --config $SO101_REPO/rosbag_to_lerobot/config/so101_30hz.yaml \
  --setup $SO101_SETUP \
  --repo-id local/so101_test

# Convert and push to the Hub
pixi run -e lerobot convert -- \
  --input-dir ~/.ros/so101_episodes/pick_and_place \
  --config $SO101_REPO/rosbag_to_lerobot/config/so101_30hz.yaml \
  --setup $SO101_SETUP \
  --repo-id <hf-username>/so101-pick-and-place \
  --push-hub
```

If the output already exists, conversion stops, pass `--overwrite` to rebuild from scratch. The `observation.state` topic resolves per episode: `/follower/joint_states` for physical-follower recordings, `/follower_sim/joint_states` for Isaac Sim follower recordings — so directories mixing both sources convert in a single run. Pass an explicit `--joint-states-topic` to pin one topic for every episode (single-arm setups only). Bimanual recordings convert with `--config .../bimanual_30hz.yaml --setup bimanual`. See the [rosbag_to_lerobot README](rosbag_to_lerobot/README.md).

---

## Training

Train with the LeRobot CLI inside the `lerobot` env:

```bash
pixi shell -e lerobot
lerobot-train \
  --dataset.repo_id=<hf-username>/so101-pick-and-place \
  --policy.type=act \
  --output_dir=outputs/train/act_so101_pick_place \
  --policy.device=cuda
```

See the [LeRobot training docs](https://huggingface.co/docs/lerobot/il_robots#train-a-policy) for all options.

---

## Inference

Deploy trained LeRobot policies, all modes publish to the same `/follower/forward_controller/commands` topic as teleop which means both a physical arm and Isaac Sim arm can recieve this.

**Option A — bringup + sync inference together** (`policy_type` defaults to `act`). Any LeRobot policy runs, but the forward pass executes inline on the timer thread — best for fast models (`act`, `smolvla`). Heavy VLAs work but stutter because each forward pass blocks the control loop; use Option B for those.

```bash
ros2 launch so101_bringup inference.launch.py \
  setup:=$SO101_SETUP \
  use_inference:=true \
  repo_id:=your-org/your-dual-overhead-act-policy
```

**Option B — async inference offloaded to a GPU server** (LeRobot policy via [`policy_server`](policy_server/README.md) over ZMQ/gRPC). The forward pass runs in a separate inference thread while an action queue + temporal aggregation hide its latency — the recommended path for most VLAs. The server can run on a remote GPU box or on `localhost` using your local GPUs. Bring up the follower + cameras, then run the client in the `lerobot` env:

```bash
# Launch arm control to read ROS2 topics
ros2 launch so101_bringup follower_vision.launch.py \
  setup:=$SO101_SETUP

# Inference server for ACT / SmolVLA to ROS2 topics
pixi run -e lerobot async_infer -- --ros-args \
    -p repo_id:="your-org/your-dual-overhead-smolvla-policy" \
    -p setup:=$SO101_SETUP \
    -p policy_type:=smolvla \
    -p task:="Pick up the cube and place it in the container." \
    -p server_address:=192.168.1.100:8090

# Inference server for  Pi0.5  to ROS2 topics — set actions_per_chunk to ~half the policy's chunk size
pixi run -e lerobot async_infer -- --ros-args \
    -p repo_id:="your-org/your-dual-overhead-xvla-policy" \
    -p setup:=$SO101_SETUP \
    -p policy_type:=pi05 \
    -p task:="Pick up the cube and place it in the container." \
    -p server_address:=192.168.1.100:8090 \
    -p actions_per_chunk:=16
```

The policy's image features and state dimension must exactly match the selected `setup`. See the [so101_inference README](so101_inference/README.md) and [policy_server README](policy_server/README.md) for all parameters.

---

## Visualization

### RViz

Launched by default with `teleop.launch.py` (`use_teleop_rviz:=true`). Shows arm meshes animating live from `/tf`, plus camera image panels.


### Rerun

Runs in the browser via Pixi. Two variants — both show camera feeds plus `state/position/<joint>` and `action/position/<joint>` time-series plots:

| Task | What it adds |
|------|--------------|
| `use_rerun:=true` / `pixi run bridge` | 2D camera views + joint/command plots |
| `use_rerun_3d:=true` / `pixi run bridge-3d` | 2D + animated SO-101 arm mesh (URDF + `/tf`) |

```bash
# During teleop / recording / inference, swap RViz for rerun:
ros2 launch so101_bringup teleop.launch.py \
  setup:=$SO101_SETUP \
  use_rerun_3d:=true use_teleop_rviz:=false

# Or standalone
pixi run bridge-3d -- --setup $SO101_SETUP
```

Rerun launches its own web viewer by default. Pass `--viewer native` to open the native Rerun app instead.

---

## Packages

| Package | Lang | Purpose |
|---------|------|---------|
| [`so101_bringup`](so101_bringup/) | Python (launch) | Top-level launch files, controller/camera configs, RViz, TF layout |
| [`so101_description`](so101_description/) | Xacro/URDF | Robot model, STL meshes, `ros2_control` hardware macros |
| [`so101_teleop`](so101_teleop/) | C++ | Leader→follower relay to the forward-command interface |
| [`episode_recorder`](episode_recorder/) | C++ | Lifecycle MCAP recorder with profile-derived topics + keyboard control |
| [`rosbag_to_lerobot`](rosbag_to_lerobot/) | Python | Convert MCAP episodes to LeRobot v3.0 datasets (Pixi `lerobot` env) |
| [`so101_inference`](so101_inference/) | Python | Sync (on-device) and async (remote GPU) policy inference nodes |
| [`policy_server`](policy_server/) | Python | Remote GPU inference server (ZMQ/gRPC) |
| [`feetech_ros2_driver`](feetech_ros2_driver/) | C++ | **Submodule** — Feetech STS3215 `ros2_control` hardware interface |
| [`scripts/`](scripts/) | Python | Rerun bridges (2D, 3D) and episode replay viewer |

---

## Extra Configuration

### Launch arguments

These are the cross-cutting arguments accepted by the top-level launch files (`teleop.launch.py`, `recording_session.launch.py`, `inference.launch.py`, …). They split into three groups.

**Robot hardware**

| Argument | Default | Description |
|----------|---------|-------------|
| `hardware_type` | `real` | Selects which `ros2_control` hardware plugin the URDF/xacro compiles in. `real` → the `feetech_ros2_driver` talks to the physical STS3215 servos over USB. `mock` → `mock_components/GenericSystem`, a fake plant that simulates the 6 joints in software with no hardware attached. Use `mock` to test launch files, the recorder, or inference wiring without the arms plugged in. (Source comments mention `mujoco` but no plugin is wired for it — treat it as unimplemented.) |
| `use_follower` | `true` | Start the follower `ros2_control` stacks (one per follower in the setup). Set `false` when Isaac Sim is the follower (single-pair setups only); leave `true` to run the physical follower and the Isaac Sim script together (both follow the same command topic, joint states stay on `/follower/joint_states` and `/follower_sim/joint_states`). In the recording session `false` also switches the recorder to `/follower_sim/joint_states`. |

Namespaces, USB ports, and world placement of every arm come from the setup YAML — there are no per-arm launch arguments.

**Cameras and setups**

A `camera_supervisor` node enforces both timeouts and **shuts down the whole launch** if any camera misses them.

| Argument | Default | Description |
|----------|---------|-------------|
| `use_cameras` | `true` | Master gate. When `true`, spawns every camera driver in the selected setup plus the supervisor. When `false`, no camera nodes start — useful for motor-only flows or when cameras are temporarily down. |
| `use_sim_cameras` | `false` | Recording-session gate for Isaac streams. When `true`, starts raw-to-JPEG republishers plus the same fail-fast watchdog. It is mutually exclusive with `use_cameras`; use `use_cameras:=false use_sim_cameras:=true` for simulation. |
| `setup` | required | The canonical rig: `monomanual` = 1 arm pair + wrist + 1 overhead; `monomanual_dual_overhead` = 1 arm pair + wrist + 2 overheads; `bimanual` = 2 arm pairs + 2 wrists + 1 shared overhead. One YAML in `so101_bringup/config/setups/` fully defines the setup (arms, logical camera contract, physical rig, sim geometry). This single value is enforced identically across bringup, recording, Rerun, conversion, and inference — it determines which arms spawn, which image topics get recorded, which become LeRobot features, and which the policy expects at inference. Use the same setup end-to-end. |
| `setup_config_file` | *(setup YAML in the repo)* | Optional absolute path to an edited copy of a setup YAML — e.g. your machine's device paths. At launch, each camera device is validated to exist and be a RW character device before any driver starts — a bad path fails immediately. |
| `camera_startup_timeout_s` | `10.0` | **Cold-start deadline.** The supervisor waits this long for every camera in the setup to publish its first frame. If any stream doesn't appear in time, the supervisor exits → the whole launch shuts down. Raise this on a slow USB bus or cold boot. The recorder also independently refuses to start an episode until every topic is fresh (`start_gate_max_age_s`). |
| `camera_stale_timeout_s` | `1.0` | **Runtime freshness limit.** Once cameras are up, the supervisor flags a stream as dead if no frame arrives for this long → launch shuts down. 1.0 s is conservative for 30 fps cameras; tighten to catch USB drops faster, loosen for flaky hardware. (The recorder and inference node have *separate* freshness gates — `start_gate_max_age_s` and `max_age_s` respectively — so this is the system-wide watchdog, not the only check.) |

**Visualization Options**

| Argument | Default | Description |
|----------|---------|-------------|
| `use_teleop_rviz` | `true` | `teleop.launch.py` only — launches RViz with the teleop config, both arm meshes animating from `/tf` plus camera image panels. Best for debugging the robot model / TF tree. Disable if RViz crashes (e.g. `inotify` watch limit) or isn't needed. (The recording session never starts RViz.) |
| `use_rerun` | `false` | Launches the **2D Rerun bridge** (Pixi task `bridge`) — camera feeds + `state/position/<joint>` and `action/position/<joint>` time-series plots in a browser. Lighter than RViz. |
| `use_rerun_3d` | `false` | Launches the **3D Rerun bridge** (Pixi task `bridge-3d`) — everything in 2D plus an animated SO-101 arm mesh from the URDF + `/tf`. Use as the richer alternative to RViz; typically `use_teleop_rviz:=false use_rerun_3d:=true` to swap them. |

**How they combine in practice**

```bash
# Minimal real-hardware teleop - cameras + RViz, all defaults
ros2 launch so101_bringup teleop.launch.py \
  setup:=monomanual_dual_overhead

# Bimanual rig
ros2 launch so101_bringup teleop.launch.py setup:=bimanual

# No-camera motor test on real arms
ros2 launch so101_bringup teleop.launch.py \
  use_cameras:=false

# Fully headless - mock arms, no cameras, no RViz
ros2 launch so101_bringup teleop.launch.py \
  hardware_type:=mock use_cameras:=false use_teleop_rviz:=false

# Real recording with the 3D Rerun bridge
ros2 launch so101_bringup recording_session.launch.py \
  setup:=monomanual_dual_overhead \
  use_rerun_3d:=true
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
