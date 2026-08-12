# SO-101 ROS - Physical AI


ROS 2 stack for the SO-101 robot arm in a leader/follower configuration: Feetech STS3215 servos via `ros2_control`, leader-to-follower teleop, multi-camera episode recording for imitation learning, LeRobot dataset conversion, policy inference, and live visualization — all on real hardware.

**End-to-end workflow:** [teleop](#teleop) → [record episodes](#data-collection) → [convert to LeRobot](#lerobot-dataset-conversion) → [train](#training) → [run policies](#inference)

The repository also contains a separate, external Isaac Lab reinforcement-learning
project for state and deployable three-camera SO-101 object-in-cup PPO training.
Its visual actor consumes the same wrist/two-overhead RGB streams and six absolute
joint positions available on the real robot, while a privileged state is used only
by the training critic. It uses this repository's generated robot USD and supplied
STL assets without modifying Isaac Lab or putting ROS 2 in the training loop. See
[isaaclab/README.md](isaaclab/README.md) for asset preparation, visual PPO,
multi-GPU/live playback, export, and shadow-mode deployment. The design,
reward equations, algorithm support, neural-network configuration, and task/model
extension process are in
[isaaclab/METHODOLOGY.md](isaaclab/METHODOLOGY.md).

---

## Requirements

- **Ubuntu 24.04** + **ROS 2 Jazzy**
- Two SO-101 arms for hardware teleop, or one leader plus Isaac Sim 6.0.1 for simulated-follower teleop
- Two or three cameras, matching the `single_overhead` or `dual_overhead` profile
- `rosdep`, `colcon`, and [Pixi](https://pixi.sh/)

> **Before launching ROS**, complete LeRobot motor setup + calibration for both arms and create the udev rules / camera rig YAML. ROS uses the calibration stored by LeRobot — do not command the arms until this is done.
> **[→ Full hardware setup guide (docs/hardware.md)](docs/hardware.md)**

---

## Installation

```bash
# Clone (includes submodules)
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone --recurse-submodules https://github.com/legalaspro/so101-ros-physical-ai.git
cd ~/ros2_ws

# Install deps and build
sudo apt update
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

The Rerun/LeRobot/visualization tooling runs in two [Pixi](https://pixi.sh/) environments declared in [`pixi.toml`](pixi.toml):

```bash
pixi install --all
```

- **`default`** — Rerun bridges and episode replay (`bridge`, `bridge-3d`, `replay`, `viewer`)
- **`lerobot`** — LeRobot 0.6.1 dataset conversion and policy inference (`convert`, `infer`, `async_infer`)

---

## Quick start

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

# Point at your camera rig and the repo root (for the Pixi visualization tasks)
export SO101_CAMERA_PROFILE=dual_overhead
export SO101_CAMERA_RIG=/absolute/path/to/camera_rig.yaml
export SO101_RERUN_ENV_DIR=~/ros2_ws/src/so101-ros-physical-ai
```

Then pick a workflow below.

---

## Teleop

### Physical leader and follower

Mirror the physical leader arm to the physical follower through the `forward_controller` position interface at 30 Hz.
Run this in one terminal, replacing the camera-rig path with the YAML for the connected physical cameras:

```bash
source /opt/ros/jazzy/setup.bash
source /home/anon/Documents/so101-ros2/install/setup.bash

export SO101_CAMERA_PROFILE=dual_overhead
export SO101_CAMERA_RIG=/absolute/path/to/camera_rig.yaml

ros2 launch so101_bringup teleop.launch.py \
  camera_profile:=$SO101_CAMERA_PROFILE \
  camera_rig_config_file:=$SO101_CAMERA_RIG
```

This launch starts the leader and follower `ros2_control` stacks, the 30 Hz command relay, the selected physical
cameras, and RViz. For arm-only teleop without cameras, omit the camera variables and run:

```bash
source /opt/ros/jazzy/setup.bash
source /home/anon/Documents/so101-ros2/install/setup.bash

ros2 launch so101_bringup teleop.launch.py \
  use_cameras:=false
```

Useful visualization overrides are `use_teleop_rviz:=false`, `use_rerun:=true`, and `use_rerun_3d:=true`. See
[Visualization](#visualization).

### Isaac Sim follower

Isaac Sim can replace the follower while the physical leader and existing 30 Hz relay remain unchanged. This uses the
ROS 2 Bridge Action Graph directly; it does not use `mock_components`, an Isaac Lab task, or a `ros2_control` simulator
plugin.

**Terminal 1 — Isaac Sim follower:**

```bash
source /opt/ros/jazzy/setup.bash
ISAACSIM_BUILD=/home/anon/Documents/isaacsim/_build/linux-x86_64/release
SO101_REPO=/home/anon/Documents/so101-ros2
source "$SO101_REPO/install/setup.bash"

"$ISAACSIM_BUILD/python.sh" "$SO101_REPO/scripts/isaac_sim_teleop.py" \
  --camera-profile dual_overhead \
  --camera-rig-config "$SO101_REPO/so101_bringup/config/cameras/isaac_dual_overhead.yaml"
```

The first run expands the follower Xacro with `use_ros2_control:=false`, validates its six joints, and imports a
fixed-base USD into the ignored `build/isaacsim_so101/` directory. Use `--rebuild-asset` after changing the Xacro or
meshes. The importer creates no symlinks. The default `dual_overhead` camera profile also converts the two supplied
mount STLs into that cache, assembles the left/right supports, and creates all three RTX cameras at 640×480 and 30 Hz.

The editable first-pass calibration is
[`so101_bringup/config/cameras/isaac_dual_overhead.yaml`](so101_bringup/config/cameras/isaac_dual_overhead.yaml). It
owns the support placement, wrist transform, camera look-at targets, FOV, topics, and frame ids. The camera
contract is fixed: wrist publishes on `/follower/image_raw`, left `overhead_1` on
`/static_camera_1/image_raw`, and right `overhead_2` on `/static_camera_2/image_raw`; each also publishes its matching
`camera_info`. No physical camera-rig file is used for these simulated streams. Press **Play** after the scene is ready.

**Terminal 2 — physical leader and command relay:**

```bash
source /opt/ros/jazzy/setup.bash
SO101_REPO=/home/anon/Documents/so101-ros2
source "$SO101_REPO/install/setup.bash"

ros2 launch so101_bringup teleop.launch.py \
  use_follower:=false \
  use_cameras:=false \
  use_teleop_rviz:=false
```

Both terminals must use the same `ROS_DOMAIN_ID` and DDS implementation. The Isaac Sim process subscribes to
`/follower/forward_controller/commands` and publishes `/follower/joint_states` and `/clock` while the timeline plays.

Before moving the physical leader, run the direct-message smoke test:

```bash
ros2 topic pub --once /follower/forward_controller/commands \
  std_msgs/msg/Float64MultiArray \
  "{data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}"
ros2 topic echo --once /follower/joint_states
ros2 topic info --verbose /follower/joint_states
```

The initial position-drive values are `100 Nm/rad` stiffness and `2 Nm*s/rad` damping. Override them with
`--joint-stiffness` and `--joint-damping` if the imported arm is underdamped or too slow. The URDF's `10 Nm` effort
limits supply the drive maximum force.

---

## Data collection

Record teleoperated episodes (joint states + camera frames + commands) to timestamped MCAP episodes.

### Physical dataset recording

This workflow uses the physical leader, physical follower, and physical cameras. The recording launch starts both arm
stacks, the teleop relay, camera drivers, camera watchdog, and episode recorder.

**Terminal 1 — physical recording session:**

```bash
source /opt/ros/jazzy/setup.bash
source /home/anon/Documents/so101-ros2/install/setup.bash

export SO101_CAMERA_PROFILE=dual_overhead
export SO101_CAMERA_RIG=/absolute/path/to/camera_rig.yaml

ros2 launch so101_bringup recording_session.launch.py \
  camera_profile:=$SO101_CAMERA_PROFILE \
  camera_rig_config_file:=$SO101_CAMERA_RIG \
  experiment_name:=pick_and_place \
  task:="Pick up the cube and place it in the container."
```

Wait until the launch reports that all required topics are ready. Then start the keyboard controller.

**Terminal 2 — episode controls:**

```bash
source /opt/ros/jazzy/setup.bash
source /home/anon/Documents/so101-ros2/install/setup.bash

ros2 run episode_recorder teleop_episode_keyboard
```

The keyboard controls the recorder, not the arm:

- **r** or **Right Arrow**: start recording an episode.
- **s** or **Left Arrow**: stop and save the current episode.
- **d** or **Backspace**: discard the current episode.
- **t**: change the recorder's task description.
- **h**: show the key bindings.
- **q**: quit the keyboard controller.

Press **r**, operate the leader arm, and then press **s** to save the episode or **d** to discard it. Repeat for each
episode. Episodes are stored in `~/.ros/so101_episodes/pick_and_place/` for the example above. Ctrl-C while recording
discards the in-progress episode.

### Isaac Sim dataset recording

This workflow keeps the physical leader but replaces the follower and all three cameras with Isaac Sim. It needs three
terminals.

**Terminal 1 — Isaac Sim follower and cameras:**

```bash
source /opt/ros/jazzy/setup.bash
ISAACSIM_BUILD=/home/anon/Documents/isaacsim/_build/linux-x86_64/release
SO101_REPO=/home/anon/Documents/so101-ros2
source "$SO101_REPO/install/setup.bash"

"$ISAACSIM_BUILD/python.sh" "$SO101_REPO/scripts/isaac_sim_teleop.py" \
  --camera-profile dual_overhead \
  --camera-rig-config "$SO101_REPO/so101_bringup/config/cameras/isaac_dual_overhead.yaml"
```

Wait for `Isaac Sim is ready`, then press **Play**. Isaac Sim now owns the simulated follower joint states and the three
raw camera streams.

**Terminal 2 — physical leader, relay, JPEG republishers, watchdog, and recorder:**

```bash
source /opt/ros/jazzy/setup.bash
SO101_REPO=/home/anon/Documents/so101-ros2
source "$SO101_REPO/install/setup.bash"

ros2 launch so101_bringup recording_session.launch.py \
  camera_profile:=dual_overhead \
  use_follower:=false \
  use_cameras:=false \
  use_sim_cameras:=true \
  experiment_name:=pick_and_place \
  task:="Pick up the cube and place it in the container."
```

This republishes Isaac's three raw images as the existing `image_raw/compressed` JPEG topics; it does not use Isaac's
H.264 output. The camera watchdog monitors the raw streams and terminates collection if any selected stream stalls.
Wait until all five required topics are reported ready before starting an episode.

**Terminal 3 — episode controls:**

```bash
source /opt/ros/jazzy/setup.bash
source /home/anon/Documents/so101-ros2/install/setup.bash

ros2 run episode_recorder teleop_episode_keyboard
```

Press **r**, operate the physical leader, and press **s** to save or **d** to discard. The keyboard controls only the
recorder; the physical leader continues to control the simulated arm. Convert the result with `so101_30hz.yaml` and
`--camera-profile dual_overhead` as shown in [LeRobot dataset conversion](#lerobot-dataset-conversion).

**Review episodes** in the browser (replays MCAP through ROS so images/joints/actions decode with the live stack):

```bash
pixi run replay -- \
  --episodes_root ~/.ros/so101_episodes/pick_and_place \
  --camera-profile $SO101_CAMERA_PROFILE
```

See the [episode_recorder README](episode_recorder/README.md) for details.

### Record PPO demonstrations for VLA training

Use the follower-only recording stack so teleop cannot publish competing commands:

```bash
ros2 launch so101_bringup follower_recording.launch.py \
  camera_profile:=dual_overhead \
  camera_rig_config_file:=$SO101_CAMERA_RIG \
  experiment_name:=ppo_pick_and_place \
  task:="Pick up the cube and place it in the container."
```

In a second terminal, launch the exported PPO actor and validate it in shadow
mode before arming:

```bash
ros2 launch so101_inference rsl_rl_infer.launch.py \
  model_dir:="$ARTIFACT_DIR" camera_profile:=dual_overhead

ros2 service call /so101_rl/set_enabled \
  std_srvs/srv/SetBool "{data: true}"
```

Only after PPO is armed, start `teleop_episode_keyboard` and press **r**. The
recorder stores the 30 Hz absolute controller targets published by PPO; do not
run the leader/follower teleop relay during these episodes.

When converting these PPO-recorded episodes to a LeRobot dataset, pass
`--dataset-source ppo` so the Hub dataset is tagged `reinforcement-learning`
rather than `teleoperation` (see `rosbag_to_lerobot/README.md`).

---

## LeRobot dataset conversion

Convert recorded MCAP episodes into LeRobot v3.0 datasets (local or on the Hub) using the `lerobot` Pixi env:

```bash
# Local dataset
pixi run -e lerobot convert -- \
  --input-dir ~/.ros/so101_episodes/pick_and_place \
  --config ~/ros2_ws/src/so101-ros-physical-ai/rosbag_to_lerobot/config/so101_30hz.yaml \
  --camera-profile $SO101_CAMERA_PROFILE \
  --repo-id local/so101_test

# Convert and push to the Hub
pixi run -e lerobot convert -- \
  --input-dir ~/.ros/so101_episodes/pick_and_place \
  --config ~/ros2_ws/src/so101-ros-physical-ai/rosbag_to_lerobot/config/so101_30hz.yaml \
  --camera-profile $SO101_CAMERA_PROFILE \
  --repo-id <hf-username>/so101-pick-and-place \
  --push-hub
```

If the output already exists, conversion stops. Pass `--overwrite` to rebuild from scratch. See the [rosbag_to_lerobot README](rosbag_to_lerobot/README.md).

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

Deploy trained LeRobot policies on the follower arm. All modes publish to the same `/follower/forward_controller/commands` topic as teleop.

**Option A — bringup + sync inference together** (`policy_type` defaults to `act`). Any LeRobot policy runs on-device, but the forward pass executes inline on the timer thread — best for fast models (`act`, `smolvla`). Heavy VLAs work but stutter because each forward pass blocks the control loop; use Option B for those.

```bash
ros2 launch so101_bringup inference.launch.py \
  camera_profile:=$SO101_CAMERA_PROFILE \
  camera_rig_config_file:=$SO101_CAMERA_RIG \
  use_inference:=true \
  repo_id:=your-org/your-dual-overhead-act-policy
```

**Option B — async inference offloaded to a GPU server** (LeRobot policy via [`policy_server`](policy_server/README.md) over ZMQ/gRPC). The forward pass runs in a separate inference thread while an action queue + temporal aggregation hide its latency — the recommended path for heavy VLAs. The server can run on a remote GPU box or on `localhost` using your local GPUs. Bring up the follower + cameras, then run the client in the `lerobot` env:

```bash
# Arm control over ROS
ros2 launch so101_bringup follower_vision.launch.py \
  camera_profile:=$SO101_CAMERA_PROFILE \
  camera_rig_config_file:=$SO101_CAMERA_RIG

# Inference server for ACT / SmolVLA
pixi run -e lerobot async_infer -- --ros-args \
    -p repo_id:="your-org/your-dual-overhead-smolvla-policy" \
    -p camera_profile:=$SO101_CAMERA_PROFILE \
    -p policy_type:=smolvla \
    -p task:="Pick up the cube and place it in the container." \
    -p server_address:=192.168.1.100:8090

# Inference server for  Pi0.5 / x-VLA — set actions_per_chunk to ~half the policy's chunk size
pixi run -e lerobot async_infer -- --ros-args \
    -p repo_id:="your-org/your-dual-overhead-xvla-policy" \
    -p camera_profile:=$SO101_CAMERA_PROFILE \
    -p policy_type:=pi05 \
    -p task:="Pick up the cube and place it in the container." \
    -p server_address:=192.168.1.100:8090 \
    -p actions_per_chunk:=16
```

The policy's image features must exactly match the selected `camera_profile` (no feature-rename). See the [so101_inference README](so101_inference/README.md) and [policy_server README](policy_server/README.md) for all parameters.

---

## Visualization

### RViz (native 3D)

Launched by default with `teleop.launch.py` (`use_teleop_rviz:=true`). Shows both arm meshes animating live from `/tf`, plus camera image panels. Best for debugging the robot model and TF tree.


### Rerun

Runs in the browser via Pixi. Two variants — both show camera feeds plus `state/position/<joint>` and `action/position/<joint>` time-series plots:

| Task | What it adds |
|------|--------------|
| `use_rerun:=true` / `pixi run bridge` | 2D camera views + joint/command plots |
| `use_rerun_3d:=true` / `pixi run bridge-3d` | 2D + animated SO-101 arm mesh (URDF + `/tf`) |

```bash
# During teleop / recording / inference, swap RViz for the 3D bridge:
ros2 launch so101_bringup teleop.launch.py \
  camera_profile:=$SO101_CAMERA_PROFILE \
  camera_rig_config_file:=$SO101_CAMERA_RIG \
  use_rerun_3d:=true use_teleop_rviz:=false

# Or standalone
pixi run bridge-3d -- --camera-profile $SO101_CAMERA_PROFILE
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
| [`policy_server`](policy_server/) | Python | Remote GPU inference server (ZMQ/gRPC) — not a ROS package |
| [`feetech_ros2_driver`](feetech_ros2_driver/) | C++ | **Submodule** — Feetech STS3215 `ros2_control` hardware interface |
| [`scripts/`](scripts/) | Python | Rerun bridges (2D, 3D) and episode replay viewer |

---

## Configuration

### Launch arguments

These are the cross-cutting arguments accepted by the top-level launch files (`teleop.launch.py`, `recording_session.launch.py`, `inference.launch.py`, …). They split into three groups.

**Robot hardware**

| Argument | Default | Description |
|----------|---------|-------------|
| `hardware_type` | `real` | Selects which `ros2_control` hardware plugin the URDF/xacro compiles in. `real` → the `feetech_ros2_driver` talks to the physical STS3215 servos over USB. `mock` → `mock_components/GenericSystem`, a fake plant that simulates the 6 joints in software with no hardware attached. Use `mock` to test launch files, the recorder, or inference wiring without the arms plugged in. (Source comments mention `mujoco` but no plugin is wired for it — treat it as unimplemented.) |
| `use_follower` | `true` | Start the follower `ros2_control` stack. Set `false` when Isaac Sim owns the follower command and joint-state topics. |
| `leader_usb_port` | `/dev/so101_leader` | USB device path for the **leader** arm. These are udev-symlink stable names (created from the rules in `docs/hardware.md`), not raw `/dev/ttyUSB0` which can reorder on replug. Only used by teleop/recording launches (the leader is absent during inference). |
| `follower_usb_port` | `/dev/so101_follower` | USB device path for the **follower** arm — the one that actually moves. Used by every launch that brings up `ros2_control`. |

**Cameras (strict subsystem)**

The camera stack is deliberately fail-fast: a `camera_supervisor` node enforces both timeouts and **shuts down the whole launch** if any camera misses them.

| Argument | Default | Description |
|----------|---------|-------------|
| `use_cameras` | `true` | Master gate. When `true`, spawns every camera driver in the selected profile plus the supervisor. When `false`, no camera nodes start — useful for motor-only flows or when cameras are temporarily down. |
| `use_sim_cameras` | `false` | Recording-session gate for Isaac streams. When `true`, starts raw-to-JPEG republishers plus the same fail-fast watchdog. It is mutually exclusive with `use_cameras`; use `use_cameras:=false use_sim_cameras:=true` for simulation. |
| `camera_profile` | required | The **logical** camera set. `single_overhead` = wrist + 1 overhead; `dual_overhead` = wrist + 2 overheads. This single value is enforced identically across recording, Rerun, conversion, and inference — it determines which image topics get recorded, which become LeRobot features, and which the policy expects at inference. Use the same profile end-to-end. |
| `camera_rig_config_file` | required | Absolute path to the **external physical-rig YAML** (schema in `docs/hardware.md` §6.1). Not in the repo because it's machine-specific: it maps each profile camera id (`wrist`, `overhead_1`, `overhead_2`) to a real `/dev/...` device symlink via udev. At launch, each device is validated to exist and be a RW character device before any driver starts — a bad path fails immediately. |
| `camera_startup_timeout_s` | `10.0` | **Cold-start deadline.** The supervisor waits this long for every camera in the profile to publish its first frame. If any stream doesn't appear in time, the supervisor exits → the whole launch shuts down. Raise this on a slow USB bus or cold boot. The recorder also independently refuses to start an episode until every topic is fresh (`start_gate_max_age_s`). |
| `camera_stale_timeout_s` | `1.0` | **Runtime freshness limit.** Once cameras are up, the supervisor flags a stream as dead if no frame arrives for this long → launch shuts down. 1.0 s is conservative for 30 fps cameras; tighten to catch USB drops faster, loosen for flaky hardware. (The recorder and inference node have *separate* freshness gates — `start_gate_max_age_s` and `max_age_s` respectively — so this is the system-wide watchdog, not the only check.) |

**Visualization (pick one)**

| Argument | Default | Description |
|----------|---------|-------------|
| `use_teleop_rviz` | `true` | Launches RViz with the teleop config — both arm meshes animating from `/tf` plus camera image panels. Best for debugging the robot model / TF tree. Disable if RViz crashes (e.g. `inotify` watch limit) or isn't needed. |
| `use_rerun` | `false` | Launches the **2D Rerun bridge** (Pixi task `bridge`) — camera feeds + `state/position/<joint>` and `action/position/<joint>` time-series plots in a browser. Lighter than RViz. |
| `use_rerun_3d` | `false` | Launches the **3D Rerun bridge** (Pixi task `bridge-3d`) — everything in 2D plus an animated SO-101 arm mesh from the URDF + `/tf`. Use as the richer alternative to RViz; typically `use_teleop_rviz:=false use_rerun_3d:=true` to swap them. |

**How they combine in practice**

```bash
# Minimal real-hardware teleop (cameras + RViz, all defaults)
ros2 launch so101_bringup teleop.launch.py \
  camera_profile:=dual_overhead \
  camera_rig_config_file:=$SO101_CAMERA_RIG

# No-camera motor test on real arms
ros2 launch so101_bringup teleop.launch.py \
  use_cameras:=false

# Fully headless (mock arms, no cameras, no RViz) — pure launch/file smoke test
ros2 launch so101_bringup teleop.launch.py \
  hardware_type:=mock use_cameras:=false use_teleop_rviz:=false

# Real recording with Rerun 3D instead of RViz
ros2 launch so101_bringup recording_session.launch.py \
  camera_profile:=dual_overhead \
  camera_rig_config_file:=$SO101_CAMERA_RIG \
  use_teleop_rviz:=false use_rerun_3d:=true
```

> **Subtlety:** `camera_profile` and `camera_rig_config_file` are two halves of one config. The profile is the portable "what cameras, logically" (shipped in the repo, shared across machines); the rig file is the machine-specific "which `/dev/` node each one is" (external, never committed). Passing a profile without a rig file, or a rig file whose devices don't exist, fails at `load_camera_setup()` before any driver starts — fail fast, never record a partial camera set.

### Config files

- `so101_bringup/config/ros2_control/` — `forward_controller` + `joint_state_broadcaster` parameters
- `so101_bringup/config/cameras/profiles/` — the two canonical camera profiles; physical device paths and driver overrides live in the external rig YAML
- `so101_bringup/config/cameras/isaac_dual_overhead.yaml` — editable simulated mount/camera calibration used by the Isaac script
- `so101_teleop/config/teleop.yaml` — relay rate, stale timeout, joint order
- `episode_recorder/config/recorder.yaml` — MCAP storage and timing

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
