# SO-101 ROS Physical AI

> ROS 2 Jazzy · ros2_control · LeRobot · Rerun

[![ROS 2 Jazzy](https://img.shields.io/badge/ROS%202-Jazzy-blue?logo=ros)](https://docs.ros.org/en/jazzy/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Rerun](https://img.shields.io/badge/Rerun-Visualization-purple)](https://www.rerun.io/)

Complete ROS 2 stack for the SO-101 robot arm in a leader/follower configuration. Feetech STS3215 servo driver via ros2_control, leader-to-follower teleoperation, multi-camera support, episode recording for imitation learning, conversion to LeRobot datasets, policy training, policy inference, and live Rerun visualization — all on real hardware.

> Sync inference supports ACT and SmolVLA on-device. The default async environment supports ACT, SmolVLA, and Pi0.5 through `policy_server`; other LeRobot policies require their own optional dependency extras. PRs and issues welcome.

<p>
  <a href="https://www.youtube.com/watch?v=l6kWDoHxczc">
    <img src="https://img.youtube.com/vi/l6kWDoHxczc/hqdefault.jpg" alt="SO-101 async inference demo" width="720" />
  </a>
</p>

**Main demo:** run async policy inference on the real SO-101 arm with a ROS 2 client on the robot and the policy hosted on a remote GPU server via `policy_server`, including ACT and VLA-style LeRobot models trained on datasets collected with this repo's end-to-end pipeline. See [so101_inference](so101_inference/README.md) and [policy_server](policy_server/README.md) for details.

## What you can do with this repo

<table>
  <tr>
    <td width="33%" align="center"><strong>Teleoperate the robot</strong></td>
    <td width="33%" align="center"><strong>Record imitation learning episodes</strong></td>
    <td width="33%" align="center"><strong>Run learned policies</strong></td>
  </tr>
  <tr>
    <td width="33%" align="center">
      <a href="so101_teleop/README.md">
        <img src="docs/assets/gifs/so101_teleop.gif" alt="SO-101 teleop demo" />
      </a>
    </td>
    <td width="33%" align="center">
      <a href="episode_recorder/README.md">
        <img src="docs/assets/gifs/ros2_episode_recorder.gif" alt="SO-101 episode recorder demo" />
      </a>
    </td>
    <td width="33%" align="center">
      <a href="so101_inference/README.md">
        <img src="https://img.youtube.com/vi/l6kWDoHxczc/hqdefault.jpg" alt="SO-101 policy inference demo" />
      </a>
    </td>
  </tr>
  <tr>
    <td width="33%" align="center">
      <strong><a href="so101_teleop/README.md">Teleop</a></strong><br>
      Mirror the leader arm to the follower with the default and recommended <code>forward_controller</code>.
    </td>
    <td width="33%" align="center">
      <strong><a href="episode_recorder/README.md">Episode recorder</a></strong><br>
      Save synchronized robot + camera episodes with keyboard controls and optional live Rerun.
    </td>
    <td width="33%" align="center">
      <strong><a href="so101_inference/README.md">Policy inference</a></strong><br>
      Run LeRobot policies locally or on a remote GPU over ZMQ/gRPC.
    </td>
  </tr>
</table>

**End-to-end workflow:** [teleoperate the robot](so101_teleop/README.md) → [record episodes](episode_recorder/README.md) → [convert rosbags to LeRobot datasets](rosbag_to_lerobot/README.md) → [train policies](#training-lerobot) → [run learned policies](so101_inference/README.md)

## Best first things to try

After the [hardware setup guide](docs/hardware.md) and [installation](#installation), these are the fastest ways to get a feel for the repo:

> **Important**
> Before using the real arms with ROS 2, you must first complete the LeRobot motor setup and calibration steps for both arms. These steps write the required persistent values to the servo motors, including IDs and calibration-related settings stored in EEPROM.
>
> Do **not** teleoperate, plan, or command the real arms from ROS until this is done. ROS uses the motor calibration completed through LeRobot.
>
> **[→ See `docs/hardware.md` → `5. Motor Calibration`](docs/hardware.md#5-motor-calibration)**

Choose one camera profile and point at the external device rig file you created
during hardware setup:

```bash
export SO101_CAMERA_PROFILE=dual_overhead
export SO101_CAMERA_RIG=/absolute/path/to/camera_rig.yaml
export SO101_RERUN_ENV_DIR=~/ros2_ws/src/so101-ros-physical-ai
```

1. **Teleoperate the robot**

   ```bash
   ros2 launch so101_bringup teleop.launch.py \
     camera_profile:=$SO101_CAMERA_PROFILE \
     camera_rig_config_file:=$SO101_CAMERA_RIG
   ```

2. **Record a demonstration episode**

   ```bash
   ros2 launch so101_bringup recording_session.launch.py \
     camera_profile:=$SO101_CAMERA_PROFILE \
     camera_rig_config_file:=$SO101_CAMERA_RIG \
     experiment_name:=pick_and_place \
     task:="Pick up the cube and place it in the container." \
     use_rerun:=true
   ```

3. **Convert recorded episodes into a LeRobot dataset**

   ```bash
   pixi run -e lerobot convert -- \
     --input-dir ~/.ros/so101_episodes/pick_and_place \
     --config ~/ros2_ws/src/so101-ros-physical-ai/rosbag_to_lerobot/config/so101_30hz.yaml \
     --camera-profile $SO101_CAMERA_PROFILE \
     --repo-id local/so101_test
   ```

4. **Then dive deeper**

   - [Teleop guide](so101_teleop/README.md)
   - [Episode recorder guide](episode_recorder/README.md)
   - [LeRobot dataset conversion](rosbag_to_lerobot/README.md)
   - [Training with LeRobot](#training-lerobot)
   - [Policy inference](so101_inference/README.md)
   - [Remote GPU policy server](policy_server/README.md)

---

## Features

| Feature                           | Description                                                                                  |
| --------------------------------- | -------------------------------------------------------------------------------------------- |
| **Leader/Follower Teleop**        | Real-time six-joint mirroring from leader arm to the follower forward controller              |
| **ros2_control + Feetech Driver** | Hardware interface for STS3215 servos with configurable joint limits and calibration         |
| **Multi-Camera Pipeline**         | Strict single/dual overhead profiles with external physical-rig configuration and fail-fast supervision |
| **Episode Recording**             | Record joint states + camera frames into timestamped episodes for imitation learning         |
| **Rerun Visualization**           | Live visualization of observations, actions, and camera feeds via ROS-to-Rerun bridge (Pixi) |
| **Policy Inference**              | Sync: ACT and SmolVLA on-device. Async: ACT, SmolVLA, and Pi0.5 by default, offloaded to a remote GPU server via ZMQ/gRPC |
| **URDF/Xacro Model**              | Full SO-101 description with STL meshes, separate leader/follower end-effectors              |

---

## Packages

| Package               | Language        | Description                                                                                      |
| --------------------- | --------------- | ------------------------------------------------------------------------------------------------ |
| `so101_bringup`       | Python (launch) | Top-level launch files, controller/camera configs, ros2_control, recording, and TF layout        |
| `so101_description`   | Xacro/URDF      | Robot model, STL meshes, and ros2_control hardware interface macros                               |
| `so101_teleop`        | C++             | Leader-to-follower teleoperation through the canonical six-joint forward-command interface       |
| `episode_recorder`    | C++             | Lifecycle MCAP recorder with profile-derived topics and keyboard-driven episode control       |
| `rosbag_to_lerobot`   | Python          | Convert rosbag episodes to [LeRobot](https://github.com/huggingface/lerobot) v3.0 datasets (local or Hub) — runs in Pixi `lerobot` env |
| `so101_inference`     | Python          | Policy inference — sync (ACT, SmolVLA on-device) and async (ACT, SmolVLA, and Pi0.5 by default via a remote GPU server). See [so101_inference README](so101_inference/README.md) |
| `policy_server`       | Python          | GPU-side inference server — serves ACT, SmolVLA, and Pi0.5 by default over ZMQ or gRPC. Other policy families need their LeRobot extra. See [policy_server README](policy_server/README.md) |
| `feetech_ros2_driver` | C++             | **Submodule** — Feetech STS3215 ros2_control hardware interface                                  |
| `scripts/`            | Python          | Live ROS-to-Rerun bridge and ROS-backed episode replay viewer (run inside the Pixi environment)   |

---

## Repository Structure

```
so101-ros-physical-ai/
├── so101_bringup/
│   ├── launch/              # teleop, recording_session, leader, follower, cameras ...
│   ├── config/
│   │   ├── ros2_control/    # forward controller + joint-state broadcaster YAML
│   │   └── cameras/profiles # immutable logical camera profiles
│   └── rviz/                # leader, follower, teleop RViz configs
├── so101_description/
│   ├── urdf/                # Xacro: arm, end-effectors (leader/follower), ros2_control
│   └── meshes/              # STL meshes for all links
├── so101_teleop/
│   ├── src/                 # teleop.cpp
│   ├── config/              # teleop.yaml
│   └── launch/
├── episode_recorder/
│   ├── src/                 # episode_recorder.cpp, teleop_episode_keyboard.cpp
│   ├── config/              # recorder.yaml (storage/timing only)
│   └── launch/
├── rosbag_to_lerobot/
│   ├── rosbag_to_lerobot/   # Python package (bag_reader, converter, decoders, cli)
│   ├── config/
│   │   ├── so101_30hz.yaml  # 30 Hz timing/non-camera conversion config
│   │   └── so101_50hz.yaml  # 50 Hz timing/non-camera conversion config
│   └── test/
├── so101_inference/
│   └── so101_inference/     # LeRobot policy inference node + utils (runs in Pixi lerobot env)
├── feetech_ros2_driver/     # (submodule) Feetech ros2_control plugin
├── scripts/
│   ├── so101_ros2_to_rerun.py
│   ├── so101_ros2_rerun_3d_bridge.py
│   └── so101_episode_viewer_ros2.py
├── docs/
│   ├── hardware.md          # Full hardware setup guide (udev, calibration, cameras)
│   └── assets/
│       └── 99-so101.rules.example  # Example udev rules template
├── pixi.toml                # Pixi envs: default (Rerun bridge) + lerobot (dataset conversion + inference)
└── LICENSE
```

---

## Requirements

- **Ubuntu 24.04** + **ROS 2 Jazzy**
- Two SO-101 arms (leader + follower) with Feetech STS3215 servos, connected via USB
- `rosdep`, `colcon`
- Two or three cameras, matching the required `single_overhead` or `dual_overhead` profile
- [Pixi](https://pixi.sh/) for the repository's Rerun, LeRobot conversion/training, and ROS inference workflows

The root [`pixi.toml`](pixi.toml) and `pixi.lock` are the authoritative environment definition for this repository. Pixi provisions the complete ROS/LeRobot working environment on both declared platforms (`linux-64` and `linux-aarch64`), including Python 3.12, LeRobot 0.6.1, the workflow extras, and ffmpeg. Install or refresh it from the repository root with:

```bash
pixi install --all
```

There is deliberately no repository-wide `uv` project or root `pyproject.toml`; do not substitute `uv sync` at the repository root. LeRobot extras are selected by workflow: `[dataset]` for conversion, `[dataset-viz]` for the dataset viewer, `[training]` plus the policy extra for training, `[async]` for remote inference, `[smolvla]` for SmolVLA, and `[pi]` for Pi0/Pi0.5. The only supported standalone `uv` path is the remote [`policy_server`](policy_server/README.md), on a host/container that already supplies its required system and native libraries.

### Hardware Setup & Calibration

> **Warning**
> Before running on real hardware you **must**:
>
> 1. Complete LeRobot motor setup + calibration for both arms so the required persistent values are written to the motors.
> 2. Set up udev rules so your devices appear as `/dev/so101_leader`, `/dev/so101_follower`, `/dev/cam_wrist`, and `/dev/cam_overhead_1`; `dual_overhead` also requires `/dev/cam_overhead_2`.
> 3. Create the external physical-rig YAML with the stable camera device paths.
>
> **[→ Full hardware setup guide (docs/hardware.md)](docs/hardware.md)**

Follow the [LeRobot SO-101 guide](https://huggingface.co/docs/lerobot/so101) to assemble and configure the arms, then complete these steps before launching ROS:

| Step                         | Action                                                                                               | Notes                                                   |
| ---------------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| **1. Motor setup**           | Run the LeRobot "setup motors" routine (writes servo IDs and baudrate to EEPROM)                    | One-time per arm                                        |
| **2. Calibrate**             | Run LeRobot calibration for both arms (`calibrate` command for follower and leader)                  | Required per robot; stores calibration-related values   |
| **3. Udev rules**            | Create stable device symlinks using the [example template](docs/assets/99-so101.rules.example)      | See [docs/hardware.md](docs/hardware.md)                |
| **4. Camera rig**            | Map the canonical camera roles to their stable `/dev/cam_*` paths                                   | No camera calibration is used                           |
| **5. Launch ROS**            | Pass `camera_profile` and the absolute `camera_rig_config_file`                                      | Only after steps 1-4                                    |

---

## Installation

```bash
# Clone (includes submodules)
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone --recurse-submodules https://github.com/legalaspro/so101-ros-physical-ai.git

# Install dependencies and build
cd ~/ros2_ws
sudo apt update
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

---

## Usage

### Teleop (Leader to Follower)

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch so101_bringup teleop.launch.py \
  camera_profile:=dual_overhead \
  camera_rig_config_file:=/absolute/path/to/camera_rig.yaml
```

Common overrides:

```bash
# Disable cameras
ros2 launch so101_bringup teleop.launch.py use_cameras:=false

# Disable RViz
ros2 launch so101_bringup teleop.launch.py \
  camera_profile:=single_overhead \
  camera_rig_config_file:=/absolute/path/to/camera_rig.yaml \
  use_teleop_rviz:=false
```

### Episode Recording

Record teleoperation episodes for imitation learning. In one terminal, launch the recording session:

```bash
export SO101_RERUN_ENV_DIR=~/ros2_ws/src/so101-ros-physical-ai
ros2 launch so101_bringup recording_session.launch.py \
  camera_profile:=dual_overhead \
  camera_rig_config_file:=/absolute/path/to/camera_rig.yaml \
  experiment_name:=pick_and_place \
  task:="Pick up the cube and place it in the container." \
  use_rerun:=true
```

In a second terminal, run the interactive keyboard controller:

```bash
ros2 run episode_recorder teleop_episode_keyboard
```

Keys: **r** — start recording, **s** — save & stop, **d** / Backspace — discard episode, **q** — quit, **h** — help. Episodes are saved to `~/.ros/so101_episodes/` by default.

After recording, you can review your episodes with the built-in episode viewer powered by [Gradio](https://www.gradio.app/) and [gradio-rerun](https://pypi.org/project/gradio-rerun/), which makes it easy to browse, select, and visualize episodes in the browser:

```bash
# Replay the MCAP through ROS so compressed images, joint states, and actions
# are decoded through the same message types used by the live stack.
pixi run replay -- \
  --episodes_root ~/.ros/so101_episodes/pick_and_place \
  --camera-profile dual_overhead
```

### LeRobot Dataset Conversion

The repo ships a second Pixi environment (`lerobot`) that bundles [LeRobot 0.6.1](https://github.com/huggingface/lerobot/releases/tag/v0.6.1), ffmpeg, and the extras needed to convert this recorder's ROS 2 MCAP episode directories into fresh LeRobot v3.0 datasets. Existing LeRobot v2/v2.1 datasets are not accepted as input, and this repository does not provide a v2-to-v3 migration command. See the full [rosbag_to_lerobot README](rosbag_to_lerobot/README.md) for details.

If the resolved output already exists, conversion stops without changing it. Pass `--overwrite` only when you intend to delete that complete output directory and rebuild the dataset from the source MCAP episodes; the converter does not append, merge, or resume.

**Local conversion** (no Hugging Face account needed):

```bash
pixi run -e lerobot convert -- \
  --input-dir  ~/.ros/so101_episodes/pick_and_place \
  --config     ~/ros2_ws/src/so101-ros-physical-ai/rosbag_to_lerobot/config/so101_30hz.yaml \
  --camera-profile dual_overhead \
  --repo-id    local/so101_test
```

**Push to Hugging Face Hub:**

```bash
# Authenticate (once)
pixi run -e lerobot -- hf auth login
# or: export HF_TOKEN="hf_..."

# Verify
pixi run -e lerobot -- hf auth whoami

# Convert & push
pixi run -e lerobot convert -- \
  --input-dir  ~/.ros/so101_episodes/pick_and_place \
  --config     ~/ros2_ws/src/so101-ros-physical-ai/rosbag_to_lerobot/config/so101_30hz.yaml \
  --camera-profile dual_overhead \
  --repo-id    <hf-username>/so101-pick-and-place \
  --push-hub
```

**Visualize** the converted dataset:

```bash
# Local — enter the lerobot shell and use the built-in visualizer
pixi shell -e lerobot
lerobot-dataset-viz --repo-id local/so101_test --episode-index 0

# Online — after pushing to the Hub, open:
# https://huggingface.co/spaces/lerobot/visualize_dataset
```

### Training (LeRobot)

Once you have a LeRobot dataset (local or on the Hub), train a policy using the [LeRobot](https://github.com/huggingface/lerobot) training CLI:

```bash
pixi shell -e lerobot

lerobot-train \
  --dataset.repo_id=<hf-username>/so101-pick-and-place \
  --policy.type=act \
  --output_dir=outputs/train/act_so101_pick_place \
  --job_name=act_so101_pick_place \
  --policy.device=cuda
```

Optional flags: `--wandb.enable=true`, `--policy.repo_id=<hf-username>/<policy-name>` (auto-push checkpoint to Hub). See the [LeRobot training docs](https://huggingface.co/docs/lerobot/il_robots#train-a-policy) for the full list of options.

### Policy Inference (LeRobot)

Deploy trained LeRobot policies on the real SO-101 follower arm. Two inference modes are available: **synchronous** supports ACT and SmolVLA on-device, while **asynchronous** offloads ACT, SmolVLA, or Pi0.5 to a remote GPU server via ZMQ/gRPC with the default environment. Other LeRobot policy families require their corresponding optional extras. See the full [so101_inference README](so101_inference/README.md) for all parameters and details.

The selected policy must have exactly the image features produced by the
selected camera profile. There are no feature-rename parameters.

**Terminal 1 — bring up the follower arm and cameras:**

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch so101_bringup follower_vision.launch.py \
  camera_profile:=dual_overhead \
  camera_rig_config_file:=/absolute/path/to/camera_rig.yaml
```

**Terminal 2 — run the policy (Pixi `lerobot` env):**

```bash
cd ~/ros2_ws/src/so101-ros-physical-ai

# Synchronous — ACT policy on-device
pixi run -e lerobot infer -- --ros-args \
    -p repo_id:="your-org/your-dual-overhead-act-policy" \
    -p camera_profile:=dual_overhead

# Synchronous — SmolVLA on-device
pixi run -e lerobot infer -- --ros-args \
    -p repo_id:="your-org/your-dual-overhead-smolvla-policy" \
    -p camera_profile:=dual_overhead \
    -p policy_type:=smolvla \
    -p task:="Pick up the cube and place it in the container."

# Asynchronous — SmolVLA offloaded to a remote GPU server
pixi run -e lerobot async_infer -- --ros-args \
    -p repo_id:="your-org/your-dual-overhead-smolvla-policy" \
    -p camera_profile:=dual_overhead \
    -p policy_type:=smolvla \
    -p task:="Pick up the cube and place it in the container." \
    -p server_address:=192.168.1.100:8090 \
    -p fps:=50.0 -p actions_per_chunk:=50 -p chunk_size_threshold:=0.6
```

Or run bringup and synchronous inference together. `repo_id`, `camera_profile`,
and `task` are required; `policy_type` defaults to `act`:

```bash
ros2 launch so101_bringup inference.launch.py \
  camera_profile:=dual_overhead \
  camera_rig_config_file:=/absolute/path/to/camera_rig.yaml \
  use_inference:=true \
  repo_id:=your-org/your-dual-overhead-smolvla-policy \
  policy_type:=smolvla \
  task:="Pick up the cube and place it in the container." \
  rerun_env_dir:=/absolute/path/to/so101-ros-physical-ai
```

### Rerun (Live Visualization)

The repo ships a [Pixi](https://pixi.sh/) environment with a `bridge` task. Rerun can be added to teleop, recording, and inference sessions:

```bash
# Set once
export SO101_RERUN_ENV_DIR=~/ros2_ws/src/so101-ros-physical-ai

# Teleop with Rerun instead of RViz
ros2 launch so101_bringup teleop.launch.py \
  camera_profile:=dual_overhead \
  camera_rig_config_file:=/absolute/path/to/camera_rig.yaml \
  use_rerun:=true use_teleop_rviz:=false

# Recording session with Rerun
ros2 launch so101_bringup recording_session.launch.py \
  camera_profile:=dual_overhead \
  camera_rig_config_file:=/absolute/path/to/camera_rig.yaml \
  experiment_name:=pick_and_place use_rerun:=true
```

Or run the bridge standalone:

```bash
cd ~/ros2_ws/src/so101-ros-physical-ai
pixi run bridge -- --camera-profile dual_overhead
```

The bridge launches its own web viewer by default. Pass `--viewer native` to
open the native Rerun application instead.

#### 3D bridge (animated URDF + TF)

The 3D variant adds an animated SO-101 arm mesh (from the follower's URDF and
live `/tf`) to the same camera views and state/action plots. Use it when you want
to see the arm move in 3D during teleop, recording, or inference:

```bash
cd ~/ros2_ws/src/so101-ros-physical-ai
pixi run bridge-3d -- --camera-profile dual_overhead
```

Or from a launch file, swap `use_rerun` for `use_rerun_3d`:

```bash
ros2 launch so101_bringup teleop.launch.py \
  camera_profile:=dual_overhead \
  camera_rig_config_file:=/absolute/path/to/camera_rig.yaml \
  use_rerun_3d:=true use_teleop_rviz:=false
```

The 3D bridge subscribes to `/follower/robot_description`, `/tf`, `/tf_static`,
the profile's camera topics, `/follower/joint_states`, and
`/follower/forward_controller/commands`. It defaults to the `follower/` TF
prefix; pass `--tf-prefix ""` if your `robot_state_publisher` runs unprefixed.

## Configuration

### Launch Arguments (teleop.launch.py)

| Argument                 | Default               | Description                                      |
| ------------------------ | --------------------- | ------------------------------------------------ |
| `hardware_type`          | `real`                | `real` or `mock`                                 |
| `use_cameras`            | `true`                | Enable the strict camera subsystem               |
| `camera_profile`         | required              | `single_overhead` or `dual_overhead`              |
| `camera_rig_config_file` | required              | Absolute external physical-rig YAML              |
| `camera_startup_timeout_s` | `10.0`              | Deadline for every selected camera stream        |
| `camera_stale_timeout_s` | `1.0`                 | Runtime image-stream stale limit                  |
| `use_teleop_rviz`        | `true`                | Launch RViz with teleop config                   |
| `use_rerun`              | `false`               | Launch profile-matched Rerun bridge              |
| `use_rerun_3d`           | `false`               | Launch the 3D Rerun bridge (animated URDF + TF)  |
| `leader_usb_port`        | `/dev/so101_leader`   | Leader arm USB device                            |
| `follower_usb_port`      | `/dev/so101_follower` | Follower arm USB device                          |

### Launch Arguments (inference.launch.py)

| Argument                 | Default                | Description                                      |
| ------------------------ | ---------------------- | ------------------------------------------------ |
| `hardware_type`          | `real`                 | `real` or `mock`                                 |
| `follower_usb_port`      | `/dev/so101_follower`  | Follower arm USB device                          |
| `camera_profile`         | required               | Camera and policy input schema                   |
| `camera_rig_config_file` | required               | Absolute external physical-rig YAML              |
| `repo_id`                | required               | Policy Hub repo or local checkpoint path         |
| `policy_type`            | `act`                  | Synchronous policy architecture: `act` or `smolvla` |
| `task`                   | required               | Runtime policy instruction                       |
| `use_inference`          | `false`                | Start local synchronous inference                |
| `use_rerun`              | `false`                | Launch profile-matched Rerun bridge              |
| `rerun_env_dir`          | `$SO101_RERUN_ENV_DIR` | Repository root containing `pixi.toml`           |

### Hardware Configs

- `so101_bringup/config/ros2_control/` — forward-controller and joint-state-broadcaster parameters
- `so101_bringup/config/cameras/profiles/` — the only two logical camera profiles; physical devices and driver overrides live in the required external rig YAML.

This is a hard cutover. Repository code no longer understands the former
camera manifests, topic aliases, feature keys, CLI overrides, or policy
defaults. Existing bags remain under their original recording directories and
existing checkpoints remain in local/Hugging Face storage, but both require
historical tooling outside this repository if they use the former schema.

---

## Author

Dmitri Manajev

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.
