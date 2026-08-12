# so101_inference

**Video:** https://www.youtube.com/watch?v=fpGmTwjTmzM

<p>
  <a href="https://www.youtube.com/watch?v=fpGmTwjTmzM">
    <img src="https://img.youtube.com/vi/fpGmTwjTmzM/hqdefault.jpg" alt="SO-101 inference demo" width="480" />
  </a>
</p>

ROS 2 inference package for the SO-101 robot arm. Runs [LeRobot 0.6.1](https://github.com/huggingface/lerobot/releases/tag/v0.6.1) policies on the robot or through a remote policy server, consuming camera images and joint states and publishing joint commands at high frequency.

## Features

- **Synchronous (local) inference** — loads the policy on the robot's own GPU/CPU and runs forward passes in a tight control loop. Supports ACT and SmolVLA policies.
- **Asynchronous (remote) inference** — offloads the policy to a remote server via ZeroMQ or gRPC, while the robot keeps executing actions from a local queue. The default environment includes ACT, SmolVLA, and Pi0.5 support. Other LeRobot policy families require their corresponding optional dependency extras on the server.
- **Explicit camera profiles** — every run selects `single_overhead` or `dual_overhead`; both use fixed canonical observation keys and ROS topics.
- **Action chunking & aggregation** — the async node manages an action queue with configurable chunk sizes, refill thresholds, and aggregation strategies (`weighted_average`, `latest_only`, `average`, `conservative`).
- **Compressed image support** — the async node can subscribe to `CompressedImage` topics and forward raw JPEG bytes to the server (decoded server-side) for lower bandwidth usage.
- **Stale-data gating** — observations older than `max_age_s` are automatically discarded to prevent the robot from acting on outdated sensor data.
- **Pluggable transports** — ZeroMQ (`zmq`) and gRPC (`grpc`) transport backends, selected via a single parameter.
- **Built-in telemetry** — periodic summaries of FPS, queue depth, actions executed, observations sent/dropped, and round-trip latency.
- **RSL-RL visual PPO deployment** — loads repository-exported TorchScript actors,
  enforces the dual-overhead manifest/checksum, runs 20 Hz local GPU inference,
  and starts in safety-gated shadow mode.

## Nodes

| Node | Executable | Description |
|------|-----------|-------------|
| `lerobot_inference_node` | `lerobot_inference_node` | Synchronous local inference — loads the policy on-device |
| `async_ros2_inference_client` | `async_inference_node` | Asynchronous remote inference — offloads policy to a server |
| `rsl_rl_inference_node` | `rsl_rl_inference_node` | Local visual-PPO inference with explicit arming and hold-on-failure |

### RSL-RL visual PPO (local, shadow-first)

Export the checkpoint through `isaaclab/export`, rebuild this ROS package, then:

```bash
ros2 launch so101_inference rsl_rl_infer.launch.py \
  model_dir:=/absolute/path/to/artifact-dir camera_profile:=dual_overhead
```

The node validates `policy_manifest.json` and the TorchScript checksum, subscribes
to all three raw RGB topics and `/follower/joint_states`, bilinearly resizes to
160x120, applies `RGB / 255 - 0.5`, and computes bounded absolute joint targets at
20 Hz. It publishes nothing in shadow mode. After simulation acceptance and
recorded-real-input shadow checks, arm with:

```bash
ros2 service call /so101_rl/set_enabled std_srvs/srv/SetBool "{data: true}"
```

All sources must be fresh and within 50 ms source-timestamp skew. Manual disable,
stale/skewed inputs, inference failure, NaN/Inf, or an invalid action causes one
measured-position hold command, disarming, and an explicit-rearm requirement.

## Quick Start

### Prerequisites

The robot hardware stack (follower arm + cameras) must already be running and publishing on the expected ROS 2 topics. Build the workspace and source it, or use the Pixi environment:

```bash
# Build (if not using pixi)
cd ~/ros2_ws && colcon build --packages-select so101_inference
source install/setup.bash
```

The root Pixi environment is authoritative for the robot-side ROS client and local inference on both `linux-64` and `linux-aarch64`; it pins LeRobot 0.6.1, includes ffmpeg, and installs `[async]`, `[smolvla]`, and `[pi]`. The `[pi]` extra supplies the dependencies shared by Pi0 and Pi0.5, while ACT uses LeRobot's base dependencies. Installing another policy family requires its upstream LeRobot extra.

Do not use a standalone `uv` environment for the ROS inference package. `uv sync` or `uv pip install .` is supported only for the remote `policy_server`, and only when that server host/container already supplies the required system and native libraries; see the [policy server installation guide](../policy_server/README.md).

### Synchronous Inference (on-device)

Run a locally-loaded ACT policy:

```bash
pixi run -e lerobot infer -- --ros-args \
    -p repo_id:="your-org/your-canonical-camera-policy" \
    -p camera_profile:=dual_overhead
```

Run a SmolVLA policy locally:

```bash
pixi run -e lerobot infer -- --ros-args \
    -p repo_id:="your-org/your-canonical-camera-smolvla-policy" \
    -p camera_profile:=dual_overhead \
    -p policy_type:=smolvla \
    -p task:="Pick up the cube and place it in the container." \
    -p fps:=50.0
```

### Asynchronous Inference (remote server)

Run a SmolVLA policy offloaded to a remote GPU server:

```bash
pixi run -e lerobot async_infer -- --ros-args \
    -p repo_id:="your-org/your-canonical-camera-smolvla-policy" \
    -p camera_profile:=dual_overhead \
    -p policy_type:=smolvla \
    -p task:="Pick up the cube and place it in the container." \
    -p server_address:=192.168.1.100:8090 \
    -p fps:=50.0 \
    -p actions_per_chunk:=50 \
    -p chunk_size_threshold:=0.6
```

For VLA policies, pass a task that matches the requested and trained behaviour.
The standalone nodes retain a demo fallback, while the combined bringup launch
requires an explicit task.

ACT policy with ZeroMQ transport (default):

```bash
pixi run -e lerobot async_infer -- --ros-args \
    -p repo_id:="your-org/your-canonical-camera-policy" \
    -p camera_profile:=single_overhead \
    -p server_address:=10.0.0.42:8090 \
    -p actions_per_chunk:=100 \
    -p chunk_size_threshold:=0.5
```

Using gRPC transport instead of ZeroMQ:

```bash
pixi run -e lerobot async_infer -- --ros-args \
    -p transport_type:=grpc \
    -p server_address:=10.0.0.42:50051 \
    -p repo_id:="your-org/your-canonical-camera-policy" \
    -p camera_profile:=dual_overhead
```

Both `repo_id` and `camera_profile` are required; neither has a compatibility
default. Profiles select a fixed schema:

| Profile | Required LeRobot image keys | ROS image topics |
|---------|------------------------------|------------------|
| `single_overhead` | `observation.images.wrist`, `observation.images.overhead_1` | `/follower/image_raw`, `/static_camera_1/image_raw` |
| `dual_overhead` | `observation.images.wrist`, `observation.images.overhead_1`, `observation.images.overhead_2` | Above plus `/static_camera_2/image_raw` |

The loaded policy must declare exactly the image keys selected by the profile
and a six-value `observation.state`. Any noncanonical or mismatched image schema
is rejected before commands can be published; migrate the checkpoint/dataset
schema instead of renaming cameras at runtime. Checkpoint-owned preprocessors
and normalization metadata are still loaded normally.

Every selected image stream and the joint-state stream must be present and
newer than `max_age_s`. With async `use_compressed:=true`, the node appends
`/compressed` to every profile topic and forwards JPEG bytes to the server.

## Parameters

### Synchronous Node (`infer`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `repo_id` | string | required | Hugging Face model repo ID or local policy path |
| `camera_profile` | string | required | `single_overhead` or `dual_overhead`; must match the policy input schema |
| `policy_type` | string | `act` | Policy architecture: `act` or `smolvla` |
| `task` | string | `Put the green cube in the cup.` | Task description (used by VLA models) |
| `fps` | float | `50.0` | Control loop frequency |
| `max_age_s` | float | `0.2` | Max sensor data age before it's considered stale |
| `fwd_topic` | string | `/follower/forward_controller/commands` | Topic to publish joint commands |
| `joints_topic` | string | `/follower/joint_states` | Topic to subscribe for joint states |

### Asynchronous Node (`async_infer`)

All parameters from the synchronous node plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `transport_type` | string | `zmq` | Transport backend: `zmq` or `grpc` |
| `server_address` | string | `127.0.0.1:8090` | Policy server `host:port` |
| `policy_device` | string | `cuda` | Device for policy inference on the server |
| `actions_per_chunk` | int | `100` | Number of actions requested per inference call |
| `chunk_size_threshold` | float | `0.5` | Queue fill ratio below which a new observation is sent (0.0–1.0) |
| `aggregate_fn_name` | string | `weighted_average` | Action aggregation strategy: `weighted_average`, `latest_only`, `average`, `conservative` |
| `use_compressed` | bool | `false` | Subscribe to `CompressedImage` topics and send JPEG bytes to the server |

## Architecture

```
 cameras + joints ──► inference node ──► arm commands
                           │
                      sync: on-device policy (ACT/SmolVLA)
                      async: ZMQ/gRPC ──► remote GPU server
```

## License

Apache-2.0 — see [LICENSE](../LICENSE).
