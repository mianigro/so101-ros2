# SO-101 Inference

ROS 2 inference package for the SO-101 robot arm. Runs [LeRobot 0.6.1](https://github.com/huggingface/lerobot/releases/tag/v0.6.1) policies on the robot or through a remote policy server, consuming camera images and joint states and publishing joint commands at high frequency.

## Features

- **Synchronous (local) inference** — loads the policy on the robot's own GPU/CPU and runs forward passes in a tight control loop. Supports ACT and SmolVLA policies.
- **Asynchronous (remote) inference** — offloads the policy to a remote server via ZeroMQ or gRPC, while the robot keeps executing actions from a local queue. The default environment includes ACT, SmolVLA, and Pi0.5 support. Other LeRobot policy families require their corresponding optional dependency extras on the server.
- **Setups** — every run selects `monomanual`, `monomanual_dual_overhead`, or `bimanual`; each uses fixed canonical observation keys and ROS topics.
- **Action chunking & aggregation** — the async node manages an action queue with configurable chunk sizes, refill thresholds, and aggregation strategies (`weighted_average`, `latest_only`, `average`, `conservative`).
- **Compressed image support** — the async node can subscribe to `CompressedImage` topics and forward raw JPEG bytes to the server (decoded server-side) for lower bandwidth usage.
- **Stale-data gating** — observations older than `max_age_s` are automatically discarded to prevent the robot from acting on outdated sensor data.
- **Pluggable transports** — ZeroMQ (`zmq`) and gRPC (`grpc`) transport backends, selected via a single parameter.
- **Built-in telemetry** — periodic summaries of FPS, queue depth, actions executed, observations sent/dropped, and round-trip latency.

## Nodes

| Node | Executable | Description |
|------|-----------|-------------|
| `lerobot_inference_node` | `lerobot_inference_node` | Synchronous local inference — loads the policy on-device |
| `async_ros2_inference_client` | `async_inference_node` | Asynchronous remote inference — offloads policy to a server |

## Quick Start

### Prerequisites

The robot hardware stack (follower arm + cameras) must already be running and publishing on the expected ROS 2 topics. Build the workspace and source it, or use the Pixi environment:

```bash
# Build if not using pixi
cd ~/Documents/so101-ros2 && colcon build --packages-select so101_inference
source install/setup.bash
```
### Synchronous Inference

This is for small fast VLAs, like the ACT or SmolVLA policy:

```bash
pixi run -e lerobot infer -- --ros-args \
    -p repo_id:="your-org/your-canonical-camera-policy" \
    -p setup:=monomanual_dual_overhead

pixi run -e lerobot infer -- --ros-args \
    -p repo_id:="your-org/your-canonical-camera-smolvla-policy" \
    -p setup:=monomanual_dual_overhead \
    -p policy_type:=smolvla \
    -p task:="Pick up the cube and place it in the container."
```

### Asynchronous Inference (remote server)

This runs larger models async on a remote or localhost GPU server.

```bash
pixi run -e lerobot async_infer -- --ros-args \
    -p repo_id:="your-org/your-canonical-camera-smolvla-policy" \
    -p setup:=monomanual_dual_overhead \
    -p policy_type:=smolvla \
    -p task:="Pick up the cube and place it in the container." \
    -p server_address:=192.168.1.100:8090 \
    -p actions_per_chunk:=50 \
    -p chunk_size_threshold:=0.6
```

Using gRPC transport instead of ZeroMQ:

```bash
pixi run -e lerobot async_infer -- --ros-args \
    -p transport_type:=grpc \
    -p server_address:=10.0.0.42:50051 \
    -p repo_id:="your-org/your-canonical-camera-policy" \
    -p setup:=monomanual_dual_overhead
```

Both `repo_id` and `setup` are required; neither has a compatibility
default. Setups select a fixed schema:

| Setup | Required LeRobot image keys | ROS image topics |
|-------|------------------------------|------------------|
| `monomanual` | `observation.images.wrist`, `observation.images.overhead_1` | `/follower/image_raw`, `/static_camera_1/image_raw` |
| `monomanual_dual_overhead` | `observation.images.wrist`, `observation.images.overhead_1`, `observation.images.overhead_2` | Above plus `/static_camera_2/image_raw` |
| `bimanual` | `observation.images.wrist_left`, `observation.images.wrist_right`, `observation.images.overhead_1` | `/follower_left/image_raw`, `/follower_right/image_raw`, `/static_camera_1/image_raw` |

The loaded policy must declare exactly the image keys selected by the setup and a matching `observation.state` (6 values per follower: 6 for the monomanual setups, 12 for bimanual).

Every selected image stream and every follower's joint-state stream must be present. With async `use_compressed:=true`, the node appends `/compressed` to every setup topic and forwards JPEG bytes to the server.

## Parameters

### Synchronous Node (`infer`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `repo_id` | string | required | Hugging Face model repo ID or local policy path |
| `setup` | string | required | `monomanual`, `monomanual_dual_overhead`, or `bimanual`; must match the policy input schema |
| `policy_type` | string | `act` | Policy architecture: `act` or `smolvla` |
| `task` | string | `Put the green cube in the cup.` | Task description (used by VLA models) |
| `max_age_s` | float | `0.2` | Max sensor data age before it's considered stale |
| `fwd_topics` | string[] | *(from setup)* | Command topic per follower |
| `joints_topics` | string[] | *(from setup)* | Joint-state topic per follower |

### Asynchronous Node (`async_infer`)

The control loop is fixed at the canonical 30 Hz. All configurable parameters
from the synchronous node plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `transport_type` | string | `zmq` | Transport backend: `zmq` or `grpc` |
| `server_address` | string | `127.0.0.1:8090` | Policy server `host:port` |
| `policy_device` | string | `cuda` | Device for policy inference on the server |
| `actions_per_chunk` | int | `100` | Number of actions requested per inference call |
| `chunk_size_threshold` | float | `0.5` | Queue fill ratio below which a new observation is sent (0.0–1.0) |
| `aggregate_fn_name` | string | `weighted_average` | Action aggregation strategy: `weighted_average`, `latest_only`, `average`, `conservative` |
| `use_compressed` | bool | `false` | Subscribe to `CompressedImage` topics and send JPEG bytes to the server |

## License

Apache-2.0 — see [LICENSE](../LICENSE).
