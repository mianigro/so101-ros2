# rosbag_to_lerobot

**Video:** https://www.youtube.com/watch?v=ffXxCkYJ6as

<p>
  <a href="https://www.youtube.com/watch?v=ffXxCkYJ6as">
    <img src="../docs/assets/gifs/ros2_lerobot_dataset.gif" alt="ROS 2 to LeRobot dataset conversion demo" height="240" />
  </a>
</p>

Convert ROS 2 rosbag episode directories recorded as MCAP into [LeRobot v3.0](https://github.com/huggingface/lerobot) datasets — locally or pushed straight to the Hugging Face Hub.

## Scope and Dataset Version

This is a one-way source conversion: **ROS 2 MCAP episodes → a new LeRobot v3.0 dataset**. The input must be episode directories produced by the ROS recorder, not an existing LeRobot dataset. LeRobot v2/v2.1 input is unsupported, and this package does not migrate v2 datasets or append them to v3 datasets.

Each run creates a fresh dataset. If the resolved target already exists, conversion stops and preserves it unless `--overwrite` is supplied. With `--overwrite`, the converter deletes the entire resolved output directory—the explicit `--output-dir`, or LeRobot's cache path for `--repo-id` when no output directory is given—then rebuilds it from the MCAP source episodes. It does not append, merge, or resume an existing dataset.

## How It Works

The converter reads bags sequentially using the absolute controller-command topic
as its reference. Every 30 Hz command emits one dataset frame; joint state and
camera features are sampled as-of that command timestamp with a configurable
freshness window (`max_age_s`). Frames where any feature is missing or stale are
dropped, and conversion rejects a bag whose median or end-to-end average command
cadence is not within 10% of 30 Hz.

### Alignment Guarantee

Non-reference features are sampled as-of the command timestamp using the latest
message **≤ reference time**. This prevents future observations from being paired
with an action. The `action` values are copied unchanged from the absolute-radian
`Float64MultiArray` controller targets.

## Prerequisites

This package runs inside the **`lerobot` Pixi environment** defined by the root `pixi.toml` and lockfile. That is the authoritative conversion environment on both `linux-64` and `linux-aarch64`: it pins LeRobot 0.6.1 and provides ffmpeg, `[dataset]`, and `[dataset-viz]` alongside the ROS-aware task wrapper. Run `pixi install -e lerobot` at the repository root before using the converter. A bare LeRobot or standalone `uv` installation is not the supported path for MCAP conversion because it does not provision the complete ROS/native environment.

```bash
# Make sure Pixi is installed — https://pixi.sh
# All commands below are run from the repo root (so101-ros2/)
```

> **Note:** The ROS 2 environment must be sourceable (and your workspace built if you rely on custom message types), because the converter uses ROS 2 Python libraries (`rosbag2_py`, message definitions, `rclpy.serialization`) at runtime.

---

## Quick Start — Local Conversion

No Hugging Face account required. Use `local/` as the repo-id prefix to keep everything on disk:

```bash
pixi run -e lerobot convert -- \
  --input-dir  ~/.ros/so101_episodes/pick_and_place \
  --config     rosbag_to_lerobot/config/so101_30hz.yaml \
  --camera-profile single_overhead \
  --repo-id    local/so101_test
```

`--camera-profile` is required. `single_overhead` emits `wrist` and
`overhead_1`; `dual_overhead` additionally emits `overhead_2`. The converter
checks every episode for the exact selected camera set and canonical 30 Hz
command cadence before it deletes or creates output.

If `--output-dir` is omitted, LeRobot writes to its default cache location (typically `~/.cache/huggingface/lerobot/<repo-id>/`).

### Useful flags

| Flag | Description |
|------|-------------|
| `--overwrite` | Delete the complete resolved output directory and rebuild it; never append or resume |
| `--sync-p95` | Collect p95 sync latency stats (slightly more overhead) |
| `--vcodec <codec>` | LeRobot RGB encoder name (`libsvtav1` default; also `h264`, `hevc`, `h264_nvenc`) |
| `--use-videos` / `--no-use-videos` | MP4 video (default) vs individual images |
| `--dataset-source` | Provenance tag set for the Hub dataset: `teleop` (default, imitation-learning) or `ppo` (reinforcement-learning). See [Publishing](#publishing-to-hugging-face-hub) |

---

## Publishing to Hugging Face Hub

### 1. Authenticate

Pick **one** of:

```bash
# Interactive browser login
pixi run -e lerobot -- hf auth login

# Or export your token directly
export HF_TOKEN="hf_..."
```

Verify you're logged in:

```bash
pixi run -e lerobot -- hf auth whoami
```

### 2. Convert & Push

```bash
pixi run -e lerobot convert -- \
  --input-dir  ~/.ros/so101_episodes/pick_and_place_2 \
  --config     rosbag_to_lerobot/config/so101_30hz.yaml \
  --camera-profile dual_overhead \
  --repo-id    <hf-username>/so101-pick-and-place \
  --push-hub
```

The `--push-hub` flag finalizes the dataset and uploads it to `https://huggingface.co/datasets/<repo-id>`.

### Dataset provenance tags

The Hub dataset is tagged from `--dataset-source` so its provenance is accurate:

| `--dataset-source` | Recorded from | Hub tags |
|--------------------|---------------|----------|
| `teleop` (default) | Human teleoperation | `teleoperation`, `imitation-learning` |
| `ppo`              | Autonomous rollouts of an exported PPO policy | `reinforcement-learning` |

Both sources share the same 30 Hz command contract; only the provenance tag set differs. When pushing episodes recorded by a PPO policy (see the top-level README's "Record PPO demonstrations for VLA training"), pass `--dataset-source ppo` so the dataset is not mislabeled as teleoperation.

---

## Visualization

### Local (LeRobot CLI)

Enter the lerobot shell, then use the built-in dataset visualizer:

```bash
pixi shell -e lerobot
lerobot-dataset-viz --repo-id local/so101_test --episode-index 0
```

### Online (Hugging Face Space)

After pushing a dataset to the Hub, visualize it at:

👉 **<https://huggingface.co/spaces/lerobot/visualize_dataset>**

Enter your `repo-id` (e.g. `<hf-username>/so101-pick-and-place`) and browse episodes interactively.

For a live example, see [legalaspro/so101-ros-physical-ai-test](https://huggingface.co/datasets/legalaspro/so101-ros-physical-ai-test).

---

## Configuration

Conversion uses the canonical
[`config/so101_30hz.yaml`](config/so101_30hz.yaml), plus the required camera
profile. The rate and command reference are validated rather than treated as
alternate conversion modes.

| Field | Purpose |
|-------|---------|
| `robot_type` | Robot identifier written into the dataset metadata |
| `fps` | Canonical dataset rate; must be `30` |
| `reference_topic` | Must be `/follower/forward_controller/commands` |
| `task` | Task label written to every frame |
| `default_max_age_s` | Default freshness window for feature sampling |
| `features[]` | Non-camera features; camera entries are rejected because `--camera-profile` owns them |

### `stamp_src` — Timestamp Source

Each feature declares a `stamp_src` that controls which timestamp is used for synchronization:

| Value | Behaviour |
|-------|-----------|
| `bag` **(default)** | Use the bag-level receive timestamp (from `read_next()`). Recommended for real hardware. |
| `header` | Use the message's `header.stamp`. Useful in simulation where clocks are perfectly synced. Falls back to `bag` if the message type has no header. |

The default is `bag` because it reflects the moment the message was actually written to the bag, which gives the most consistent alignment across topics on real hardware. Camera drivers in particular report header timestamps that lag behind the true capture time by varying amounts — after benchmarking several ROS 2 camera drivers (`v4l2_camera`, `usb_cam`, `camera_ros`/libcamera, `gscam`) I found that `camera_ros` and `gscam` performed best in terms of stable Hz and low header-to-bag delay; I selected `gscam` as slightly better overall (~30-35 ms delay). Even so, the bag timestamp remains the most reliable common reference for cross-topic synchronization. In simulation, where all clocks are perfectly synchronized, `header` can be the better choice.

### Supported message types

`sensor_msgs/msg/Image`, `sensor_msgs/msg/CompressedImage`, `sensor_msgs/msg/JointState`, `std_msgs/msg/Float64MultiArray`.

---

## CLI Reference

```
pixi run -e lerobot convert -- --help
```

```
usage: convert --input-dir DIR --config FILE --camera-profile PROFILE --repo-id ID [options]

  --input-dir       Directory containing episode bag subdirectories
  --config          Path to YAML config file
  --camera-profile  single_overhead or dual_overhead
  --repo-id         HuggingFace repo ID (e.g. user/dataset_name or local/name)
  --output-dir      Override default output location
  --use-videos      Store as MP4 video (default) / --no-use-videos for images
  --vcodec          LeRobot RGB encoder (libsvtav1 | libaom-av1 | h264 |
                    hevc | auto | a supported hardware encoder)
  --push-hub        Push final dataset to HuggingFace Hub
  --dataset-source  Provenance tags: teleop (default) | ppo
  --sync-p95        Collect p95 sync stats
  --overwrite       Delete the complete output directory and rebuild from MCAP
```

---

## License

Apache License 2.0 — see [LICENSE](../LICENSE) for details.
