# Policy Server

**Video:** https://www.youtube.com/watch?v=l6kWDoHxczc

<p>
  <a href="https://www.youtube.com/watch?v=l6kWDoHxczc">
    <img src="https://img.youtube.com/vi/l6kWDoHxczc/hqdefault.jpg" alt="Vast.ai policy server and ROS 2 async inference demo" width="480" />
  </a>
</p>

GPU-side inference server for SO-101. Loads [LeRobot 0.6.1](https://github.com/huggingface/lerobot/releases/tag/v0.6.1) policies and serves action predictions over ZMQ or gRPC. The default installation supports ACT, SmolVLA, and Pi0.5; other policy families require their corresponding LeRobot dependency extra. This is the simplest way to host larger policies on a remote GPU machine (for example on Vast.ai) and connect them to the ROS 2 async inference client in [`so101_inference`](../so101_inference/README.md).

## Prerequisites

The policy server declares `lerobot[async,smolvla,pi]==0.6.1`, so installing the server also installs the default Python policy extras. ACT uses LeRobot's base installation; `[async]` provides the gRPC runtime, `[smolvla]` provides SmolVLA dependencies, and `[pi]` covers Pi0/Pi0.5. Add another upstream policy extra explicitly before selecting that policy.

> **Important:** The standalone `uv` path is for the policy server only. Use it only when the host or container already supplies the required native/system stack, including a compatible C/C++ runtime, ffmpeg/native codec libraries when needed, and compatible NVIDIA drivers/CUDA runtime for GPU inference. `uv` installs Python packages; it does not provision those host libraries. For the complete ROS client, conversion, training, and visualization environment, use the root Pixi workspace instead.

The standalone server does not require ROS 2 unless you intentionally colocate ROS-side components on the same host.

## Install

### Option 1 — `uv sync` (development)

Clone the repository, enter the standalone project directory, and let `uv` create and synchronize its local environment:

```bash
REPO_DIR="/workspace/repo"

# Clone or update
if [ -d "${REPO_DIR}/.git" ]; then
  git -C "${REPO_DIR}" pull
else
  git clone --depth 1 \
    https://github.com/legalaspro/so101-ros-physical-ai.git "${REPO_DIR}"
fi

cd "${REPO_DIR}/policy_server"
uv sync --locked
uv run --locked policy-server --help
```

### Option 2 — `uv pip install .`

Use this when you manage the virtual environment yourself:

```bash
cd /path/to/so101-ros-physical-ai/policy_server
uv venv
uv pip install .
```

### Option 3 — Direct from GitHub

One-liner, no local clone. Re-run to update.

```bash
uv pip install \
  "policy_server @ git+https://github.com/legalaspro/so101-ros-physical-ai.git#subdirectory=policy_server"
```

## Deploy on vast.ai

1. Create an instance using the **PyTorch** template
2. Set `PROVISIONING_SCRIPT` to the [provisioning gist](https://gist.github.com/legalaspro/d81fabb628f600cc27bc33ce5f5c130d)
3. Open TCP port **8090** in the instance config

The script installs LeRobot + policy_server automatically on first boot.

## Usage

```bash
# Console script
policy-server --transport=zmq --host=0.0.0.0 --port=8090

# Or as a module
python -m policy_server --transport=zmq --host=0.0.0.0 --port=8090
```

## Run async inference from ROS 2

Once the server is running and TCP port `8090` is reachable from the robot, start the async ROS 2 client from the `so101_inference` package:

```bash
pixi run -e lerobot async_infer -- --ros-args \
  -p repo_id:="your-org/your-canonical-camera-smolvla-policy" \
  -p camera_profile:=dual_overhead \
  -p policy_type:=smolvla \
  -p server_address:=<vast-ai-public-ip>:8090 \
  -p actions_per_chunk:=50 \
  -p chunk_size_threshold:=0.6
```

The server and client run at the fixed canonical 30 Hz. The client sends the canonical LeRobot feature schema selected by
`camera_profile`. After loading the checkpoint, the server compares that schema
with `policy.config.input_features` and rejects setup unless the image keys
match exactly and `observation.state` has six values. This prevents legacy
camera renames or a single/dual-profile mismatch from reaching action
publication. Processor and normalization configuration stored in the checkpoint
is still loaded through LeRobot 0.6.1's `make_pre_post_processors` API.

For more async inference options and transports, see the [`so101_inference` README](../so101_inference/README.md).
