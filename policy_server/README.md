# Policy Server

GPU-side inference server for SO-101. Loads [LeRobot 0.6.1](https://github.com/huggingface/lerobot/releases/tag/v0.6.1) policies and serves action predictions over ZMQ or gRPC. The default installation supports ACT, SmolVLA, and Pi0.5; other policy families require their corresponding LeRobot dependency extra. This is the simplest way to host larger policies on a remote GPU machine and connect them to the ROS 2 async inference client in [`so101_inference`](../so101_inference/README.md).

## Prerequisites

The policy server declares `lerobot==0.6.1`, so installing the server also installs the default Python policy extras. ACT uses LeRobot's base installation; `[async]` provides the gRPC runtime, `[smolvla]` provides SmolVLA dependencies, and `[pi]` covers Pi0/Pi0.5. Add another upstream policy extra explicitly before selecting that policy.

## Usage

```bash
# Console script
policy-server --transport=zmq --host=0.0.0.0 --port=8090

# Or as a module
python -m policy_server --transport=zmq --host=0.0.0.0 --port=8090
```

## Tests

The inference engine has a transport-independent pytest suite (no GPU or ROS
needed):

```bash
cd policy_server
uv run --with pytest pytest tests/
```

## Run async inference from ROS 2

Once the server is running and TCP port `8090` is reachable from the robot, start the async ROS 2 client from the `so101_inference` package:

```bash
pixi run -e lerobot async_infer -- --ros-args \
  -p repo_id:="your-org/your-canonical-camera-smolvla-policy" \
  -p setup:=monomanual_dual_overhead \
  -p policy_type:=smolvla \
  -p server_address:=<vast-ai-public-ip>:8090 \
  -p actions_per_chunk:=50 \
  -p chunk_size_threshold:=0.6
```

The server and client run at the fixed canonical 30 Hz. The client sends the canonical LeRobot feature schema selected by `setup`. After loading the checkpoint, the server compares that schema with `policy.config.input_features` and rejects the setup unless the image keys match exactly and `observation.state` matches the setup's dimension (six values per follower). This prevents camera renames or a setup mismatch from reaching action publication. Processor and normalization configuration stored in the checkpoint is still loaded through LeRobot 0.6.1's `make_pre_post_processors` API.

For more async inference options and transports, see the [`so101_inference` README](../so101_inference/README.md).
