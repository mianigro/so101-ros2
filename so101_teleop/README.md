# SO-101 Teleop

Leader-to-follower teleoperation package for the SO-101 arm. It subscribes to the leader `/joint_states` topic and sends six-joint absolute position commands to the follower at the canonical 30 Hz. This can be recieved either by the follow robot arm or the Isaac Sim robot arm.

## Quick start

Recommended full-stack launch:

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch so101_bringup teleop.launch.py \
  setup:=monomanual_dual_overhead
```

This uses the `forward_controller` (`ForwardCommandController`), matching the command contract used by recording and inference. Bimanual setups spawn one relay per leader/follower pair, namespaced under each follower (`/follower_left/...`, `/follower_right/...`).

## Package-only launch

Use this only if the leader and follower stacks are already running:

```bash
ros2 launch so101_teleop teleop.launch.py
```

## Main files

- `launch/teleop.launch.py` — forward-controller teleop node
- `config/teleop.yaml` — stale timeout and joint list
- `src/teleop.cpp` — six-joint follower command relay

## Useful launch args

- `leader_namespace` — default: `leader`
- `follower_namespace` — default: `follower`
- `params_file` — custom teleop parameter file
