# episode_recorder

**Video:** https://www.youtube.com/watch?v=lXkloZll4PA

<p>
  <a href="https://www.youtube.com/watch?v=lXkloZll4PA">
    <img src="../docs/assets/gifs/ros2_episode_recorder.gif" alt="ROS 2 episode recorder demo" height="240" />
  </a>
</p>

Minimalistic ROS 2 episode recorder for imitation learning. It records one of two strict camera profiles into rosbag2 episodes (MCAP by default) and supports keyboard-driven start / stop / discard control.

- `single_overhead`: wrist + overhead 1
- `dual_overhead`: wrist + overhead 1 + overhead 2

Both profiles also require follower joint states and forward-controller commands. Camera membership is fixed in the node; the storage YAML cannot add or remove topics.
Recording cannot start until every required topic is live and fresh. If the
recorder process exits while an episode is active—for example because the
camera supervisor shuts down the session—the incomplete episode is discarded
instead of being presented as valid data.

## Quick start

Recommended full-stack launch:

```bash
export SO101_RERUN_ENV_DIR=~/ros2_ws/src/so101-ros-physical-ai
ros2 launch so101_bringup recording_session.launch.py \
  camera_profile:=single_overhead \
  camera_rig_config_file:=/absolute/path/to/camera_rig.yaml \
  experiment_name:=pick_and_place \
  task:="Pick up the cube and place it in the container." \
  use_rerun:=true
```

`use_rerun:=true` is recommended if you want to see the recording live during the session.

In a second terminal, run the keyboard controller:

```bash
ros2 run episode_recorder teleop_episode_keyboard
```

Episodes are saved under `~/.ros/so101_episodes/` by default.

## Package-only launch

Use this if the robot stack is already running and you only want the recorder:

```bash
ros2 launch episode_recorder recorder.launch.py \
  camera_profile:=dual_overhead \
  experiment_name:=pick_and_place \
  task:="Pick up the cube and place it in the container."
```

## Keyboard controls

- `r` or `→` — start recording
- `s` or `←` — stop and save
- `d` or `Backspace` — discard current episode
- `t` — edit the recorder `task` parameter
- `h` — help
- `q` — quit

## Main files

- `launch/recorder.launch.py` — lifecycle recorder launch with auto-configure and auto-activate
- `config/recorder.yaml` — storage, timing gate, and default experiment settings
- `src/episode_recorder.cpp` — recorder lifecycle node and bag writing logic
- `src/teleop_episode_keyboard.cpp` — interactive keyboard client for start / stop / discard

## Useful launch args

- `camera_profile` — required: `single_overhead` or `dual_overhead`
- `params_file` — YAML config file for storage and timing settings
- `root_dir` — default output root, usually `~/.ros/so101_episodes`
- `experiment_name` — subfolder under `root_dir`
- `task` — task label stored in rosbag metadata
- `recorder_ns` — optional recorder namespace
