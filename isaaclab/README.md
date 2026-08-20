# SO-101 Isaac Lab environments

This directory is a repository-owned Isaac Lab project providing the SO-101
visual-manipulation simulation environments consumed by
[`pi05-self-improve/`](../pi05-self-improve/) for VLA data collection and
self-improvement rollouts. It contains no policy training code.

Isaac Lab supplies simulation, rendering, and managers. This project owns the
task environments, assets, and the camera/joint contract:

| Task | Goal | Episode | Fixed task | Randomized task |
|---|---|---:|---|---|
| Object in cup | Put the cube into the cup and release it stably | 15 s | `SO101-Object-In-Cup-Vision-Fixed-v0` | `SO101-Object-In-Cup-Vision-v0` |

Both IDs are registered against `isaaclab.envs:ManagerBasedRLEnv`. The fixed
task uses nominal geometry, appearance, physics, deterministic resets, and no
observation/action latency; the randomized task adds layout curriculum plus
yaw, mass, friction, actuator, joint noise, camera calibration, appearance,
lighting, image corruption, and zero/one-step camera/action latency.

## How `pi05-self-improve` consumes the environments

`pi05-self-improve/rollout_sim.py` bootstraps Isaac Sim through
`so101_rl.runtime`, then builds a task with `parse_env_cfg`/`gym.make` and
overrides two things on top of the registered configuration:

- cameras render at the dataset resolution (480x640) instead of the default
  120x160 policy resolution;
- the delta-action term is replaced by an absolute `JointPositionAction` in
  SO-101 joint order, which is what VLA action chunks command.

Frames are read directly from the scene cameras, and episode outcomes come
from the termination manager's scripted oracle (`success`, `dropped`,
`invalid`, `time_out`) — no reward or policy machinery is involved.

## Prerequisites

- repository: cloned anywhere (examples use `/path/to/so101-ros2`);
- Isaac Lab: `~/Documents/IsaacLab` — override with `SO101_ISAACLAB_ROOT`;
- source-built Isaac Sim: `~/Documents/isaacsim` — override with
  `SO101_ISAACSIM_PYTHON`, pointing at
  `_build/linux-x86_64/release/python.sh` inside the source build;
- setup: `monomanual_dual_overhead` (visual environments are locked to it);
- a CUDA GPU for rendering;
- `cube.stl` is manipulated and `cup.stl` remains fixed during an episode;
- the generated repository SO-101 USD is authoritative.

Start from the repository root:

```bash
cd /path/to/so101-ros2
export ISAACLAB_PYTHON=~/Documents/IsaacLab/.venv/bin/python
```

The entry points automatically switch to the configured source-built Isaac Sim
runtime when required.

## 1. Prepare and validate assets

The visual task uses these teleop-generated assets:

```text
build/isaacsim_so101/so101_follower/so101_follower.usda
build/isaacsim_so101/camera_rig/cam_mount_bottom.usd
build/isaacsim_so101/camera_rig/cam_mount_top.usd
```

If they are missing, build and source the ROS workspace, then generate them:

```bash
source /opt/ros/jazzy/setup.bash
SO101_REPO=/home/anon/Documents/so101-ros2
source "$SO101_REPO/install/setup.bash"

/home/anon/Documents/isaacsim/_build/linux-x86_64/release/python.sh \
  "$SO101_REPO/scripts/isaac_sim_teleop.py" \
  --setup monomanual_dual_overhead \
  --headless --max-frames 1 \
  --rebuild-asset
```

Use `--rebuild-asset` only after changing robot Xacro/meshes, camera mount
meshes, or camera rig configuration.

The Isaac expansion enables `simulation_contact_pads:=true`. It imports the
two invisible inner-jaw pad links without fixed-joint merging so PhysX can
address them independently; the normal ROS description keeps the option false.
The generated articulation must still expose only the canonical six movable
joints. Rebuilding after the pad or grasp-geometry change is mandatory.

Prepare and validate the cube and cup:

```bash
"$ISAACLAB_PYTHON" isaaclab/prepare_assets
"$ISAACLAB_PYTHON" isaaclab/prepare_assets --validate-only
```

Expected outputs:

```text
build/isaaclab_assets/cube.usda
build/isaaclab_assets/cup.usda
build/isaaclab_assets/manifest.json
```

Repeat preparation after changing either STL or regenerating the robot USD.

## 2. Inspect the environment

From a graphical desktop terminal, run:

```bash
"$ISAACLAB_PYTHON" isaaclab/live --num_envs 4
```

`live` opens the task with zero (or `--policy random`) actions and the native
Isaac Sim visualizer; it defaults to `SO101-Object-In-Cup-Vision-Fixed-v0`.
Before collecting rollouts, verify:

- robot, table, cube, and open cup geometry;
- the cube fits through the cup opening;
- wrist camera attachment and both overhead camera poses;
- camera optical axes, FOV, image content, and support geometry;
- joint ordering, limits, initial pose, gripper direction, actions, and resets.

## Task reference

Canonical joint and action order:

```text
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
```

The environment steps at 30 Hz over 120 Hz physics. The registered default
action term converts normalized outputs clipped to `[-1, 1]` into
measured-position deltas:

```text
arm target delta      0.033333 rad * action
gripper target delta  0.10 rad * action
```

`pi05-self-improve` replaces this term with absolute joint-position commands
for VLA rollouts; the delta contract above documents the registered default.

The actor observation surface (also overridden/ignored by VLA rollouts, which
read cameras directly) is:

```text
joint_state  [N, 6]            absolute joint positions in radians
wrist       [N, 3, 120, 160]  RGB / 255 - 0.5
overhead_1  [N, 3, 120, 160]  RGB / 255 - 0.5
overhead_2  [N, 3, 120, 160]  RGB / 255 - 0.5
```

## Files and outputs

| Purpose | Location |
|---|---|
| Source cube/cup meshes | `isaaclab/assets/source/` |
| Environment package | `isaaclab/source/so101_rl/` |
| Focused tests | `isaaclab/tests/` |
| Generated cube/cup assets | `build/isaaclab_assets/` |
| Generated robot and camera supports | `build/isaacsim_so101/` |

Generated assets and local environments are ignored by Git.

## Tests

Run the focused Isaac Lab suite through the source-runtime bootstrap:

```bash
"$ISAACLAB_PYTHON" isaaclab/test
```

The suite checks asset geometry, the shared observation/action contract,
task configuration, reset spacing, camera calibration, and pickup semantics.
It does not prove rendered-camera correctness or sim-to-real transfer.

## Troubleshooting

### No Isaac Sim window

Use `isaaclab/live`. Run from a desktop session with `DISPLAY`; remote
sessions require an Isaac Lab livestream visualizer.

### Missing assets

Repeat asset generation and [asset preparation](#1-prepare-and-validate-assets).
Regenerate after robot, camera mount, cube, or cup geometry changes.

### Wrong camera views

Compare the fixed task with sim teleop at the same joint pose and inspect the
`sim:` section of
`so101_bringup/config/setups/monomanual_dual_overhead.yaml`. Nominal
calibration is not proof of pixel-perfect real calibration.

### CUDA out of memory

Reduce `--num_envs` from 64 to 32, 16, or 4. Each environment renders three
views.
