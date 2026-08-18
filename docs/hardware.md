# Hardware Setup

> **Required before running on real hardware.**

This document focuses on **ROS-side hardware integration**:

- stable device naming (serial + cameras),
- permissions,
- LeRobot motor setup and calibration as prerequisites.

---

## 1. Motor Setup (One-Time per Arm)

Each servo must have a unique ID and correct baudrate written to EEPROM.
Follow the official [LeRobot SO-101 guide](https://huggingface.co/docs/lerobot/so101) for:

- **setup motors** (IDs / baudrate)
- **calibration** (offsets / limits)

> This repo assumes your servos are already configured and responding correctly.

---

## 2. Identify Devices (Quick)

Plug in the devices and confirm the kernel sees them:

```bash
ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || true
ls -l /dev/video* 2>/dev/null || true
```

---

## 3. Udev Rules (Recommended)

This stack assumes stable device symlinks created by udev:

| Device                     | Default path             | Required by |
| -------------------------- | ------------------------ | ----------- |
| Leader arm                 | `/dev/so101_leader`      | monomanual setups |
| Follower arm               | `/dev/so101_follower`    | monomanual setups |
| Wrist camera               | `/dev/cam_wrist`         | monomanual setups |
| Overhead camera 1          | `/dev/cam_overhead_1`    | every setup |
| Overhead camera 2          | `/dev/cam_overhead_2`    | `monomanual_dual_overhead` |
| Left leader / follower     | `/dev/so101_leader_left`, `/dev/so101_follower_left` | `bimanual` |
| Right leader / follower    | `/dev/so101_leader_right`, `/dev/so101_follower_right` | `bimanual` |
| Left / right wrist cameras | `/dev/cam_wrist_left`, `/dev/cam_wrist_right` | `bimanual` |

Camera membership is never inferred. Select exactly one setup
(`monomanual`, `monomanual_dual_overhead`, or `bimanual`); every camera and
arm device in the selected setup is required.

### 3.1 Query Device Properties

```bash
# Arms — replace /dev/ttyACM0 with the device you see:
udevadm info --query=property --name=/dev/ttyACM0 | \
  egrep 'ID_VENDOR_ID|ID_MODEL_ID|ID_SERIAL_SHORT|ID_PATH'
```

```bash
# Cameras — query each physical camera separately:
ls -l /dev/v4l/by-id/
# pick the node you want, then:
udevadm info --query=property --name=/dev/videoX | \
  egrep 'ID_VENDOR_ID|ID_MODEL_ID|ID_SERIAL_SHORT|ID_PATH'
```

> If `ID_SERIAL_SHORT` is missing or identical across cameras, match on
> `ID_PATH` instead. `ID_PATH` identifies the physical USB port, so reconnect
> that camera to the same port to retain its stable name.

### 3.2 Edit the Example Rules File

Template: [`docs/assets/99-so101.rules.example`](assets/99-so101.rules.example).
Replace placeholders with values from 3.1.

**Arms (tty):**

```
SUBSYSTEM=="tty", ENV{ID_VENDOR_ID}=="XXXX", ENV{ID_MODEL_ID}=="YYYY", ENV{ID_SERIAL_SHORT}=="SERIAL_LEADER",   SYMLINK+="so101_leader",   GROUP="dialout", MODE="0660"
SUBSYSTEM=="tty", ENV{ID_VENDOR_ID}=="XXXX", ENV{ID_MODEL_ID}=="YYYY", ENV{ID_SERIAL_SHORT}=="SERIAL_FOLLOWER", SYMLINK+="so101_follower", GROUP="dialout", MODE="0660"
```

**Cameras (video4linux):**

```
ACTION=="add|change", SUBSYSTEM=="video4linux", KERNEL=="video*", ENV{ID_SERIAL_SHORT}=="SERIAL_WRIST",    ATTR{index}=="0", SYMLINK+="cam_wrist",    GROUP="video", MODE="0660"
ACTION=="add|change", SUBSYSTEM=="video4linux", KERNEL=="video*", ENV{ID_SERIAL_SHORT}=="SERIAL_OVERHEAD_1", ATTR{index}=="0", SYMLINK+="cam_overhead_1", GROUP="video", MODE="0660"
ACTION=="add|change", SUBSYSTEM=="video4linux", KERNEL=="video*", ENV{ID_SERIAL_SHORT}=="SERIAL_OVERHEAD_2", ATTR{index}=="0", SYMLINK+="cam_overhead_2", GROUP="video", MODE="0660"
```

> Many USB cameras expose multiple `/dev/video*` devices (video + metadata).
> `ATTR{index}=="0"` selects the main video stream.

When serials are unavailable, replace the `ENV{ID_SERIAL_SHORT}==...` match for
that camera with its queried path, for example:

```
ACTION=="add|change", SUBSYSTEM=="video4linux", KERNEL=="video*", ENV{ID_PATH}=="PATH_OVERHEAD_1", ATTR{index}=="0", SYMLINK+="cam_overhead_1", GROUP="video", MODE="0660"
ACTION=="add|change", SUBSYSTEM=="video4linux", KERNEL=="video*", ENV{ID_PATH}=="PATH_OVERHEAD_2", ATTR{index}=="0", SYMLINK+="cam_overhead_2", GROUP="video", MODE="0660"
```

### 3.3 Install and Reload

```bash
sudo cp docs/assets/99-so101.rules.example /etc/udev/rules.d/99-so101.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Verify:

```bash
ls -l /dev/so101_leader /dev/so101_follower
ls -l /dev/cam_wrist /dev/cam_overhead_1
ls -l /dev/cam_overhead_2
# bimanual:
ls -l /dev/so101_leader_left /dev/so101_leader_right \
      /dev/so101_follower_left /dev/so101_follower_right \
      /dev/cam_wrist_left /dev/cam_wrist_right
```

---

## 4. Permissions (dialout + video)

```bash
sudo usermod -aG dialout,video $USER
```

Log out / in (or reboot), then verify:

```bash
groups | grep -E 'dialout|video'
```

> `dialout` is needed for serial ports (arms), `video` for cameras.
> With the udev rules above (`GROUP="dialout"` / `GROUP="video"`, `MODE="0660"`),
> you should not need `sudo` or `chmod` hacks.

---

## 5. Motor Calibration

Before using the real arms from ROS, complete the LeRobot motor setup and calibration steps for both arms. Those steps write the required persistent values to the servo motors, including IDs and calibration-related settings stored in EEPROM.

Servo motors keep persistent values in EEPROM, so those settings survive power cycles and only change when a tool explicitly writes new ones. That means calibration values already stored on the motors can remain the source of truth instead of being duplicated into multiple ROS config files.

For the follower gripper, this project sets these protection values by default in `so101_description/urdf/ros2_control/so101_ros2_control.xacro` to reduce the risk of overloading or damaging the motor:

- `max_torque_limit: 500`
- `protection_current: 250`
- `overload_torque: 25`

---

## 6. Setups

The repository owns exactly three canonical setups. Each one is fully
described by a single YAML in `so101_bringup/config/setups/` — the arm
topology (namespaces, USB ports, world placement), the logical camera
contract (nodes, topics, frames), the physical rig (device paths), and the
Isaac Sim camera geometry (`sim:` section; bimanual has none):

| Setup | Arms | Cameras | LeRobot image features |
| ----- | ---- | ------- | ---------------------- |
| `monomanual` | 1 leader + 1 follower | wrist, overhead 1 | `observation.images.wrist`, `observation.images.overhead_1` |
| `monomanual_dual_overhead` | 1 leader + 1 follower | wrist, overhead 1, overhead 2 | the monomanual set plus `observation.images.overhead_2` |
| `bimanual` | 2 leaders + 2 followers (`_left`/`_right`) | left wrist, right wrist, overhead 1 | `observation.images.wrist_left`, `observation.images.wrist_right`, `observation.images.overhead_1` |

The monomanual image and topic contract is fixed:

| Camera | Node | Raw image |
| ------ | ---- | --------- |
| Wrist | `/follower/cam_wrist` | `/follower/image_raw` |
| Overhead 1 | `/static_camera_1/cam_overhead_1` | `/static_camera_1/image_raw` |
| Overhead 2 | `/static_camera_2/cam_overhead_2` | `/static_camera_2/image_raw` |

The bimanual wrists publish `/follower_left/image_raw` and
`/follower_right/image_raw`; the shared overhead keeps
`/static_camera_1/image_raw`. Bimanual datasets concatenate both followers:
`observation.state` and `action` are 12-dimensional (`left.*` / `right.*`
labels; use `rosbag_to_lerobot/config/bimanual_30hz.yaml` when converting).

### 6.1 Setup file schema

The setup file owns every name; topics and frames are concrete strings, so
nothing anywhere else renders templates. To run on your machine, either edit
the device paths in the canonical file or copy it and pass the copy via
`setup_config_file:=/absolute/path/to/copy.yaml`.

```yaml
setup: monomanual            # must match the file name
schema_version: 1
arms:
  leaders:
    - namespace: leader
      usb_port: /dev/so101_leader
      tf_xyz: [-0.5, -0.5, 0.0]
      tf_yaw_deg: 90.0
  followers:
    - namespace: follower
      usb_port: /dev/so101_follower
      tf_xyz: [0.0, 0.0, 0.0]
      tf_yaw_deg: 0.0
cameras:
  profile:
    - id: wrist
      node_name: cam_wrist
      namespace: follower
      image_topic: /follower/image_raw
      frame_id: follower/wrist_camera_optical_frame
      feature: wrist
      default_backend: gscam
    - id: overhead_1
      node_name: cam_overhead_1
      namespace: static_camera_1
      image_topic: /static_camera_1/image_raw
      frame_id: follower/static_camera_1_optical_frame
      feature: overhead_1
      default_backend: gscam
  rig:
    wrist:
      device: /dev/cam_wrist
    overhead_1:
      device: /dev/cam_overhead_1
sim:                        # Isaac Sim geometry (monomanual setups only)
  ...
```

Camera calibration is not part of this stack. The rig section does not accept
intrinsics, CameraInfo URLs, or camera transforms. Overhead cameras publish
images but are not placed in the robot TF tree.

### 6.2 Launching a setup

`setup` is required whenever `use_cameras:=true` (the optional
`setup_config_file` selects an edited external copy):

```bash
ros2 launch so101_bringup teleop.launch.py setup:=monomanual_dual_overhead
```

Before any driver starts, launch validates the setup schema and selected
device nodes. It then requires every selected image stream within 10 seconds and
keeps checking them with a one-second stale limit. A driver or supervisor exit
shuts down the whole launch. There is no camera discovery, fallback topic, or
partial-setup operation.

The standard backend is `gscam`; its pipeline is constructed from each
camera's `device`. A machine that needs a different backend must declare that
driver's package, executable, parameters, parameter-name mapping, and remaps in
the camera's rig entry. There are no built-in alternate-driver presets.

The same `setup` selects the spawned arms, recorder topics, Rerun
subscriptions, conversion features, and inference inputs. Commands and
datasets use the fixed 30 Hz contract in `so101_30hz.yaml`
(`bimanual_30hz.yaml` for bimanual).

---

## 7. Verification Checklist

Before launching teleop:

- [ ] LeRobot motor setup and calibration were completed for every arm before first ROS use
- [ ] Leader / follower udev symlinks exist for the selected setup
- [ ] User is in `dialout` and `video` groups
- [ ] `/dev/cam_wrist` and `/dev/cam_overhead_1` exist (monomanual setups)
- [ ] `/dev/cam_overhead_2` exists for `monomanual_dual_overhead`
- [ ] The bimanual `_left`/`_right` device symlinks exist for `bimanual`
- [ ] The `setup` argument matches the physically connected rig

Sanity checks:

```bash
ls -l /dev/so101_leader /dev/so101_follower 2>/dev/null || true
ls -l /dev/cam_wrist /dev/cam_overhead_1 /dev/cam_overhead_2 2>/dev/null || true
ls -l /dev/cam_wrist_left /dev/cam_wrist_right 2>/dev/null || true
```
