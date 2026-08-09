# Hardware Setup

> **Required before running on real hardware.**
> The joint override examples shipped in this repo came from one specific
> robot and **will not match yours**. Physical camera calibration is not
> checked in at all.

This document focuses on **ROS-side hardware integration**:

- stable device naming (serial + cameras),
- permissions,
- LeRobot motor setup and calibration as prerequisites,
- optional ROS-side joint overrides when you explicitly want them.

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

| Device                     | Default path             | Membership |
| -------------------------- | ------------------------ | ---------- |
| Leader arm                 | `/dev/so101_leader`      | Required |
| Follower arm               | `/dev/so101_follower`    | Required |
| Wrist camera               | `/dev/cam_wrist`         | Both camera profiles |
| Overhead camera 1          | `/dev/cam_overhead_1`    | Both camera profiles |
| Overhead camera 2          | `/dev/cam_overhead_2`    | `dual_overhead` |

Camera membership is never inferred. Select exactly `single_overhead` or
`dual_overhead`; every camera in the selected profile is required.

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
ls -l /dev/cam_overhead_2  # required for dual_overhead
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

## 5. Calibration, EEPROM, and Optional Joint Config Overrides

Before using the real arms from ROS, complete the LeRobot motor setup and calibration steps for both arms. Those steps write the required persistent values to the servo motors, including IDs and calibration-related settings stored in EEPROM.

Servo motors keep persistent values in EEPROM, so those settings survive power cycles and only change when a tool explicitly writes new ones. That means calibration values already stored on the motors can remain the source of truth instead of being duplicated into multiple ROS config files.

### Parameter precedence

**Precedence:** `joint_config_file` overrides `URDF/Xacro` defaults.

On initialization, the driver writes the resulting values for supported parameters to the motors. If you provide a `joint_config_file`, those values override the URDF/Xacro defaults and replace the older stored values for the same registers.

If you use a `joint_config_file`, each joint entry must:

- include the correct `id`
- match a joint defined in the `ros2_control` / xacro description

Common parameters:

- `id`: motor ID on the bus
- `homing_offset`: joint zero alignment, written to the servo EEPROM
- `range_min` / `range_max`: joint travel limits, written to the servo EEPROM
- `p_coefficient` / `i_coefficient` / `d_coefficient`, `return_delay_time`, `max_torque_limit`, `protection_current`, `overload_torque`: optional tuning and protection settings written to the servo EEPROM
- `acceleration`: optional motion parameter used by the driver and not part of LeRobot calibration output

For the follower gripper, this project sets these protection values by default in `so101_description/urdf/ros2_control/so101_ros2_control.xacro` to reduce the risk of overloading or damaging the motor:

- `max_torque_limit: 500`
- `protection_current: 250`
- `overload_torque: 25`

LeRobot calibration does not produce these safety values. If needed, you can still override them per robot through `joint_config_file`.

Use `joint_config_file` when you want explicit per-robot overrides, to reapply known-good values during bringup, or to set extra tuning/protection parameters not produced by LeRobot.

Optional override examples live here:

```text
so101_bringup/config/hardware/
├── leader_joints.yaml
├── follower_joints.yaml
├── lerobot_leader_arm.json
└── lerobot_follower_arm.json
```

The YAML files in this repo are examples of override files, while the included `lerobot_*.json` files show raw LeRobot calibration output for reference.

---

## 6. Camera Configuration

The repository owns exactly two immutable logical profiles:

| Profile | Required cameras | LeRobot image features |
| ------- | ---------------- | ---------------------- |
| `single_overhead` | wrist, overhead 1 | `observation.images.wrist`, `observation.images.overhead_1` |
| `dual_overhead` | wrist, overhead 1, overhead 2 | the single profile plus `observation.images.overhead_2` |

Their ROS contract is fixed:

| Camera | Node | Raw image | CameraInfo | Optical frame |
| ------ | ---- | --------- | ---------- | ------------- |
| Wrist | `/follower/cam_wrist` | `/follower/image_raw` | `/follower/camera_info` | `follower/wrist_camera_optical_frame` |
| Overhead 1 | `/static_camera_1/cam_overhead_1` | `/static_camera_1/image_raw` | `/static_camera_1/camera_info` | `follower/static_camera_1_optical_frame` |
| Overhead 2 | `/static_camera_2/cam_overhead_2` | `/static_camera_2/image_raw` | `/static_camera_2/camera_info` | `follower/static_camera_2_optical_frame` |

### 6.1 External physical rig file

The profiles do not contain machine-specific device paths or calibration.
Create one external YAML file for the physical rig and pass its absolute path
at every camera-enabled launch. No physical calibration file is shipped in the
repository.

```yaml
schema_version: 1
cameras:
  wrist:
    device: /dev/cam_wrist
    camera_info_url: file:///absolute/path/to/wrist_camera_info.yaml
  overhead_1:
    device: /dev/cam_overhead_1
    camera_info_url: file:///absolute/path/to/overhead_1_camera_info.yaml
    transform:
      parent_frame: base_link
      translation: [CALIBRATED_X, CALIBRATED_Y, CALIBRATED_Z]
      rotation_xyzw: [CALIBRATED_QX, CALIBRATED_QY, CALIBRATED_QZ, CALIBRATED_QW]
  overhead_2:
    device: /dev/cam_overhead_2
    camera_info_url: file:///absolute/path/to/overhead_2_camera_info.yaml
    transform:
      parent_frame: base_link
      translation: [CALIBRATED_X, CALIBRATED_Y, CALIBRATED_Z]
      rotation_xyzw: [CALIBRATED_QX, CALIBRATED_QY, CALIBRATED_QZ, CALIBRATED_QW]
```

The capitalized values are placeholders, not defaults. Generate independent
intrinsics and extrinsics for each physical camera with
[`so101_camera_calibration`](../so101_camera_calibration/README.md). The loader
requires positive focal lengths, an absolute `file://` URL, and a normalized
quaternion. It does not accept a copied camera-1 calibration for camera 2 as a
substitute for measuring camera 2.

### 6.2 Launching a profile

Both arguments are required whenever `use_cameras:=true`:

```bash
ros2 launch so101_bringup teleop.launch.py \
  camera_profile:=dual_overhead \
  camera_rig_config_file:=/absolute/path/to/camera_rig.yaml
```

Before any driver starts, launch validates the profile, rig schema, selected
device nodes, per-camera calibration, and overhead transforms. It then requires
every selected Image and valid nonzero CameraInfo stream within 10 seconds and
keeps checking them with a one-second stale limit. A driver, TF publisher, or
supervisor exit shuts down the whole launch. There is no camera discovery,
fallback topic, or partial-profile operation.

The standard backend is `gscam`; its pipeline is constructed from each
camera's `device`. A machine that needs a different backend must declare that
driver's package, executable, parameters, parameter-name mapping, and remaps in
the external rig entry. There are no built-in alternate-driver presets.

The same `camera_profile` selects recorder topics, Rerun subscriptions,
conversion features, and inference inputs. Converter timing is the only choice
left to its two configs: `so101_30hz.yaml` or `so101_50hz.yaml`.

### 6.3 Calibration bootstrap

Production does not start with missing calibration. The dedicated bootstrap is
the one exception and launches only the selected physical camera:

```bash
ros2 launch so101_bringup camera_calibration_bootstrap.launch.py \
  camera_profile:=dual_overhead \
  camera_id:=overhead_2 \
  camera_rig_config_file:=/absolute/path/to/camera_rig.yaml
```

Use it only while producing the CameraInfo and transform blocks, then return to
the strict production launch.

### 6.4 Hard-cutover boundary

Old-format bags and checkpoints are not stored in this repository and are not
modified by this change. Recorder output normally lives under
`~/.ros/so101_episodes`; downloaded datasets and policies normally live under
the Hugging Face cache or in their Hub repositories. Anything using the former
topics or feature schema needs historical tooling outside this repository. No
aliases, remaps, or migration utilities are provided here.

---

## 7. Verification Checklist

Before launching teleop:

- [ ] LeRobot motor setup and calibration were completed for both arms before first ROS use
- [ ] Leader / follower udev symlinks exist
- [ ] User is in `dialout` and `video` groups
- [ ] If using `joint_config_file`, it points to the intended override YAML with correct joint IDs
- [ ] `/dev/cam_wrist` and `/dev/cam_overhead_1` exist
- [ ] `/dev/cam_overhead_2` exists for `dual_overhead`
- [ ] Every selected camera has its own intrinsic CameraInfo YAML
- [ ] Every selected overhead camera has its own measured quaternion transform
- [ ] `camera_profile` and the absolute external rig path are supplied

Sanity checks:

```bash
ls -l /dev/so101_leader /dev/so101_follower 2>/dev/null || true
ls -l /dev/cam_wrist /dev/cam_overhead_1 2>/dev/null || true
ls -l /dev/cam_overhead_2 2>/dev/null || true
```

Then run:

```bash
ros2 launch so101_bringup teleop.launch.py hardware_type:=real \
  camera_profile:=dual_overhead \
  camera_rig_config_file:=/absolute/path/to/camera_rig.yaml
```
