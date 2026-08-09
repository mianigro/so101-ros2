# so101_camera_calibration

Camera intrinsic and hand-eye calibration tools for the SO-101 arm.
Both tools expose a **Viser web UI** at `http://localhost:8080`, so
calibration can be done headlessly without a desktop GUI.

> **Intrinsic calibration** (section 1) is the useful part — standard
> ChArUco-based OpenCV calibration, and the output YAML drops straight into
> `camera_info_url` for any ROS 2 camera driver.
>
> ⚠️ **Hand-eye calibration** (section 2) is experimental. On a low-cost
> arm like the SO-101, servo repeatability (~1–2°) sets a hard accuracy
> floor, and the workflow here is a first pass rather than a validated
> pipeline. Treat the result as a starting point, not ground truth.
> If you work in computer vision / robotics and can spot issues or suggest
> improvements for cheap arms like the SO-101, contributions are very
> welcome.

> **Frame naming.** In this README `base_link` refers to the robot base
> frame; in the running system this is typically `follower/base_link`.
> The validated rig publishes each overhead transform directly to its
> canonical optical frame, such as `static_camera_1_optical_frame`, which is
> also the frame used by OpenCV and the calibration output.

## Prerequisites

Create an external physical rig file. During calibration the selected camera
may omit `camera_info_url`, and an overhead camera may omit `transform`:

```yaml
schema_version: 1
cameras:
  wrist:
    device: /dev/cam_wrist
  overhead_1:
    device: /dev/cam_overhead_1
  overhead_2:
    device: /dev/cam_overhead_2
```

The production camera launcher is intentionally stricter: every selected
camera needs its own valid intrinsic YAML, and every selected overhead camera
needs a calibrated quaternion transform. The file must be outside the
repository and supplied by absolute path.

Start the follower without cameras when the arm is needed for hand-eye
calibration:

```bash
ros2 launch so101_bringup follower_vision.launch.py use_cameras:=false
```

In another terminal, start exactly the camera being calibrated. Use
`single_overhead` for `wrist` or `overhead_1`; `overhead_2` belongs to the
`dual_overhead` profile:

```bash
ros2 launch so101_bringup camera_calibration_bootstrap.launch.py \
  camera_profile:=dual_overhead \
  camera_id:=overhead_2 \
  camera_rig_config_file:=/absolute/path/to/camera_rig.yaml
```

You also need a printed ChArUco board — see
[Printing a Target](#printing-a-target).

## 1. Camera Intrinsic Calibration

Calibrates focal lengths, principal point, and distortion.

<video src="https://github.com/user-attachments/assets/d333c7da-e61c-4738-8384-039c7fbca979" controls width="720"></video>


```bash
ros2 launch so101_camera_calibration intrinsic_calibration.launch.py \
  camera_role:=overhead_1
```

Open `http://localhost:8080`. Move the board around and vary tilt:

- cover center, corners, and edges
- rotate and tilt the board
- fill as much of the coverage grid as possible

Outputs:

- `/tmp/overhead_1_camera_info.yaml` — ROS CameraInfo format
- `/tmp/overhead_1_camera_info.npz` — NumPy archive

Calibrate overhead camera 2 in a separate run by selecting its topic and name:

```bash
ros2 launch so101_camera_calibration intrinsic_calibration.launch.py \
  camera_role:=overhead_2
```

Calibrate `wrist` the same way with `camera_role:=wrist`. Do not reuse one
physical camera's intrinsic YAML for another camera. Copy each resulting YAML
to a persistent machine-local location and add its absolute `file://` URL to
the corresponding rig entry:

```yaml
camera_info_url: file:///absolute/path/to/overhead_1_camera_info.yaml
```

## 2. Hand-Eye (Extrinsic) Calibration

Estimates the transform from `base_link` to the selected overhead optical
frame. Calibrate each overhead camera independently:

```bash
ros2 launch so101_camera_calibration handeye_calibration.launch.py \
  camera_role:=overhead_1

ros2 launch so101_camera_calibration handeye_calibration.launch.py \
  camera_role:=overhead_2
```

### Manual Calibration (Recommended)

<video src="https://github.com/user-attachments/assets/cf9792f5-6f94-478e-8a59-db8f60bd06aa" controls width="720"></video>

1. Toggle **Manual EE: ON** to enable the IK gizmo
2. Drag the gripper to a pose where the ChArUco board is clearly visible
3. Click **📷 Take Sample**
4. Repeat for **20–30 diverse poses** — vary position *and* tilt
5. Click **🧮 Compute Calibration**
6. Verify **Park** and **Horaud** agree closely (for example within ~1 cm)
7. Click **✅ Save Calibration**

The results are saved separately under
`~/.ros2/robokin_calibrations/overhead_1_transform.yaml` and
`overhead_2_transform.yaml`.

> **Tip:** Rotational variety is critical. Pure translations give degenerate
> solutions — always tilt the gripper at different angles between samples.
> Only take samples when ChArUco detection looks clean and reprojection
> error is low.

### Auto-Calibration

Drives the arm through preconfigured joint target poses from
`config/calibration_poses.yaml` and captures a sample at each one. Useful
as a **sanity check**. It is typically less accurate than manual collection
because of servo repeatability and lighting sensitivity. Click
**🤖 Auto-Calibrate** in the UI — make sure the workspace is clear first.

### Applying the Result

The saved file is already a valid fragment of the external rig schema. Merge
the selected camera's `transform` block into the same camera entry that holds
its device and intrinsic URL:

```yaml
schema_version: 1
cameras:
  overhead_1:
    device: /dev/cam_overhead_1
    camera_info_url: file:///absolute/path/to/overhead_1_camera_info.yaml
    transform:
      parent_frame: base_link
      translation: [0.1752, 0.0255, 0.5618]
      rotation_xyzw: [0.0, 0.0, 0.0, 1.0]
```

The values above only illustrate the schema; they are not a usable physical
calibration. The production loader rejects a missing or non-normalized
quaternion. Keeping the quaternion directly avoids the Euler-angle gimbal-lock
failure that affected the removed launch-argument interface.

## Printing a Target

Two generators are included — one per calibration type.

**Hand-eye target** (small, attaches to the gripper):

```bash
python3 scripts/gen_charuco_handeye.py
```

Creates `/tmp/charuco_handeye_A4.png` + `.pdf` (A4, 300 DPI, 4×5 squares at
**15 mm**, `DICT_4X4_50`). Attach to the gripper facing the camera.

**Intrinsic target** (larger and denser, covers the image better):

```bash
python3 scripts/gen_charuco_intrinsic.py
```

Creates `/tmp/charuco_intrinsic_A4.png` + `.pdf` (A4, 300 DPI, 8×6 squares at
**25 mm** (18 mm markers), `DICT_5X5_250`). Use it handheld — move it around
the image to cover corners/edges and vary tilt.

Both: print at **actual size** (no scale-to-fit). Verify with a ruler that
one square matches the expected millimetres. The node parameters
(`squares_x/y`, `square_length`, `marker_length`, `aruco_dict`) must match
the generated board.

## Package Structure

```
so101_camera_calibration/
├── config/
│   └── calibration_poses.yaml          # Joint poses for auto-calibrate
├── launch/
│   ├── handeye_calibration.launch.py
│   └── intrinsic_calibration.launch.py
├── scripts/
│   ├── gen_charuco_handeye.py
│   └── gen_charuco_intrinsic.py
└── so101_camera_calibration/
    ├── camera_intrinsic_calibration_node.py
    └── handeye_calibration_node.py
```

`cartesian_motion_node` (the IK/trajectory service used by the hand-eye
launch file) now lives in [`so101_kinematics`](../so101_kinematics), with
its services in [`so101_kinematics_msgs`](../so101_kinematics_msgs).

## Tips

- **Lighting matters.** Use consistent artificial light. Matte-print the
  ChArUco board to avoid specular reflections.
- **Rotation variety > position variety.** Tilt the gripper at different
  angles between samples.
- **Trust Park and Horaud.** Tsai-Lenz often gets depth wrong; Andreff and
  Daniilidis can be unstable. If Park and Horaud agree within ~1 cm, the
  calibration is usually in good shape.
- **Typical values** for a camera ~60 cm above the base:
  `z ≈ 0.55–0.60 m`, `x ≈ 0.15–0.20 m` (mount-dependent).

## Troubleshooting

- **No green corners detected** — improve lighting, flatten the print, or
  move the board closer.
- **Coverage grid stays incomplete** — push the board into the image corners
  and vary tilt more aggressively.
- **Park and Horaud disagree strongly** — collect more samples with more
  rotational diversity.
- **Auto-calibration results are inconsistent** — switch to manual calibration;
  servo repeatability is the likely culprit.
