#!/usr/bin/env python3
"""Run an Isaac Sim SO-101 follower controlled by the existing ROS 2 teleop topic."""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
XACRO_PATH = REPO_ROOT / "so101_description" / "urdf" / "so101_arm.urdf.xacro"
DESCRIPTION_PATH = REPO_ROOT / "so101_description"
DEFAULT_ASSET_DIR = REPO_ROOT / "build" / "isaacsim_so101"
DEFAULT_CAMERA_RIG_CONFIG = (
    REPO_ROOT
    / "so101_bringup"
    / "config"
    / "cameras"
    / "isaacsim_profiles"
    / "isaac_dual_overhead.yaml"
)

ROBOT_PRIM_PATH = "/World/SO101"
ARTICULATION_PRIM_PATH = f"{ROBOT_PRIM_PATH}/Geometry"
GRIPPER_PRIM_PATH = (
    f"{ARTICULATION_PRIM_PATH}/base_link/shoulder_link/upper_arm_link/"
    "lower_arm_link/wrist_link/gripper_link"
)
ACTION_GRAPH_PATH = "/SO101_ROS2_ActionGraph"
COMMAND_TOPIC = "/follower/forward_controller/commands"
JOINT_STATE_TOPIC = "/follower/joint_states"
CAMERA_NAMES_BY_PROFILE = {
    "none": (),
    "single_overhead": ("wrist", "overhead_1"),
    "dual_overhead": ("wrist", "overhead_1", "overhead_2"),
}
EXPECTED_CAMERA_INTERFACES = {
    "wrist": (
        "/follower/image_raw",
        "/follower/camera_info",
        "follower/wrist_camera_optical_frame",
    ),
    "overhead_1": (
        "/static_camera_1/image_raw",
        "/static_camera_1/camera_info",
        "follower/static_camera_1_optical_frame",
    ),
    "overhead_2": (
        "/static_camera_2/image_raw",
        "/static_camera_2/camera_info",
        "follower/static_camera_2_optical_frame",
    ),
}
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
CONTACT_PAD_SPECS = {
    "fixed_jaw_contact_pad_link": {
        "joint": "fixed_jaw_contact_pad_joint",
        "parent": "gripper_link",
        "size": [0.0005, 0.018, 0.025],
        "origin": [-0.00805, -0.000218, -0.0925],
    },
    "moving_jaw_contact_pad_link": {
        "joint": "moving_jaw_contact_pad_joint",
        "parent": "moving_jaw_so101_v1_link",
        "size": [0.0005, 0.025, 0.018],
        # Origin X calibrated so the pad face is flush with the moving-jaw
        # finger mesh at the grasp angle (pad protrudes ~0.1 mm, not the
        # ~2.7 mm invisible gap produced by the previous -0.01215 value).
        "origin": [-0.00996, -0.0691, 0.0190],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import the SO-101 follower and connect it to the Isaac Sim ROS 2 bridge."
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=DEFAULT_ASSET_DIR,
        help="Generated URDF/USD cache directory (default: repo build directory).",
    )
    parser.add_argument(
        "--rebuild-asset",
        action="store_true",
        help="Rebuild generated robot and camera-mount USD assets.",
    )
    parser.add_argument(
        "--camera-profile",
        choices=tuple(CAMERA_NAMES_BY_PROFILE),
        default="dual_overhead",
        help="Simulated camera set to publish (default: dual_overhead).",
    )
    parser.add_argument(
        "--camera-rig-config",
        type=Path,
        default=DEFAULT_CAMERA_RIG_CONFIG,
        help="Isaac camera calibration YAML (default: photographed dual-overhead rig).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Expand and validate the follower URDF without starting Isaac Sim.",
    )
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim without its GUI.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Exit after this many frames; zero keeps running until Isaac Sim closes.",
    )
    parser.add_argument(
        "--publish-clock",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Publish Isaac Sim time on /clock.",
    )
    parser.add_argument(
        "--physics-hz",
        type=float,
        default=120.0,
        help="Physics update frequency.",
    )
    parser.add_argument(
        "--joint-stiffness",
        type=float,
        default=100.0,
        help="Initial position-drive stiffness in Nm/rad.",
    )
    parser.add_argument(
        "--joint-damping",
        type=float,
        default=2.0,
        help="Initial position-drive damping in Nm*s/rad.",
    )
    args, _ = parser.parse_known_args()
    if args.physics_hz <= 0:
        parser.error("--physics-hz must be positive")
    if args.max_frames < 0:
        parser.error("--max-frames must be non-negative")
    if args.joint_stiffness < 0 or args.joint_damping < 0:
        parser.error("joint stiffness and damping must be non-negative")
    return args


def _require_mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{where} must be a mapping")
    return value


def _require_vector(value: Any, length: int, where: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise RuntimeError(f"{where} must contain {length} numbers")
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{where} must contain {length} numbers") from error
    if not all(math.isfinite(item) for item in result):
        raise RuntimeError(f"{where} must contain finite numbers")
    return result


def _require_positive(value: Any, where: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{where} must be a positive number") from error
    if not math.isfinite(result) or result <= 0:
        raise RuntimeError(f"{where} must be a positive number")
    return result


def resolve_description_uri(uri: Any) -> Path:
    prefix = "package://so101_description/"
    if not isinstance(uri, str) or not uri.startswith(prefix):
        raise RuntimeError(f"camera support mesh must use {prefix}: {uri!r}")
    unresolved = DESCRIPTION_PATH / uri.removeprefix(prefix)
    relative_parts = unresolved.relative_to(DESCRIPTION_PATH).parts
    current = DESCRIPTION_PATH
    for part in relative_parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"camera support mesh path must not contain symlinks: {unresolved}")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(DESCRIPTION_PATH.resolve())
    except ValueError as error:
        raise RuntimeError(f"camera support mesh escapes so101_description: {uri}") from error
    if not resolved.is_file():
        raise RuntimeError(f"camera support mesh does not exist: {resolved}")
    return resolved


def binary_stl_bounds(path: Path) -> tuple[list[float], list[float]]:
    data = path.read_bytes()
    if len(data) < 84:
        raise RuntimeError(f"camera support STL is truncated: {path}")
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + triangle_count * 50:
        raise RuntimeError(f"camera support STL must be binary and structurally valid: {path}")
    lower = [math.inf, math.inf, math.inf]
    upper = [-math.inf, -math.inf, -math.inf]
    for triangle in struct.iter_unpack("<12fH", data[84:]):
        for offset in (3, 6, 9):
            for axis in range(3):
                coordinate = float(triangle[offset + axis])
                lower[axis] = min(lower[axis], coordinate)
                upper[axis] = max(upper[axis], coordinate)
    return lower, upper


def load_camera_rig(config_path: Path, profile: str) -> dict[str, Any] | None:
    if profile == "none":
        return None
    if config_path.is_symlink():
        raise RuntimeError(f"camera rig config must not be a symlink: {config_path}")
    if not config_path.is_file():
        raise RuntimeError(f"camera rig config does not exist: {config_path}")
    try:
        data = _require_mapping(
            yaml.safe_load(config_path.read_text(encoding="utf-8")), "camera rig"
        )
    except yaml.YAMLError as error:
        raise RuntimeError(f"camera rig config is not valid YAML: {error}") from error
    if data.get("schema_version") != 1:
        raise RuntimeError("camera rig schema_version must be 1")

    resolution = _require_mapping(data.get("resolution"), "camera rig resolution")
    for dimension in ("width", "height"):
        value = resolution.get(dimension)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"camera rig resolution {dimension} must be a positive integer")
    if not math.isclose(_require_positive(data.get("fps"), "camera rig fps"), 30.0):
        raise RuntimeError("camera rig fps must remain 30 for the current data contract")

    optics = _require_mapping(data.get("optics"), "camera rig optics")
    aperture = _require_positive(optics.get("horizontal_aperture_mm"), "horizontal aperture")
    near_clip = _require_positive(optics.get("near_clip_m"), "near clip")
    far_clip = _require_positive(optics.get("far_clip_m"), "far clip")
    if near_clip >= far_clip:
        raise RuntimeError("camera near clip must be less than far clip")
    f_stop = float(optics.get("f_stop", 0.0))
    if not math.isfinite(f_stop) or f_stop < 0:
        raise RuntimeError("camera f_stop must be finite and non-negative")

    support = _require_mapping(data.get("support"), "camera rig support")
    bottom_mesh = resolve_description_uri(support.get("bottom_mesh"))
    top_mesh = resolve_description_uri(support.get("top_mesh"))
    mesh_scale = _require_positive(support.get("mesh_scale"), "support mesh_scale")
    if not math.isclose(mesh_scale, 0.001, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            "supplied camera support meshes must use scale 0.001 "
            "(millimetres to metres)"
        )
    assembly_center = _require_vector(support.get("assembly_center_mm"), 3, "assembly_center_mm")
    top_translation = _require_vector(support.get("top_translation_mm"), 3, "top_translation_mm")
    expected_height = _require_positive(support.get("expected_height_mm"), "expected_height_mm")
    bottom_lower, bottom_upper = binary_stl_bounds(bottom_mesh)
    top_lower, top_upper = binary_stl_bounds(top_mesh)
    insertion_depth = bottom_upper[1] - top_translation[1]
    if insertion_depth <= 0.0 or top_translation[1] <= bottom_lower[1]:
        raise RuntimeError("support top must slide into the bottom mesh")
    assembled_height = max(bottom_upper[1], top_translation[1] + top_upper[1]) - min(
        bottom_lower[1], top_translation[1] + top_lower[1]
    )
    if not math.isclose(assembled_height, expected_height, abs_tol=1e-3):
        raise RuntimeError(
            "camera support height mismatch: "
            f"configured={expected_height}, measured={assembled_height} mm"
        )
    _require_vector(support.get("color_rgb"), 3, "support color_rgb")
    post_centers = _require_mapping(support.get("post_centers_m"), "support post_centers_m")
    yaw_degrees = _require_mapping(support.get("yaw_deg"), "support yaw_deg")
    for camera_id in ("overhead_1", "overhead_2"):
        _require_vector(post_centers.get(camera_id), 3, f"{camera_id} post center")
        float(yaw_degrees.get(camera_id))
    colliders = _require_mapping(support.get("colliders"), "support colliders")
    _require_vector(colliders.get("foot_size_m"), 3, "support foot collider")
    _require_vector(colliders.get("post_size_m"), 3, "support post collider")
    _require_positive(colliders.get("post_center_z_m"), "support post_center_z_m")

    cameras = _require_mapping(data.get("cameras"), "camera rig cameras")
    if set(cameras) != set(EXPECTED_CAMERA_INTERFACES):
        raise RuntimeError(f"camera rig must define exactly: {sorted(EXPECTED_CAMERA_INTERFACES)}")
    for camera_id, expected_interface in EXPECTED_CAMERA_INTERFACES.items():
        camera = _require_mapping(cameras[camera_id], f"camera {camera_id}")
        actual_interface = (
            camera.get("image_topic"),
            camera.get("camera_info_topic"),
            camera.get("frame_id"),
        )
        if actual_interface != expected_interface:
            raise RuntimeError(
                f"camera {camera_id} must keep interface {expected_interface}, "
                f"got {actual_interface}"
            )
        fov = _require_positive(
            camera.get("horizontal_fov_deg"), f"{camera_id} horizontal_fov_deg"
        )
        if fov >= 179:
            raise RuntimeError(f"camera {camera_id} horizontal_fov_deg must be less than 179")
        _require_vector(camera.get("housing_size_m"), 3, f"{camera_id} housing_size_m")
        if camera_id == "wrist":
            if camera.get("parent_link") != "gripper_link":
                raise RuntimeError("wrist camera must be fixed to gripper_link")
            _require_vector(camera.get("translation_m"), 3, "wrist translation_m")
            quaternion = _require_vector(
                camera.get("orientation_wxyz"), 4, "wrist orientation_wxyz"
            )
            quaternion_norm = math.sqrt(sum(value * value for value in quaternion))
            if not math.isclose(quaternion_norm, 1.0, abs_tol=1e-5):
                raise RuntimeError("wrist orientation_wxyz must be a unit quaternion")
        else:
            position = _require_vector(camera.get("position_m"), 3, f"{camera_id} position_m")
            target = _require_vector(camera.get("look_at_m"), 3, f"{camera_id} look_at_m")
            if position == target:
                raise RuntimeError(f"camera {camera_id} position and look-at target must differ")

    data["_config_path"] = config_path.resolve()
    support["_bottom_mesh_path"] = bottom_mesh
    support["_top_mesh_path"] = top_mesh
    support["_measured_height_mm"] = assembled_height
    support["_insertion_depth_mm"] = insertion_depth
    support["_assembly_center_mm"] = assembly_center
    data["_selected_camera_names"] = CAMERA_NAMES_BY_PROFILE[profile]
    data["_horizontal_aperture_mm"] = aperture
    data["_near_clip_m"] = near_clip
    data["_far_clip_m"] = far_clip
    data["_f_stop"] = f_stop
    return data


def expand_follower_urdf(asset_dir: Path) -> Path:
    xacro = shutil.which("xacro")
    if xacro is None:
        raise RuntimeError(
            "xacro was not found; source /opt/ros/jazzy/setup.bash "
            "and this workspace first"
        )

    asset_dir.mkdir(parents=True, exist_ok=True)
    output_path = asset_dir / "so101_follower.urdf"
    temporary_path = asset_dir / "so101_follower.urdf.tmp"
    command = [
        xacro,
        str(XACRO_PATH),
        "variant:=follower",
        "use_ros2_control:=false",
        "add_transmissions:=false",
        "simulation_contact_pads:=true",
    ]
    with temporary_path.open("w", encoding="utf-8") as output:
        result = subprocess.run(
            command, stdout=output, stderr=subprocess.PIPE, text=True, check=False
        )
    if result.returncode != 0:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"xacro failed:\n{result.stderr.strip()}")
    temporary_path.replace(output_path)
    validate_follower_urdf(output_path)
    return output_path


def validate_follower_urdf(urdf_path: Path) -> None:
    robot = ET.parse(urdf_path).getroot()
    if robot.tag != "robot" or robot.attrib.get("name") != "so101_arm":
        raise RuntimeError(
            f"unexpected robot in generated URDF: {robot.tag} "
            f"{robot.attrib.get('name')}"
        )
    if robot.find("ros2_control") is not None:
        raise RuntimeError("generated simulation URDF unexpectedly contains ros2_control")

    movable_joints = {
        joint.attrib["name"]
        for joint in robot.findall("joint")
        if joint.attrib.get("type") in {"continuous", "prismatic", "revolute"}
    }
    if movable_joints != set(JOINT_NAMES):
        raise RuntimeError(
            "generated URDF joint mismatch: "
            f"missing={sorted(set(JOINT_NAMES) - movable_joints)}, "
            f"unexpected={sorted(movable_joints - set(JOINT_NAMES))}"
        )
    if not robot.findall(".//collision") or not robot.findall(".//inertial"):
        raise RuntimeError("generated URDF must contain collision and inertial data")

    for link_name, spec in CONTACT_PAD_SPECS.items():
        link = robot.find(f"./link[@name='{link_name}']")
        if link is None:
            raise RuntimeError(f"generated follower URDF is missing {link_name}")
        if link.find("visual") is not None:
            raise RuntimeError(f"{link_name} must remain collision-only")
        box = link.find("./collision/geometry/box")
        if box is None:
            raise RuntimeError(f"{link_name} must use a box collision")
        size = [float(value) for value in box.attrib.get("size", "").split()]
        if len(size) != 3 or any(
            not math.isclose(actual, expected, abs_tol=1e-12)
            for actual, expected in zip(size, spec["size"])
        ):
            raise RuntimeError(f"unexpected {link_name} collision size: {size}")
        mass = link.find("./inertial/mass")
        if mass is None or not math.isclose(
            float(mass.attrib.get("value", "nan")), 0.0001, abs_tol=1e-12
        ):
            raise RuntimeError(f"{link_name} must have a 0.1 g mass")
        joint = robot.find(f"./joint[@name='{spec['joint']}']")
        if joint is None or joint.attrib.get("type") != "fixed":
            raise RuntimeError(f"{link_name} must be attached by a fixed joint")
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or parent.attrib.get("link") != spec["parent"]:
            raise RuntimeError(f"unexpected parent for {spec['joint']}")
        if child is None or child.attrib.get("link") != link_name:
            raise RuntimeError(f"unexpected child for {spec['joint']}")
        origin = joint.find("origin")
        xyz = [] if origin is None else [
            float(value) for value in origin.attrib.get("xyz", "").split()
        ]
        if len(xyz) != 3 or any(
            not math.isclose(actual, expected, abs_tol=1e-12)
            for actual, expected in zip(xyz, spec["origin"])
        ):
            raise RuntimeError(f"unexpected origin for {spec['joint']}: {xyz}")

    wrist_joint = robot.find("./joint[@name='wrist_camera_joint']")
    if wrist_joint is None:
        raise RuntimeError("generated follower URDF is missing wrist_camera_joint")
    parent = wrist_joint.find("parent")
    origin = wrist_joint.find("origin")
    if parent is None or parent.attrib.get("link") != "gripper_link":
        raise RuntimeError("wrist_camera_joint must be fixed to gripper_link")
    if origin is None:
        raise RuntimeError("wrist_camera_joint is missing its calibrated origin")
    xyz = [float(value) for value in origin.attrib.get("xyz", "").split()]
    rpy = [float(value) for value in origin.attrib.get("rpy", "").split()]
    expected_xyz = [0.0025, -0.0720574, 0.0041503]
    expected_rpy = [math.pi / 2.0, math.radians(65.0), math.pi / 2.0]
    if len(xyz) != 3 or any(
        not math.isclose(actual, expected, abs_tol=1e-6)
        for actual, expected in zip(xyz, expected_xyz)
    ):
        raise RuntimeError(f"unexpected wrist camera translation in generated URDF: {xyz}")
    if len(rpy) != 3 or any(
        not math.isclose(actual, expected, abs_tol=1e-5)
        for actual, expected in zip(rpy, expected_rpy)
    ):
        raise RuntimeError(f"unexpected wrist camera rotation in generated URDF: {rpy}")


def newest_description_mtime() -> float:
    inputs = list((DESCRIPTION_PATH / "urdf").rglob("*.xacro"))
    inputs.extend((DESCRIPTION_PATH / "meshes").rglob("*.stl"))
    return max(path.stat().st_mtime for path in inputs)


async def convert_mesh_to_usd(asset_converter, input_path: Path, output_path: Path) -> None:
    def progress_callback(_progress: float, _total_steps: int) -> None:
        return None

    context = asset_converter.AssetConverterContext()
    context.ignore_materials = True
    context.single_mesh = True
    context.smooth_normals = True
    task = asset_converter.get_instance().create_converter_task(
        str(input_path), str(output_path), progress_callback, context
    )
    while not await task.wait_until_finished():
        await asyncio.sleep(0.1)
    if not output_path.is_file():
        raise RuntimeError(f"camera support converter did not create: {output_path}")


def import_or_reuse_camera_mounts(args, camera_rig, asset_converter) -> dict[str, Path]:
    if camera_rig is None:
        return {}
    support = camera_rig["support"]
    cache_dir = args.asset_dir / "camera_rig"
    if cache_dir.is_symlink():
        raise RuntimeError(f"refusing to use a symlinked camera asset directory: {cache_dir}")
    if args.rebuild_asset and cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    converted: dict[str, Path] = {}
    for part in ("bottom", "top"):
        source = support[f"_{part}_mesh_path"]
        output = cache_dir / f"cam_mount_{part}.usd"
        if output.is_symlink():
            raise RuntimeError(f"camera support cache must not be a symlink: {output}")
        if output.exists() and output.stat().st_mtime < source.stat().st_mtime:
            raise RuntimeError(f"stale camera support USD: {output}; rerun with --rebuild-asset")
        if not output.exists():
            asyncio.get_event_loop().run_until_complete(
                convert_mesh_to_usd(asset_converter, source, output)
            )
        converted[part] = output.resolve()
    return converted


def _yaw_quaternion(degrees: float) -> list[float]:
    half_angle = math.radians(degrees) / 2.0
    return [math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)]


def create_camera_supports(
    camera_rig,
    converted_mounts,
    stage,
    stage_utils,
    xform_prim_type,
    cube_type,
    gf,
    usd_geom,
    usd_physics,
) -> None:
    if camera_rig is None:
        return
    support = camera_rig["support"]
    mesh_scale = float(support["mesh_scale"])
    assembly_center = support["_assembly_center_mm"]
    top_translation = support["top_translation_mm"]
    # Center the native mesh cross-section on the support root, then rotate
    # native +Y to stage +Z with +90 degrees about X.
    bottom_translation = [
        -assembly_center[0] * mesh_scale,
        assembly_center[2] * mesh_scale,
        -assembly_center[1] * mesh_scale,
    ]
    native_top_from_center = [
        top_translation[index] - assembly_center[index] for index in range(3)
    ]
    top_stage_translation = [
        native_top_from_center[0] * mesh_scale,
        -native_top_from_center[2] * mesh_scale,
        native_top_from_center[1] * mesh_scale,
    ]
    upright_orientation = [math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]
    color = support["color_rgb"]
    colliders = support["colliders"]

    usd_geom.Xform.Define(stage, "/World/CameraRig")
    for camera_id in ("overhead_1", "overhead_2"):
        label = "Left" if camera_id == "overhead_1" else "Right"
        root_path = f"/World/CameraRig/{label}Support"
        usd_geom.Xform.Define(stage, root_path)
        root = xform_prim_type(
            root_path,
            positions=[support["post_centers_m"][camera_id]],
            orientations=[_yaw_quaternion(float(support["yaw_deg"][camera_id]))],
            reset_xform_op_properties=True,
        )
        if not root.valid:
            raise RuntimeError(f"failed to create camera support root: {root_path}")

        bottom_path = f"{root_path}/Bottom"
        top_path = f"{root_path}/Top"
        stage_utils.add_reference_to_stage(str(converted_mounts["bottom"]), bottom_path)
        stage_utils.add_reference_to_stage(str(converted_mounts["top"]), top_path)
        xform_prim_type(
            bottom_path,
            translations=[bottom_translation],
            orientations=[upright_orientation],
            scales=[[mesh_scale, mesh_scale, mesh_scale]],
            reset_xform_op_properties=True,
        )
        xform_prim_type(
            top_path,
            translations=[top_stage_translation],
            orientations=[upright_orientation],
            scales=[[mesh_scale, mesh_scale, mesh_scale]],
            reset_xform_op_properties=True,
        )

        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith(f"{root_path}/") and prim.IsA(usd_geom.Gprim):
                usd_geom.Gprim(prim).CreateDisplayColorAttr([gf.Vec3f(*color)])

        foot = cube_type(
            f"{root_path}/FootCollider",
            sizes=[1.0],
            translations=[[0.0, 0.0, colliders["foot_size_m"][2] / 2.0]],
            scales=[colliders["foot_size_m"]],
        )
        post = cube_type(
            f"{root_path}/PostCollider",
            sizes=[1.0],
            translations=[[0.0, 0.0, colliders["post_center_z_m"]]],
            scales=[colliders["post_size_m"]],
        )
        for collider in (foot, post):
            prim = collider.prims[0]
            usd_physics.CollisionAPI.Apply(prim)
            usd_geom.Imageable(prim).MakeInvisible()


def create_sim_cameras(
    camera_rig, rtx_camera_type, cube_type, transform_utils
) -> list[dict[str, Any]]:
    if camera_rig is None:
        return []
    width = int(camera_rig["resolution"]["width"])
    height = int(camera_rig["resolution"]["height"])
    fps = float(camera_rig["fps"])
    aperture = camera_rig["_horizontal_aperture_mm"]
    vertical_aperture = aperture * height / width
    selected_names = camera_rig["_selected_camera_names"]
    specs: list[dict[str, Any]] = []

    for camera_id in selected_names:
        config = camera_rig["cameras"][camera_id]
        if camera_id == "wrist":
            prim_path = f"{GRIPPER_PRIM_PATH}/WristCamera"
            sensor = rtx_camera_type(prim_path, tick_rate=fps)
            sensor.camera.set_local_poses(
                translations=[config["translation_m"]],
                orientations=[config["orientation_wxyz"]],
            )
        else:
            prim_name = "OverheadCamera1" if camera_id == "overhead_1" else "OverheadCamera2"
            prim_path = f"/World/CameraRig/{prim_name}"
            orientation = transform_utils.look_at_quaternion(
                eye=config["position_m"], target=config["look_at_m"]
            ).numpy()
            sensor = rtx_camera_type(
                prim_path,
                tick_rate=fps,
                positions=[config["position_m"]],
                orientations=[orientation],
            )

        focal_length = aperture / (
            2.0 * math.tan(math.radians(float(config["horizontal_fov_deg"])) / 2.0)
        )
        sensor.camera.set_focal_lengths([focal_length])
        sensor.camera.set_apertures(
            horizontal_apertures=[aperture], vertical_apertures=[vertical_aperture]
        )
        sensor.camera.set_aperture_offsets(
            horizontal_offsets=[0.0], vertical_offsets=[0.0]
        )
        sensor.camera.set_clipping_ranges(
            near_distances=[camera_rig["_near_clip_m"]],
            far_distances=[camera_rig["_far_clip_m"]],
        )
        sensor.camera.set_fstops([camera_rig["_f_stop"]])

        housing_size = config["housing_size_m"]
        cube_type(
            f"{prim_path}/Housing",
            sizes=[1.0],
            colors="#111111",
            translations=[[0.0, 0.0, housing_size[2] / 2.0]],
            scales=[housing_size],
        )
        specs.append(
            {
                "id": camera_id,
                "prim_path": prim_path,
                "image_topic": config["image_topic"],
                "camera_info_topic": config["camera_info_topic"],
                "frame_id": config["frame_id"],
                "width": width,
                "height": height,
                "fps": fps,
            }
        )
    return specs


def run_isaac_sim(args: argparse.Namespace, urdf_path: Path, camera_rig) -> None:
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})
    exit_code = 0
    try:
        import omni.graph.core as og
        import omni.kit.app
        import omni.timeline
        import usdrt.Sdf
        from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
        from isaacsim.core.experimental.objects import Cube, DistantLight, GroundPlane
        from isaacsim.core.experimental.prims import Articulation, XformPrim
        import isaacsim.core.experimental.utils.app as app_utils
        import isaacsim.core.experimental.utils.stage as stage_utils
        import isaacsim.core.experimental.utils.transform as transform_utils
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.utils.viewports import set_camera_view
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        extension_manager = omni.kit.app.get_app().get_extension_manager()
        extensions = [
            "omni.scene.optimizer.core",
            "isaacsim.robot.schema",
            "isaacsim.asset.importer.urdf",
            "isaacsim.ros2.bridge",
        ]
        if camera_rig is not None:
            extensions.extend(["omni.kit.asset_converter", "isaacsim.sensors.experimental.rtx"])
        for extension in extensions:
            extension_manager.set_extension_enabled_immediate(extension, True)
        simulation_app.update()

        usd_path = import_or_reuse_asset(args, urdf_path, URDFImporter, URDFImporterConfig)
        validate_generated_robot_usd(usd_path, Usd, UsdGeom, UsdPhysics)
        converted_mounts = {}
        rtx_camera_type = None
        if camera_rig is not None:
            import omni.kit.asset_converter
            from isaacsim.sensors.experimental.rtx import RtxCamera

            converted_mounts = import_or_reuse_camera_mounts(
                args, camera_rig, omni.kit.asset_converter
            )
            rtx_camera_type = RtxCamera

        stage_utils.create_new_stage()
        stage_utils.set_stage_units(meters_per_unit=1.0)
        GroundPlane("/World/GroundPlane", positions=[0.0, 0.0, 0.0])
        light = DistantLight("/World/DistantLight")
        light.set_intensities(300)
        stage_utils.add_reference_to_stage(usd_path=str(usd_path), path=ROBOT_PRIM_PATH)
        create_camera_supports(
            camera_rig,
            converted_mounts,
            stage_utils.get_current_stage(backend="usd"),
            stage_utils,
            XformPrim,
            Cube,
            Gf,
            UsdGeom,
            UsdPhysics,
        )
        set_camera_view(
            eye=[0.65, 0.65, 0.45],
            target=[0.0, 0.0, 0.16],
            camera_prim_path="/OmniverseKit_Persp",
        )
        simulation_app.update()

        camera_specs = create_sim_cameras(
            camera_rig, rtx_camera_type, Cube, transform_utils
        )
        simulation_app.update()

        articulation = Articulation(ARTICULATION_PRIM_PATH)
        imported_joints = list(articulation.dof_names)
        if set(imported_joints) != set(JOINT_NAMES):
            raise RuntimeError(
                "imported articulation joint mismatch: "
                f"expected={JOINT_NAMES}, actual={imported_joints}"
            )

        create_ros_action_graph(
            og, simulation_app, usdrt.Sdf, args.publish_clock, camera_specs
        )
        SimulationManager.setup_simulation(dt=1.0 / args.physics_hz, device="cpu")

        domain_id = os.environ.get("ROS_DOMAIN_ID", "0")
        rmw = os.environ.get("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp (Isaac Sim default)")
        print(f"SO-101 USD: {usd_path}", flush=True)
        print(f"Imported DOF order: {imported_joints}", flush=True)
        print(f"ROS_DOMAIN_ID={domain_id}; RMW_IMPLEMENTATION={rmw}", flush=True)
        print(f"Subscribing: {COMMAND_TOPIC} [std_msgs/msg/Float64MultiArray]", flush=True)
        print(f"Publishing: {JOINT_STATE_TOPIC} [sensor_msgs/msg/JointState]", flush=True)
        for camera in camera_specs:
            print(
                f"Publishing: {camera['image_topic']} [sensor_msgs/msg/Image] "
                f"and {camera['camera_info_topic']} [sensor_msgs/msg/CameraInfo] "
                f"at {camera['width']}x{camera['height']} {camera['fps']:g} Hz",
                flush=True,
            )

        timeline = omni.timeline.get_timeline_interface()
        if args.headless:
            app_utils.play()
            simulation_app.update()
        else:
            print(
                "Isaac Sim is ready; press Play to activate physics and ROS camera streams.",
                flush=True,
            )
        frame_count = 0
        while simulation_app.is_running():
            simulation_app.update()
            if timeline.is_playing():
                frame_count += 1
                if args.max_frames and frame_count >= args.max_frames:
                    break
    except Exception as error:
        exit_code = 1
        print(f"Isaac Sim setup/runtime error: {error!r}", file=sys.stderr, flush=True)
        raise
    finally:
        simulation_app.close(exit_code=exit_code)


def import_or_reuse_asset(args, urdf_path, importer_type, config_type) -> Path:
    asset_dir = args.asset_dir.resolve()
    robot_name = urdf_path.stem
    robot_output_dir = asset_dir / robot_name
    expected_usd = robot_output_dir / f"{robot_name}.usda"

    if robot_output_dir.is_symlink():
        raise RuntimeError(
            f"refusing to use a symlinked Isaac Sim asset directory: {robot_output_dir}"
        )
    if args.rebuild_asset and robot_output_dir.exists():
        shutil.rmtree(robot_output_dir)
    if expected_usd.exists():
        if expected_usd.stat().st_mtime < newest_description_mtime():
            raise RuntimeError("cached SO-101 USD is stale; rerun with --rebuild-asset")
        return expected_usd
    if robot_output_dir.exists():
        raise RuntimeError(
            f"incomplete generated asset directory: {robot_output_dir}; "
            "rerun with --rebuild-asset"
        )

    config = config_type(
        urdf_path=str(urdf_path),
        usd_path=str(asset_dir),
        # The collision-only jaw pads must remain separate rigid bodies so the
        # two PhysX contact sensors can address them independently.
        merge_fixed_joints=False,
        merge_mesh=False,
        collision_from_visuals=False,
        allow_self_collision=False,
        ros_package_paths=[{"name": "so101_description", "path": str(DESCRIPTION_PATH)}],
        robot_type="Manipulator",
        fix_base=True,
        joint_drive_type="force",
        joint_target_type="position",
        override_joint_stiffness=args.joint_stiffness,
        override_joint_damping=args.joint_damping,
        run_asset_transformer=True,
        run_multi_physics_conversion=True,
    )
    usd_path = Path(importer_type(config=config).import_urdf()).resolve()
    if not usd_path.is_file():
        raise RuntimeError(f"URDF importer did not produce the expected USD file: {usd_path}")
    return usd_path


def validate_generated_robot_usd(usd_path, usd, usd_geom, usd_physics) -> None:
    """Validate the independently addressable pad and articulation contract."""
    stage = usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"could not open generated robot USD: {usd_path}")
    prims = list(stage.Traverse())

    movable_joints = {
        prim.GetName()
        for prim in prims
        if prim.IsA(usd_physics.RevoluteJoint)
        or prim.IsA(usd_physics.PrismaticJoint)
    }
    if movable_joints != set(JOINT_NAMES):
        raise RuntimeError(
            "generated USD joint mismatch: "
            f"missing={sorted(set(JOINT_NAMES) - movable_joints)}, "
            f"unexpected={sorted(movable_joints - set(JOINT_NAMES))}"
        )

    for link_name, spec in CONTACT_PAD_SPECS.items():
        matches = [prim for prim in prims if prim.GetName() == link_name]
        if len(matches) != 1:
            raise RuntimeError(
                f"generated USD must contain exactly one {link_name}; got {len(matches)}"
            )
        pad = matches[0]
        if not pad.HasAPI(usd_physics.RigidBodyAPI):
            raise RuntimeError(f"{link_name} is not an independently addressable rigid body")
        geometry = [
            prim
            for prim in usd.PrimRange(pad)
            if prim != pad and prim.IsA(usd_geom.Gprim)
        ]
        if not geometry or any(
            not prim.HasAPI(usd_physics.CollisionAPI) for prim in geometry
        ):
            raise RuntimeError(f"{link_name} must contain collision-only geometry")

        fixed_joint = [
            prim
            for prim in prims
            if prim.GetName() == spec["joint"] and prim.IsA(usd_physics.FixedJoint)
        ]
        if len(fixed_joint) != 1:
            raise RuntimeError(f"generated USD is missing fixed joint {spec['joint']}")


def create_ros_action_graph(
    og, simulation_app, sdf, publish_clock: bool, camera_specs: list[dict[str, Any]]
) -> None:
    keys = og.Controller.Keys
    nodes = [
        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
        ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
        ("CommandQoS", "isaacsim.ros2.bridge.ROS2QoSProfile"),
        ("CommandSubscriber", "isaacsim.ros2.bridge.ROS2Subscriber"),
        ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
        ("ReadJointState", "isaacsim.sensors.physics.IsaacReadJointState"),
        ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
    ]
    connections = [
        ("OnPlaybackTick.outputs:tick", "CommandSubscriber.inputs:execIn"),
        ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
        ("OnPlaybackTick.outputs:tick", "ReadJointState.inputs:execIn"),
        ("ROS2Context.outputs:context", "CommandSubscriber.inputs:context"),
        ("ROS2Context.outputs:context", "PublishJointState.inputs:context"),
        ("CommandQoS.outputs:qosProfile", "CommandSubscriber.inputs:qosProfile"),
        ("ReadJointState.outputs:execOut", "PublishJointState.inputs:execIn"),
        ("ReadJointState.outputs:jointNames", "PublishJointState.inputs:jointNames"),
        ("ReadJointState.outputs:jointPositions", "PublishJointState.inputs:jointPositions"),
        ("ReadJointState.outputs:jointVelocities", "PublishJointState.inputs:jointVelocities"),
        ("ReadJointState.outputs:jointEfforts", "PublishJointState.inputs:jointEfforts"),
        ("ReadJointState.outputs:jointDofTypes", "PublishJointState.inputs:jointDofTypes"),
        (
            "ReadJointState.outputs:stageMetersPerUnit",
            "PublishJointState.inputs:stageMetersPerUnit",
        ),
        ("ReadJointState.outputs:sensorTime", "PublishJointState.inputs:sensorTime"),
    ]
    values = [
        ("CommandQoS.inputs:history", "keepLast"),
        ("CommandQoS.inputs:depth", 10),
        ("CommandQoS.inputs:reliability", "reliable"),
        ("CommandQoS.inputs:durability", "volatile"),
        ("CommandSubscriber.inputs:topicName", COMMAND_TOPIC),
        ("ArticulationController.inputs:robotPath", ARTICULATION_PRIM_PATH),
        ("ArticulationController.inputs:jointNames", JOINT_NAMES),
        ("ReadJointState.inputs:prim", [sdf.Path(ARTICULATION_PRIM_PATH)]),
        ("PublishJointState.inputs:topicName", JOINT_STATE_TOPIC),
        ("PublishJointState.inputs:queueSize", 10),
    ]

    if camera_specs:
        nodes.append(("CameraRunOnce", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"))
        connections.append(("OnPlaybackTick.outputs:tick", "CameraRunOnce.inputs:execIn"))
        for camera in camera_specs:
            prefix = camera["id"].title().replace("_", "")
            render_node = f"{prefix}RenderProduct"
            image_node = f"{prefix}RGBPublish"
            info_node = f"{prefix}CameraInfoPublish"
            nodes.extend(
                [
                    (render_node, "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                    (image_node, "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    (info_node, "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                ]
            )
            connections.extend(
                [
                    ("CameraRunOnce.outputs:step", f"{render_node}.inputs:execIn"),
                    (f"{render_node}.outputs:execOut", f"{image_node}.inputs:execIn"),
                    (f"{render_node}.outputs:execOut", f"{info_node}.inputs:execIn"),
                    (
                        f"{render_node}.outputs:renderProductPath",
                        f"{image_node}.inputs:renderProductPath",
                    ),
                    (
                        f"{render_node}.outputs:renderProductPath",
                        f"{info_node}.inputs:renderProductPath",
                    ),
                    ("ROS2Context.outputs:context", f"{image_node}.inputs:context"),
                    ("ROS2Context.outputs:context", f"{info_node}.inputs:context"),
                ]
            )
            values.extend(
                [
                    (f"{render_node}.inputs:cameraPrim", [sdf.Path(camera["prim_path"])]),
                    (f"{render_node}.inputs:width", camera["width"]),
                    (f"{render_node}.inputs:height", camera["height"]),
                    (f"{render_node}.inputs:enabled", True),
                    (f"{image_node}.inputs:type", "rgb"),
                    (f"{image_node}.inputs:topicName", camera["image_topic"]),
                    (f"{image_node}.inputs:frameId", camera["frame_id"]),
                    (f"{image_node}.inputs:resetSimulationTimeOnStop", True),
                    (f"{info_node}.inputs:topicName", camera["camera_info_topic"]),
                    (f"{info_node}.inputs:frameId", camera["frame_id"]),
                    (f"{info_node}.inputs:resetSimulationTimeOnStop", True),
                ]
            )

    if publish_clock:
        nodes.extend(
            [
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ]
        )
        connections.extend(
            [
                ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                ("ROS2Context.outputs:context", "PublishClock.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
            ]
        )
        values.append(("PublishClock.inputs:topicName", "/clock"))

    og.Controller.edit(
        {"graph_path": ACTION_GRAPH_PATH, "evaluator_name": "execution"},
        {keys.CREATE_NODES: nodes, keys.CONNECT: connections, keys.SET_VALUES: values},
    )

    subscriber = og.Controller.node(f"{ACTION_GRAPH_PATH}/CommandSubscriber")
    for attribute, value in (
        ("inputs:messagePackage", "std_msgs"),
        ("inputs:messageSubfolder", "msg"),
        ("inputs:messageName", "Float64MultiArray"),
    ):
        og.Controller.attribute(attribute, subscriber).set(value)
        simulation_app.update()

    try:
        controller = og.Controller.node(f"{ACTION_GRAPH_PATH}/ArticulationController")
        og.Controller.connect(
            og.Controller.attribute("outputs:data", subscriber),
            og.Controller.attribute("inputs:positionCommand", controller),
        )
    except Exception as error:
        raise RuntimeError(
            "Isaac Sim did not create Float64MultiArray outputs:data; "
            "direct bridge smoke test failed"
        ) from error


def main() -> int:
    args = parse_args()
    args.asset_dir = args.asset_dir.resolve()
    args.camera_rig_config = args.camera_rig_config.expanduser()
    camera_rig = load_camera_rig(args.camera_rig_config, args.camera_profile)
    urdf_path = expand_follower_urdf(args.asset_dir)
    print(f"Validated follower URDF: {urdf_path}")
    if camera_rig is not None:
        print(
            f"Validated {args.camera_profile} camera rig: {camera_rig['_config_path']} "
            f"({camera_rig['support']['_measured_height_mm']:.3f} mm supports, "
            f"{camera_rig['support']['_insertion_depth_mm']:.3f} mm insertion, "
            "no source symlinks)"
        )
    if args.validate_only:
        return 0
    if os.environ.get("ROS_DISTRO") not in {"jazzy", "humble"}:
        raise RuntimeError("source ROS 2 Jazzy or Humble before starting Isaac Sim")
    run_isaac_sim(args, urdf_path, camera_rig)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
