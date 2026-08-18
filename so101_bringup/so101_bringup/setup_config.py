"""Strict setup loading: one canonical YAML fully describes one hardware rig.

Each file in ``config/setups`` defines a complete setup: the arm topology
(namespaces, USB ports, world placement), the logical camera contract
(node/namespace/topic/frame names), the physical camera rig (device paths),
and the Isaac Sim camera geometry (``sim:`` section). Topics and frames are
concrete strings — the setup file owns the namespaces, so nothing renders
templates anywhere else in the stack.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import stat
from typing import Any

import yaml


CANONICAL_SETUPS = ("monomanual", "monomanual_dual_overhead", "bimanual")

_SETUP_KEYS = {"setup", "schema_version", "arms", "cameras", "sim"}
_ARM_KEYS = {"namespace", "usb_port", "tf_xyz", "tf_yaw_deg"}
_PROFILE_KEYS = {
    "id", "node_name", "namespace", "image_topic", "frame_id",
    "feature", "default_backend",
}
_CAMERA_KEYS = {"device", "gscam_config", "driver"}
_DRIVER_KEYS = {
    "package", "executable", "parameters", "remappings", "device_parameter",
    "frame_id_parameter",
}

ARM_JOINTS = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll",
    "gripper",
)


class SetupConfigError(ValueError):
    """Raised when a setup file is invalid."""


def evaluate_camera_streams(
    last_image: dict[str, float | None],
    now: float,
    stale_timeout_s: float,
) -> tuple[list[str], list[str]]:
    """Return missing and stale stream labels for the supervisor."""
    missing = [f"{name}:image" for name, received in last_image.items() if received is None]
    if missing:
        return missing, []
    stale = [
        f"{name}:image" for name, received in last_image.items()
        if now - received > stale_timeout_s
    ]
    return [], stale


@dataclass(frozen=True)
class ArmSpec:
    role: str  # "leader" | "follower"
    namespace: str
    usb_port: str
    tf_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tf_yaw_deg: float = 0.0

    @property
    def frame_prefix(self) -> str:
        return f"{self.namespace}/"


@dataclass(frozen=True)
class Setup:
    name: str
    leaders: tuple[ArmSpec, ...]
    followers: tuple[ArmSpec, ...]
    cameras: tuple[dict[str, str], ...]
    rig: dict[str, dict[str, Any]]
    sim: dict[str, Any] | None

    @property
    def pairs(self) -> int:
        return len(self.followers)

    def joint_state_topics(self) -> tuple[str, ...]:
        return tuple(f"/{arm.namespace}/joint_states" for arm in self.followers)

    def command_topics(self) -> tuple[str, ...]:
        return tuple(
            f"/{arm.namespace}/forward_controller/commands" for arm in self.followers
        )


@dataclass(frozen=True)
class CameraSpec:
    camera_id: str
    node_name: str
    namespace: str
    image_topic: str
    frame_id: str
    feature: str
    backend: str
    device: str
    parameters: dict[str, Any]
    remappings: tuple[tuple[str, str], ...]
    driver_package: str
    driver_executable: str


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SetupConfigError(f"{where} must be a mapping")
    return value


def _only_keys(data: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise SetupConfigError(f"{where} has unknown keys: {', '.join(unknown)}")


def _nonempty_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SetupConfigError(f"{where} must be a non-empty string")
    return value.strip()


def _load_yaml(path: Path, where: str) -> dict[str, Any]:
    if not path.is_file():
        raise SetupConfigError(f"{where} does not exist: {path}")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), where)
    except yaml.YAMLError as exc:
        raise SetupConfigError(f"{where} is not valid YAML: {exc}") from exc


def default_setups_dir() -> Path:
    """Locate the installed so101_bringup setups directory."""
    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory("so101_bringup")) / "config" / "setups"


def normalize_namespace(value: str) -> str:
    value = value.strip().strip("/")
    if not value or "//" in value:
        raise SetupConfigError("namespace must name one ROS namespace")
    return value


def _parse_arm(role: str, index: int, raw: Any) -> ArmSpec:
    arm = _mapping(raw, f"arms.{role}s[{index}]")
    _only_keys(arm, _ARM_KEYS, f"arms.{role}s[{index}]")
    namespace = normalize_namespace(
        _nonempty_string(arm.get("namespace"), f"{role}s[{index}].namespace")
    )
    usb_port = _nonempty_string(arm.get("usb_port"), f"{role}s[{index}].usb_port")
    tf_xyz_raw = arm.get("tf_xyz", [0.0, 0.0, 0.0])
    if (
        not isinstance(tf_xyz_raw, list)
        or len(tf_xyz_raw) != 3
        or not all(isinstance(v, (int, float)) for v in tf_xyz_raw)
    ):
        raise SetupConfigError(f"{role}s[{index}].tf_xyz must be three numbers")
    tf_yaw_raw = arm.get("tf_yaw_deg", 0.0)
    if not isinstance(tf_yaw_raw, (int, float)):
        raise SetupConfigError(f"{role}s[{index}].tf_yaw_deg must be a number")
    return ArmSpec(
        role=role,
        namespace=namespace,
        usb_port=usb_port,
        tf_xyz=(float(tf_xyz_raw[0]), float(tf_xyz_raw[1]), float(tf_xyz_raw[2])),
        tf_yaw_deg=float(tf_yaw_raw),
    )


def load_setup(
    setup_name: str,
    setups_dir: str | os.PathLike[str] | None = None,
    *,
    path: str | os.PathLike[str] | None = None,
) -> Setup:
    """Load and validate one canonical setup YAML.

    The file is located either explicitly via ``path`` (e.g. an edited external
    copy of a canonical setup) or as ``<setups_dir>/<setup_name>.yaml``.
    """
    if setup_name not in CANONICAL_SETUPS:
        allowed = ", ".join(CANONICAL_SETUPS)
        raise SetupConfigError(f"setup must be one of: {allowed}")
    if path is None:
        if setups_dir is None:
            raise SetupConfigError("either setups_dir or path is required")
        file_path = Path(setups_dir) / f"{setup_name}.yaml"
    else:
        file_path = Path(path)
        if not str(path).strip():
            raise SetupConfigError("setup_config_file must be a non-empty path")
    data = _load_yaml(file_path, "setup config")
    _only_keys(data, _SETUP_KEYS, "setup config")
    for required in ("setup", "schema_version", "arms", "cameras"):
        if required not in data:
            raise SetupConfigError(f"setup config is missing required key: {required}")
    if data["setup"] != setup_name:
        raise SetupConfigError(f"setup config file must declare setup: {setup_name}")
    if data["schema_version"] != 1:
        raise SetupConfigError("setup config schema_version must be 1")

    arms = _mapping(data["arms"], "arms")
    _only_keys(arms, {"leaders", "followers"}, "arms")
    parsed_arms: list[ArmSpec] = []
    for role in ("leader", "follower"):
        group = arms.get(f"{role}s")
        if not isinstance(group, list) or not group:
            raise SetupConfigError(f"arms.{role}s must be a non-empty list")
        parsed_arms.extend(_parse_arm(role, index, raw) for index, raw in enumerate(group))
    namespaces = [arm.namespace for arm in parsed_arms]
    if len(namespaces) != len(set(namespaces)):
        raise SetupConfigError("arm namespaces must be unique across leaders and followers")
    leaders = tuple(arm for arm in parsed_arms if arm.role == "leader")
    followers = tuple(arm for arm in parsed_arms if arm.role == "follower")
    if len(leaders) != len(followers):
        raise SetupConfigError(
            "arms.leaders and arms.followers must define one leader per follower"
        )

    cameras_section = _mapping(data["cameras"], "cameras")
    _only_keys(cameras_section, {"profile", "rig"}, "cameras")
    profile = cameras_section.get("profile")
    if not isinstance(profile, list) or not profile:
        raise SetupConfigError("cameras.profile must be a non-empty list")
    parsed_profile: list[dict[str, str]] = []
    for index, raw in enumerate(profile):
        camera = _mapping(raw, f"cameras.profile[{index}]")
        _only_keys(camera, _PROFILE_KEYS, f"cameras.profile[{index}]")
        parsed_profile.append({
            key: _nonempty_string(camera.get(key), f"cameras.profile {key}")
            for key in _PROFILE_KEYS
        })
    ids = [camera["id"] for camera in parsed_profile]
    if len(ids) != len(set(ids)):
        raise SetupConfigError(f"{setup_name} contains duplicate camera ids")
    features = [camera["feature"] for camera in parsed_profile]
    if len(features) != len(set(features)):
        raise SetupConfigError(f"{setup_name} contains duplicate feature names")
    for key in ("image_topic", "frame_id"):
        values = [camera[key] for camera in parsed_profile]
        if len(values) != len(set(values)):
            raise SetupConfigError(f"{setup_name} contains duplicate {key} values")
    node_locations = [
        (normalize_namespace(camera["namespace"]), camera["node_name"])
        for camera in parsed_profile
    ]
    if len(node_locations) != len(set(node_locations)):
        raise SetupConfigError(
            f"{setup_name} contains duplicate (namespace, node_name) camera pairs"
        )
    for camera in parsed_profile:
        if camera["default_backend"] != "gscam":
            raise SetupConfigError(
                f"canonical setup camera {camera['id']} must default to gscam"
            )
        for key in ("image_topic", "frame_id"):
            if "{" in camera[key] or "}" in camera[key]:
                raise SetupConfigError(
                    f"camera {camera['id']} {key} must be concrete; setup files "
                    f"do not use templates: {camera[key]!r}"
                )
        if not camera["image_topic"].startswith("/"):
            raise SetupConfigError(f"camera {camera['id']} image_topic must be absolute")

    rig = _mapping(cameras_section.get("rig"), "cameras.rig")
    profile_ids = set(ids)
    rig_ids = set(rig)
    if rig_ids != profile_ids:
        missing = sorted(profile_ids - rig_ids)
        extra = sorted(rig_ids - profile_ids)
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise SetupConfigError(
            f"cameras.rig ids must match cameras.profile ids exactly: "
            + ", ".join(details)
        )
    for camera_id, raw in rig.items():
        _only_keys(_mapping(raw, f"cameras.rig {camera_id}"), _CAMERA_KEYS, f"cameras.rig {camera_id}")

    sim = data.get("sim")
    if sim is not None:
        sim = dict(_mapping(sim, "sim"))

    return Setup(
        name=setup_name,
        leaders=leaders,
        followers=followers,
        cameras=tuple(parsed_profile),
        rig=rig,
        sim=sim,
    )


def _validate_device(path: str, camera_id: str) -> None:
    try:
        mode = os.stat(path).st_mode
    except OSError as exc:
        raise SetupConfigError(f"{camera_id} device is unavailable: {path}: {exc.strerror}") from exc
    if not stat.S_ISCHR(mode):
        raise SetupConfigError(f"{camera_id} device is not a character device: {path}")
    if not os.access(path, os.R_OK | os.W_OK):
        raise SetupConfigError(f"{camera_id} device is not readable/writable: {path}")


def _default_gscam(device: str) -> str:
    return (
        f"v4l2src device={device} io-mode=2 do-timestamp=true ! "
        "image/jpeg,width=640,height=480,framerate=30/1 ! jpegdec ! videoconvert"
    )


def load_camera_setup(
    setup_name: str,
    setups_dir: str | os.PathLike[str] | None = None,
    *,
    path: str | os.PathLike[str] | None = None,
    validate_devices: bool = True,
) -> list[CameraSpec]:
    """Join a setup's logical camera contract with its physical rig."""
    setup = load_setup(setup_name, setups_dir, path=path)

    result: list[CameraSpec] = []
    for logical in setup.cameras:
        camera_id = logical["id"]
        raw = _mapping(setup.rig[camera_id], f"cameras.rig {camera_id}")
        device = _nonempty_string(raw.get("device"), f"{camera_id}.device")
        if validate_devices:
            _validate_device(device, camera_id)

        namespace = normalize_namespace(logical["namespace"])
        topic = logical["image_topic"]
        frame_id = logical["frame_id"].strip("/")

        backend = logical["default_backend"]
        driver_package = "gscam"
        driver_executable = "gscam_node"
        parameters: dict[str, Any] = {
            "use_sim_time": False,
            "camera_name": logical["node_name"],
            "frame_id": frame_id,
            "use_sensor_data_qos": True,
            "sync_sink": False,
            "use_gst_timestamps": True,
            "image_encoding": "rgb8",
            "gscam_config": raw.get("gscam_config") or _default_gscam(device),
        }
        remappings: tuple[tuple[str, str], ...] = (
            ("camera/image_raw", "image_raw"),
            ("camera/image_raw/compressed", "image_raw/compressed"),
        )
        if raw.get("driver") is not None:
            if raw.get("gscam_config") is not None:
                raise SetupConfigError(
                    f"{camera_id} cannot set both gscam_config and a custom driver"
                )
            driver = _mapping(raw["driver"], f"{camera_id}.driver")
            _only_keys(driver, _DRIVER_KEYS, f"{camera_id}.driver")
            driver_package = _nonempty_string(driver.get("package"), f"{camera_id}.driver.package")
            driver_executable = _nonempty_string(driver.get("executable"), f"{camera_id}.driver.executable")
            backend = f"custom:{driver_package}/{driver_executable}"
            parameters = dict(_mapping(driver.get("parameters", {}), f"{camera_id}.driver.parameters"))
            for key, value in (
                (driver.get("device_parameter", "video_device"), device),
                (driver.get("frame_id_parameter", "frame_id"), frame_id),
            ):
                if key:
                    parameters[str(key)] = value
            raw_remaps = driver.get("remappings", [])
            if not isinstance(raw_remaps, list) or not all(
                isinstance(item, list) and len(item) == 2 and all(isinstance(v, str) for v in item)
                for item in raw_remaps
            ):
                raise SetupConfigError(f"{camera_id}.driver.remappings must be pairs of strings")
            remappings = tuple((item[0], item[1]) for item in raw_remaps)

        result.append(CameraSpec(
            camera_id=camera_id,
            node_name=logical["node_name"],
            namespace=namespace,
            image_topic=topic,
            frame_id=frame_id,
            feature=logical["feature"],
            backend=backend,
            device=device,
            parameters=parameters,
            remappings=remappings,
            driver_package=driver_package,
            driver_executable=driver_executable,
        ))
    return result


def layout_transforms(setup: Setup) -> tuple[tuple[float, float, float, float, str], ...]:
    """Return (x, y, z, yaw_rad, child_frame) world placements for every arm."""
    return tuple(
        (
            arm.tf_xyz[0],
            arm.tf_xyz[1],
            arm.tf_xyz[2],
            math.radians(arm.tf_yaw_deg),
            f"{arm.namespace}/base_link",
        )
        for arm in (*setup.followers, *setup.leaders)
    )
