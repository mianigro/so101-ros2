"""Strict camera-profile and physical-rig loading."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import stat
from typing import Any
from urllib.parse import unquote, urlparse

import yaml


CANONICAL_PROFILES = ("single_overhead", "dual_overhead")
_PROFILE_KEYS = {
    "id", "node_name", "namespace", "image_topic", "camera_info_topic", "frame_id",
    "feature", "default_backend", "extrinsics"
}
_CAMERA_KEYS = {"device", "camera_info_url", "gscam_config", "driver", "transform"}
_DRIVER_KEYS = {
    "package", "executable", "parameters", "remappings", "device_parameter",
    "frame_id_parameter", "camera_info_url_parameter",
}
_TRANSFORM_KEYS = {"parent_frame", "translation", "rotation_xyzw"}


class CameraConfigError(ValueError):
    """Raised when a camera profile or physical rig is invalid."""


def intrinsics_are_valid(width: int, height: int, matrix: Any) -> bool:
    """Return whether dimensions and a 3x3 pinhole matrix are usable."""
    return (
        width > 0
        and height > 0
        and len(matrix) == 9
        and matrix[0] > 0.0
        and matrix[4] > 0.0
    )


def evaluate_camera_streams(
    last_image: dict[str, float | None],
    last_info: dict[str, float | None],
    now: float,
    stale_timeout_s: float,
    require_camera_info: bool,
) -> tuple[list[str], list[str]]:
    """Return missing and stale stream labels for the supervisor."""
    missing = [f"{name}:image" for name, received in last_image.items() if received is None]
    if require_camera_info:
        missing.extend(
            f"{name}:camera_info" for name, received in last_info.items() if received is None
        )
    if missing:
        return missing, []
    stale = [
        f"{name}:image" for name, received in last_image.items()
        if now - received > stale_timeout_s
    ]
    if require_camera_info:
        stale.extend(
            f"{name}:camera_info" for name, received in last_info.items()
            if now - received > stale_timeout_s
        )
    return [], stale


@dataclass(frozen=True)
class CameraSpec:
    camera_id: str
    node_name: str
    namespace: str
    image_topic: str
    camera_info_topic: str
    frame_id: str
    feature: str
    backend: str
    device: str
    camera_info_url: str
    parameters: dict[str, Any]
    remappings: tuple[tuple[str, str], ...]
    driver_package: str
    driver_executable: str
    transform: dict[str, Any] | None


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CameraConfigError(f"{where} must be a mapping")
    return value


def _only_keys(data: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise CameraConfigError(f"{where} has unknown keys: {', '.join(unknown)}")


def _nonempty_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CameraConfigError(f"{where} must be a non-empty string")
    return value.strip()


def _load_yaml(path: Path, where: str) -> dict[str, Any]:
    if not path.is_file():
        raise CameraConfigError(f"{where} does not exist: {path}")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), where)
    except yaml.YAMLError as exc:
        raise CameraConfigError(f"{where} is not valid YAML: {exc}") from exc


def normalize_namespace(value: str) -> str:
    value = value.strip().strip("/")
    if not value or "//" in value:
        raise CameraConfigError("follower_namespace must name one ROS namespace")
    return value


def normalize_frame_prefix(value: str) -> str:
    value = value.strip().strip("/")
    return f"{value}/" if value else ""


def _render(value: str, follower_namespace: str, frame_prefix: str) -> str:
    try:
        return value.format(
            follower_namespace=follower_namespace,
            frame_prefix=frame_prefix,
        )
    except (KeyError, ValueError) as exc:
        raise CameraConfigError(f"invalid profile template {value!r}: {exc}") from exc


def load_profile(profile_name: str, profiles_dir: str | os.PathLike[str]) -> list[dict[str, str]]:
    if profile_name not in CANONICAL_PROFILES:
        allowed = ", ".join(CANONICAL_PROFILES)
        raise CameraConfigError(f"camera_profile must be one of: {allowed}")
    path = Path(profiles_dir) / f"{profile_name}.yaml"
    data = _load_yaml(path, "camera profile")
    _only_keys(data, {"profile", "cameras"}, "camera profile")
    if data.get("profile") != profile_name:
        raise CameraConfigError(f"camera profile file must declare profile: {profile_name}")
    cameras = data.get("cameras")
    if not isinstance(cameras, list):
        raise CameraConfigError("camera profile cameras must be a list")
    parsed: list[dict[str, str]] = []
    for index, raw in enumerate(cameras):
        cam = _mapping(raw, f"camera profile cameras[{index}]")
        _only_keys(cam, _PROFILE_KEYS, f"camera profile cameras[{index}]")
        parsed.append({key: _nonempty_string(cam.get(key), f"camera profile {key}") for key in _PROFILE_KEYS})
    ids = [camera["id"] for camera in parsed]
    if len(ids) != len(set(ids)):
        raise CameraConfigError(f"{profile_name} contains duplicate camera ids")
    features = [camera["feature"] for camera in parsed]
    if len(features) != len(set(features)):
        raise CameraConfigError(f"{profile_name} contains duplicate feature names")
    for key in ("node_name", "image_topic", "camera_info_topic", "frame_id"):
        values = [camera[key] for camera in parsed]
        if len(values) != len(set(values)):
            raise CameraConfigError(f"{profile_name} contains duplicate {key} values")
    for camera in parsed:
        if camera["default_backend"] != "gscam":
            raise CameraConfigError(f"canonical profile camera {camera['id']} must default to gscam")
        if camera["extrinsics"] not in {"robot_description", "external"}:
            raise CameraConfigError(f"camera {camera['id']} has invalid extrinsics source")
    return parsed


def _resolve_calibration_url(url: str) -> Path:
    parsed = urlparse(url)
    if parsed.scheme != "file":
        raise CameraConfigError("camera_info_url must be an absolute file:// URL")
    path = Path(unquote(parsed.path))
    if not path.is_absolute():
        raise CameraConfigError("camera_info_url must contain an absolute path")
    return path


def validate_calibration(url: str, camera_id: str) -> None:
    path = _resolve_calibration_url(_nonempty_string(url, f"{camera_id}.camera_info_url"))
    data = _load_yaml(path, f"{camera_id} calibration")
    required = {
        "image_width", "image_height", "camera_name", "camera_matrix",
        "distortion_model", "distortion_coefficients", "rectification_matrix",
        "projection_matrix",
    }
    missing = sorted(required - set(data))
    if missing:
        raise CameraConfigError(f"{camera_id} calibration missing keys: {', '.join(missing)}")
    if not isinstance(data["image_width"], int) or data["image_width"] <= 0:
        raise CameraConfigError(f"{camera_id} calibration image_width must be positive")
    if not isinstance(data["image_height"], int) or data["image_height"] <= 0:
        raise CameraConfigError(f"{camera_id} calibration image_height must be positive")
    _nonempty_string(data["camera_name"], f"{camera_id}.camera_name")
    _nonempty_string(data["distortion_model"], f"{camera_id}.distortion_model")
    matrix = _mapping(data["camera_matrix"], f"{camera_id}.camera_matrix")
    if matrix.get("rows") != 3 or matrix.get("cols") != 3:
        raise CameraConfigError(f"{camera_id} camera_matrix must be 3x3")
    values = matrix.get("data")
    if not isinstance(values, list) or len(values) != 9:
        raise CameraConfigError(f"{camera_id} camera_matrix.data must have 9 values")
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        raise CameraConfigError(f"{camera_id} camera_matrix contains invalid values")
    if not intrinsics_are_valid(data["image_width"], data["image_height"], values):
        raise CameraConfigError(f"{camera_id} calibration focal lengths must be positive")
    for key, rows, cols, length in (
        ("distortion_coefficients", 1, None, None),
        ("rectification_matrix", 3, 3, 9),
        ("projection_matrix", 3, 4, 12),
    ):
        item = _mapping(data[key], f"{camera_id}.{key}")
        item_values = item.get("data")
        if item.get("rows") != rows or (cols is not None and item.get("cols") != cols):
            raise CameraConfigError(f"{camera_id} {key} has invalid dimensions")
        if not isinstance(item_values, list) or (length is not None and len(item_values) != length):
            raise CameraConfigError(f"{camera_id} {key}.data has invalid length")
        if not item_values or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in item_values):
            raise CameraConfigError(f"{camera_id} {key} contains invalid values")


def _validate_device(path: str, camera_id: str) -> None:
    try:
        mode = os.stat(path).st_mode
    except OSError as exc:
        raise CameraConfigError(f"{camera_id} device is unavailable: {path}: {exc.strerror}") from exc
    if not stat.S_ISCHR(mode):
        raise CameraConfigError(f"{camera_id} device is not a character device: {path}")
    if not os.access(path, os.R_OK | os.W_OK):
        raise CameraConfigError(f"{camera_id} device is not readable/writable: {path}")


def _transform(raw: Any, camera_id: str, frame_prefix: str) -> dict[str, Any]:
    data = _mapping(raw, f"{camera_id}.transform")
    _only_keys(data, _TRANSFORM_KEYS, f"{camera_id}.transform")
    parent = _nonempty_string(data.get("parent_frame"), f"{camera_id}.transform.parent_frame")
    if parent != "base_link":
        raise CameraConfigError(
            f"{camera_id}.transform.parent_frame must be base_link"
        )
    translation = data.get("translation")
    rotation = data.get("rotation_xyzw")
    if not isinstance(translation, list) or len(translation) != 3:
        raise CameraConfigError(f"{camera_id}.transform.translation must have 3 numbers")
    if not isinstance(rotation, list) or len(rotation) != 4:
        raise CameraConfigError(f"{camera_id}.transform.rotation_xyzw must have 4 numbers")
    numbers = translation + rotation
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in numbers):
        raise CameraConfigError(f"{camera_id}.transform values must be finite numbers")
    norm = math.sqrt(sum(float(v) ** 2 for v in rotation))
    if abs(norm - 1.0) > 1e-3:
        raise CameraConfigError(f"{camera_id}.transform quaternion must be normalized")
    return {
        "parent_frame": f"{frame_prefix}{parent}",
        "translation": [float(v) for v in translation],
        "rotation_xyzw": [float(v) for v in rotation],
    }


def _default_gscam(device: str) -> str:
    return (
        f"v4l2src device={device} io-mode=2 do-timestamp=true ! "
        "image/jpeg,width=640,height=480,framerate=30/1 ! jpegdec ! videoconvert"
    )


def load_camera_setup(
    profile_name: str,
    profiles_dir: str | os.PathLike[str],
    rig_path: str | os.PathLike[str],
    follower_namespace: str,
    frame_prefix: str,
    *,
    calibration_mode: bool = False,
    calibration_camera_id: str = "",
    validate_devices: bool = True,
) -> list[CameraSpec]:
    profile = load_profile(profile_name, profiles_dir)
    rig_file = Path(rig_path)
    if not str(rig_path).strip():
        raise CameraConfigError("camera_rig_config_file is required when cameras are enabled")
    if not rig_file.is_absolute():
        raise CameraConfigError("camera_rig_config_file must be an absolute external path")
    rig = _load_yaml(rig_file, "camera rig config")
    _only_keys(rig, {"schema_version", "cameras"}, "camera rig config")
    if rig.get("schema_version") != 1:
        raise CameraConfigError("camera rig config schema_version must be 1")
    physical = _mapping(rig.get("cameras"), "camera rig config cameras")
    known_ids = {
        camera["id"]
        for canonical_name in CANONICAL_PROFILES
        for camera in load_profile(canonical_name, profiles_dir)
    }
    unknown_ids = sorted(set(physical) - known_ids)
    if unknown_ids:
        raise CameraConfigError(f"camera rig config has unknown camera ids: {', '.join(unknown_ids)}")

    follower_namespace = normalize_namespace(follower_namespace)
    frame_prefix = normalize_frame_prefix(frame_prefix)
    selected = profile
    if calibration_mode and calibration_camera_id:
        selected = [cam for cam in profile if cam["id"] == calibration_camera_id]
        if not selected:
            raise CameraConfigError(
                f"calibration_camera_id {calibration_camera_id!r} is not in {profile_name}"
            )

    result: list[CameraSpec] = []
    for logical in selected:
        camera_id = logical["id"]
        if camera_id not in physical:
            raise CameraConfigError(f"camera rig config is missing profile camera: {camera_id}")
        raw = _mapping(physical[camera_id], f"camera rig {camera_id}")
        _only_keys(raw, _CAMERA_KEYS, f"camera rig {camera_id}")
        device = _nonempty_string(raw.get("device"), f"{camera_id}.device")
        if validate_devices:
            _validate_device(device, camera_id)
        camera_info_url = raw.get("camera_info_url", "")
        if not calibration_mode:
            validate_calibration(camera_info_url, camera_id)
        elif camera_info_url:
            validate_calibration(camera_info_url, camera_id)

        namespace = _render(logical["namespace"], follower_namespace, frame_prefix).strip("/")
        topic = _render(logical["image_topic"], follower_namespace, frame_prefix)
        camera_info_topic = _render(logical["camera_info_topic"], follower_namespace, frame_prefix)
        frame_id = _render(logical["frame_id"], follower_namespace, frame_prefix).strip("/")
        if not topic.startswith("/") or not camera_info_topic.startswith("/"):
            raise CameraConfigError(f"camera {camera_id} profile topics must be absolute")
        transform = None
        if logical["extrinsics"] == "external" and not calibration_mode:
            transform = _transform(raw.get("transform"), camera_id, frame_prefix)
        elif logical["extrinsics"] == "external" and raw.get("transform") is not None:
            transform = _transform(raw["transform"], camera_id, frame_prefix)

        backend = logical["default_backend"]
        driver_package = "gscam"
        driver_executable = "gscam_node"
        parameters: dict[str, Any] = {
            "use_sim_time": False,
            "camera_name": logical["node_name"],
            "frame_id": frame_id,
            "camera_info_url": camera_info_url,
            "use_sensor_data_qos": True,
            "sync_sink": False,
            "use_gst_timestamps": True,
            "image_encoding": "rgb8",
            "gscam_config": raw.get("gscam_config") or _default_gscam(device),
        }
        remappings: tuple[tuple[str, str], ...] = (
            ("camera/image_raw", "image_raw"),
            ("camera/camera_info", "camera_info"),
            ("camera/image_raw/compressed", "image_raw/compressed"),
        )
        if raw.get("driver") is not None:
            if raw.get("gscam_config") is not None:
                raise CameraConfigError(
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
                (driver.get("camera_info_url_parameter", "camera_info_url"), camera_info_url),
            ):
                if key:
                    parameters[str(key)] = value
            raw_remaps = driver.get("remappings", [])
            if not isinstance(raw_remaps, list) or not all(
                isinstance(item, list) and len(item) == 2 and all(isinstance(v, str) for v in item)
                for item in raw_remaps
            ):
                raise CameraConfigError(f"{camera_id}.driver.remappings must be pairs of strings")
            remappings = tuple((item[0], item[1]) for item in raw_remaps)

        result.append(CameraSpec(
            camera_id=camera_id,
            node_name=logical["node_name"],
            namespace=namespace,
            image_topic=topic,
            camera_info_topic=camera_info_topic,
            frame_id=frame_id,
            feature=logical["feature"],
            backend=backend,
            device=device,
            camera_info_url=camera_info_url,
            parameters=parameters,
            remappings=remappings,
            driver_package=driver_package,
            driver_executable=driver_executable,
            transform=transform,
        ))
    return result
