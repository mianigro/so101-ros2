"""Strict camera-profile and physical-rig loading."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat
from typing import Any

import yaml


CANONICAL_PROFILES = ("single_overhead", "dual_overhead")
_PROFILE_KEYS = {
    "id", "node_name", "namespace", "image_topic", "frame_id",
    "feature", "default_backend",
}
_CAMERA_KEYS = {"device", "gscam_config", "driver"}
_DRIVER_KEYS = {
    "package", "executable", "parameters", "remappings", "device_parameter",
    "frame_id_parameter",
}


class CameraConfigError(ValueError):
    """Raised when a camera profile or physical rig is invalid."""


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
    for key in ("node_name", "image_topic", "frame_id"):
        values = [camera[key] for camera in parsed]
        if len(values) != len(set(values)):
            raise CameraConfigError(f"{profile_name} contains duplicate {key} values")
    for camera in parsed:
        if camera["default_backend"] != "gscam":
            raise CameraConfigError(f"canonical profile camera {camera['id']} must default to gscam")
    return parsed


def _validate_device(path: str, camera_id: str) -> None:
    try:
        mode = os.stat(path).st_mode
    except OSError as exc:
        raise CameraConfigError(f"{camera_id} device is unavailable: {path}: {exc.strerror}") from exc
    if not stat.S_ISCHR(mode):
        raise CameraConfigError(f"{camera_id} device is not a character device: {path}")
    if not os.access(path, os.R_OK | os.W_OK):
        raise CameraConfigError(f"{camera_id} device is not readable/writable: {path}")


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

    result: list[CameraSpec] = []
    for logical in profile:
        camera_id = logical["id"]
        if camera_id not in physical:
            raise CameraConfigError(f"camera rig config is missing profile camera: {camera_id}")
        raw = _mapping(physical[camera_id], f"camera rig {camera_id}")
        _only_keys(raw, _CAMERA_KEYS, f"camera rig {camera_id}")
        device = _nonempty_string(raw.get("device"), f"{camera_id}.device")
        if validate_devices:
            _validate_device(device, camera_id)

        namespace = _render(logical["namespace"], follower_namespace, frame_prefix).strip("/")
        topic = _render(logical["image_topic"], follower_namespace, frame_prefix)
        frame_id = _render(logical["frame_id"], follower_namespace, frame_prefix).strip("/")
        if not topic.startswith("/"):
            raise CameraConfigError(f"camera {camera_id} profile image topic must be absolute")

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
