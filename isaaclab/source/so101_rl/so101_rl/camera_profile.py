"""Validated nominal camera contract shared by visual training and export."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import yaml

from .paths import CAMERA_PROFILE_PATH

CAMERA_NAMES = ("wrist", "overhead_1", "overhead_2")
POLICY_IMAGE_WIDTH = 160
POLICY_IMAGE_HEIGHT = 120
POLICY_FREQUENCY_HZ = 30.0


def _vector(mapping: dict[str, Any], key: str, length: int) -> tuple[float, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"camera profile field {key!r} must contain {length} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"camera profile field {key!r} contains a non-finite value")
    return result


def load_camera_profile(path: Path = CAMERA_PROFILE_PATH) -> dict[str, Any]:
    """Load the checked-in dual-overhead profile and enforce the deployment schema."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"camera profile does not exist: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("camera profile must be a schema_version 1 mapping")
    cameras = data.get("cameras")
    if not isinstance(cameras, dict) or tuple(cameras) != CAMERA_NAMES:
        raise ValueError(f"camera profile must define cameras in order {CAMERA_NAMES}")
    resolution = data.get("resolution")
    if resolution != {"width": 640, "height": 480}:
        raise ValueError("dual_overhead source resolution must remain 640x480")
    if not math.isclose(float(data.get("fps", 0.0)), 30.0):
        raise ValueError("dual_overhead cameras must remain 30 Hz")

    wrist = cameras["wrist"]
    _vector(wrist, "translation_m", 3)
    quaternion = _vector(wrist, "orientation_wxyz", 4)
    if not math.isclose(sum(value * value for value in quaternion), 1.0, abs_tol=2e-5):
        raise ValueError("wrist orientation_wxyz must be a unit quaternion")
    if wrist.get("parent_link") != "gripper_link":
        raise ValueError("wrist camera must remain attached to gripper_link")
    for name in CAMERA_NAMES[1:]:
        camera = cameras[name]
        if _vector(camera, "position_m", 3) == _vector(camera, "look_at_m", 3):
            raise ValueError(f"{name} eye and look-at positions must differ")
    for name in CAMERA_NAMES:
        fov = float(cameras[name].get("horizontal_fov_deg", 0.0))
        if not 0.0 < fov < 179.0:
            raise ValueError(f"{name} horizontal FOV is invalid: {fov}")
    data["_path"] = path
    return data


def camera_profile_sha256(path: Path = CAMERA_PROFILE_PATH) -> str:
    """Return the byte-exact nominal camera calibration hash."""
    return hashlib.sha256(path.expanduser().resolve().read_bytes()).hexdigest()


def wxyz_to_xyzw(quaternion: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    """Convert the teleop YAML quaternion convention to Isaac Lab's XYZW order."""
    if len(quaternion) != 4:
        raise ValueError("a quaternion must contain four values")
    w, x, y, z = (float(value) for value in quaternion)
    return (x, y, z, w)


def _matrix_to_xyzw(matrix: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
    """Convert a right-handed 3x3 rotation matrix to a normalized XYZW quaternion."""
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        w = 0.25 * scale
        x = (matrix[2][1] - matrix[1][2]) / scale
        y = (matrix[0][2] - matrix[2][0]) / scale
        z = (matrix[1][0] - matrix[0][1]) / scale
    else:
        axis = max(range(3), key=lambda index: matrix[index][index])
        if axis == 0:
            scale = 2.0 * math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2])
            w = (matrix[2][1] - matrix[1][2]) / scale
            x = 0.25 * scale
            y = (matrix[0][1] + matrix[1][0]) / scale
            z = (matrix[0][2] + matrix[2][0]) / scale
        elif axis == 1:
            scale = 2.0 * math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2])
            w = (matrix[0][2] - matrix[2][0]) / scale
            x = (matrix[0][1] + matrix[1][0]) / scale
            y = 0.25 * scale
            z = (matrix[1][2] + matrix[2][1]) / scale
        else:
            scale = 2.0 * math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1])
            w = (matrix[1][0] - matrix[0][1]) / scale
            x = (matrix[0][2] + matrix[2][0]) / scale
            y = (matrix[1][2] + matrix[2][1]) / scale
            z = 0.25 * scale
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    return (x / norm, y / norm, z / norm, w / norm)


def look_at_opengl_xyzw(
    eye: tuple[float, ...] | list[float],
    target: tuple[float, ...] | list[float],
) -> tuple[float, ...]:
    """Match teleop's Z-up OpenGL look-at quaternion, returned in XYZW order."""
    backward = [float(eye[i]) - float(target[i]) for i in range(3)]
    length = math.sqrt(sum(value * value for value in backward))
    if length <= 1e-12:
        raise ValueError("look-at eye and target must differ")
    z_axis = [value / length for value in backward]
    up = (0.0, 0.0, 1.0)
    x_axis = [
        up[1] * z_axis[2] - up[2] * z_axis[1],
        up[2] * z_axis[0] - up[0] * z_axis[2],
        up[0] * z_axis[1] - up[1] * z_axis[0],
    ]
    x_length = math.sqrt(sum(value * value for value in x_axis))
    if x_length <= 1e-12:
        up = (0.0, 1.0, 0.0)
        x_axis = [
            up[1] * z_axis[2] - up[2] * z_axis[1],
            up[2] * z_axis[0] - up[0] * z_axis[2],
            up[0] * z_axis[1] - up[1] * z_axis[0],
        ]
        x_length = math.sqrt(sum(value * value for value in x_axis))
    x_axis = [value / x_length for value in x_axis]
    y_axis = [
        z_axis[1] * x_axis[2] - z_axis[2] * x_axis[1],
        z_axis[2] * x_axis[0] - z_axis[0] * x_axis[2],
        z_axis[0] * x_axis[1] - z_axis[1] * x_axis[0],
    ]
    rotation = tuple(
        tuple((x_axis, y_axis, z_axis)[column][row] for column in range(3))
        for row in range(3)
    )
    return _matrix_to_xyzw(rotation)


def focal_length_mm(horizontal_fov_deg: float, horizontal_aperture_mm: float) -> float:
    """Compute the physical focal length used by sim teleop."""
    return horizontal_aperture_mm / (
        2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0)
    )
