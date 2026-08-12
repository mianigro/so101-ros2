"""Portable visual-policy manifest construction and validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .camera_profile import (
    CAMERA_NAMES,
    POLICY_FREQUENCY_HZ,
    POLICY_IMAGE_HEIGHT,
    POLICY_IMAGE_WIDTH,
    camera_profile_sha256,
)

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
DELTA_SCALES_RAD = (0.05, 0.05, 0.05, 0.05, 0.05, 0.15)
VISION_TASK_IDS = {
    "SO101-Object-In-Cup-Vision-Fixed-v0",
    "SO101-Object-In-Cup-Vision-v0",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_policy_manifest(
    task_id: str,
    joint_limits_rad: list[list[float]],
    torchscript_path: Path,
    onnx_path: Path,
) -> dict:
    """Build the complete actor/deployment contract for an exported checkpoint."""
    if task_id not in VISION_TASK_IDS:
        raise ValueError(f"refusing to export a non-vision task: {task_id}")
    if len(joint_limits_rad) != len(JOINT_NAMES) or any(
        len(bounds) != 2 or bounds[0] >= bounds[1] for bounds in joint_limits_rad
    ):
        raise ValueError("joint limits must contain six ordered [lower, upper] pairs")
    camera_inputs = [
        {
            "key": f"observation.images.{name}",
            "name": name,
            "shape_chw": [3, POLICY_IMAGE_HEIGHT, POLICY_IMAGE_WIDTH],
        }
        for name in CAMERA_NAMES
    ]
    model_checksum = sha256_file(torchscript_path)
    return {
        "schema_version": 1,
        "task_id": task_id,
        "camera_profile": "dual_overhead",
        "camera_calibration_sha256": camera_profile_sha256(),
        "actor_observations": {
            "cameras": camera_inputs,
            "camera_order": list(CAMERA_NAMES),
            "image_preprocessing": {
                "source_shape_hwc": [480, 640, 3],
                "resize": "bilinear",
                "output_shape_chw": [3, POLICY_IMAGE_HEIGHT, POLICY_IMAGE_WIDTH],
                "channel_order": "RGB",
                "normalization": "RGB / 255 - 0.5",
            },
            "joint_state": {
                "key": "observation.state",
                "names": list(JOINT_NAMES),
                "shape": [len(JOINT_NAMES)],
                "units": "rad",
                "representation": "absolute_position",
            },
        },
        "policy": {
            "frequency_hz": POLICY_FREQUENCY_HZ,
            "action_order": list(JOINT_NAMES),
            "action_representation": "normalized_joint_position_delta",
            "normalized_action_clip": [-1.0, 1.0],
            "delta_scales_rad": list(DELTA_SCALES_RAD),
            "joint_limits_rad": joint_limits_rad,
            "joint_limit_safety_margin": 0.98,
        },
        "artifacts": {
            "torchscript": torchscript_path.name,
            "torchscript_sha256": model_checksum,
            "onnx": onnx_path.name,
            "onnx_sha256": sha256_file(onnx_path),
        },
        "model_checksum": model_checksum,
    }


def write_policy_manifest(manifest: dict, output_path: Path) -> None:
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
