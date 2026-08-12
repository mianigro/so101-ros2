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
from .visual_contract import (
    SO101_ACTION_DELTA_SCALES_RAD,
    SO101_ACTOR_OBSERVATION_GROUPS,
    SO101_ARM_ACTION_PATTERN,
    SO101_ARM_DELTA_RAD,
    SO101_GRIPPER_DELTA_RAD,
    SO101_JOINT_NAMES,
    SO101_NORMALIZED_ACTION_CLIP,
)


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
    if not task_id:
        raise ValueError("task ID must not be empty")
    if len(joint_limits_rad) != len(SO101_JOINT_NAMES) or any(
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
                "names": list(SO101_JOINT_NAMES),
                "shape": [len(SO101_JOINT_NAMES)],
                "units": "rad",
                "representation": "absolute_position",
            },
        },
        "policy": {
            "frequency_hz": POLICY_FREQUENCY_HZ,
            "action_order": list(SO101_JOINT_NAMES),
            "action_representation": "normalized_joint_position_delta",
            "normalized_action_clip": list(SO101_NORMALIZED_ACTION_CLIP),
            "delta_scales_rad": list(SO101_ACTION_DELTA_SCALES_RAD),
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


def validate_deployable_contract(task_id: str, env_cfg, agent_cfg) -> None:
    """Reject a task whose actor cannot use the real SO-101 deployment interface."""
    errors: list[str] = []
    if not task_id:
        errors.append("task ID is empty")

    actor_groups = tuple(agent_cfg.obs_groups.get("actor", ()))
    if actor_groups != SO101_ACTOR_OBSERVATION_GROUPS:
        errors.append(
            f"actor groups are {actor_groups}, expected {SO101_ACTOR_OBSERVATION_GROUPS}"
        )

    action = env_cfg.actions.joint_delta
    if tuple(action.joint_names) != SO101_JOINT_NAMES or not action.preserve_order:
        errors.append("joint action order does not match the six-joint SO-101 contract")
    expected_scale = {
        SO101_ARM_ACTION_PATTERN: SO101_ARM_DELTA_RAD,
        "gripper": SO101_GRIPPER_DELTA_RAD,
    }
    expected_clip = {
        SO101_ARM_ACTION_PATTERN: (-SO101_ARM_DELTA_RAD, SO101_ARM_DELTA_RAD),
        "gripper": (-SO101_GRIPPER_DELTA_RAD, SO101_GRIPPER_DELTA_RAD),
    }
    if action.scale != expected_scale:
        errors.append(f"action scale is {action.scale}, expected {expected_scale}")
    if action.clip != expected_clip:
        errors.append(f"action clip is {action.clip}, expected {expected_clip}")
    if not action.use_zero_offset:
        errors.append("joint actions must use the current position as their zero offset")
    if float(agent_cfg.clip_actions) != SO101_NORMALIZED_ACTION_CLIP[1]:
        errors.append("runner normalized-action clipping must remain [-1, 1]")

    policy_period = float(env_cfg.sim.dt) * int(env_cfg.decimation)
    expected_period = 1.0 / POLICY_FREQUENCY_HZ
    if abs(policy_period - expected_period) > 1.0e-9:
        errors.append(
            f"policy period is {policy_period:g}s, expected {expected_period:g}s"
        )

    joint_term = env_cfg.observations.joint_state.absolute_joint_positions
    joint_cfg = joint_term.params["asset_cfg"]
    if tuple(joint_cfg.joint_names) != SO101_JOINT_NAMES or not joint_cfg.preserve_order:
        errors.append("joint observation order does not match the action order")

    for camera_name in CAMERA_NAMES:
        group = getattr(env_cfg.observations, camera_name)
        sensor_name = group.rgb.params["sensor_cfg"].name
        if sensor_name != f"{camera_name}_camera":
            errors.append(
                f"{camera_name} actor group reads {sensor_name}, expected {camera_name}_camera"
            )
        camera = getattr(env_cfg.scene, f"{camera_name}_camera")
        if (camera.width, camera.height) != (
            POLICY_IMAGE_WIDTH,
            POLICY_IMAGE_HEIGHT,
        ):
            errors.append(
                f"{camera_name} size is {(camera.width, camera.height)}, expected "
                f"{(POLICY_IMAGE_WIDTH, POLICY_IMAGE_HEIGHT)}"
            )
        if abs(float(camera.update_period) - expected_period) > 1.0e-9:
            errors.append(
                f"{camera_name} period is {camera.update_period:g}s, "
                f"expected {expected_period:g}s"
            )

    if errors:
        details = "\n  - ".join(errors)
        raise ValueError(
            f"task {task_id!r} violates the deployable SO-101 contract:\n  - {details}"
        )


def write_policy_manifest(manifest: dict, output_path: Path) -> None:
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
