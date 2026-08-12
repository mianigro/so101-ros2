"""Pure preprocessing and safety contract for exported RSL-RL actors."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as functional

EXPECTED_CAMERAS = ("wrist", "overhead_1", "overhead_2")
EXPECTED_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
EXPECTED_IMAGE_SHAPE = (3, 120, 160)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_policy_manifest(model_dir: Path, camera_profile: str) -> dict:
    """Load and strictly validate the exported actor/deployment contract."""
    model_dir = model_dir.expanduser().resolve()
    manifest_path = model_dir / "policy_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"policy manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported policy manifest schema")
    if manifest.get("camera_profile") != camera_profile or camera_profile != "dual_overhead":
        raise ValueError("RSL-RL visual policy requires camera_profile='dual_overhead'")

    observations = manifest.get("actor_observations", {})
    if observations.get("camera_order") != list(EXPECTED_CAMERAS):
        raise ValueError("manifest camera order does not match the real SO-101 rig")
    cameras = observations.get("cameras", [])
    if [camera.get("name") for camera in cameras] != list(EXPECTED_CAMERAS):
        raise ValueError("manifest camera inputs are incomplete or reordered")
    if any(tuple(camera.get("shape_chw", ())) != EXPECTED_IMAGE_SHAPE for camera in cameras):
        raise ValueError("manifest camera shape must be [3, 120, 160]")
    preprocessing = observations.get("image_preprocessing", {})
    if preprocessing.get("normalization") != "RGB / 255 - 0.5":
        raise ValueError("manifest image normalization is unsupported")
    if preprocessing.get("resize") != "bilinear":
        raise ValueError("manifest image resize must be bilinear")
    joint_state = observations.get("joint_state", {})
    if joint_state.get("key") != "observation.state":
        raise ValueError("manifest joint input key is invalid")
    if joint_state.get("names") != list(EXPECTED_JOINTS):
        raise ValueError("manifest joint order does not match the controller")

    policy = manifest.get("policy", {})
    if not math.isclose(float(policy.get("frequency_hz", 0.0)), 20.0):
        raise ValueError("manifest policy frequency must be 20 Hz")
    if policy.get("action_order") != list(EXPECTED_JOINTS):
        raise ValueError("manifest action order does not match the controller")
    scales = policy.get("delta_scales_rad", [])
    limits = policy.get("joint_limits_rad", [])
    if len(scales) != 6 or len(limits) != 6 or any(len(item) != 2 for item in limits):
        raise ValueError("manifest action scales or joint limits are malformed")

    artifacts = manifest.get("artifacts", {})
    policy_path = model_dir / str(artifacts.get("torchscript", ""))
    if not policy_path.is_file():
        raise FileNotFoundError(f"TorchScript policy does not exist: {policy_path}")
    checksum = _sha256(policy_path)
    if checksum != artifacts.get("torchscript_sha256") or checksum != manifest.get(
        "model_checksum"
    ):
        raise ValueError("TorchScript policy checksum does not match its manifest")
    manifest["_model_dir"] = str(model_dir)
    manifest["_policy_path"] = str(policy_path)
    return manifest


def preprocess_rgb(image: np.ndarray, device: torch.device) -> torch.Tensor:
    """Resize one 640x480 HWC uint8 RGB image into the actor's normalized CHW batch."""
    if image.dtype != np.uint8 or image.shape != (480, 640, 3):
        raise ValueError(
            f"expected a 640x480 RGB uint8 frame, got shape={image.shape}, dtype={image.dtype}"
        )
    tensor = torch.from_numpy(np.ascontiguousarray(image)).to(
        device=device, dtype=torch.float32
    )
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    tensor = functional.interpolate(
        tensor, size=(120, 160), mode="bilinear", align_corners=False
    )
    return tensor / 255.0 - 0.5


def timestamps_within_skew(
    timestamp_ns: Mapping[str, int | None], maximum_skew_s: float
) -> bool:
    """Require every source timestamp and enforce maximum newest-oldest skew."""
    if not timestamp_ns or any(value is None or value <= 0 for value in timestamp_ns.values()):
        return False
    values = [int(value) for value in timestamp_ns.values() if value is not None]
    return (max(values) - min(values)) * 1e-9 <= maximum_skew_s


def safe_absolute_targets(
    normalized_action: np.ndarray,
    measured_positions: np.ndarray,
    delta_scales_rad: np.ndarray,
    joint_limits_rad: np.ndarray,
    safety_margin: float,
) -> np.ndarray:
    """Convert normalized deltas to finite absolute targets inside margin-adjusted limits."""
    action = np.asarray(normalized_action, dtype=np.float64).reshape(-1)
    measured = np.asarray(measured_positions, dtype=np.float64).reshape(-1)
    scales = np.asarray(delta_scales_rad, dtype=np.float64).reshape(-1)
    limits = np.asarray(joint_limits_rad, dtype=np.float64)
    if action.shape != (6,) or measured.shape != (6,) or scales.shape != (6,):
        raise ValueError("SO-101 action, state, and scale vectors must each contain six values")
    if limits.shape != (6, 2):
        raise ValueError("SO-101 joint limits must have shape [6, 2]")
    if not 0.0 < safety_margin <= 1.0:
        raise ValueError("joint-limit safety margin must be in (0, 1]")
    if not all(
        np.all(np.isfinite(value)) for value in (action, measured, scales, limits)
    ):
        raise ValueError("policy action contract contains NaN or Inf")
    if np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("joint lower limits must be less than upper limits")

    normalized = np.clip(action, -1.0, 1.0)
    delta = np.clip(normalized * scales, -scales, scales)
    targets = measured + delta
    midpoint = limits.mean(axis=1)
    half_range = (limits[:, 1] - limits[:, 0]) * 0.5 * safety_margin
    return np.clip(targets, midpoint - half_range, midpoint + half_range)
