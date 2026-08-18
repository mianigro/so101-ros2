"""Tests for deployment preprocessing, timestamp gating, and action safety."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from so101_inference.rsl_rl_policy import (
    EXPECTED_CAMERAS,
    EXPECTED_JOINTS,
    REQUIRED_SETUP,
    load_policy_manifest,
    preprocess_rgb,
    safe_absolute_targets,
    timestamps_within_skew,
)


def _write_manifest(
    directory: Path, *, schema_version: int = 2, frequency_hz: float = 30.0
):
    policy_path = directory / "policy.pt"
    policy_path.write_bytes(b"policy")
    checksum = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": schema_version,
        "setup": REQUIRED_SETUP,
        "actor_observations": {
            "camera_order": list(EXPECTED_CAMERAS),
            "cameras": [
                {"name": name, "shape_chw": [3, 120, 160]}
                for name in EXPECTED_CAMERAS
            ],
            "image_preprocessing": {
                "normalization": "RGB / 255 - 0.5",
                "resize": "bilinear",
            },
            "joint_state": {
                "key": "observation.state",
                "names": list(EXPECTED_JOINTS),
            },
        },
        "policy": {
            "frequency_hz": frequency_hz,
            "action_order": list(EXPECTED_JOINTS),
            "action_representation": "normalized_joint_position_delta",
            "normalized_action_clip": [-1.0, 1.0],
            "delta_scales_rad": [1.0 / 30.0] * 5 + [0.10],
            "joint_limits_rad": [[-2.0, 2.0]] * 6,
            "joint_limit_safety_margin": 0.98,
        },
        "deployment": {
            "controller_command": {
                "topic": "/follower/forward_controller/commands",
                "message_type": "std_msgs/msg/Float64MultiArray",
                "representation": "absolute_joint_position",
                "units": "rad",
                "names": list(EXPECTED_JOINTS),
            }
        },
        "artifacts": {"torchscript": policy_path.name, "torchscript_sha256": checksum},
        "model_checksum": checksum,
    }
    (directory / "policy_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class RslRlPolicyTests(unittest.TestCase):
    def test_real_rgb_preprocessing_matches_training_contract(self):
        white = np.full((480, 640, 3), 255, dtype=np.uint8)
        processed = preprocess_rgb(white, torch.device("cpu"))
        self.assertEqual(tuple(processed.shape), (1, 3, 120, 160))
        self.assertTrue(torch.allclose(processed, torch.full_like(processed, 0.5)))

    def test_action_clipping_and_absolute_target_validation(self):
        action = np.array([2.0, -2.0, 0.5, 0.0, 1.0, -1.0])
        measured = np.array([0.90, -0.90, 0.0, 0.0, 0.0, 0.0])
        scales = np.array([1.0 / 30.0] * 5 + [0.10])
        limits = np.array([[-1.0, 1.0]] * 6)
        targets = safe_absolute_targets(action, measured, scales, limits, 0.98)
        self.assertAlmostEqual(targets[0], 0.90 + 1.0 / 30.0)
        self.assertAlmostEqual(targets[1], -0.90 - 1.0 / 30.0)
        self.assertAlmostEqual(targets[2], 1.0 / 60.0)
        with self.assertRaises(ValueError):
            safe_absolute_targets(
                np.array([np.nan] * 6), measured, scales, limits, 0.98
            )
        with self.assertRaisesRegex(ValueError, "absolute target exceeds"):
            safe_absolute_targets(
                np.ones(6),
                np.array([0.97, 0.0, 0.0, 0.0, 0.0, 0.0]),
                scales,
                limits,
                0.98,
            )

    def test_manifest_requires_schema_two_and_30hz_absolute_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _write_manifest(directory)
            loaded = load_policy_manifest(directory, REQUIRED_SETUP)
            self.assertEqual(loaded["schema_version"], 2)

            for frequency_hz in (20.0, 50.0):
                _write_manifest(directory, frequency_hz=frequency_hz)
                with self.subTest(frequency_hz=frequency_hz):
                    with self.assertRaisesRegex(ValueError, "frequency must be 30 Hz"):
                        load_policy_manifest(directory, REQUIRED_SETUP)

            _write_manifest(directory, schema_version=1)
            with self.assertRaisesRegex(ValueError, "unsupported policy manifest schema"):
                load_policy_manifest(directory, REQUIRED_SETUP)

    def test_manifest_rejects_other_setups(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _write_manifest(directory)
            with self.assertRaisesRegex(ValueError, "requires setup"):
                load_policy_manifest(directory, "bimanual")

    def test_fifty_millisecond_timestamp_gate(self):
        base = 1_000_000_000
        self.assertTrue(
            timestamps_within_skew(
                {"wrist": base, "overhead_1": base + 20_000_000, "joints": base},
                0.05,
            )
        )
        self.assertFalse(
            timestamps_within_skew(
                {"wrist": base, "overhead_1": base + 50_000_001, "joints": base},
                0.05,
            )
        )


if __name__ == "__main__":
    unittest.main()
