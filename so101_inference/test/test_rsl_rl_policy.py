"""Tests for deployment preprocessing, timestamp gating, and action safety."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from so101_inference.rsl_rl_policy import (
    preprocess_rgb,
    safe_absolute_targets,
    timestamps_within_skew,
)


class RslRlPolicyTests(unittest.TestCase):
    def test_real_rgb_preprocessing_matches_training_contract(self):
        white = np.full((480, 640, 3), 255, dtype=np.uint8)
        processed = preprocess_rgb(white, torch.device("cpu"))
        self.assertEqual(tuple(processed.shape), (1, 3, 120, 160))
        self.assertTrue(torch.allclose(processed, torch.full_like(processed, 0.5)))

    def test_action_clipping_and_joint_limit_margin(self):
        action = np.array([2.0, -2.0, 0.5, 0.0, 1.0, -1.0])
        measured = np.array([0.97, -0.97, 0.0, 0.0, 0.0, 0.0])
        scales = np.array([0.05] * 5 + [0.15])
        limits = np.array([[-1.0, 1.0]] * 6)
        targets = safe_absolute_targets(action, measured, scales, limits, 0.98)
        self.assertTrue(np.all(targets <= 0.98))
        self.assertTrue(np.all(targets >= -0.98))
        self.assertAlmostEqual(targets[2], 0.025)
        with self.assertRaises(ValueError):
            safe_absolute_targets(
                np.array([np.nan] * 6), measured, scales, limits, 0.98
            )

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
