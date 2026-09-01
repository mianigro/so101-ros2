# Copyright 2026 Dmitri Manajev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Condition-driver tests for the M3/M4 campaigns (no GPU, no sockets)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from so101_icl.conditions import (  # noqa: E402
    DemoPackBuilder,
    apply_condition,
    bare_prompt,
    condition_prompt,
)


class FakeTransport:
    def __init__(self):
        self.calls = []

    def set_demo_pack(self, frames, traj=None, traj_ok=None, k_max=4):
        self.calls.append(("set", frames.shape, traj.shape, k_max))
        return {"status": "ok", "k": frames.shape[0]}

    def clear(self):
        self.calls.append(("clear",))
        return {"status": "ok"}


class TestPrompts(unittest.TestCase):
    def test_bare_prompt_first_sentence(self):
        prompt = "pick up the cube. Place it gently in the red cup on the left."
        self.assertEqual(bare_prompt(prompt), "pick up the cube.")

    def test_bare_prompt_no_punctuation(self):
        self.assertEqual(bare_prompt("pick up the cube"), "pick up the cube")

    def test_bare_prompt_empty(self):
        self.assertEqual(bare_prompt("   "), "")

    def test_condition_prompt(self):
        prompt = "pick up the cube. Carefully."
        self.assertEqual(condition_prompt("full_icl", prompt), prompt)
        self.assertEqual(condition_prompt("prompt_enriched", prompt), prompt)
        self.assertEqual(condition_prompt("bare_prompt", prompt), "pick up the cube.")


class TestApplyCondition(unittest.TestCase):
    def test_full_icl_requires_builder(self):
        with self.assertRaises(ValueError):
            apply_condition("full_icl", FakeTransport())

    def test_unknown_condition(self):
        with self.assertRaises(ValueError):
            apply_condition("nope", FakeTransport())

    def test_prompt_conditions_clear(self):
        for condition in ("prompt_enriched", "bare_prompt"):
            transport = FakeTransport()
            reply = apply_condition(condition, transport)
            self.assertEqual(transport.calls, [("clear",)])
            self.assertEqual(reply["condition"], condition)

    def test_full_icl_with_fake_builder(self):
        import numpy as np

        class FakeBuilder:
            k = 2
            k_max = 4

            def build(self, group):
                return {
                    "frames": np.zeros((4, 6, 3, 224, 224), np.float32),
                    "traj": np.zeros((4, 16, 64), np.float32),
                    "traj_ok": np.ones(4, np.float32),
                }

        transport = FakeTransport()
        reply = apply_condition("full_icl", transport, FakeBuilder(), group="g")
        self.assertEqual(reply["status"], "ok")
        set_call = transport.calls[0]
        self.assertEqual(set_call[0], "set")
        self.assertEqual(set_call[1], (2, 6, 3, 224, 224))  # sliced to k
        self.assertEqual(set_call[3], 4)  # k_max padding


class TestBuilderRegistryHandling(unittest.TestCase):
    """Registry-facing behavior that needs no dataset on disk."""

    def test_resolve_group_unknown(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reg.json"
            path.write_text(json.dumps({
                "version": 1, "datasets": [],
                "groups": {"task_a": [[0, 0, 10]]}, "splits": {"train": ["task_a"]},
            }))
            builder = DemoPackBuilder(path)
            self.assertEqual(builder.resolve_group(None), "task_a")
            self.assertEqual(builder.resolve_group("task_a"), "task_a")
            with self.assertRaises(ValueError):
                builder.resolve_group("missing")


class _FakeEpisodes:
    def __init__(self):
        import pandas as pd

        self._df = pd.DataFrame([{
            "episode_index": 0, "dataset_from_index": 0,
            "dataset_to_index": 10, "length": 10,
        }])

    def to_pandas(self):
        return self._df


class _FakeDataset:
    """Minimal LeRobotDataset stand-in for DemoPackBuilder._bundle."""

    def __init__(self, image_keys):
        self.meta = type("Meta", (), {})()
        self.meta.info = {"features": {
            **{k: {"shape": [3, 224, 224]} for k in image_keys},
            "observation.state": {"shape": [7]},
            "action": {"shape": [7]},
        }}
        self.meta.stats = {}
        self.meta.episodes = _FakeEpisodes()
        self.hf_dataset = type("HF", (), {"select_columns": lambda self, cols: None})()

    def __getitem__(self, idx):
        import torch

        return torch.zeros(3, 224, 224)


class TestBuilderCameraResolution(unittest.TestCase):
    """The demo camera must resolve to a real feature key or fail loudly."""

    def _registry(self, tmp, camera_rename):
        import json

        path = Path(tmp) / "reg.json"
        path.write_text(json.dumps({
            "version": 1,
            "datasets": [{
                "repo_id": "local/fake", "root": str(tmp),
                "camera_rename": camera_rename,
            }],
            "groups": {"task_a": [[0, 0, 10]]}, "splits": {"train": ["task_a"]},
        }))
        return path

    def _builder(self, registry_path, image_keys, **kwargs):
        from unittest.mock import patch

        import so101_icl.conditions as conditions

        builder = DemoPackBuilder(registry_path, **kwargs)
        patcher_ds = patch.object(
            conditions, "open_local_dataset", return_value=_FakeDataset(image_keys))
        patcher_norm = patch.object(conditions, "TrajNormalizer", return_value=None)
        patcher_ds.start()
        patcher_norm.start()
        self.addCleanup(patcher_ds.stop)
        self.addCleanup(patcher_norm.stop)
        return builder

    def test_default_demo_camera_is_wrist(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            builder = self._builder(self._registry(tmp, {}), [])
            self.assertEqual(
                builder._demo_camera, "observation.images.left_wrist_0_rgb")

    def test_camera_resolved_through_reverse_rename(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            registry = self._registry(tmp, {
                "observation.images.wrist": "observation.images.left_wrist_0_rgb",
            })
            builder = self._builder(
                registry, ["observation.images.wrist", "observation.images.front"])
            bundle = builder._bundle(0)
            self.assertEqual(bundle["camera"], "observation.images.wrist")

    def test_missing_camera_raises_with_available_keys(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            registry = self._registry(tmp, {})
            builder = self._builder(registry, ["observation.images.front"])
            with self.assertRaises(ValueError) as ctx:
                builder._bundle(0)
            message = str(ctx.exception)
            self.assertIn("demo camera", message)
            self.assertIn("observation.images.front", message)  # lists what exists

    def test_no_silent_first_key_fallback(self):
        """Explicit non-existent camera must never fall back to another view."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            registry = self._registry(tmp, {})
            builder = self._builder(
                registry, ["observation.images.front"],
                demo_camera="observation.images.base_0_rgb")
            with self.assertRaises(ValueError):
                builder._bundle(0)


if __name__ == "__main__":
    unittest.main()
