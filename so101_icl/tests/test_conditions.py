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
    CONDITIONS,
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

    def test_conditions_tuple(self):
        self.assertEqual(CONDITIONS, ("full_icl", "prompt_enriched", "bare_prompt"))


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


if __name__ == "__main__":
    unittest.main()
