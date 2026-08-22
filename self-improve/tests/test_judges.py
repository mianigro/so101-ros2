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

"""VLM verdict parsing + oracle verdict tests (no model weights needed)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from self_improve import contract
from self_improve.judges import Verdict
from self_improve.judges.scripted import oracle_verdicts
from self_improve.judges.vlm_qwen import parse_verdict


class ParseVerdictTests(unittest.TestCase):
    def test_plain_json(self) -> None:
        verdict = parse_verdict(
            '{"success": true, "confidence": 0.9, "reason": "cube in cup"}')
        self.assertTrue(verdict.success)
        self.assertEqual(verdict.confidence, 0.9)
        self.assertEqual(verdict.reason, "cube in cup")

    def test_json_with_preamble(self) -> None:
        verdict = parse_verdict(
            'Looking at the images, the cube is inside the cup.\n'
            '{"success": true, "confidence": 0.85, "reason": "cube in cup"}')
        self.assertTrue(verdict.success)

    def test_json_with_wrapping_prose(self) -> None:
        verdict = parse_verdict(
            'Sure! {"success": false, "confidence": 0.7, '
            '"reason": "cube on table"} hope that helps')
        self.assertFalse(verdict.success)
        self.assertEqual(verdict.confidence, 0.7)

    def test_confidence_clamped(self) -> None:
        verdict = parse_verdict(
            '{"success": true, "confidence": 5, "reason": "x"}')
        self.assertEqual(verdict.confidence, 1.0)

    def test_boolean_fallback(self) -> None:
        self.assertTrue(parse_verdict("Yes.").success)
        self.assertFalse(parse_verdict("false").success)

    def test_garbage_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_verdict("the cube seems maybe done?")


class VerdictTests(unittest.TestCase):
    def test_roundtrip(self) -> None:
        verdict = Verdict(success=True, confidence=0.75, reason="ok")
        restored = Verdict.from_dict(verdict.to_dict())
        self.assertEqual(restored.success, True)
        self.assertEqual(restored.confidence, 0.75)
        self.assertEqual(restored.reason, "ok")


class OracleVerdictTests(unittest.TestCase):
    def test_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp)
            contract.append_jsonl(round_dir / contract.ROLL_FILENAME, {
                "episode_index": 0, "file": "rollouts_raw/episode_000000.npz",
                "source": "sim", "task": "pick cube", "success": True,
                "reason": "success", "steps": 300,
                "fps": contract.CONTROL_FREQUENCY_HZ,
            })
            contract.append_jsonl(round_dir / contract.ROLL_FILENAME, {
                "episode_index": 1, "file": "rollouts_raw/episode_000001.npz",
                "source": "sim", "task": "pick cube", "success": False,
                "reason": "timeout", "steps": 450,
                "fps": contract.CONTROL_FREQUENCY_HZ,
            })
            verdicts = oracle_verdicts(round_dir)
            self.assertEqual(len(verdicts), 2)
            self.assertTrue(verdicts[0]["oracle"]["success"])
            self.assertEqual(verdicts[1]["oracle"]["reason"], "timeout")


if __name__ == "__main__":
    unittest.main()
