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

"""Offline-eval report + ablation-flag plumbing tests (no GPU, no datasets)."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from so101_icl.eval_icl import _offline_report, main  # noqa: E402


def _rows():
    return [
        ("k=1", 1.0, 2.0, 1.5),
        ("k=2", 0.8, 2.0, 1.5),
        ("k=4", 0.7, 2.0, 1.5),
        ("F=4", 0.9, 2.0, 1.5),
        ("F=6", 0.7, 2.0, 1.5),
        ("F=8", 0.65, 2.0, 1.5),
        ("traj=on", 0.7, 2.0, 1.5),
        ("traj=off", 1.1, 2.0, 1.5),
    ]


class TestOfflineReport(unittest.TestCase):
    def test_renders_all_ablation_rows(self):
        report = _offline_report(_rows(), n_k_runs=3, per_group={}, split_note=None)
        for label in ("k=1", "k=4", "F=8", "traj=on", "traj=off"):
            self.assertIn(f"| {label} |", report)
        self.assertIn("median demo/zeroed ratio (k ablation)", report)

    def test_per_group_table(self):
        per_group = {
            "pick up the cube": {"demo": [0.7, 0.9], "zeroed": [2.0, 2.0]},
            "place in the cup": {"demo": [1.0], "zeroed": [2.0]},
        }
        report = _offline_report(_rows(), n_k_runs=3, per_group=per_group, split_note=None)
        self.assertIn("### per-group (k=4)", report)
        self.assertIn("| pick up the cube |", report)
        self.assertIn("| place in the cup |", report)

    def test_split_fallback_warning_lands_in_report(self):
        report = _offline_report(_rows()[:3], n_k_runs=3, per_group={},
                                 split_note="WARNING: split 'test' is empty")
        self.assertIn("WARNING: split 'test' is empty", report)


class TestOfflineCliPlumbing(unittest.TestCase):
    """--frames/--traj flags resolve from the stage YAML when not given."""

    STAGE = """
base_checkpoint: lerobot/pi05_base
dataset: {demo_camera: observation.images.base_0_rgb}
eval:
  offline_trials: 12
  ablations: {k: [1, 2], frames: [4, 6], traj: [on, off]}
"""

    def test_flags_default_from_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "stage.yaml"
            cfg.write_text(self.STAGE)
            with mock.patch("so101_icl.eval_icl.run_offline", return_value="report") as run:
                rc = main(["offline", "--adapter", "a", "--registry", "r", "--config", str(cfg)])
            self.assertEqual(rc, 0)
            run.assert_called_once()
            kwargs = run.call_args.kwargs
            self.assertEqual(kwargs["trials"], 12)
            self.assertEqual(kwargs["ablate_k"], (1, 2))
            self.assertEqual(kwargs["ablate_frames"], (4, 6))
            self.assertEqual(kwargs["ablate_traj"], (True, False))

    def test_cli_flags_override_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "stage.yaml"
            cfg.write_text(self.STAGE)
            with mock.patch("so101_icl.eval_icl.run_offline", return_value="report") as run:
                main(["offline", "--adapter", "a", "--registry", "r", "--config", str(cfg),
                      "--k", "3", "--frames", "8", "--traj", "off"])
            kwargs = run.call_args.kwargs
            self.assertEqual(kwargs["ablate_k"], (3,))
            self.assertEqual(kwargs["ablate_frames"], (8,))
            self.assertEqual(kwargs["ablate_traj"], (False,))


if __name__ == "__main__":
    unittest.main()
