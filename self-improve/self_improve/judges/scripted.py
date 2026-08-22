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

"""Scripted (oracle) verdicts — the sim ground truth recorded at rollout time.

Sim episodes are labeled in ``rollouts.jsonl`` by the termination manager's
``success`` term (cube settled in the cup, gripper released, 15 consecutive
steps).  This module converts that manifest into the uniform verdict format
used by the dataset builder and the VLM cross-check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from . import Verdict
from .. import contract


def oracle_verdicts(round_dir: Path) -> Dict[int, dict]:
    """{episode_index: {"oracle": Verdict dict, "task": str, "steps": int}}."""
    entries: List[dict] = contract.read_jsonl(
        Path(round_dir) / contract.ROLL_FILENAME)
    verdicts: Dict[int, dict] = {}
    for entry in entries:
        verdict = Verdict(
            success=bool(entry.get("success")),
            confidence=1.0,
            reason=str(entry.get("reason", "")),
        )
        verdicts[int(entry["episode_index"])] = {
            "oracle": verdict.to_dict(),
            "task": str(entry.get("task", "")),
            "steps": int(entry.get("steps", 0)),
        }
    return verdicts
