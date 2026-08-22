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

"""Success judges for the VLA self-improvement loop.

* scripted — ground truth from the sim termination oracle (sim rollouts are
  labeled at rollout time; this module repackages the manifest).
* vlm — Qwen3-VL judges final camera frames against the task text (real
  rollouts; also a cross-check for sim rounds).

Every judge returns a ``Verdict``; verdicts land in the round's
``verdicts.jsonl`` and gate which episodes enter the training dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Verdict:
    success: bool
    confidence: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "success": bool(self.success),
            "confidence": round(float(self.confidence), 3),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Verdict":
        return cls(
            success=bool(data["success"]),
            confidence=float(data.get("confidence", 1.0)),
            reason=str(data.get("reason", "")),
        )


__all__ = ["Verdict"]
