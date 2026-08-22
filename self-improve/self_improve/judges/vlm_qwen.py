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

"""Qwen3-VL success judge: "is the task done?" from final camera frames.

The self-improvement workflow needs a reward signal for autonomous rollouts;
in the real world there is no scripted oracle, so a small local VLM judges the
final overhead (+ wrist) frames against the task text. Output is a strict-JSON
``{"success": bool, "confidence": 0-1, "reason": str}``; a human confirms
verdicts before data enters the training set (supervised-assist).
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

import numpy as np

from . import Verdict

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

PROMPT_TEMPLATE = (
    "You are judging whether a robot arm completed its task.\n"
    "Task: {task}\n"
    "The images show the final state of the workspace after the robot "
    "finished its attempt. Decide whether the task is complete.\n"
    "Answer with ONLY a JSON object, no other text:\n"
    '{{"success": true|false, "confidence": 0.0-1.0, "reason": "short explanation"}}'
)


def parse_verdict(text: str) -> Verdict:
    """Extract the JSON verdict from a model reply (robust to prose).

    Raises ``ValueError`` when no parseable verdict is present.
    """
    candidates: List[str] = []
    for match in re.finditer(r"\{[^{}]*\}", text, flags=re.DOTALL):
        candidates.append(match.group(0))
    try:
        # Prefer the last JSON-looking object (models sometimes preamble).
        for candidate in reversed(candidates):
            data = json.loads(candidate)
            if "success" in data:
                return Verdict(
                    success=_to_bool(data["success"]),
                    confidence=_clamp01(data.get("confidence", 0.5)),
                    reason=str(data.get("reason", ""))[:500],
                )
    except (json.JSONDecodeError, TypeError):
        pass
    lowered = text.strip().lower()
    if lowered.startswith(("true", "yes")):
        return Verdict(True, 0.6, "parsed from boolean prefix")
    if lowered.startswith(("false", "no")):
        return Verdict(False, 0.6, "parsed from boolean prefix")
    raise ValueError(f"no JSON verdict in model reply: {text[:200]!r}")


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "yes", "1")


def _clamp01(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.5
    return min(1.0, max(0.0, number))


def frames_to_pil(frames: List[np.ndarray]):
    """uint8 HWC numpy frames -> PIL images (lazy import keeps tests light)."""
    from PIL import Image

    return [Image.fromarray(np.ascontiguousarray(frame)) for frame in frames]


class QwenVLMJudge:
    """Local Qwen3-VL judge over final-state frames."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID,
                 device: str = "cuda") -> None:
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        logger.info("loading VLM judge %s on %s ...", self.model_id, self.device)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype="auto",
            device_map=self.device,
        )
        self._model.eval()
        logger.info("VLM judge ready")

    def judge(self, frames: List[np.ndarray], task: str) -> Verdict:
        """Judge final-state frames (HWC uint8) against the task text."""
        if self._model is None:
            raise RuntimeError("judge model not loaded; call load() first")
        import torch  # noqa: F401 - used implicitly by the model call

        images = frames_to_pil(frames)
        prompt = PROMPT_TEMPLATE.format(task=task)
        messages = [{
            "role": "user",
            "content": [
                *[{"type": "image", "image": image} for image in images],
                {"type": "text", "text": prompt},
            ],
        }]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        generated = self._model.generate(
            **inputs, max_new_tokens=128, do_sample=False)
        generated = generated[:, inputs["input_ids"].shape[-1]:]
        text = self._processor.batch_decode(
            generated, skip_special_tokens=True)[0]
        logger.debug("VLM reply: %s", text)
        return parse_verdict(text)


def judge_from_dataset(
    judge: QwenVLMJudge,
    dataset,
    *,
    camera_names=("overhead_1", "wrist"),
    interactive: bool = False,
) -> List[dict]:
    """Judge every episode of a converted LeRobot dataset by its final frame.

    Returns verdicts.jsonl-ready entries with the human-confirmation field
    (``approved``): VLM verdict by default, overridden interactively when
    ``interactive`` is set.
    """
    from .. import contract

    rows_by_episode: dict = {}
    for row in range(len(dataset)):
        episode_index = int(dataset.hf_dataset[row]["episode_index"])
        rows_by_episode.setdefault(episode_index, []).append(row)

    entries: List[dict] = []
    for episode_index in sorted(rows_by_episode):
        last_row = rows_by_episode[episode_index][-1]
        item = dataset[last_row]
        task = item.get("task")
        if task is None:
            task = dataset.meta.episodes[episode_index].get("task", "")
        if isinstance(task, (list, tuple)):
            task = task[0] if task else ""
        task = str(task)

        frames = []
        for camera in camera_names:
            key = contract.camera_feature_key(camera)
            if key in item:
                frame = item[key]
                array = frame.numpy() if hasattr(frame, "numpy") else frame
                array = np.asarray(array)
                if array.shape[0] in (1, 3) and array.shape[-1] != 3:
                    array = np.transpose(array, (1, 2, 0))
                frames.append(array)
        verdict = judge.judge(frames, task)
        approved = verdict.success
        if interactive:
            reply = input(
                f"episode {episode_index}: VLM says "
                f"{'SUCCESS' if verdict.success else 'FAILURE'} "
                f"(confidence {verdict.confidence:.2f}) — {verdict.reason}\n"
                f"  keep for training? [Y/n] ").strip().lower()
            approved = reply not in ("n", "no")
        entries.append({
            "episode_index": episode_index,
            "task": task,
            "vlm": verdict.to_dict(),
            "approved": bool(approved),
        })
        logger.info("episode %d: %s (confidence %.2f)",
                    episode_index,
                    "SUCCESS" if verdict.success else "FAILURE",
                    verdict.confidence)
    return entries
