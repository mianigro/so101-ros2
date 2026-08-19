#!/usr/bin/env python3
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

"""Run the Qwen3-VL judge over a round's rollout episodes.

Modes:
* sim cross-check — judge each recorded episode's final frames and compare
  with the scripted oracle already stored in ``rollouts.jsonl``;
* real authoritative — judge every episode of a converted real-rollout
  LeRobot dataset; verdicts (with optional human confirmation) are written to
  ``verdicts.jsonl`` and gate the dataset builder's ``approved`` filter.

Run inside the pixi ``lerobot`` env.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from pi05_selfimprove import contract  # noqa: E402
from pi05_selfimprove.judges.scripted import oracle_verdicts  # noqa: E402
from pi05_selfimprove.judges.vlm_qwen import (  # noqa: E402
    DEFAULT_MODEL_ID,
    QwenVLMJudge,
    judge_from_dataset,
)

logger = logging.getLogger("judge_rollouts")


def final_frames_from_npz(path: Path, camera_names) -> list:
    import numpy as np

    frames = []
    with np.load(path) as data:
        for camera in camera_names:
            key = contract.camera_feature_key(camera)
            if key in data.files:
                frames.append(data[key][-1])
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-dir", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", default=None,
                        help="real-phase converted rollout dataset "
                             "(authoritative mode)")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--interactive", action="store_true",
                        help="confirm/correct each verdict (real phase)")
    parser.add_argument("--cameras", default="overhead_1,wrist")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    camera_names = tuple(args.cameras.split(","))
    judge = QwenVLMJudge(args.model_id, device=args.device)
    judge.load()

    verdict_path = args.round_dir / contract.VERDICT_FILENAME
    entries = []
    if args.dataset_repo_id:
        from lerobot.datasets import LeRobotDataset

        dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root,
                                 return_uint8=True)
        entries = judge_from_dataset(judge, dataset,
                                     camera_names=camera_names,
                                     interactive=args.interactive)
    else:
        oracles = oracle_verdicts(args.round_dir)
        if not oracles:
            logger.error("no rollouts.jsonl under %s", args.round_dir)
            return 2
        agreement = 0
        for episode_index, info in sorted(oracles.items()):
            path = (args.round_dir / "rollouts_raw" /
                    f"episode_{episode_index:06d}.npz")
            frames = final_frames_from_npz(path, camera_names)
            verdict = judge.judge(frames, info["task"])
            agrees = verdict.success == info["oracle"]["success"]
            agreement += int(agrees)
            entries.append({
                "episode_index": episode_index,
                "task": info["task"],
                "oracle": info["oracle"],
                "vlm": verdict.to_dict(),
                "agreement": bool(agrees),
                # Sim training data stays gated by the scripted oracle.
                "approved": bool(info["oracle"]["success"]),
            })
            logger.info("episode %d: vlm=%s oracle=%s agree=%s",
                        episode_index, verdict.success,
                        info["oracle"]["success"], agrees)
        rate = agreement / max(1, len(entries))
        logger.info("VLM/oracle agreement: %.0f%% (%d/%d)", 100 * rate,
                    agreement, len(entries))
        meta_path = args.round_dir / contract.META_FILENAME
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        meta["vlm_oracle_agreement"] = round(rate, 3)
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    for entry in entries:
        contract.append_jsonl(verdict_path, entry)
    approved = sum(1 for entry in entries if entry.get("approved"))
    logger.info("wrote %d verdicts to %s (%d approved)", len(entries),
                verdict_path, approved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
