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

"""BC training wrapper: round 0 baseline and round N self-improvement.

Both are plain ``lerobot-train`` fine-tuning runs of a flow-matching VLA
(pi0.5 by default); the only difference is where the weights and the data
come from:

* round 0 — initialize from the published base checkpoint
  (``lerobot/pi05_base``) and train on the human teleop dataset.  This is
  "Phase 1: the baseline" — pure behavioral cloning / mimicry.
* round N — initialize from either the published base policy or round N-1's
  checkpoint and train on the round's mixed dataset (robot's own successes +
  teleop anchor). This is the "refine" step of the self-improvement loop: BC
  re-trained on filtered autonomous data, so the policy learns to recover from
  its own mistakes.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import RoundConfig

logger = logging.getLogger(__name__)


def training_dataset_repo_id(config: RoundConfig) -> str:
    """Round 0 trains on teleop data; later rounds on the mixed dataset."""
    if config.round_index == 0:
        if not config.dataset.teleop_repo_ids:
            raise ValueError("round 0 needs at least one teleop dataset")
        return config.dataset.teleop_repo_ids[0]
    return config.mixed_repo_id()


def init_weights_source(config: RoundConfig) -> str:
    """Resolve the input policy selected explicitly by ``train.init_from``."""
    if config.train.init_from == "base":
        return config.train.base_repo_id
    if config.train.init_from == "previous":
        if not config.checkpoint_in:
            raise ValueError(
                f"round {config.round_index} uses train.init_from=previous "
                "but has no checkpoint_in")
        return config.checkpoint_in
    raise ValueError(
        f"unknown train.init_from {config.train.init_from!r}; expected "
        "'base' or 'previous'")


def build_train_command(config: RoundConfig, output_dir: Path,
                        *, dataset_repo_id: Optional[str] = None,
                        policy_path: Optional[str] = None) -> List[str]:
    """Assemble the lerobot-train command line for one round."""
    dataset = dataset_repo_id or training_dataset_repo_id(config)
    weights = policy_path or init_weights_source(config)
    command = [
        "lerobot-train",
        f"--policy.path={weights}",
        f"--dataset.repo_id={dataset}",
        f"--steps={config.train.steps}",
        f"--batch_size={config.train.batch_size}",
        f"--save_freq={config.train.save_steps}",
        f"--num_workers={config.train.num_workers}",
        f"--output_dir={output_dir}",
        # Published base configs default to push_to_hub=true but this wrapper
        # produces local round checkpoints unless extra_args opts back in and
        # supplies a policy repo id.
        "--policy.push_to_hub=false",
    ]
    if config.train.freeze_vision_encoder:
        command.append("--policy.freeze_vision_encoder=true")
    if config.rollout.image_key_map:
        rename_map = json.dumps(
            config.rollout.image_key_map,
            sort_keys=True,
            separators=(",", ":"),
        )
        command.append(f"--rename_map={rename_map}")
    command.extend(config.train.extra_args)
    return command


def find_latest_checkpoint(train_output_dir: Path) -> Optional[Path]:
    """Newest ``checkpoints/<step>`` directory with a model in it."""
    checkpoint_root = Path(train_output_dir) / "checkpoints"
    if not checkpoint_root.is_dir():
        return None
    candidates = [
        path for path in checkpoint_root.iterdir()
        if path.is_dir() and (path / "model.safetensors").exists()
    ]
    if not candidates:
        return None
    def step_of(path: Path) -> int:
        try:
            return int(path.name)
        except ValueError:
            return -1
    return max(candidates, key=step_of)


def run_training(config: RoundConfig, rounds_root: Path,
                 *, dry_run: bool = False,
                 dataset_repo_id: Optional[str] = None,
                 policy_path: Optional[str] = None) -> Dict[str, Any]:
    """Run (or print) the round's lerobot-train command and record artifacts.

    Returns a dict with the command, dataset, weight source, and the latest
    checkpoint path — also appended to the round's ``round_meta.json``.
    """
    train_dir = config.round_dir(rounds_root) / "checkpoint"
    output_dir = train_dir / "run"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    command = build_train_command(
        config, output_dir,
        dataset_repo_id=dataset_repo_id,
        policy_path=policy_path,
    )

    record: Dict[str, Any] = {
        "command": command,
        "dataset_repo_id": (dataset_repo_id
                            or training_dataset_repo_id(config)),
        "init_from": (policy_path or init_weights_source(config)),
    }
    if dry_run:
        record["dry_run"] = True
        logger.info("dry run: %s", " ".join(command))
        return record

    logger.info("launching: %s", " ".join(command))
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise RuntimeError(
            f"lerobot-train failed with exit code {completed.returncode}")

    latest = find_latest_checkpoint(output_dir)
    if latest is None:
        raise RuntimeError(
            f"training finished but no checkpoint found under {output_dir}")
    record["checkpoint_path"] = str(latest)
    logger.info("round %d checkpoint: %s", config.round_index, latest)

    meta_path = config.round_dir(rounds_root) / "round_meta.json"
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    meta["train"] = record
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return record


def main(argv: Optional[list] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="BC train one self-improvement round (wraps lerobot-train)",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rounds-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "rounds")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the lerobot-train command and exit")
    parser.add_argument("--dataset-repo-id", default=None,
                        help="override the training dataset")
    parser.add_argument("--policy-path", default=None,
                        help="override the pretrained weight source")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config = RoundConfig.load(args.config)
    record = run_training(
        config, args.rounds_root,
        dry_run=args.dry_run,
        dataset_repo_id=args.dataset_repo_id,
        policy_path=args.policy_path,
    )
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
