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

"""Filter successful rollouts, mix with teleop data, write a LeRobot dataset.

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

from self_improve.config import RoundConfig  # noqa: E402
from self_improve.dataset_tools.build_round_dataset import (  # noqa: E402
    build_round_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="round YAML")
    parser.add_argument("--rounds-root", type=Path,
                        default=PROJECT_ROOT / "rounds")
    parser.add_argument("--dataset-root", type=Path, default=None,
                        help="root dir for the created dataset (default: "
                             "LeRobot cache)")
    parser.add_argument("--real-dataset-repo-id", default=None,
                        help="real-phase converted rollout dataset")
    parser.add_argument("--min-successes", type=int, default=None)
    parser.add_argument("--teleop-episode-cap", type=int, default=None)
    parser.add_argument("--vcodec", default="libsvtav1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config = RoundConfig.load(args.config)
    if args.min_successes is not None:
        config.judge.min_successes = args.min_successes
    if args.teleop_episode_cap is not None:
        config.dataset.teleop_episode_cap = args.teleop_episode_cap

    summary = build_round_dataset(
        repo_id=config.mixed_repo_id(),
        round_dir=config.round_dir(args.rounds_root),
        teleop_repo_ids=config.dataset.teleop_repo_ids,
        teleop_episode_cap=config.dataset.teleop_episode_cap,
        min_successes=config.judge.min_successes,
        root=args.dataset_root,
        real_dataset_repo_id=args.real_dataset_repo_id,
        vcodec=args.vcodec,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
