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

"""Measure a checkpoint's sim success rate (no recording).

Spawns ``serve_policy`` for the given checkpoint, runs ``rollout_sim --eval``,
and prints/writes the summary.  Run inside the pixi ``lerobot`` env.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from pi05_selfimprove.loop import PolicyServerProcess  # noqa: E402

logger = logging.getLogger("eval_policy")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True,
                        help="checkpoint dir or HF repo id to evaluate")
    parser.add_argument("--task", default="SO101-Object-In-Cup-Vision-Fixed-v0")
    parser.add_argument("--task-prompt",
                        default="pick up the cube and place it in the cup")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-episodes", type=int, default=24)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8660)
    parser.add_argument("--policy-type", default="pi05")
    parser.add_argument("--actions-per-chunk", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument(
        "--image-key-map", action="append", default=None, metavar="MAP",
        help="camera key remap for zero-shot base checkpoints, forwarded to "
             "rollout_sim: the preset 'pi05_base' or repeatable "
             "client=policy pairs like "
             "observation.images.wrist=observation.images.left_wrist_0_rgb")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    server = PolicyServerProcess(
        args.repo_id,
        host=args.host,
        port=args.port,
        policy_type=args.policy_type,
        actions_per_chunk=args.actions_per_chunk,
        device=args.device,
    )
    server.start()
    try:
        argv = [
            str(PROJECT_ROOT / "rollout_sim.py"),
            "--eval",
            "--task", args.task,
            "--task-prompt", args.task_prompt,
            "--num-envs", str(args.num_envs),
            "--num-episodes", str(args.num_episodes),
            "--server-host", args.host,
            "--server-port", str(args.port),
        ]
        if args.metrics_out:
            argv += ["--metrics-out", str(args.metrics_out)]
        for entry in args.image_key_map or []:
            argv += ["--image-key-map", entry]
        logger.info("launching: %s", " ".join(argv))
        subprocess.run(argv, check=True)
    finally:
        server.stop()

    if args.metrics_out and args.metrics_out.exists():
        print(json.dumps(json.loads(args.metrics_out.read_text()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
