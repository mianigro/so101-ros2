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

"""Autonomous pi0.5 rollouts in Isaac Sim (the "generate" step).

Re-executes itself into Isaac Sim's Python exactly like the isaaclab/ entry
points, then drives the registered SO-101 task with a VLA served by
``serve_policy`` over the rollout wire protocol.  Start the policy server
first (see README).  Every completed episode is dumped under
``<rounds-root>/round_N/rollouts_raw`` with a scripted success verdict.

Examples:
  pi05-self-improve/rollout_sim --config pi05-self-improve/configs/round_1.yaml \
      --rounds-root pi05-self-improve/rounds
  pi05-self-improve/rollout_sim --config pi05-self-improve/configs/round_1.yaml \
      --visualizer kit  # open the native Isaac Sim window
  pi05-self-improve/rollout_sim --eval --num-episodes 24  # metrics only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "isaaclab" / "source" / "so101_rl"))

from so101_rl.runtime import launch_isaac_sim_before_task_imports  # noqa: E402

launch_isaac_sim_before_task_imports(Path(__file__))

from pi05_selfimprove import contract, sim_rollout  # noqa: E402
from pi05_selfimprove.config import RoundConfig  # noqa: E402
from pi05_selfimprove.wire import RolloutClient  # noqa: E402

logger = logging.getLogger("rollout_sim")

#: Preset camera remap for zero-shot base checkpoints (see contract).
PI05_BASE_IMAGE_KEY_MAP = contract.PI05_BASE_IMAGE_KEY_MAP
resolve_image_key_map = contract.resolve_image_key_map


def build_features() -> dict:
    """Feature schema advertised to the policy server (mirrors the rig)."""
    features = {
        contract.STATE_FEATURE_KEY: {
            "dtype": "float32",
            "shape": (contract.NUM_JOINTS,),
            "names": [f"{joint}.pos" for joint in contract.SO101_JOINT_NAMES],
        },
    }
    for camera in contract.CAMERA_KEYS:
        features[contract.camera_feature_key(camera)] = {
            "dtype": "image",
            "shape": (contract.IMAGE_HEIGHT, contract.IMAGE_WIDTH,
                      contract.IMAGE_CHANNELS),
            "names": ["height", "width", "channels"],
        }
    return features


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="round YAML (defaults otherwise apply)")
    parser.add_argument("--rounds-root", type=Path,
                        default=PROJECT_ROOT / "rounds")
    parser.add_argument("--eval", action="store_true",
                        help="measure success rate only; no recording")
    parser.add_argument("--metrics-out", type=Path, default=None,
                        help="write the summary JSON here")
    parser.add_argument("--task", default=None,
                        help="override the Isaac Sim task id")
    parser.add_argument("--task-prompt", default=None,
                        help="override the language instruction sent to the VLA")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--actions-per-chunk", type=int, default=None)
    parser.add_argument("--server-host", default=None)
    parser.add_argument("--server-port", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--visualizer", "--viz", default=None,
        help="Isaac Lab visualizer already consumed during simulator startup "
             "(use 'kit' for the native Isaac Sim window)",
    )
    parser.add_argument("--headless", action="store_true", default=False,
                        help=argparse.SUPPRESS)  # compat flag, see runtime.py
    parser.add_argument("--max-wall-time-min", type=float, default=240.0)
    parser.add_argument(
        "--image-key-map", action="append", default=None, metavar="MAP",
        help="camera key remap for zero-shot base checkpoints. Either the "
             "preset name 'pi05_base' or repeatable client=policy pairs "
             "like observation.images.wrist=observation.images.left_wrist_0_rgb")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config = RoundConfig.load(args.config) if args.config else RoundConfig()
    rollout = config.rollout
    if args.task:
        rollout.task_id = args.task
    if args.task_prompt:
        rollout.task_prompt = args.task_prompt
    if args.num_envs:
        rollout.num_envs = args.num_envs
    if args.num_episodes:
        rollout.num_episodes = args.num_episodes
    if args.max_episode_steps:
        rollout.max_episode_steps = args.max_episode_steps
    if args.actions_per_chunk:
        rollout.actions_per_chunk = args.actions_per_chunk
    if args.server_host:
        rollout.server_host = args.server_host
    if args.server_port:
        rollout.server_port = args.server_port
    if args.seed is not None:
        rollout.seed = args.seed

    joint_map = config.resolve_joint_map()
    image_key_map = config.rollout.image_key_map or {}
    if args.image_key_map:
        image_key_map.update(resolve_image_key_map(args.image_key_map))
    client = RolloutClient(rollout.server_host, rollout.server_port,
                           timeout_s=600.0)
    client.hello(build_features(), rollout.actions_per_chunk,
                 rollout.task_prompt,
                 image_key_map=image_key_map or None,
                 action_dim=contract.NUM_JOINTS)
    logger.info("policy server handshake ok: chunk=%d action_dim=%d",
                client.chunk_size, client.action_dim)

    env = sim_rollout.make_env(
        rollout.task_id,
        rollout.num_envs,
        image_width=contract.IMAGE_WIDTH,
        image_height=contract.IMAGE_HEIGHT,
        seed=rollout.seed,
        device=args.device,
    )
    try:
        round_dir = None if args.eval else config.round_dir(args.rounds_root)
        summary = sim_rollout.run_rollouts(
            env,
            client,
            task=rollout.task_prompt,
            joint_map=joint_map,
            num_episodes=rollout.num_episodes,
            max_episode_steps=rollout.max_episode_steps,
            actions_per_chunk=rollout.actions_per_chunk,
            chunk_size_threshold=rollout.chunk_size_threshold,
            aggregate_fn_name=rollout.aggregate,
            round_dir=round_dir,
            max_wall_time_min=args.max_wall_time_min,
        )
    finally:
        env.close()
        client.close()

    logger.info("rollout summary: %s", json.dumps(summary))
    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
