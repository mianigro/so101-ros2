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

"""
Command-line interface for rosbat-to-lerobot conversion.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from lerobot.configs import VALID_VIDEO_CODECS

from rosbag_to_lerobot.config import load_config
from rosbag_to_lerobot.setups import CAMERA_NAMES_BY_SETUP
from rosbag_to_lerobot.converter import (
    convert_all_bags,
    DATASET_TAGS_BY_SOURCE,
    DEFAULT_DATASET_SOURCE,
)

logger = logging.getLogger(__name__)


def main() -> None:

    parser = argparse.ArgumentParser(
        prog="ros2 run rosbag_to_lerobot convert",
        description="Convert ROS 2 rosbag episodes to LeRobot v3.0 datasets",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing episode bag subdirectories",
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="Path to YAML config file."
    )
    parser.add_argument(
        "--setup",
        required=True,
        choices=tuple(CAMERA_NAMES_BY_SETUP),
        help="Required setup recorded in every input episode",
    )
    parser.add_argument(
        "--joint-states-topic",
        default="auto",
        help=(
            "Primary joint-states topic recorded as observation.state. 'auto' "
            "(default) resolves per episode: /follower/joint_states for the "
            "physical follower, /follower_sim/joint_states for Isaac Sim "
            "follower recordings, so mixed directories convert in one run. "
            "Pass an explicit topic to pin it for every episode "
            "(single-arm setups only)."
        ),
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace repo ID (e.g., user/dataset_name)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Output directory for the LeRobot dataset. If omitted, uses LeRobot default (~/.cache/huggingface/lerobot/<repo_id>).",
    )
    parser.add_argument(
        "--use-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store images as MP4 video (default) vs individual images",
    )
    parser.add_argument(
        "--vcodec",
        type=str,
        default="libsvtav1",
        choices=sorted(VALID_VIDEO_CODECS),
        help=(
            "Video codec for encoding (default: libsvtav1). "
            "Use h264 for faster CPU encoding; hardware encoders require compatible hardware."
        ),
    )
    parser.add_argument(
        "--push-hub",
        action="store_true",
        default=False,
        help="Push final dataset to HuggingFace Hub",
    )
    parser.add_argument(
        "--dataset-source",
        choices=tuple(DATASET_TAGS_BY_SOURCE),
        default=DEFAULT_DATASET_SOURCE,
        help=(
            "Provenance of the recorded episodes, used to tag the dataset on the "
            "Hub when --push-hub is set. 'teleop' = human teleoperation "
            "(imitation-learning, default); 'ppo' = autonomous rollouts of an "
            "exported PPO policy (reinforcement-learning); 'autonomous' = "
            "self-improvement rollouts of a VLA policy "
            "(reinforcement-learning, self-improvement). All use the same 30 Hz "
            "command contract."
        ),
    )
    parser.add_argument(
        "--sync-p95",
        action="store_true",
        default=False,
        help="Collect p95 sync stats (slightly more overhead). Default: off.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="If set, delete any existing dataset directory before writing.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Basic path validation
    if not args.input_dir.exists() or not args.input_dir.is_dir():
        logger.error(
            "--input-dir does not exist or is not a directory: %s", args.input_dir
        )
        sys.exit(2)
    if not args.config.exists() or not args.config.is_file():
        logger.error("--config does not exist or is not a file: %s", args.config)
        sys.exit(2)

    try:
        cfg = load_config(
            args.config, args.setup, joint_states_topic=args.joint_states_topic
        )

        output_dir = args.output_dir.expanduser() if args.output_dir else None
        convert_all_bags(
            cfg=cfg,
            input_dir=args.input_dir,
            output_dir=output_dir,
            repo_id=args.repo_id,
            use_videos=args.use_videos,
            vcodec=args.vcodec,
            push_to_hub=args.push_hub,
            dataset_source=args.dataset_source,
            collect_p95=args.sync_p95,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        logger.error("Conversion failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
