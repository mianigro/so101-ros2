"""Canonical setup definitions for SO-101 visualization tools.

Thin facade over ``rosbag_to_lerobot.setups``, which derives the camera
contract from the so101_bringup setup YAMLs (the single source of truth for
the whole repository). Bag-metadata helpers live here because only the
visualization tools need them.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Append the rosbag_to_lerobot project directory (not the repo root, which
# would resolve rosbag_to_lerobot as a namespace package) so the regular
# package inside it is importable.
sys.path.append(str(Path(__file__).resolve().parent.parent / "rosbag_to_lerobot"))

from rosbag_to_lerobot.setups import (  # noqa: E402
    CAMERA_NAMES_BY_SETUP,
    RAW_IMAGE_TOPICS,
    camera_names,
    command_topics,
    follower_label,
    joint_state_topics,
)
import yaml  # noqa: E402

SETUP_CAMERA_NAMES: dict[str, tuple[str, ...]] = dict(CAMERA_NAMES_BY_SETUP)

COMPRESSED_IMAGE_TOPICS: dict[str, str] = {
    name: f"{topic}/compressed" for name, topic in RAW_IMAGE_TOPICS.items()
}


def image_topics(setup: str, *, compressed: bool) -> dict[str, str]:
    """Return canonical camera-name to ROS-topic mappings for a setup."""
    available = COMPRESSED_IMAGE_TOPICS if compressed else RAW_IMAGE_TOPICS
    return {name: available[name] for name in camera_names(setup)}


def recorded_topics(bag_dir: Path) -> set[str]:
    """Read topic names from a rosbag2 episode's metadata file."""
    metadata_path = bag_dir / "metadata.yaml"
    if not metadata_path.is_file():
        raise ValueError(f"Missing rosbag metadata: {metadata_path}")

    raw = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    info = raw.get("rosbag2_bagfile_information", {})
    entries = info.get("topics_with_message_count", [])
    topics = {
        entry.get("topic_metadata", {}).get("name")
        for entry in entries
        if isinstance(entry, dict)
    }
    return {topic for topic in topics if isinstance(topic, str) and topic}


def detect_recorded_setup(bag_dir: Path) -> str:
    """Detect a complete canonical setup and reject partial/extra camera sets."""
    topics = recorded_topics(bag_dir)
    present = topics.intersection(COMPRESSED_IMAGE_TOPICS.values())

    matches = [
        setup
        for setup in SETUP_CAMERA_NAMES
        if present == set(image_topics(setup, compressed=True).values())
    ]
    if len(matches) == 1:
        return matches[0]

    raise ValueError(
        f"{bag_dir}: camera topics do not form a complete canonical setup; "
        f"found={sorted(present)}"
    )
