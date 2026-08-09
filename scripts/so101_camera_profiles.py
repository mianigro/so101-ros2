"""Canonical camera-profile definitions for SO-101 visualization tools."""

from __future__ import annotations

from pathlib import Path

import yaml

PROFILE_CAMERA_NAMES: dict[str, tuple[str, ...]] = {
    "single_overhead": ("wrist", "overhead_1"),
    "dual_overhead": ("wrist", "overhead_1", "overhead_2"),
}

RAW_IMAGE_TOPICS: dict[str, str] = {
    "wrist": "/follower/image_raw",
    "overhead_1": "/static_camera_1/image_raw",
    "overhead_2": "/static_camera_2/image_raw",
}

COMPRESSED_IMAGE_TOPICS: dict[str, str] = {
    name: f"{topic}/compressed" for name, topic in RAW_IMAGE_TOPICS.items()
}

TF_FRAME_SUFFIXES: dict[str, str] = {
    "wrist": "wrist_camera_optical_frame",
    "overhead_1": "static_camera_1_optical_frame",
    "overhead_2": "static_camera_2_optical_frame",
}


def camera_names(camera_profile: str) -> tuple[str, ...]:
    """Return ordered canonical camera names for a profile."""
    try:
        return PROFILE_CAMERA_NAMES[camera_profile]
    except KeyError as exc:
        choices = ", ".join(PROFILE_CAMERA_NAMES)
        raise ValueError(
            f"camera_profile must be one of: {choices} (got {camera_profile!r})"
        ) from exc


def image_topics(camera_profile: str, *, compressed: bool) -> dict[str, str]:
    """Return canonical camera-name to ROS-topic mappings for a profile."""
    available = COMPRESSED_IMAGE_TOPICS if compressed else RAW_IMAGE_TOPICS
    return {name: available[name] for name in camera_names(camera_profile)}


def tf_frames(camera_profile: str, *, prefix: str = "follower/") -> dict[str, str]:
    """Return canonical camera-name to TF-frame mappings for a profile."""
    return {
        name: f"{prefix}{TF_FRAME_SUFFIXES[name]}"
        for name in camera_names(camera_profile)
    }


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


def detect_recorded_profile(bag_dir: Path) -> str:
    """Detect a complete canonical profile and reject partial/extra camera sets."""
    topics = recorded_topics(bag_dir)
    present = topics.intersection(COMPRESSED_IMAGE_TOPICS.values())

    matches = [
        profile
        for profile in PROFILE_CAMERA_NAMES
        if present == set(image_topics(profile, compressed=True).values())
    ]
    if len(matches) == 1:
        return matches[0]

    raise ValueError(
        f"{bag_dir}: camera topics do not form a complete canonical profile; "
        f"found={sorted(present)}"
    )
