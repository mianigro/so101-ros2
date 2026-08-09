"""Canonical SO-101 camera profiles shared by conversion config and preflight."""

from __future__ import annotations

from collections.abc import Mapping

CAMERA_NAMES_BY_PROFILE: dict[str, tuple[str, ...]] = {
    "single_overhead": ("wrist", "overhead_1"),
    "dual_overhead": ("wrist", "overhead_1", "overhead_2"),
}

CAMERA_TOPICS: dict[str, str] = {
    "wrist": "/follower/image_raw/compressed",
    "overhead_1": "/static_camera_1/image_raw/compressed",
    "overhead_2": "/static_camera_2/image_raw/compressed",
}

COMPRESSED_IMAGE_TYPE = "sensor_msgs/msg/CompressedImage"


def camera_names(camera_profile: str) -> tuple[str, ...]:
    """Return the ordered camera names for a required profile."""
    try:
        return CAMERA_NAMES_BY_PROFILE[camera_profile]
    except KeyError as exc:
        choices = ", ".join(CAMERA_NAMES_BY_PROFILE)
        raise ValueError(
            f"camera_profile must be one of: {choices} (got {camera_profile!r})"
        ) from exc


def camera_topics(camera_profile: str) -> tuple[str, ...]:
    """Return the ordered compressed-image topics for a profile."""
    return tuple(CAMERA_TOPICS[name] for name in camera_names(camera_profile))


def validate_profile_topics(
    camera_profile: str,
    topic_types: Mapping[str, str],
    source: str,
) -> None:
    """Require exactly the canonical camera set and message type for a bag."""
    expected = set(camera_topics(camera_profile))
    present = set(topic_types).intersection(CAMERA_TOPICS.values())
    if present != expected:
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError(
            f"{source}: camera topics do not match profile {camera_profile!r}: "
            + ", ".join(details)
        )

    wrong_types = {
        topic: topic_types[topic]
        for topic in expected
        if topic_types[topic] != COMPRESSED_IMAGE_TYPE
    }
    if wrong_types:
        raise ValueError(
            f"{source}: canonical camera topics must use {COMPRESSED_IMAGE_TYPE}: "
            f"{wrong_types}"
        )
