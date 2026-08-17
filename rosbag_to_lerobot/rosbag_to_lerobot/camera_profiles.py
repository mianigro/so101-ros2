"""Canonical SO-101 camera profiles shared by conversion config and preflight.

The single source of truth for the camera-name -> ROS-topic mapping is the
camera-profile YAML shipped with ``so101_bringup``
(``so101_bringup/config/cameras/profiles/*.yaml``). The mappings below are
derived from those files at import time; do not add topic literals to this
module. The profile directory is located via, in order:

1. ``$SO101_CAMERA_PROFILES_DIR``
2. a repository checkout found by walking up from this file
3. the installed ``so101_bringup`` package share (``ament_index``)

The wrist camera's topic is templated on the follower namespace in the YAML;
it is rendered here with the default ``follower`` namespace that recording,
conversion, and inference all use.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import yaml

DEFAULT_FOLLOWER_NAMESPACE = "follower"
COMPRESSED_IMAGE_TYPE = "sensor_msgs/msg/CompressedImage"

_PROFILE_KEYS = ("profile", "cameras")
_CAMERA_KEYS = ("id", "feature", "image_topic")


def default_profiles_dir() -> Path:
    """Locate the so101_bringup camera-profile directory."""
    override = os.environ.get("SO101_CAMERA_PROFILES_DIR")
    if override:
        return Path(override).expanduser().resolve()

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "so101_bringup" / "config" / "cameras" / "profiles"
        if candidate.is_dir():
            return candidate

    try:
        from ament_index_python.packages import get_package_share_directory
    except ImportError:
        get_package_share_directory = None

    if get_package_share_directory is not None:
        try:
            return (
                Path(get_package_share_directory("so101_bringup"))
                / "config"
                / "cameras"
                / "profiles"
            )
        except Exception as exc:  # ament_index raises its own package-not-found
            raise RuntimeError(
                "Cannot locate the so101_bringup camera profiles. Set "
                "SO101_CAMERA_PROFILES_DIR to "
                "<repo>/so101_bringup/config/cameras/profiles."
            ) from exc

    raise RuntimeError(
        "Cannot locate the so101_bringup camera profiles. Set "
        "SO101_CAMERA_PROFILES_DIR to "
        "<repo>/so101_bringup/config/cameras/profiles."
    )


def _render_topic(template: str, profile: str) -> str:
    try:
        topic = template.format(follower_namespace=DEFAULT_FOLLOWER_NAMESPACE)
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"camera profile {profile!r} has an unsupported image_topic "
            f"template {template!r}: {exc}"
        ) from exc
    if "{" in topic or "}" in topic:
        raise ValueError(
            f"camera profile {profile!r} leaves an unresolved placeholder in "
            f"image_topic {topic!r}"
        )
    if not topic.startswith("/"):
        raise ValueError(
            f"camera profile {profile!r} image_topic must be absolute: {topic!r}"
        )
    return topic


def _load_contract(profiles_dir: Path) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    names: dict[str, tuple[str, ...]] = {}
    topics: dict[str, str] = {}
    for path in sorted(profiles_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not all(key in data for key in _PROFILE_KEYS):
            raise ValueError(f"{path}: not a camera profile (needs {_PROFILE_KEYS})")
        profile = data["profile"]
        if profile != path.stem:
            raise ValueError(f"{path}: profile key {profile!r} must match the file name")
        entries: list[tuple[str, str]] = []
        for index, raw in enumerate(data["cameras"]):
            if not isinstance(raw, dict) or not all(key in raw for key in _CAMERA_KEYS):
                raise ValueError(
                    f"{path}: cameras[{index}] must define {_CAMERA_KEYS}"
                )
            entries.append(
                (str(raw["feature"]), _render_topic(str(raw["image_topic"]), profile))
            )
        if len(entries) != len({name for name, _ in entries}):
            raise ValueError(f"{path}: duplicate camera feature names")
        names[profile] = tuple(name for name, _ in entries)
        for name, topic in entries:
            if name in topics and topics[name] != topic:
                raise ValueError(
                    f"{path}: camera {name!r} maps to {topic!r} but an earlier "
                    f"profile mapped it to {topics[name]!r}"
                )
            topics[name] = topic
    if not names:
        raise ValueError(f"{profiles_dir}: no camera profiles found")
    return names, topics


def load_contract(
    profiles_dir: str | os.PathLike[str] | None = None,
) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    """Load (profile -> camera names, camera name -> raw topic) from the YAMLs."""
    return _load_contract(Path(profiles_dir) if profiles_dir else default_profiles_dir())


CAMERA_NAMES_BY_PROFILE, RAW_IMAGE_TOPICS = load_contract()

CAMERA_TOPICS: dict[str, str] = {
    name: f"{topic}/compressed" for name, topic in RAW_IMAGE_TOPICS.items()
}


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
