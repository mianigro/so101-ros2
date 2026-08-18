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

"""Canonical SO-101 setups shared by conversion config and preflight.

The single source of truth for the camera-name -> ROS-topic mapping and the
arm topic contract is the setup YAML shipped with ``so101_bringup``
(``so101_bringup/config/setups/*.yaml``): one file per hardware rig,
defining arms, the logical camera contract, the physical rig, and Isaac Sim
geometry. The mappings below are derived from those files at import time; do
not add topic literals to this module. The setups directory is located via,
in order:

1. ``$SO101_SETUPS_DIR``
2. a repository checkout found by walking up from this file
3. the installed ``so101_bringup`` package share (``ament_index``)
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import yaml

COMPRESSED_IMAGE_TYPE = "sensor_msgs/msg/CompressedImage"

ARM_JOINT_NAMES = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll",
    "gripper",
)

_SETUP_KEYS = ("setup", "cameras")
_CAMERA_KEYS = ("id", "feature", "image_topic")


def default_setups_dir() -> Path:
    """Locate the so101_bringup setups directory."""
    override = os.environ.get("SO101_SETUPS_DIR")
    if override:
        return Path(override).expanduser().resolve()

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "so101_bringup" / "config" / "setups"
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
                / "setups"
            )
        except Exception as exc:  # ament_index raises its own package-not-found
            raise RuntimeError(
                "Cannot locate the so101_bringup setups. Set "
                "SO101_SETUPS_DIR to <repo>/so101_bringup/config/setups."
            ) from exc

    raise RuntimeError(
        "Cannot locate the so101_bringup setups. Set "
        "SO101_SETUPS_DIR to <repo>/so101_bringup/config/setups."
    )


def _load_setup_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(key in data for key in _SETUP_KEYS):
        raise ValueError(f"{path}: not a setup config (needs {_SETUP_KEYS})")
    setup = data["setup"]
    if setup != path.stem:
        raise ValueError(f"{path}: setup key {setup!r} must match the file name")
    return data


def load_contract(
    setups_dir: str | os.PathLike[str] | None = None,
) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    """Load (setup -> camera names, camera name -> raw topic) from the YAMLs."""
    directory = Path(setups_dir) if setups_dir else default_setups_dir()
    names: dict[str, tuple[str, ...]] = {}
    topics: dict[str, str] = {}
    for path in sorted(directory.glob("*.yaml")):
        data = _load_setup_yaml(path)
        setup = data["setup"]
        entries: list[tuple[str, str]] = []
        for index, raw in enumerate(data["cameras"]["profile"]):
            if not isinstance(raw, dict) or not all(key in raw for key in _CAMERA_KEYS):
                raise ValueError(
                    f"{path}: cameras.profile[{index}] must define {_CAMERA_KEYS}"
                )
            topic = str(raw["image_topic"])
            if "{" in topic or "}" in topic:
                raise ValueError(
                    f"setup {setup!r} leaves an unresolved placeholder in "
                    f"image_topic {topic!r}"
                )
            if not topic.startswith("/"):
                raise ValueError(
                    f"setup {setup!r} image_topic must be absolute: {topic!r}"
                )
            entries.append((str(raw["feature"]), topic))
        if len(entries) != len({name for name, _ in entries}):
            raise ValueError(f"{path}: duplicate camera feature names")
        names[setup] = tuple(name for name, _ in entries)
        for name, topic in entries:
            if name in topics and topics[name] != topic:
                raise ValueError(
                    f"{path}: camera {name!r} maps to {topic!r} but an earlier "
                    f"setup mapped it to {topics[name]!r}"
                )
            topics[name] = topic
    if not names:
        raise ValueError(f"{directory}: no setups found")
    return names, topics


CAMERA_NAMES_BY_SETUP, RAW_IMAGE_TOPICS = load_contract()

CAMERA_TOPICS: dict[str, str] = {
    name: f"{topic}/compressed" for name, topic in RAW_IMAGE_TOPICS.items()
}


def _arm_namespaces(setup: str, role: str, setups_dir: Path | None = None) -> tuple[str, ...]:
    directory = setups_dir if setups_dir is not None else default_setups_dir()
    data = _load_setup_yaml(directory / f"{setup}.yaml")
    arms = data.get("arms")
    if not isinstance(arms, dict) or not isinstance(arms.get(role), list):
        raise ValueError(f"setup {setup!r} must define arms.{role}")
    return tuple(str(arm["namespace"]) for arm in arms[role])


def camera_names(setup: str) -> tuple[str, ...]:
    """Return the ordered camera names for a required setup."""
    try:
        return CAMERA_NAMES_BY_SETUP[setup]
    except KeyError as exc:
        choices = ", ".join(CAMERA_NAMES_BY_SETUP)
        raise ValueError(
            f"setup must be one of: {choices} (got {setup!r})"
        ) from exc


def camera_topics(setup: str) -> tuple[str, ...]:
    """Return the ordered compressed-image topics for a setup."""
    return tuple(CAMERA_TOPICS[name] for name in camera_names(setup))


def joint_state_topics(setup: str) -> tuple[str, ...]:
    """Return the follower joint-state topics defined by a setup."""
    return tuple(
        f"/{ns}/joint_states" for ns in _arm_namespaces(setup, "followers")
    )


def command_topics(setup: str) -> tuple[str, ...]:
    """Return the follower command topics defined by a setup."""
    return tuple(
        f"/{ns}/forward_controller/commands"
        for ns in _arm_namespaces(setup, "followers")
    )


def follower_count(setup: str) -> int:
    """Return the number of follower arms defined by a setup."""
    return len(_arm_namespaces(setup, "followers"))


def follower_label(namespace: str) -> str:
    """Dataset-label prefix for one follower ("left." for follower_left)."""
    if namespace.startswith("follower_"):
        return f"{namespace[len('follower_'):]}."
    return ""


def state_names(setup: str) -> tuple[str, ...]:
    """Return the dataset labels of the concatenated follower joint state."""
    labels: tuple[str, ...] = ()
    for namespace in _arm_namespaces(setup, "followers"):
        labels += tuple(
            f"{follower_label(namespace)}{joint}" for joint in ARM_JOINT_NAMES
        )
    return labels


def validate_setup_topics(
    setup: str,
    topic_types: Mapping[str, str],
    source: str,
) -> None:
    """Require exactly the canonical camera set and message type for a bag."""
    expected = set(camera_topics(setup))
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
            f"{source}: camera topics do not match setup {setup!r}: "
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
