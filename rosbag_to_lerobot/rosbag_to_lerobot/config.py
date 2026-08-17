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
YAML configuration loader and dataclasses for rosbag_to_lerobot
Decoder-driven design:
- msg_type selects the decoder (registered in rosbag_to_lerobot.decoders)
- YAML only carries mapping + alignment + optional decoding hints (names/shape)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from rosbag_to_lerobot.camera_profiles import (
    CAMERA_TOPICS,
    COMPRESSED_IMAGE_TYPE,
    camera_names,
    camera_topics,
)

_ALLOWED_STAMP_SRC = {"header", "bag"}
CANONICAL_FPS = 30
COMMAND_TOPIC = "/follower/forward_controller/commands"


@dataclass(frozen=True)
class FeatureSpec:
    """Specification for a single feature to extract from the bag."""

    key: str
    topic: str
    msg_type: str  # e.g. "sensor_msgs/msg/Image"

    # Timestamp source used for alignment (converter)
    stamp_src: str = "bag"  # "header" | "bag"

    # Freshness window for as-of sampling; if None, use Config.default_max_age_s
    max_age_s: Optional[float] = None

    # Optional hints used by decoders and/or feature schema
    names: Optional[List[str]] = None  # ordering for JointState/arrays
    shape: Optional[List[int]] = None  # for images: [H, W, C] (HWC)


@dataclass(frozen=True)
class Config:
    """Top-level conversion configuration."""

    camera_profile: str
    robot_type: str
    fps: int
    reference_topic: str
    task: str
    default_max_age_s: float = 0.2
    features: List[FeatureSpec] = field(default_factory=list)

    def by_topic(self) -> Dict[str, FeatureSpec]:
        return {f.topic: f for f in self.features}

    def by_key(self) -> Dict[str, FeatureSpec]:
        return {f.key: f for f in self.features}

    def reference_spec(self) -> FeatureSpec:
        for f in self.features:
            if f.topic == self.reference_topic:
                return f
        raise ValueError(
            f"reference_topic '{self.reference_topic}' is not listed in features"
        )

    def validate(self) -> None:
        expected_camera_names = camera_names(self.camera_profile)
        if not self.robot_type:
            raise ValueError("robot_type must be set")
        if self.fps != CANONICAL_FPS:
            raise ValueError(
                f"SO-101 datasets must use the canonical {CANONICAL_FPS} Hz rate "
                f"(got {self.fps})"
            )
        if self.default_max_age_s <= 0:
            raise ValueError(
                f"default_max_age_s must be > 0 (got {self.default_max_age_s})"
            )
        if not self.features:
            raise ValueError("features list is empty")

        # Unique keys
        keys = [f.key for f in self.features]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate feature.key entries found")

        # Reference topic must exist
        topics = [f.topic for f in self.features]
        if len(topics) != len(set(topics)):
            raise ValueError("Duplicate feature.topic entries found")
        if self.reference_topic not in topics:
            raise ValueError(
                f"reference_topic '{self.reference_topic}' is not listed in features"
            )
        if self.reference_topic != COMMAND_TOPIC:
            raise ValueError(
                f"reference_topic must be the absolute controller command topic "
                f"{COMMAND_TOPIC!r}"
            )

        reference = self.reference_spec()
        if (
            reference.key != "action"
            or reference.msg_type != "std_msgs/msg/Float64MultiArray"
            or reference.stamp_src != "bag"
        ):
            raise ValueError(
                "the reference feature must be the bag-timestamped absolute action "
                "Float64MultiArray"
            )

        for f in self.features:
            if f.stamp_src not in _ALLOWED_STAMP_SRC:
                raise ValueError(
                    f"Invalid stamp_src '{f.stamp_src}' for feature '{f.key}'"
                )
            if f.max_age_s is not None and f.max_age_s <= 0:
                raise ValueError(
                    f"max_age_s must be > 0 for '{f.key}' (got {f.max_age_s})"
                )

        expected_camera_keys = {
            f"observation.images.{name}" for name in expected_camera_names
        }
        actual_camera_specs = [
            f for f in self.features if f.key.startswith("observation.images.")
        ]
        if {f.key for f in actual_camera_specs} != expected_camera_keys:
            raise ValueError(
                "Image features do not match camera_profile "
                f"{self.camera_profile!r}"
            )
        if {f.topic for f in actual_camera_specs} != set(
            camera_topics(self.camera_profile)
        ):
            raise ValueError(
                "Image topics do not match camera_profile "
                f"{self.camera_profile!r}"
            )


def load_config(
    path: str | Path, camera_profile: str, joint_states_topic: Optional[str] = None
) -> Config:
    """Load a YAML configuration file and return a validated :class:`Config`.

    Args:
        path (str | Path): Filesystem path to the timing/non-camera YAML config.
        camera_profile (str): ``single_overhead`` or ``dual_overhead``.
        joint_states_topic (Optional[str]): Overrides the ``observation.state``
            feature's topic. Pass ``/follower_sim/joint_states`` when converting
            Isaac Sim follower recordings.

    Returns:
        Config: Parsed configuration object.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    required = ("robot_type", "fps", "reference_topic", "task", "features")
    for k in required:
        if k not in raw:
            raise ValueError(f"Missing required field: {k}")

    camera_names_for_profile = camera_names(camera_profile)
    raw_features = raw["features"]
    for feat in raw_features:
        key = feat.get("key", "")
        topic = feat.get("topic", "")
        if key.startswith("observation.images.") or topic in CAMERA_TOPICS.values():
            raise ValueError(
                "Timing config must not define camera features; select them with "
                "camera_profile"
            )

    if joint_states_topic is not None:
        for feat in raw_features:
            if feat.get("key") == "observation.state":
                feat["topic"] = joint_states_topic

    features = [
        FeatureSpec(
            key=f"observation.images.{name}",
            topic=CAMERA_TOPICS[name],
            msg_type=COMPRESSED_IMAGE_TYPE,
            stamp_src="bag",
            shape=[480, 640, 3],
        )
        for name in camera_names_for_profile
    ]
    features.extend(FeatureSpec(**feat) for feat in raw_features)

    cfg = Config(
        camera_profile=camera_profile,
        robot_type=raw["robot_type"],
        fps=raw["fps"],
        reference_topic=raw["reference_topic"],
        task=raw["task"],
        default_max_age_s=raw.get("default_max_age_s", 0.2),
        features=features,
    )
    cfg.validate()
    return cfg
