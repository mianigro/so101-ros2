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
- A feature may list several topics (multi-arm setups): the decoded vectors are
  concatenated in listed order into one dataset column.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from rosbag_to_lerobot.setups import (
    CAMERA_TOPICS,
    COMPRESSED_IMAGE_TYPE,
    camera_names,
    camera_topics,
    command_topics,
    follower_count,
    joint_state_topics,
)

_ALLOWED_STAMP_SRC = {"header", "bag"}
CANONICAL_FPS = 30
SIM_FOLLOWER_JOINT_STATES = "/follower_sim/joint_states"
JOINT_STATE_AUTO = "auto"


@dataclass(frozen=True)
class FeatureSpec:
    """Specification for a single feature to extract from the bag."""

    key: str
    topics: Tuple[str, ...]
    msg_type: str  # e.g. "sensor_msgs/msg/Image"

    # Timestamp source used for alignment (converter)
    stamp_src: str = "bag"  # "header" | "bag"

    # Freshness window for as-of sampling; if None, use Config.default_max_age_s
    max_age_s: Optional[float] = None

    # Optional hints used by decoders and/or feature schema
    names: Optional[List[str]] = None  # ordering for JointState/arrays
    shape: Optional[List[int]] = None  # for images: [H, W, C] (HWC)

    @property
    def topic(self) -> str:
        return self.topics[0]

    def part_spec(self, index: int) -> "FeatureSpec":
        """Sub-spec used to decode one topic of a feature.

        Multi-topic features drop ``names`` for decoding: each follower's
        joint_state_broadcaster publishes the canonical arm joint order, so
        parts decode positionally while the full spec's ``names`` label the
        concatenated dataset column. Shape is dropped because each part only
        carries its share of the vector.
        """
        names = self.names if len(self.topics) == 1 else None
        return replace(self, topics=(self.topics[index],), names=names, shape=None)


@dataclass(frozen=True)
class Config:
    """Top-level conversion configuration."""

    setup: str
    robot_type: str
    fps: int
    reference_topic: str
    task: str
    default_max_age_s: float = 0.2
    features: List[FeatureSpec] = field(default_factory=list)
    # None = strict mode (the configured observation.state topics must exist in
    # every bag). Set when --joint-states-topic auto lets each episode pick the
    # first candidate present in its bag (physical vs Isaac Sim follower).
    joint_state_topic_candidates: Optional[Tuple[str, ...]] = None

    def by_topic(self) -> Dict[str, FeatureSpec]:
        return {topic: f for f in self.features for topic in f.topics}

    def by_key(self) -> Dict[str, FeatureSpec]:
        return {f.key: f for f in self.features}

    def reference_spec(self) -> FeatureSpec:
        for f in self.features:
            if self.reference_topic in f.topics:
                return f
        raise ValueError(
            f"reference_topic '{self.reference_topic}' is not listed in features"
        )

    def validate(self) -> None:
        expected_camera_names = camera_names(self.setup)
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
        topics = [topic for f in self.features for topic in f.topics]
        if len(topics) != len(set(topics)):
            raise ValueError("Duplicate feature.topic entries found")
        if self.reference_topic not in topics:
            raise ValueError(
                f"reference_topic '{self.reference_topic}' is not listed in features"
            )

        setup_commands = command_topics(self.setup)
        if self.reference_topic != setup_commands[0]:
            raise ValueError(
                f"reference_topic must be the primary controller command topic "
                f"{setup_commands[0]!r} of setup {self.setup!r}"
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
        if reference.topics != setup_commands:
            raise ValueError(
                f"action topics {list(reference.topics)} do not match the command "
                f"topics of setup {self.setup!r}: {list(setup_commands)}"
            )

        state = self.by_key().get("observation.state")
        if state is None:
            raise ValueError("features must define observation.state")
        if state.topics != joint_state_topics(self.setup):
            raise ValueError(
                f"observation.state topics {list(state.topics)} do not match the "
                f"joint-state topics of setup {self.setup!r}: "
                f"{list(joint_state_topics(self.setup))}"
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
                f"Image features do not match setup {self.setup!r}"
            )
        if {f.topic for f in actual_camera_specs} != set(
            camera_topics(self.setup)
        ):
            raise ValueError(
                f"Image topics do not match setup {self.setup!r}"
            )


def _feature_topics(feat: dict) -> Tuple[str, ...]:
    key = feat.get("key", "?")
    if "topics" in feat:
        topics = tuple(str(t) for t in feat["topics"])
    elif "topic" in feat:
        topics = (str(feat["topic"]),)
    else:
        raise ValueError(f"feature '{key}' must define 'topic' or 'topics'")
    if not topics or not all(t.startswith("/") for t in topics):
        raise ValueError(
            f"feature '{key}' topics must be absolute: {list(topics)}"
        )
    return topics


def load_config(
    path: str | Path, setup: str, joint_states_topic: Optional[str] = None
) -> Config:
    """Load a YAML configuration file and return a validated :class:`Config`.

    Args:
        path (str | Path): Filesystem path to the timing/non-camera YAML config.
        setup (str): ``monomanual``, ``monomanual_dual_overhead`` or ``bimanual``.
        joint_states_topic (Optional[str]): Overrides the ``observation.state``
            feature's primary topic. Pass ``/follower_sim/joint_states`` to pin
            Isaac Sim follower recordings, or ``auto`` to resolve the topic per
            episode (physical vs Isaac Sim follower, single-arm setups only).

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

    camera_names_for_setup = camera_names(setup)
    raw_features = raw["features"]
    for feat in raw_features:
        key = feat.get("key", "")
        if key.startswith("observation.images.") or any(
            t in CAMERA_TOPICS.values() for t in _feature_topics(feat)
        ):
            raise ValueError(
                "Timing config must not define camera features; select them with "
                "the setup"
            )

    joint_state_candidates: Optional[Tuple[str, ...]] = None
    explicit_state_override: Optional[str] = None
    state_feature = next(
        (feat for feat in raw_features if feat.get("key") == "observation.state"),
        None,
    )
    if state_feature is None:
        raise ValueError("features must define observation.state")
    if joint_states_topic == JOINT_STATE_AUTO:
        primary = _feature_topics(state_feature)
        candidates = list(primary)
        if follower_count(setup) == 1:
            candidates.append(SIM_FOLLOWER_JOINT_STATES)
        joint_state_candidates = tuple(candidates)
    elif joint_states_topic is not None:
        if len(_feature_topics(state_feature)) != 1:
            raise ValueError(
                "an explicit --joint-states-topic override requires a setup with "
                "a single follower; multi-arm setups always record every "
                "follower's joint states"
            )
        explicit_state_override = joint_states_topic

    features = [
        FeatureSpec(
            key=f"observation.images.{name}",
            topics=(CAMERA_TOPICS[name],),
            msg_type=COMPRESSED_IMAGE_TYPE,
            stamp_src="bag",
            shape=[480, 640, 3],
        )
        for name in camera_names_for_setup
    ]
    features.extend(
        FeatureSpec(
            key=feat["key"],
            topics=_feature_topics(feat),
            msg_type=feat["msg_type"],
            stamp_src=feat.get("stamp_src", "bag"),
            max_age_s=feat.get("max_age_s"),
            names=feat.get("names"),
            shape=feat.get("shape"),
        )
        for feat in raw_features
    )

    cfg = Config(
        setup=setup,
        robot_type=raw["robot_type"],
        fps=raw["fps"],
        reference_topic=raw["reference_topic"],
        task=raw["task"],
        default_max_age_s=raw.get("default_max_age_s", 0.2),
        features=features,
        joint_state_topic_candidates=joint_state_candidates,
    )
    cfg.validate()
    if explicit_state_override is not None:
        # The override deliberately pins a non-canonical state topic (the
        # Isaac Sim follower), so it is applied after canonical validation.
        state_spec = cfg.by_key()["observation.state"]
        resolved = replace(state_spec, topics=(explicit_state_override,))
        cfg = replace(
            cfg,
            features=[
                resolved if f.key == "observation.state" else f for f in cfg.features
            ],
        )
    return cfg


def resolve_joint_state_topic(
    cfg: Config, topic_types: Dict[str, str]
) -> Tuple[Config, str]:
    """Resolve the ``observation.state`` topics for a single bag.

    Args:
        cfg (Config): Conversion configuration, as returned by
            :func:`load_config`.
        topic_types (Dict[str, str]): ``topic -> msg_type`` map of one bag.

    Returns:
        Tuple[Config, str]: The config to use for this bag (a copy whose
        ``observation.state`` primary topic was rewritten when the Isaac Sim
        fallback was chosen) and the resolved primary topic name.

    Raises:
        ValueError: In auto mode, when no candidate topic exists in the bag
            or the chosen one's type does not match the configured msg_type.
    """
    state_spec = cfg.by_key()["observation.state"]
    if cfg.joint_state_topic_candidates is None:
        return cfg, state_spec.topic

    if all(t in topic_types for t in state_spec.topics):
        return cfg, state_spec.topic

    chosen = next(
        (
            t
            for t in cfg.joint_state_topic_candidates
            if t in topic_types
        ),
        None,
    )
    if chosen is None:
        raise ValueError(
            "no joint-states topic found in bag: expected one of "
            f"{list(cfg.joint_state_topic_candidates)} "
            "(feature=observation.state)"
        )
    if len(state_spec.topics) != 1:
        raise ValueError(
            f"cannot substitute {chosen}: setup "
            f"{cfg.setup!r} defines multiple joint-state topics"
        )
    if topic_types[chosen] != state_spec.msg_type:
        raise ValueError(
            f"type mismatch for {chosen}: config={state_spec.msg_type}, "
            f"bag={topic_types[chosen]}"
        )

    resolved = replace(state_spec, topics=(chosen,))
    features = [
        resolved if f.key == "observation.state" else f for f in cfg.features
    ]
    return replace(cfg, features=features), chosen
