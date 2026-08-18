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

"""Canonical setups and sensor-state helpers for inference nodes.

The camera-name -> topic mapping, arm topic contracts, and dataset label
convention are not defined here: they are derived from the so101_bringup
setup YAMLs via ``rosbag_to_lerobot.setups``, which is the single source of
truth for the whole repository.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from rosbag_to_lerobot.setups import (
        CAMERA_NAMES_BY_SETUP,
        RAW_IMAGE_TOPICS,
        command_topics,
        joint_state_topics,
        state_names,
    )
except ImportError:  # source checkout without the workspace built
    # Append the rosbag_to_lerobot project directory (not the repo root, which
    # would resolve rosbag_to_lerobot as a namespace package) so the regular
    # package inside it is importable.
    sys.path.append(str(Path(__file__).resolve().parents[2] / "rosbag_to_lerobot"))
    from rosbag_to_lerobot.setups import (
        CAMERA_NAMES_BY_SETUP,
        RAW_IMAGE_TOPICS,
        command_topics,
        joint_state_topics,
        state_names,
    )


CAMERA_TOPICS_BY_SETUP: dict[str, dict[str, str]] = {
    setup: {name: RAW_IMAGE_TOPICS[name] for name in names}
    for setup, names in CAMERA_NAMES_BY_SETUP.items()
}


def camera_topics_for_setup(setup: str) -> dict[str, str]:
    """Return canonical observation names and ROS topics for a required setup."""
    try:
        topics = CAMERA_TOPICS_BY_SETUP[setup]
    except KeyError as exc:
        supported = ", ".join(CAMERA_TOPICS_BY_SETUP)
        raise ValueError(
            f"setup must be one of [{supported}], got {setup!r}"
        ) from exc
    return dict(topics)


def state_dimension(setup: str) -> int:
    """Return the concatenated observation.state dimension for a setup."""
    return len(state_names(setup))


def build_lerobot_features(setup: str) -> dict[str, dict[str, Any]]:
    """Build the LeRobot dataset feature specification for a setup."""
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dimension(setup),),
            "names": [f"{label}.pos" for label in state_names(setup)],
        }
    }
    for camera_name in camera_topics_for_setup(setup):
        features[f"observation.images.{camera_name}"] = {
            "dtype": "image",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def camera_subscription_topics(setup: str, use_compressed: bool) -> dict[str, str]:
    """Return the ROS subscription topic for every configured camera."""
    suffix = "/compressed" if use_compressed else ""
    return {
        name: f"{topic}{suffix}"
        for name, topic in camera_topics_for_setup(setup).items()
    }


def validate_policy_input_features(
    input_features: Mapping[str, Any] | None,
    setup: str,
) -> None:
    """Require a loaded policy's input schema to exactly match the selected rig."""
    if not input_features:
        raise ValueError("Loaded policy does not declare input_features")

    expected_images = {
        f"observation.images.{name}"
        for name in camera_topics_for_setup(setup)
    }
    actual_images = {
        name for name in input_features if name.startswith("observation.images.")
    }
    if actual_images != expected_images:
        missing = sorted(expected_images - actual_images)
        unexpected = sorted(actual_images - expected_images)
        raise ValueError(
            f"Policy camera schema does not match setup={setup!r}; "
            f"missing={missing}, unexpected={unexpected}"
        )

    state_feature = input_features.get("observation.state")
    if state_feature is None:
        raise ValueError("Policy input schema is missing observation.state")
    state_shape = (
        state_feature.get("shape")
        if isinstance(state_feature, Mapping)
        else getattr(state_feature, "shape", None)
    )
    expected_shape = (state_dimension(setup),)
    if tuple(state_shape or ()) != expected_shape:
        raise ValueError(
            f"Policy observation.state shape does not match setup {setup!r}; "
            f"expected {expected_shape}, got {state_shape}"
        )


def streams_ready(
    stream_names: Iterable[str],
    latest_data: Mapping[str, Any],
    received_at: Mapping[str, Any],
) -> bool:
    """Return whether every named stream has both data and a receive time."""
    return all(
        latest_data.get(name) is not None and received_at.get(name) is not None
        for name in stream_names
    )


def streams_fresh(received_at: Mapping[str, Any], now: Any, max_age_s: float) -> bool:
    """Return whether every configured receive time is present and fresh."""
    return bool(received_at) and all(
        timestamp is not None and (now - timestamp).nanoseconds * 1e-9 <= max_age_s
        for timestamp in received_at.values()
    )
