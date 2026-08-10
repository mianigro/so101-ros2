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

"""Canonical camera profiles and sensor-state helpers for inference nodes."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


CAMERA_PROFILES: dict[str, dict[str, str]] = {
    "single_overhead": {
        "wrist": "/follower/image_raw",
        "overhead_1": "/static_camera_1/image_raw",
    },
    "dual_overhead": {
        "wrist": "/follower/image_raw",
        "overhead_1": "/static_camera_1/image_raw",
        "overhead_2": "/static_camera_2/image_raw",
    },
}


def camera_topics_for_profile(camera_profile: str) -> dict[str, str]:
    """Return canonical observation names and ROS topics for a required profile."""
    try:
        topics = CAMERA_PROFILES[camera_profile]
    except KeyError as exc:
        supported = ", ".join(CAMERA_PROFILES)
        raise ValueError(
            f"camera_profile must be one of [{supported}], got {camera_profile!r}"
        ) from exc
    return dict(topics)


def build_lerobot_features(camera_profile: str) -> dict[str, dict[str, Any]]:
    """Build the LeRobot dataset feature specification for a camera profile."""
    features: dict[str, dict[str, Any]] = {
        'observation.state': {
            'dtype': 'float32',
            'shape': (6,),
            'names': [
                'shoulder_pan.pos',
                'shoulder_lift.pos',
                'elbow_flex.pos',
                'wrist_flex.pos',
                'wrist_roll.pos',
                'gripper.pos',
            ],
        }
    }
    for camera_name in camera_topics_for_profile(camera_profile):
        features[f"observation.images.{camera_name}"] = {
            "dtype": "image",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def camera_subscription_topics(camera_profile: str, use_compressed: bool) -> dict[str, str]:
    """Return the ROS subscription topic for every configured camera."""
    suffix = "/compressed" if use_compressed else ""
    return {
        name: f"{topic}{suffix}"
        for name, topic in camera_topics_for_profile(camera_profile).items()
    }


def validate_policy_input_features(
    input_features: Mapping[str, Any] | None,
    camera_profile: str,
    state_dimension: int = 6,
) -> None:
    """Require a loaded policy's input schema to exactly match the selected rig."""
    if not input_features:
        raise ValueError("Loaded policy does not declare input_features")

    expected_images = {
        f"observation.images.{name}"
        for name in camera_topics_for_profile(camera_profile)
    }
    actual_images = {
        name for name in input_features if name.startswith("observation.images.")
    }
    if actual_images != expected_images:
        missing = sorted(expected_images - actual_images)
        unexpected = sorted(actual_images - expected_images)
        raise ValueError(
            "Policy camera schema does not match "
            f"camera_profile={camera_profile!r}; missing={missing}, unexpected={unexpected}"
        )

    state_feature = input_features.get("observation.state")
    if state_feature is None:
        raise ValueError("Policy input schema is missing observation.state")
    state_shape = (
        state_feature.get("shape")
        if isinstance(state_feature, Mapping)
        else getattr(state_feature, "shape", None)
    )
    if tuple(state_shape or ()) != (state_dimension,):
        raise ValueError(
            "Policy observation.state shape does not match the SO-101 arm; "
            f"expected ({state_dimension},), got {state_shape}"
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
