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

"""Tests for canonical inference camera profiles."""

from types import SimpleNamespace

import pytest

from so101_inference.camera_config import (
    build_lerobot_features,
    camera_subscription_topics,
    camera_topics_for_profile,
    streams_fresh,
    streams_ready,
    validate_policy_input_features,
)


class _Stamp:
    """Small stand-in for a ROS time object."""

    def __init__(self, seconds):
        self.nanoseconds = int(seconds * 1e9)

    def __sub__(self, other):
        return SimpleNamespace(nanoseconds=self.nanoseconds - other.nanoseconds)


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        (
            "single_overhead",
            {
                "wrist": "/follower/image_raw",
                "overhead_1": "/static_camera_1/image_raw",
            },
        ),
        (
            "dual_overhead",
            {
                "wrist": "/follower/image_raw",
                "overhead_1": "/static_camera_1/image_raw",
                "overhead_2": "/static_camera_2/image_raw",
            },
        ),
    ],
)
def test_camera_profile_has_canonical_names_and_topics(profile, expected):
    assert camera_topics_for_profile(profile) == expected


@pytest.mark.parametrize("profile", ["", "dual", "unknown", "three_camera"])
def test_unknown_camera_profiles_are_rejected(profile):
    with pytest.raises(ValueError, match="camera_profile must be one of"):
        camera_topics_for_profile(profile)


def test_dual_profile_builds_matching_lerobot_features():
    features = build_lerobot_features("dual_overhead")

    assert list(features) == [
        "observation.state",
        "observation.images.wrist",
        "observation.images.overhead_1",
        "observation.images.overhead_2",
    ]
    assert features["observation.images.overhead_2"] == {
        "dtype": "image",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    }


def test_compressed_setting_applies_to_every_profile_topic():
    assert camera_subscription_topics("dual_overhead", use_compressed=True) == {
        "wrist": "/follower/image_raw/compressed",
        "overhead_1": "/static_camera_1/image_raw/compressed",
        "overhead_2": "/static_camera_2/image_raw/compressed",
    }


def test_policy_schema_must_exactly_match_selected_camera_profile():
    valid = {
        "observation.state": SimpleNamespace(shape=(6,)),
        "observation.images.wrist": SimpleNamespace(shape=(3, 480, 640)),
        "observation.images.overhead_1": SimpleNamespace(shape=(3, 480, 640)),
        "observation.images.overhead_2": SimpleNamespace(shape=(3, 480, 640)),
    }
    validate_policy_input_features(valid, "dual_overhead")

    noncanonical_camera_names = {
        **valid,
        "observation.images.unexpected": valid["observation.images.overhead_1"],
    }
    noncanonical_camera_names.pop("observation.images.overhead_1")
    with pytest.raises(ValueError, match="missing=.*overhead_1.*unexpected=.*unexpected"):
        validate_policy_input_features(noncanonical_camera_names, "dual_overhead")

    with pytest.raises(ValueError, match="unexpected=.*overhead_2"):
        validate_policy_input_features(valid, "single_overhead")


def test_policy_schema_requires_six_joint_state_values():
    features = {
        "observation.state": {"shape": (7,)},
        "observation.images.wrist": SimpleNamespace(shape=(3, 480, 640)),
        "observation.images.overhead_1": SimpleNamespace(shape=(3, 480, 640)),
    }
    with pytest.raises(ValueError, match=r"expected \(6,\), got \(7,\)"):
        validate_policy_input_features(features, "single_overhead")


def test_stream_readiness_requires_every_camera_and_joint_state():
    names = ["wrist", "overhead_1", "overhead_2"]
    data = {"wrist": object(), "overhead_1": object(), "overhead_2": None}
    received_at = {name: object() for name in names}

    assert not streams_ready(names, data, received_at)
    data["overhead_2"] = object()
    assert streams_ready(names, data, received_at)


def test_stream_freshness_checks_all_cameras_and_joint_state():
    now = _Stamp(10.0)
    received_at = {
        "wrist": _Stamp(9.9),
        "overhead_1": _Stamp(9.85),
        "overhead_2": _Stamp(9.81),
        "joints": _Stamp(9.9),
    }

    assert streams_fresh(received_at, now, max_age_s=0.2)
    received_at["overhead_2"] = _Stamp(9.79)
    assert not streams_fresh(received_at, now, max_age_s=0.2)
