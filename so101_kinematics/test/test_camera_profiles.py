"""Tests for the Viser camera-profile contract."""

import pytest

from so101_kinematics.camera_profiles import camera_topics


def test_camera_profiles_have_exact_canonical_membership():
    assert list(camera_topics("single_overhead")) == ["wrist", "overhead_1"]
    assert list(camera_topics("dual_overhead")) == [
        "wrist",
        "overhead_1",
        "overhead_2",
    ]


def test_unknown_camera_profile_is_rejected():
    with pytest.raises(ValueError, match="camera_profile"):
        camera_topics("overhead")
