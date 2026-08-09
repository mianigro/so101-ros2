"""Tests for profile detection used by offline and replay viewers."""

from pathlib import Path

import pytest
import yaml

from so101_camera_profiles import (
    COMPRESSED_IMAGE_TOPICS,
    detect_recorded_profile,
    tf_frames,
)


def _write_metadata(bag_dir: Path, topics: list[str]) -> None:
    bag_dir.mkdir()
    metadata = {
        "rosbag2_bagfile_information": {
            "topics_with_message_count": [
                {"topic_metadata": {"name": topic}} for topic in topics
            ]
        }
    }
    (bag_dir / "metadata.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")


def test_detects_both_complete_profiles(tmp_path):
    single = tmp_path / "single"
    dual = tmp_path / "dual"
    _write_metadata(
        single,
        [
            COMPRESSED_IMAGE_TOPICS["wrist"],
            COMPRESSED_IMAGE_TOPICS["overhead_1"],
        ],
    )
    _write_metadata(dual, list(COMPRESSED_IMAGE_TOPICS.values()))

    assert detect_recorded_profile(single) == "single_overhead"
    assert detect_recorded_profile(dual) == "dual_overhead"


def test_rejects_partial_canonical_profile(tmp_path):
    partial = tmp_path / "partial"
    _write_metadata(partial, [COMPRESSED_IMAGE_TOPICS["wrist"]])

    with pytest.raises(ValueError, match="complete canonical profile"):
        detect_recorded_profile(partial)


def test_dual_profile_uses_published_optical_frames():
    assert tf_frames("dual_overhead") == {
        "wrist": "follower/wrist_camera_optical_frame",
        "overhead_1": "follower/static_camera_1_optical_frame",
        "overhead_2": "follower/static_camera_2_optical_frame",
    }
