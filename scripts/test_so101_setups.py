"""Tests for the visualization-tools setup facade."""

from pathlib import Path
import sys

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from so101_setups import (  # noqa: E402
    SETUP_CAMERA_NAMES,
    detect_recorded_setup,
    image_topics,
)


def _write_metadata(bag_dir: Path, topics):
    entries = [
        {"topic_metadata": {"name": topic}, "message_count": 10} for topic in topics
    ]
    content = {"rosbag2_bagfile_information": {"topics_with_message_count": entries}}
    (bag_dir / "metadata.yaml").parent.mkdir(parents=True, exist_ok=True)
    (bag_dir / "metadata.yaml").write_text(
        __import__("yaml").safe_dump(content), encoding="utf-8"
    )


def _compressed(setup):
    return list(image_topics(setup, compressed=True).values())


def test_detects_each_recorded_setup(tmp_path):
    for setup in SETUP_CAMERA_NAMES:
        bag = tmp_path / f"{setup}_episode" / "episode_000000"
        _write_metadata(bag, _compressed(setup))
        assert detect_recorded_setup(bag) == setup


def test_rejects_partial_and_mixed_camera_sets(tmp_path):
    # A lone wrist stream matches no complete canonical setup.
    bag = tmp_path / "partial_episode" / "episode_000000"
    _write_metadata(bag, _compressed("monomanual")[:1])
    with pytest.raises(ValueError, match="do not form a complete canonical setup"):
        detect_recorded_setup(bag)

    # Two of the three bimanual cameras match no complete canonical setup.
    bag = tmp_path / "missing_bimanual_episode" / "episode_000000"
    _write_metadata(bag, _compressed("bimanual")[:2])
    with pytest.raises(ValueError, match="do not form a complete canonical setup"):
        detect_recorded_setup(bag)


def test_bimanual_detection_uses_distinct_topic_set(tmp_path):
    bag = tmp_path / "bimanual_episode" / "episode_000000"
    _write_metadata(
        bag,
        _compressed("bimanual")
        + ["/follower_left/joint_states", "/follower_right/joint_states"],
    )
    assert detect_recorded_setup(bag) == "bimanual"
