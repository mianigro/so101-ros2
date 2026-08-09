"""Contract tests for canonical camera profiles and timing configs."""

from pathlib import Path

import pytest
import yaml

from rosbag_to_lerobot.camera_profiles import (
    CAMERA_TOPICS,
    COMPRESSED_IMAGE_TYPE,
    validate_profile_topics,
)
from rosbag_to_lerobot.config import load_config


CONFIG_DIR = Path(__file__).parents[1] / "config"


@pytest.mark.parametrize(
    ("config_name", "camera_profile", "fps", "reference_topic", "camera_names"),
    [
        (
            "so101_30hz.yaml",
            "single_overhead",
            30,
            "/follower/image_raw/compressed",
            ["wrist", "overhead_1"],
        ),
        (
            "so101_50hz.yaml",
            "dual_overhead",
            50,
            "/follower/forward_controller/commands",
            ["wrist", "overhead_1", "overhead_2"],
        ),
    ],
)
def test_timing_config_builds_profile_camera_schema(
    config_name, camera_profile, fps, reference_topic, camera_names
):
    cfg = load_config(CONFIG_DIR / config_name, camera_profile)

    camera_specs = [
        feature
        for feature in cfg.features
        if feature.key.startswith("observation.images.")
    ]
    assert cfg.fps == fps
    assert cfg.reference_topic == reference_topic
    assert [feature.key.rsplit(".", 1)[-1] for feature in camera_specs] == camera_names
    assert [feature.topic for feature in camera_specs] == [
        CAMERA_TOPICS[name] for name in camera_names
    ]


def test_profile_validation_rejects_extra_canonical_camera():
    topic_types = {
        topic: COMPRESSED_IMAGE_TYPE for topic in CAMERA_TOPICS.values()
    }

    with pytest.raises(ValueError, match="extra=.*static_camera_2"):
        validate_profile_topics("single_overhead", topic_types, "episode_000001")


def test_profile_validation_rejects_wrong_image_type():
    topic_types = {
        CAMERA_TOPICS["wrist"]: "sensor_msgs/msg/Image",
        CAMERA_TOPICS["overhead_1"]: COMPRESSED_IMAGE_TYPE,
    }

    with pytest.raises(ValueError, match="must use sensor_msgs/msg/CompressedImage"):
        validate_profile_topics("single_overhead", topic_types, "episode_000001")


def test_timing_config_cannot_redefine_camera_membership(tmp_path):
    raw = yaml.safe_load((CONFIG_DIR / "so101_30hz.yaml").read_text())
    raw["features"].append(
        {
            "key": "observation.images.overhead_2",
            "topic": CAMERA_TOPICS["overhead_2"],
            "msg_type": COMPRESSED_IMAGE_TYPE,
        }
    )
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="must not define camera features"):
        load_config(config_path, "single_overhead")
