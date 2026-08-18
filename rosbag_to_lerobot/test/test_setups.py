"""Contract tests for canonical setups and timing configs."""

from pathlib import Path

import pytest
import yaml

from rosbag_to_lerobot.buffers import LastBuffer
from rosbag_to_lerobot.setups import (
    CAMERA_TOPICS,
    COMPRESSED_IMAGE_TYPE,
    command_topics,
    validate_setup_topics,
)
from rosbag_to_lerobot.config import load_config
from rosbag_to_lerobot.converter import _validate_reference_cadence
from rosbag_to_lerobot.decoders import decode


CONFIG_DIR = Path(__file__).parents[1] / "config"


def test_timing_config_builds_command_driven_30hz_camera_schema():
    cfg = load_config(CONFIG_DIR / "so101_30hz.yaml", "monomanual_dual_overhead")

    camera_specs = [
        feature
        for feature in cfg.features
        if feature.key.startswith("observation.images.")
    ]
    assert cfg.fps == 30
    assert cfg.reference_topic == command_topics("monomanual_dual_overhead")[0]
    assert [feature.key.rsplit(".", 1)[-1] for feature in camera_specs] == [
        "wrist",
        "overhead_1",
        "overhead_2",
    ]
    assert [feature.topic for feature in camera_specs] == [
        CAMERA_TOPICS[name] for name in ("wrist", "overhead_1", "overhead_2")
    ]


def test_bimanual_timing_config_concatenates_both_followers():
    cfg = load_config(CONFIG_DIR / "bimanual_30hz.yaml", "bimanual")

    assert cfg.reference_topic == "/follower_left/forward_controller/commands"
    state = cfg.by_key()["observation.state"]
    action = cfg.by_key()["action"]
    assert state.topics == (
        "/follower_left/joint_states",
        "/follower_right/joint_states",
    )
    assert action.topics == command_topics("bimanual")
    assert len(state.names) == 12
    assert state.names[0] == "left.shoulder_pan"
    assert state.names[6] == "right.shoulder_pan"


def test_bimanual_multi_topic_features_decode_positionally_per_part():
    cfg = load_config(CONFIG_DIR / "bimanual_30hz.yaml", "bimanual")
    action = cfg.by_key()["action"]

    left = decode(
        type("Message", (), {"data": [0.1, -0.2, 0.3, -0.4, 0.5, 1.2]})(),
        action.part_spec(0),
    )
    assert left.tolist() == pytest.approx([0.1, -0.2, 0.3, -0.4, 0.5, 1.2])


def test_timing_config_rejects_noncanonical_rate_or_reference(tmp_path):
    raw = yaml.safe_load((CONFIG_DIR / "so101_30hz.yaml").read_text())
    raw["fps"] = 50
    config_path = tmp_path / "invalid_rate.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical 30 Hz"):
        load_config(config_path, "monomanual")

    raw["fps"] = 30
    raw["reference_topic"] = CAMERA_TOPICS["wrist"]
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="primary controller command topic"):
        load_config(config_path, "monomanual")


def test_bimanual_config_rejects_mono_action_topics(tmp_path):
    raw = yaml.safe_load((CONFIG_DIR / "bimanual_30hz.yaml").read_text())
    raw["features"][1]["topics"] = ["/follower/forward_controller/commands"]
    config_path = tmp_path / "mono_action.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(
        ValueError, match="do not match the command topics|is not listed in features"
    ):
        load_config(config_path, "bimanual")


def test_reference_cadence_accepts_30hz_and_rejects_20hz():
    period_30_ns = round(1.0e9 / 30.0)
    _validate_reference_cadence(
        [index * period_30_ns for index in range(10)], 30, "30hz"
    )
    with pytest.raises(ValueError, match="20.00 Hz average"):
        _validate_reference_cadence(
            [index * 50_000_000 for index in range(10)], 30, "20hz"
        )
    irregular_30hz_median = [0, 33_333_333, 66_666_666, 200_000_000, 233_333_333]
    with pytest.raises(ValueError, match="average"):
        _validate_reference_cadence(irregular_30hz_median, 30, "dropped_commands")


def test_absolute_action_decoder_preserves_controller_targets():
    cfg = load_config(CONFIG_DIR / "so101_30hz.yaml", "monomanual")
    action_spec = cfg.reference_spec()
    message = type("Message", (), {"data": [0.1, -0.2, 0.3, -0.4, 0.5, 1.2]})()
    assert decode(message, action_spec).tolist() == pytest.approx(message.data)


def test_asof_sampling_rejects_future_observations():
    buffer = LastBuffer(max_age_ns=50_000_000)
    buffer.push(110_000_000, [0.1, 0.2, 0.3])
    assert buffer.asof(100_000_000) is None
    assert buffer.summary()["miss_future"] == 1


def test_setup_validation_rejects_extra_canonical_camera():
    topic_types = {
        topic: COMPRESSED_IMAGE_TYPE for topic in CAMERA_TOPICS.values()
    }

    with pytest.raises(ValueError, match="extra=.*static_camera_2"):
        validate_setup_topics("monomanual", topic_types, "episode_000001")


def test_setup_validation_rejects_wrong_image_type():
    topic_types = {
        CAMERA_TOPICS["wrist"]: "sensor_msgs/msg/Image",
        CAMERA_TOPICS["overhead_1"]: COMPRESSED_IMAGE_TYPE,
    }

    with pytest.raises(ValueError, match="must use sensor_msgs/msg/CompressedImage"):
        validate_setup_topics("monomanual", topic_types, "episode_000001")


def test_setup_validation_accepts_bimanual_camera_set():
    topic_types = {
        CAMERA_TOPICS[name]: COMPRESSED_IMAGE_TYPE
        for name in ("wrist_left", "wrist_right", "overhead_1")
    }
    validate_setup_topics("bimanual", topic_types, "episode_000001")


def test_dataset_tags_distinguish_teleop_and_ppo_sources():
    from rosbag_to_lerobot.converter import DATASET_TAGS_BY_SOURCE

    teleop = DATASET_TAGS_BY_SOURCE["teleop"]
    assert "teleoperation" in teleop
    assert "imitation-learning" in teleop

    ppo = DATASET_TAGS_BY_SOURCE["ppo"]
    assert "reinforcement-learning" in ppo
    assert "teleoperation" not in ppo
    assert "imitation-learning" not in ppo
    assert teleop is not ppo


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
        load_config(config_path, "monomanual")
