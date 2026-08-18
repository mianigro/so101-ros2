"""Tests for per-episode observation.state topic auto-resolution."""

from pathlib import Path

import pytest

from rosbag_to_lerobot.config import (
    SIM_FOLLOWER_JOINT_STATES,
    load_config,
    resolve_joint_state_topic,
)


CONFIG_DIR = Path(__file__).parents[1] / "config"

JOINT_STATE_TYPE = "sensor_msgs/msg/JointState"
PHYSICAL_TOPIC = "/follower/joint_states"
SIM_TOPIC = SIM_FOLLOWER_JOINT_STATES


def _state_topic(cfg):
    return cfg.by_key()["observation.state"].topic


def test_auto_mode_lists_physical_topic_first():
    cfg = load_config(CONFIG_DIR / "so101_30hz.yaml", "monomanual_dual_overhead", "auto")

    assert cfg.joint_state_topic_candidates == (PHYSICAL_TOPIC, SIM_TOPIC)


def test_auto_mode_picks_sim_topic_when_physical_absent():
    cfg = load_config(CONFIG_DIR / "so101_30hz.yaml", "monomanual_dual_overhead", "auto")
    topic_types = {SIM_TOPIC: JOINT_STATE_TYPE}

    resolved, topic = resolve_joint_state_topic(cfg, topic_types)

    assert topic == SIM_TOPIC
    assert _state_topic(resolved) == SIM_TOPIC
    # The shared config stays untouched; only the per-episode copy changes.
    assert _state_topic(cfg) == PHYSICAL_TOPIC
    # Decoder hints survive the topic rewrite.
    spec = resolved.by_key()["observation.state"]
    assert spec.names == cfg.by_key()["observation.state"].names
    assert spec.msg_type == JOINT_STATE_TYPE


def test_auto_mode_prefers_physical_topic_when_both_recorded():
    cfg = load_config(CONFIG_DIR / "so101_30hz.yaml", "monomanual_dual_overhead", "auto")
    topic_types = {
        PHYSICAL_TOPIC: JOINT_STATE_TYPE,
        SIM_TOPIC: JOINT_STATE_TYPE,
    }

    resolved, topic = resolve_joint_state_topic(cfg, topic_types)

    # Dual-follower recordings keep the physical follower as state source.
    assert topic == PHYSICAL_TOPIC
    assert resolved is cfg


def test_auto_mode_raises_when_no_candidate_present():
    cfg = load_config(CONFIG_DIR / "so101_30hz.yaml", "monomanual_dual_overhead", "auto")

    with pytest.raises(ValueError, match="no joint-states topic found"):
        resolve_joint_state_topic(cfg, {})


def test_auto_mode_rejects_type_mismatch_on_chosen_topic():
    cfg = load_config(CONFIG_DIR / "so101_30hz.yaml", "monomanual_dual_overhead", "auto")

    with pytest.raises(ValueError, match="type mismatch"):
        resolve_joint_state_topic(cfg, {SIM_TOPIC: "sensor_msgs/msg/Image"})


def test_auto_mode_bimanual_requires_both_followers():
    cfg = load_config(CONFIG_DIR / "bimanual_30hz.yaml", "bimanual", "auto")

    both = {
        "/follower_left/joint_states": JOINT_STATE_TYPE,
        "/follower_right/joint_states": JOINT_STATE_TYPE,
    }
    resolved, topic = resolve_joint_state_topic(cfg, both)
    assert topic == "/follower_left/joint_states"
    assert resolved is cfg

    with pytest.raises(ValueError, match="no joint-states topic found"):
        resolve_joint_state_topic(cfg, {})


def test_explicit_pin_keeps_strict_single_topic_behavior():
    cfg = load_config(
        CONFIG_DIR / "so101_30hz.yaml", "monomanual_dual_overhead", SIM_TOPIC
    )

    assert _state_topic(cfg) == SIM_TOPIC
    assert cfg.joint_state_topic_candidates is None

    # Strict mode never falls back: the resolver is a no-op even when the
    # pinned topic is missing (the converter's validation raises instead).
    resolved, topic = resolve_joint_state_topic(cfg, {})
    assert resolved is cfg
    assert topic == SIM_TOPIC


def test_explicit_pin_rejected_for_multi_arm_setups():
    with pytest.raises(ValueError, match="single follower"):
        load_config(
            CONFIG_DIR / "bimanual_30hz.yaml", "bimanual", PHYSICAL_TOPIC
        )
