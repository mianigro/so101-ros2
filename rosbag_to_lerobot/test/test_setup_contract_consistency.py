"""Cross-source setup-contract consistency checks.

The camera-name -> topic mapping, arm topic contract, and dataset label
convention have one source of truth: the so101_bringup setup YAMLs. These
tests fail if any consumer reintroduces a private copy of the contract.
"""

from pathlib import Path
import sys

import yaml

from rosbag_to_lerobot.setups import (
    CAMERA_NAMES_BY_SETUP,
    CAMERA_TOPICS,
    RAW_IMAGE_TOPICS,
    command_topics,
    default_setups_dir,
    joint_state_topics,
    state_names,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SETUPS_DIR = REPO_ROOT / "so101_bringup" / "config" / "setups"

EXPECTED_RAW_TOPICS = {
    "wrist": "/follower/image_raw",
    "wrist_left": "/follower_left/image_raw",
    "wrist_right": "/follower_right/image_raw",
    "overhead_1": "/static_camera_1/image_raw",
    "overhead_2": "/static_camera_2/image_raw",
}
EXPECTED_NAMES = {
    "monomanual": ("wrist", "overhead_1"),
    "monomanual_dual_overhead": ("wrist", "overhead_1", "overhead_2"),
    "bimanual": ("wrist_left", "wrist_right", "overhead_1"),
}
EXPECTED_JOINT_STATE_TOPICS = {
    "monomanual": ("/follower/joint_states",),
    "monomanual_dual_overhead": ("/follower/joint_states",),
    "bimanual": ("/follower_left/joint_states", "/follower_right/joint_states"),
}
EXPECTED_COMMAND_TOPICS = {
    "monomanual": ("/follower/forward_controller/commands",),
    "monomanual_dual_overhead": ("/follower/forward_controller/commands",),
    "bimanual": (
        "/follower_left/forward_controller/commands",
        "/follower_right/forward_controller/commands",
    ),
}


def test_setups_dir_resolves_to_bringup_checkout():
    assert default_setups_dir() == SETUPS_DIR


def test_canonical_module_matches_setup_yamls():
    names = {}
    topics = {}
    for path in sorted(SETUPS_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["setup"] == path.stem
        entries = [
            (camera["feature"], camera["image_topic"])
            for camera in data["cameras"]["profile"]
        ]
        names[data["setup"]] = tuple(name for name, _ in entries)
        topics.update(dict(entries))
        for camera in data["cameras"]["profile"]:
            assert "{" not in camera["image_topic"]
            assert "}" not in camera["image_topic"]

    assert names == EXPECTED_NAMES == CAMERA_NAMES_BY_SETUP
    assert topics == EXPECTED_RAW_TOPICS == RAW_IMAGE_TOPICS
    assert CAMERA_TOPICS == {
        name: f"{topic}/compressed" for name, topic in EXPECTED_RAW_TOPICS.items()
    }
    for setup, expected in EXPECTED_JOINT_STATE_TOPICS.items():
        assert joint_state_topics(setup) == expected
    for setup, expected in EXPECTED_COMMAND_TOPICS.items():
        assert command_topics(setup) == expected


def test_state_names_match_dataset_label_convention():
    assert state_names("monomanual")[0] == "shoulder_pan"
    assert state_names("monomanual") == tuple(
        state_names("monomanual")
    )  # stable ordering
    bimanual = state_names("bimanual")
    assert len(bimanual) == 12
    assert bimanual[0] == "left.shoulder_pan"
    assert bimanual[6] == "right.shoulder_pan"


def test_so101_inference_camera_topics_derive_from_canonical_module():
    sys.path.insert(0, str(REPO_ROOT / "so101_inference"))
    try:
        from so101_inference.setup_config import CAMERA_TOPICS_BY_SETUP

        assert CAMERA_TOPICS_BY_SETUP == {
            setup: {name: RAW_IMAGE_TOPICS[name] for name in names}
            for setup, names in EXPECTED_NAMES.items()
        }
    finally:
        sys.path.remove(str(REPO_ROOT / "so101_inference"))


def test_scripts_facade_derives_from_canonical_module():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import so101_setups as scripts_setups

        assert scripts_setups.SETUP_CAMERA_NAMES == EXPECTED_NAMES
        assert scripts_setups.RAW_IMAGE_TOPICS == EXPECTED_RAW_TOPICS
        assert scripts_setups.COMPRESSED_IMAGE_TOPICS == CAMERA_TOPICS
    finally:
        sys.path.remove(str(REPO_ROOT / "scripts"))


def test_episode_recorder_has_no_hardcoded_camera_topics():
    """The recorder must resolve camera topics from the setup YAMLs."""
    source = (
        REPO_ROOT / "episode_recorder" / "src" / "episode_recorder.cpp"
    ).read_text(encoding="utf-8")
    for literal in (
        "/follower/image_raw",
        "/follower_left/image_raw",
        "/static_camera_1",
        "/static_camera_2",
    ):
        assert literal not in source, (
            f"episode_recorder.cpp hardcodes camera topic {literal!r}; camera "
            "topics must come from the so101_bringup setup YAMLs"
        )


def test_setup_files_each_describe_one_complete_rig():
    """Every canonical setup defines arms, matching rig ids, and (optionally) sim."""
    import pytest

    sys.path.insert(0, str(REPO_ROOT / "so101_bringup" / "so101_bringup"))
    try:
        from setup_config import SetupConfigError, load_setup

        for name in EXPECTED_NAMES:
            setup = load_setup(name, SETUPS_DIR)
            assert setup.name == name
            assert len(setup.leaders) == len(setup.followers)
            assert set(setup.rig) == {camera["id"] for camera in setup.cameras}
            if name == "bimanual":
                assert setup.sim is None
            else:
                assert setup.sim is not None
                assert set(setup.sim["cameras"]) == {
                    camera["id"] for camera in setup.cameras
                }

        with pytest.raises(SetupConfigError):
            load_setup("nonexistent_setup", SETUPS_DIR)
    finally:
        sys.path.remove(str(REPO_ROOT / "so101_bringup" / "so101_bringup"))
