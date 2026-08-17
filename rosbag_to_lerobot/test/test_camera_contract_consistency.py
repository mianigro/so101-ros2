"""Cross-source camera-contract consistency checks.

The camera-name -> topic mapping has one source of truth: the so101_bringup
camera-profile YAMLs. These tests fail if any consumer reintroduces a private
copy of the contract.
"""

from pathlib import Path
import sys

import yaml

from rosbag_to_lerobot.camera_profiles import (
    CAMERA_NAMES_BY_PROFILE,
    CAMERA_TOPICS,
    RAW_IMAGE_TOPICS,
    default_profiles_dir,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILES_DIR = REPO_ROOT / "so101_bringup" / "config" / "cameras" / "profiles"

EXPECTED_RAW_TOPICS = {
    "wrist": "/follower/image_raw",
    "overhead_1": "/static_camera_1/image_raw",
    "overhead_2": "/static_camera_2/image_raw",
}
EXPECTED_NAMES = {
    "single_overhead": ("wrist", "overhead_1"),
    "dual_overhead": ("wrist", "overhead_1", "overhead_2"),
}


def test_profiles_dir_resolves_to_bringup_checkout():
    assert default_profiles_dir() == PROFILES_DIR


def test_canonical_module_matches_profile_yamls():
    names = {}
    topics = {}
    for path in sorted(PROFILES_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["profile"] == path.stem
        entries = [(cam["feature"], cam["image_topic"]) for cam in data["cameras"]]
        names[data["profile"]] = tuple(name for name, _ in entries)
        topics.update(dict(entries))

    rendered = {
        name: template.format(follower_namespace="follower")
        for name, template in topics.items()
    }
    assert names == EXPECTED_NAMES == CAMERA_NAMES_BY_PROFILE
    assert rendered == EXPECTED_RAW_TOPICS == RAW_IMAGE_TOPICS
    assert CAMERA_TOPICS == {
        name: f"{topic}/compressed" for name, topic in EXPECTED_RAW_TOPICS.items()
    }


def test_so101_inference_camera_profiles_derive_from_canonical_module():
    sys.path.insert(0, str(REPO_ROOT / "so101_inference"))
    try:
        from so101_inference.camera_config import CAMERA_PROFILES

        assert CAMERA_PROFILES == {
            profile: {name: RAW_IMAGE_TOPICS[name] for name in names}
            for profile, names in EXPECTED_NAMES.items()
        }
    finally:
        sys.path.remove(str(REPO_ROOT / "so101_inference"))


def test_scripts_facade_derives_from_canonical_module():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import so101_camera_profiles as scripts_profiles

        assert scripts_profiles.PROFILE_CAMERA_NAMES == EXPECTED_NAMES
        assert scripts_profiles.RAW_IMAGE_TOPICS == EXPECTED_RAW_TOPICS
        assert scripts_profiles.COMPRESSED_IMAGE_TOPICS == CAMERA_TOPICS
    finally:
        sys.path.remove(str(REPO_ROOT / "scripts"))


def test_episode_recorder_has_no_hardcoded_camera_topics():
    """The recorder must resolve camera topics from the profile YAMLs."""
    source = (
        REPO_ROOT / "episode_recorder" / "src" / "episode_recorder.cpp"
    ).read_text(encoding="utf-8")
    for literal in (
        "/follower/image_raw",
        "/static_camera_1",
        "/static_camera_2",
    ):
        assert literal not in source, (
            f"episode_recorder.cpp hardcodes camera topic {literal!r}; camera "
            "topics must come from the so101_bringup camera-profile YAMLs"
        )
