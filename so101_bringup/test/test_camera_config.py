from pathlib import Path
import pytest

from so101_bringup.camera_config import (
    CameraConfigError,
    load_profile,
)


PROFILES = Path(__file__).parents[1] / "config" / "cameras" / "profiles"


def test_canonical_profiles_are_the_only_membership_source():
    single = load_profile("single_overhead", PROFILES)
    dual = load_profile("dual_overhead", PROFILES)
    assert [camera["id"] for camera in single] == ["wrist", "overhead_1"]
    assert [camera["feature"] for camera in dual] == ["wrist", "overhead_1", "overhead_2"]
    with pytest.raises(CameraConfigError, match="camera_profile must be one of"):
        load_profile("custom", PROFILES)
