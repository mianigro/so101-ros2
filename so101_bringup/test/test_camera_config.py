from pathlib import Path
import pytest
import yaml

import so101_bringup.camera_config as camera_config
from so101_bringup.camera_config import (
    CameraConfigError,
    evaluate_camera_streams,
    intrinsics_are_valid,
    load_camera_setup,
    load_profile,
)


PROFILES = Path(__file__).parents[1] / "config" / "cameras" / "profiles"


def _calibration(tmp_path: Path, name: str) -> str:
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump({
        "image_width": 640,
        "image_height": 480,
        "camera_name": name,
        "camera_matrix": {"rows": 3, "cols": 3, "data": [600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0]},
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {"rows": 1, "cols": 5, "data": [0.0] * 5},
        "rectification_matrix": {"rows": 3, "cols": 3, "data": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]},
        "projection_matrix": {"rows": 3, "cols": 4, "data": [600.0, 0.0, 320.0, 0.0, 0.0, 600.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]},
    }), encoding="utf-8")
    return path.as_uri()


def _rig(tmp_path: Path, *, bad_quaternion: bool = False) -> Path:
    cameras = {}
    for camera_id in ("wrist", "overhead_1", "overhead_2"):
        camera = {
            "device": f"/dev/{camera_id}",
            "camera_info_url": _calibration(tmp_path, camera_id),
        }
        if camera_id.startswith("overhead_"):
            camera["transform"] = {
                "parent_frame": "base_link",
                "translation": [0.1, 0.2, 0.3],
                "rotation_xyzw": [0.0, 0.0, 0.0, 2.0 if bad_quaternion else 1.0],
            }
        cameras[camera_id] = camera
    path = tmp_path / "rig.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 1, "cameras": cameras}), encoding="utf-8")
    return path


@pytest.fixture
def character_devices(monkeypatch):
    monkeypatch.setattr(camera_config, "_validate_device", lambda _path, _camera_id: None)


def test_canonical_profiles_are_the_only_membership_source():
    single = load_profile("single_overhead", PROFILES)
    dual = load_profile("dual_overhead", PROFILES)
    assert [camera["id"] for camera in single] == ["wrist", "overhead_1"]
    assert [camera["feature"] for camera in dual] == ["wrist", "overhead_1", "overhead_2"]
    with pytest.raises(CameraConfigError, match="camera_profile must be one of"):
        load_profile("custom", PROFILES)


def test_dual_profile_resolves_topics_frames_and_real_calibration(tmp_path, character_devices):
    cameras = load_camera_setup(
        "dual_overhead", PROFILES, _rig(tmp_path), "robot", "robot/", validate_devices=True
    )
    assert [camera.image_topic for camera in cameras] == [
        "/robot/image_raw", "/static_camera_1/image_raw", "/static_camera_2/image_raw"
    ]
    assert cameras[1].frame_id == "robot/static_camera_1_optical_frame"
    assert cameras[1].transform["parent_frame"] == "robot/base_link"
    assert all(camera.camera_info_url.startswith("file://") for camera in cameras)


def test_production_rejects_blank_calibration(tmp_path, character_devices):
    rig = _rig(tmp_path)
    data = yaml.safe_load(rig.read_text(encoding="utf-8"))
    data["cameras"]["wrist"]["camera_info_url"] = ""
    rig.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(CameraConfigError, match="wrist.camera_info_url"):
        load_camera_setup("single_overhead", PROFILES, rig, "follower", "follower/")


def test_production_rejects_non_normalized_extrinsic(tmp_path, character_devices):
    with pytest.raises(CameraConfigError, match="quaternion must be normalized"):
        load_camera_setup(
            "single_overhead", PROFILES, _rig(tmp_path, bad_quaternion=True),
            "follower", "follower/"
        )


def test_production_rejects_noncanonical_overhead_parent(tmp_path, character_devices):
    rig = _rig(tmp_path)
    data = yaml.safe_load(rig.read_text(encoding="utf-8"))
    data["cameras"]["overhead_1"]["transform"]["parent_frame"] = "world"
    rig.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(CameraConfigError, match="parent_frame must be base_link"):
        load_camera_setup(
            "single_overhead", PROFILES, rig, "follower", "follower/"
        )


def test_calibration_bootstrap_can_select_one_uncalibrated_camera(tmp_path, character_devices):
    rig = tmp_path / "bootstrap.yaml"
    rig.write_text(yaml.safe_dump({
        "schema_version": 1,
        "cameras": {"overhead_2": {"device": "/dev/overhead_2"}},
    }), encoding="utf-8")
    cameras = load_camera_setup(
        "dual_overhead", PROFILES, rig, "follower", "follower/",
        calibration_mode=True, calibration_camera_id="overhead_2",
    )
    assert [camera.camera_id for camera in cameras] == ["overhead_2"]
    assert cameras[0].camera_info_url == ""
    assert cameras[0].transform is None


def test_intrinsics_gate_rejects_zero_focal_lengths():
    assert intrinsics_are_valid(640, 480, [600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0])
    assert not intrinsics_are_valid(640, 480, [0.0] * 9)


def test_supervisor_requires_and_stales_both_stream_types():
    missing, stale = evaluate_camera_streams(
        {"wrist": 10.0}, {"wrist": None}, 10.5, 1.0, True
    )
    assert missing == ["wrist:camera_info"]
    assert stale == []

    missing, stale = evaluate_camera_streams(
        {"wrist": 10.0}, {"wrist": 8.0}, 10.5, 1.0, True
    )
    assert missing == []
    assert stale == ["wrist:camera_info"]
