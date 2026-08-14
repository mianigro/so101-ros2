"""Repository-owned paths used by the SO-101 Isaac Lab project."""

from __future__ import annotations

import os
import json
from pathlib import Path


def _repository_root() -> Path:
    override = os.environ.get("SO101_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[4]


REPOSITORY_ROOT = _repository_root()
ISAACLAB_PROJECT_ROOT = REPOSITORY_ROOT / "isaaclab"
SOURCE_ASSET_DIR = ISAACLAB_PROJECT_ROOT / "assets" / "source"
GENERATED_ASSET_DIR = REPOSITORY_ROOT / "build" / "isaaclab_assets"
CAMERA_PROFILE_PATH = (
    REPOSITORY_ROOT
    / "so101_bringup"
    / "config"
    / "cameras"
    / "isaac_dual_overhead.yaml"
)
CAMERA_SUPPORT_BOTTOM_USD_PATH = (
    REPOSITORY_ROOT
    / "build"
    / "isaacsim_so101"
    / "camera_rig"
    / "cam_mount_bottom.usd"
)
CAMERA_SUPPORT_TOP_USD_PATH = (
    REPOSITORY_ROOT
    / "build"
    / "isaacsim_so101"
    / "camera_rig"
    / "cam_mount_top.usd"
)

ROBOT_USD_PATH = (
    REPOSITORY_ROOT
    / "build"
    / "isaacsim_so101"
    / "so101_follower"
    / "so101_follower.usda"
)
CUBE_STL_PATH = SOURCE_ASSET_DIR / "cube.stl"
CUP_STL_PATH = SOURCE_ASSET_DIR / "cup.stl"
CUBE_USD_PATH = GENERATED_ASSET_DIR / "cube.usda"
CUP_USD_PATH = GENERATED_ASSET_DIR / "cup.usda"
ASSET_MANIFEST_PATH = GENERATED_ASSET_DIR / "manifest.json"


def require_task_assets() -> None:
    """Fail with one actionable error when a generated task asset is unavailable."""
    missing = [
        path
        for path in (ROBOT_USD_PATH, CUBE_USD_PATH, CUP_USD_PATH, ASSET_MANIFEST_PATH)
        if not path.is_file()
    ]
    if not missing:
        return
    formatted = "\n".join(f"  - {path}" for path in missing)
    raise FileNotFoundError(
        "SO-101 Isaac Lab assets are not prepared. Missing:\n"
        f"{formatted}\n"
        "Generate the robot USD with scripts/isaac_sim_teleop.py, then run "
        "`python isaaclab/prepare_assets`."
    )


def load_asset_manifest() -> dict:
    """Load the validated geometry contract emitted by asset preparation."""
    require_task_assets()
    return json.loads(ASSET_MANIFEST_PATH.read_text(encoding="utf-8"))


def require_vision_assets() -> None:
    """Require the robot and nominal camera rig shared by every visual task."""
    missing = [
        path
        for path in (
            ROBOT_USD_PATH,
            CAMERA_PROFILE_PATH,
            CAMERA_SUPPORT_BOTTOM_USD_PATH,
            CAMERA_SUPPORT_TOP_USD_PATH,
        )
        if not path.is_file()
    ]
    if not missing:
        return
    formatted = "\n".join(f"  - {path}" for path in missing)
    raise FileNotFoundError(
        "SO-101 visual-platform assets are unavailable. Missing:\n"
        f"{formatted}\n"
        "Run scripts/isaac_sim_teleop.py once so its YAML-derived camera support "
        "assets are authored before launching the Isaac Lab vision task."
    )
