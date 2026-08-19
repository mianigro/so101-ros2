# Copyright 2026 Dmitri Manajev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Success filtering + teleop mixing: the "refine" data step of the loop.

Takes the round's autonomous rollout episodes (sim npz dumps labeled by the
scripted oracle, or a real-robot converted dataset labeled by the VLM judge
in ``verdicts.jsonl``), keeps only the successes, and writes them — side by
side with the original human teleoperation episodes — into a fresh LeRobot
v3.0 dataset ready for ``lerobot-train``.  This mirrors the pi0.5 recipe:
the model's own successful recoveries enter the training set while the human
data anchors it (co-training), and failures are filtered out.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .. import contract
from ..config import RoundConfig

logger = logging.getLogger(__name__)


def rollout_features(image_height: int = contract.IMAGE_HEIGHT,
                     image_width: int = contract.IMAGE_WIDTH,
                     use_videos: bool = True) -> Dict[str, dict]:
    """Feature schema matching the rig's teleop datasets."""
    features: Dict[str, dict] = {
        contract.STATE_FEATURE_KEY: {
            "dtype": "float32",
            "shape": (contract.NUM_JOINTS,),
            "names": [f"{joint}.pos" for joint in contract.SO101_JOINT_NAMES],
        },
        contract.ACTION_FEATURE_KEY: {
            "dtype": "float32",
            "shape": (contract.NUM_JOINTS,),
            "names": [f"{joint}.pos" for joint in contract.SO101_JOINT_NAMES],
        },
    }
    for camera in contract.CAMERA_KEYS:
        features[contract.camera_feature_key(camera)] = {
            "dtype": "video" if use_videos else "image",
            "shape": (image_height, image_width, contract.IMAGE_CHANNELS),
            "names": ["height", "width", "channels"],
        }
    return features


# ---------------------------------------------------------------------------
# Episode selection
# ---------------------------------------------------------------------------

def select_sim_successes(round_dir: Path) -> List[Path]:
    """Successful sim npz episodes: manifest verdicts first, npz labels second."""
    manifest = contract.read_jsonl(Path(round_dir) / contract.ROLL_FILENAME)
    if manifest:
        files = []
        for entry in manifest:
            if entry.get("success"):
                files.append(Path(round_dir) / entry["file"])
        return files
    files = []
    for path in contract.iter_episode_files(Path(round_dir)):
        with np.load(path) as data:
            if bool(data[contract.NPZ_SUCCESS]):
                files.append(path)
    return files


def select_real_episodes(round_dir: Path) -> Dict[int, dict]:
    """Approved real-robot episodes from ``verdicts.jsonl`` (judge + human)."""
    verdicts = contract.read_jsonl(Path(round_dir) / contract.VERDICT_FILENAME)
    return {
        int(entry["episode_index"]): entry
        for entry in verdicts
        if entry.get("approved")
    }


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _to_hwc_uint8(image: Any) -> np.ndarray:
    array = image.numpy() if hasattr(image, "numpy") else np.asarray(image)
    if array.dtype != np.uint8:
        if array.dtype.kind == "f" and array.max() <= 1.0:
            array = (array * 255.0)
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))  # CHW -> HWC
    return np.ascontiguousarray(array)


def append_npz_episode(dataset, path: Path) -> int:
    """Write one successful sim npz episode into the dataset."""
    with np.load(path) as data:
        states = data[contract.NPZ_STATE]
        actions = data[contract.NPZ_ACTION]
        task = str(data[contract.NPZ_TASK])
        images = {key: data[key] for key in data.files
                  if key.startswith("observation.images.")}
    frames = states.shape[0]
    for index in range(frames):
        frame = {
            contract.STATE_FEATURE_KEY: states[index],
            contract.ACTION_FEATURE_KEY: actions[index],
            "task": task,
        }
        for key, stack in images.items():
            frame[key] = stack[index]
        dataset.add_frame(frame)
    dataset.save_episode()
    return frames


def append_dataset_episode(dataset, source, indices: Sequence[int]) -> int:
    """Copy frames ``indices`` from ``source`` (a loaded LeRobotDataset)."""
    if not indices:
        return 0
    written = 0
    for index in indices:
        item = source[index]
        episode_index = int(item["episode_index"])
        task = item.get("task")
        if task is None:
            task = source.meta.episodes[episode_index].get("task", "")
        if isinstance(task, (list, tuple)):
            task = task[0] if task else ""
        frame = {
            contract.STATE_FEATURE_KEY:
                np.asarray(item[contract.STATE_FEATURE_KEY].numpy(),
                           dtype=np.float32),
            contract.ACTION_FEATURE_KEY:
                np.asarray(item[contract.ACTION_FEATURE_KEY].numpy(),
                           dtype=np.float32),
            "task": str(task),
        }
        for camera in contract.CAMERA_KEYS:
            key = contract.camera_feature_key(camera)
            frame[key] = _to_hwc_uint8(item[key])
        dataset.add_frame(frame)
        written += 1
    dataset.save_episode()
    return written


def episodes_to_rows(source) -> Dict[int, List[int]]:
    """One pass over the source dataset: episode_index -> dataset row list."""
    rows_by_episode: Dict[int, List[int]] = {}
    for row in range(len(source)):
        episode_index = int(source.hf_dataset[row]["episode_index"])
        rows_by_episode.setdefault(episode_index, []).append(row)
    return rows_by_episode


def build_round_dataset(
    *,
    repo_id: str,
    round_dir: Path,
    teleop_repo_ids: Sequence[str],
    teleop_episode_cap: Optional[int] = None,
    min_successes: int = 1,
    root: Optional[Path] = None,
    dataset_roots: Optional[Dict[str, Path]] = None,
    real_dataset_repo_id: Optional[str] = None,
    vcodec: str = "libsvtav1",
    image_height: int = contract.IMAGE_HEIGHT,
    image_width: int = contract.IMAGE_WIDTH,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Create the round's co-training dataset.

    Sources, in order: successful sim npz episodes (or approved real episodes
    from ``real_dataset_repo_id``), then the teleop datasets.  ``root`` is the
    created dataset's directory (LeRobot semantics: the full dataset path);
    ``dataset_roots`` optionally maps source repo ids to their directories,
    defaulting to the LeRobot cache.  Returns a summary dict also written to
    ``<round_dir>/dataset_summary.json``.
    """
    from lerobot.configs import RGBEncoderConfig
    from lerobot.datasets import CODEBASE_VERSION, LeRobotDataset

    if CODEBASE_VERSION != "v3.0":
        raise RuntimeError(
            f"expected LeRobot dataset v3.0, got {CODEBASE_VERSION!r}")

    round_dir = Path(round_dir)
    dataset_roots = dataset_roots or {}
    summary: Dict[str, Any] = {
        "repo_id": repo_id,
        "sources": {},
    }

    # LeRobot's `root` is the dataset directory itself; the cache layout
    # nests it under the repo id.
    target = (Path(root) if root is not None
              else Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id)
    if target.exists():
        if not overwrite:
            raise RuntimeError(
                f"dataset {repo_id} already exists at {target}; pass "
                f"overwrite=True to replace it")
        shutil.rmtree(target)

    # --- pick the round's successful episodes ---------------------------
    if real_dataset_repo_id is not None:
        approved = select_real_episodes(round_dir)
        if len(approved) < min_successes:
            raise RuntimeError(
                f"only {len(approved)} approved real episodes "
                f"(< {min_successes}); refusing to build a dataset")
        summary["sources"]["real_rollouts"] = {
            "dataset": real_dataset_repo_id,
            "episodes": sorted(approved),
        }
        npz_successes: List[Path] = []
    else:
        npz_successes = select_sim_successes(round_dir)
        if len(npz_successes) < min_successes:
            raise RuntimeError(
                f"only {len(npz_successes)} successful sim episodes "
                f"(< {min_successes}); refusing to build a dataset")
        summary["sources"]["sim_rollouts"] = {
            "episodes": [str(path.name) for path in npz_successes]}

    # --- create the output dataset ---------------------------------------
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        # PyAV requires an integral frame rate (Fraction), not 30.0.
        fps=int(contract.CONTROL_FREQUENCY_HZ),
        robot_type="so101",
        features=rollout_features(image_height, image_width),
        use_videos=True,
        rgb_encoder=RGBEncoderConfig(vcodec=vcodec),
    )

    total_frames = 0
    # --- sim rollout successes ------------------------------------------
    for path in npz_successes:
        total_frames += append_npz_episode(dataset, path)
        logger.info("appended sim episode %s", path.name)
    if npz_successes:
        summary["sources"]["sim_rollouts"]["frames"] = total_frames

    # --- approved real rollout episodes ----------------------------------
    if real_dataset_repo_id is not None:
        real_frames = 0
        source = LeRobotDataset(
            real_dataset_repo_id,
            root=dataset_roots.get(real_dataset_repo_id),
            return_uint8=True,
        )
        rows_by_episode = episodes_to_rows(source)
        for episode_index in sorted(
                summary["sources"]["real_rollouts"]["episodes"]):
            real_frames += append_dataset_episode(
                dataset, source, rows_by_episode.get(episode_index, []))
        summary["sources"]["real_rollouts"]["frames"] = real_frames
        total_frames += real_frames

    # --- teleop anchor data -----------------------------------------------
    teleop_frames = 0
    teleop_episodes = 0
    for teleop_repo_id in teleop_repo_ids:
        source = LeRobotDataset(
            teleop_repo_id,
            root=dataset_roots.get(teleop_repo_id),
            return_uint8=True,
        )
        rows_by_episode = episodes_to_rows(source)
        episodes = list(range(len(source.meta.episodes)))
        if teleop_episode_cap is not None:
            episodes = episodes[:teleop_episode_cap]
        for episode_index in episodes:
            added = append_dataset_episode(
                dataset, source, rows_by_episode.get(episode_index, []))
            if added:
                teleop_episodes += 1
                teleop_frames += added
        logger.info("appended %d episodes from %s", len(episodes),
                    teleop_repo_id)
    summary["sources"]["teleop"] = {
        "repo_ids": list(teleop_repo_ids),
        "episodes": teleop_episodes,
        "frames": teleop_frames,
    }
    total_frames += teleop_frames
    dataset.finalize()

    summary["total_frames"] = total_frames
    summary_path = round_dir / "dataset_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("mixed dataset %s ready: %s", repo_id,
                json.dumps(summary["sources"]))
    return summary


def build_from_config(config: RoundConfig, rounds_root: Path,
                      *, root: Optional[Path] = None,
                      dataset_roots: Optional[Dict[str, Path]] = None,
                      real_dataset_repo_id: Optional[str] = None,
                      overwrite: bool = False) -> Dict[str, Any]:
    return build_round_dataset(
        repo_id=config.mixed_repo_id(),
        round_dir=config.round_dir(rounds_root),
        teleop_repo_ids=config.dataset.teleop_repo_ids,
        teleop_episode_cap=config.dataset.teleop_episode_cap,
        min_successes=config.judge.min_successes,
        root=root,
        dataset_roots=dataset_roots,
        real_dataset_repo_id=real_dataset_repo_id,
        overwrite=overwrite,
    )
