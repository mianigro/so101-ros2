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

"""ICL data pipeline (ICL §4.4): task registry + support/query sampler.

Two dataset regimes share this module:

- Stage 1 ``lerobot/droid_1.0.1`` subset: grouping by ``task_category``
  (86 values — never the 49,630 raw task strings), DROID camera rename,
  episode-range downloads (``download-subset`` CLI — exact shard file list
  derived from ``meta/episodes``; the repo is a single ``chunk-000`` so
  chunk filtering is meaningless).
- Stage 2 own SO-101 exports (``rosbag_to_lerobot``, ``self-improve``,
  anchor): grouping by normalized episode ``task`` string with an alias map,
  ``PI05_BASE_IMAGE_KEY_MAP`` rename.

The registry is always built from what is actually ON DISK (LeRobot v3.0
metadata is ``meta/episodes/chunk-*/*.parquet`` + ``meta/tasks.parquet`` —
never global hub listings). Camera renames happen at collation time; the
published datasets are never rewritten.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import DROID_IMAGE_KEY_MAP, PI05_BASE_IMAGE_KEY_MAP

logger = logging.getLogger(__name__)

REGISTRY_VERSION = 1


def normalize_task(task: str) -> str:
    """Normalize an episode task string into a group key (ICL §6.1)."""
    return re.sub(r"\s+", " ", task.strip().lower())


# ---------------------------------------------------------------------- #
# Task registry                                                          #
# ---------------------------------------------------------------------- #


@dataclass
class DatasetSpec:
    repo_id: str
    root: str = "~/.cache/huggingface/lerobot"
    camera_rename: dict[str, str] = field(default_factory=lambda: dict(PI05_BASE_IMAGE_KEY_MAP))
    grouping_key: str = "task"  # "task" or an episode/data column name


def _episode_group_keys(spec: DatasetSpec, meta, only: set[int] | None = None) -> dict[int, str]:
    """Map episode_index -> group key, from on-disk metadata only.

    ``only`` restricts the mapping to those episode indices (the on-disk
    subset) so the data-parquet fallback never touches missing files.
    """
    eps = meta.episodes.to_pandas()
    if only is not None:
        eps = eps[eps["episode_index"].isin(only)]
    if spec.grouping_key == "task":
        groups = {}
        for _, row in eps.iterrows():
            tasks = row["tasks"] if isinstance(row["tasks"], list) else [row["tasks"]]
            groups[int(row["episode_index"])] = normalize_task(str(tasks[0]))
        return groups

    if spec.grouping_key in eps.columns:
        return {
            int(row["episode_index"]): str(row[spec.grouping_key])
            for _, row in eps.iterrows()
        }

    # Fall back to a per-frame column in the data parquet (e.g. DROID's
    # task_category may live there): first frame of each episode decides.
    # Episodes are grouped by shard first so every parquet file is read
    # ONCE — the on-disk subset can be a single multi-GB shard, and a
    # per-episode re-read of it multiplies the whole file by the episode
    # count (the registry build "hangs" for hours).
    import pyarrow.parquet as pq

    root = Path(spec.root).expanduser() / spec.repo_id
    by_path: dict[Path, list[int]] = {}
    for _, row in eps.iterrows():
        chunk = int(row["data/chunk_index"])
        file_idx = int(row["data/file_index"])
        by_path.setdefault(
            root / "data" / f"chunk-{chunk:03d}" / f"file-{file_idx:03d}.parquet",
            [],
        ).append(int(row["episode_index"]))
    keys: dict[int, str] = {}
    for path, ep_list in by_path.items():
        wanted = set(ep_list)
        table = pq.read_table(
            path, columns=["episode_index", spec.grouping_key]
        ).to_pydict()
        for e, g in zip(table["episode_index"], table[spec.grouping_key]):
            ep = int(e)
            if ep in wanted and ep not in keys:
                keys[ep] = str(g)  # first frame decides
    return keys


def _episode_task_strings(meta) -> dict[int, str]:
    """Map episode_index -> normalized task string, from episodes metadata.

    Reads the standard lerobot v3 ``tasks`` column (same source as the
    registry's ``grouping_key: task`` path). Returns ``{}`` when the column
    is missing or empty so callers fall back to group-level behavior.
    """
    try:
        eps = meta.episodes.to_pandas()
    except Exception:
        return {}
    if "tasks" not in eps.columns:
        return {}
    out: dict[int, str] = {}
    for _, row in eps.iterrows():
        tasks = row["tasks"] if isinstance(row["tasks"], list) else [row["tasks"]]
        if not tasks:
            continue
        # Some DROID rows carry stringified empty lists ("['']") — reject
        # anything without alphanumeric content, not just empty strings.
        s = str(tasks[0])
        if not any(ch.isalnum() for ch in s):
            continue
        out[int(row["episode_index"])] = normalize_task(s)
    return out


# ---------------------------------------------------------------------- #
# On-disk subset resolution (download-subset alignment)                   #
# ---------------------------------------------------------------------- #


def on_disk_episodes(repo_id: str, root: str = "~/.cache/huggingface/lerobot") -> set[int]:
    """Episode indices whose data files exist locally (and videos, when the
    episode metadata carries per-camera file indices).

    ``download-subset`` fetches the FULL ``meta/`` but only the shards of an
    episode range; lerobot's loader treats every episode listed in ``meta/``
    as required and re-downloads whatever is missing
    (``dataset_reader.try_load`` -> ``LeRobotDataset._download``). Registry
    building and dataset loads must therefore be pinned to this set.
    """
    import pyarrow.parquet as pq

    base = Path(root).expanduser() / repo_id
    episodes: set[int] = set()
    for path in sorted((base / "data").glob("**/*.parquet")):
        episodes.update(
            int(e) for e in pq.read_table(path, columns=["episode_index"])["episode_index"].to_pylist()
        )

    # Video validation only when the episode rows carry per-camera file
    # indices (hub-published subsets like droid_1.0.1); datasets created
    # locally by lerobot have videos consistent with their data by
    # construction.
    rows: dict[int, dict] = {}
    for shard in sorted((base / "meta" / "episodes").glob("*/*.parquet")):
        for row in pq.read_table(shard).to_pylist():
            rows[int(row["episode_index"])] = row
    if rows and _video_keys_of(next(iter(rows.values()))):
        info = json.loads((base / "meta" / "info.json").read_text())
        video_tpl = info.get("video_path", DEFAULT_VIDEO_PATH)
        for ep in sorted(episodes):
            row = rows.get(ep)
            if row is None:
                episodes.discard(ep)
                continue
            for key in _video_keys_of(row):
                chunk_idx = row.get(f"videos/{key}/chunk_index")
                file_idx = row.get(f"videos/{key}/file_index")
                if chunk_idx is None or file_idx is None or not (
                    base / video_tpl.format(
                        video_key=key, chunk_index=chunk_idx, file_index=file_idx
                    )
                ).exists():
                    episodes.discard(ep)
                    break
    return episodes


def open_local_dataset(repo_id: str, root: str = "~/.cache/huggingface/lerobot"):
    """Open a LeRobotDataset pinned to the on-disk episodes, hub disabled.

    Pinning ``episodes`` makes lerobot's local-sufficiency check pass
    without downloading, and ``HF_HUB_OFFLINE=1`` turns any residual hub
    call into a loud error instead of a multi-GB pull. Frame indexing stays
    valid only for a CONTIGUOUS PREFIX of episodes (0..N) — which
    ``download-subset`` ranges always produce — so anything else is
    rejected up front.
    """
    import os

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes = sorted(on_disk_episodes(repo_id, root))
    if not episodes:
        raise FileNotFoundError(
            f"no episodes with on-disk data found under "
            f"{Path(root).expanduser() / repo_id} — run "
            "`pixi run -e lerobot icl_data download-subset` first"
        )
    if episodes != list(range(len(episodes))):
        raise ValueError(
            f"on-disk episodes of {repo_id} are not a contiguous prefix "
            f"(0..{episodes[-1]}); re-download a single episode range to "
            "restore a consistent subset"
        )
    prev = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        return LeRobotDataset(
            repo_id, root=str(Path(root).expanduser() / repo_id), episodes=episodes
        )
    finally:
        if prev is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev


def build_task_registry(
    specs: list[DatasetSpec],
    *,
    alias_map: dict[str, str] | None = None,
    holdout: dict[str, int] | None = None,
    seed: int = 42,
    out_path: Path | str | None = None,
) -> dict:
    """Scan the given datasets on disk and produce the task registry.

    Holdout picks WHOLE groups (eval/test), seeded and frozen by name into
    the registry — the sampler never mixes held-out groups into training.
    Only episodes whose files are actually on disk are registered (a
    ``download-subset`` range, not the full episode list in ``meta/``).
    """
    alias_map = alias_map or {}
    holdout = holdout or {}
    registry = {
        "version": REGISTRY_VERSION,
        "datasets": [],
        "groups": {},  # group -> [[ds_idx, episode_index, length], ...]
        "splits": {},
    }
    for ds_idx, spec in enumerate(specs):
        # NOTE: lerobot's `root` replaces the whole dataset dir (must include repo_id).
        ds = open_local_dataset(spec.repo_id, spec.root)
        on_disk = set(ds.episodes)
        registry["datasets"].append(
            {
                "repo_id": spec.repo_id,
                "root": spec.root,
                "camera_rename": spec.camera_rename,
                "grouping_key": spec.grouping_key,
            }
        )
        eps = ds.meta.episodes.to_pandas()
        keys = _episode_group_keys(spec, ds.meta, only=on_disk)
        for _, row in eps.iterrows():
            ep = int(row["episode_index"])
            if ep not in on_disk:
                continue
            group = alias_map.get(keys[ep], keys[ep])
            registry["groups"].setdefault(group, []).append(
                [ds_idx, ep, int(row["length"])]
            )

    groups = sorted(registry["groups"].keys())
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    eval_n = holdout.get("eval", 0)
    test_n = holdout.get("test", 0)
    eval_groups = [groups[i] for i in order[:eval_n]]
    test_groups = [groups[i] for i in order[eval_n : eval_n + test_n]]
    train_groups = [g for i, g in enumerate(groups) if g not in eval_groups + test_groups]

    registry["splits"] = {
        "train": train_groups,
        "eval": eval_groups,
        "test": test_groups,
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(registry, indent=2))
        logger.info(
            "wrote registry %s: %d groups (train %d / eval %d / test %d)",
            out_path, len(groups), len(train_groups), len(eval_groups), len(test_groups),
        )
    return registry


def load_task_registry(path: Path | str) -> dict:
    reg = json.loads(Path(path).read_text())
    if reg.get("version") != REGISTRY_VERSION:
        raise ValueError(f"unsupported registry version {reg.get('version')}")
    return reg


# ---------------------------------------------------------------------- #
# Stats quantile check (ICL §4.4 "recompute_stats: auto")                 #
# ---------------------------------------------------------------------- #

QUANTILE_STATS_FEATURES = ("observation.state", "action")
QUANTILE_STATS_KEYS = ("q01", "q99")


def load_stats_json(repo_id: str, root: str = "~/.cache/huggingface/lerobot") -> dict:
    path = Path(root).expanduser() / repo_id / "meta" / "stats.json"
    return json.loads(path.read_text())


def missing_quantiles(stats: dict) -> list[str]:
    """Report state/action entries lacking usable q01/q99 quantiles.

    pi05 quantile normalization needs q01/q99; a droid subset that ships only
    min/max/mean/std must be recomputed before training. VISUAL features are
    IDENTITY-normalized and ignored.
    """
    problems = []
    for feature in QUANTILE_STATS_FEATURES:
        entry = stats.get(feature)
        if entry is None:
            problems.append(f"{feature}: missing entirely")
            continue
        for key in QUANTILE_STATS_KEYS:
            value = entry.get(key)
            if value is None:
                problems.append(f"{feature}: {key} is null")
    return problems


def recompute_stats_command(repo_id: str, root: str) -> list[str]:
    """The exact in-place recompute command.

    ``--root`` is the FULL dataset directory (lerobot_edit_dataset.py:382
    ``input_path = Path(root)`` — same convention as ``LeRobotDataset``);
    ``--operation.overwrite true`` avoids the default whole-tree copy and
    recomputes stats in place (handler at :674-710).
    """
    import shutil
    import sys

    if shutil.which("lerobot-edit-dataset"):
        prefix = ["lerobot-edit-dataset"]
    else:
        sibling = Path(sys.executable).parent / "lerobot-edit-dataset"
        if sibling.exists():
            prefix = [str(sibling)]
        else:
            # same-env fallback: let main() parse the args from sys.argv
            prefix = [
                sys.executable, "-c",
                "import sys; from lerobot.scripts.lerobot_edit_dataset import main; main()",
            ]
    dataset_dir = str((Path(root).expanduser() / repo_id).resolve())
    return [
        *prefix,
        # draccus parses underscores, not dashes
        f"--repo_id={repo_id}",
        f"--root={dataset_dir}",
        # Without --new_root the handler resolves the output to
        # HF_LEROBOT_HOME/<repo>_recomputed_stats and COPYTREES; pointing it
        # back at the input dir makes _is_in_place() true instead.
        f"--new_root={dataset_dir}",
        "--operation.type=recompute_stats",
        "--operation.overwrite=true",
    ]


def check_stats(specs: list[DatasetSpec], *, fix: bool = False, assume_yes: bool = False) -> int:
    """CLI body: report (and optionally fix) missing quantile stats.

    Returns 0 only when every dataset ends up with usable q01/q99 —
    including after a successful ``--fix`` recompute.
    """
    verdicts: dict[str, bool] = {}
    for spec in specs:
        try:
            stats = load_stats_json(spec.repo_id, spec.root)
        except FileNotFoundError as e:
            print(f"{spec.repo_id}: ERROR — {e}")
            verdicts[spec.repo_id] = False
            continue
        problems = missing_quantiles(stats)
        if not problems:
            print(f"{spec.repo_id}: ok (q01/q99 present for state/action)")
            verdicts[spec.repo_id] = True
            continue
        for problem in problems:
            print(f"{spec.repo_id}: MISSING — {problem}")
        cmd = recompute_stats_command(spec.repo_id, spec.root)
        verdicts[spec.repo_id] = False
        if not fix:
            print(f"  fix with: {' '.join(cmd)}")
            continue
        if not assume_yes:
            answer = input(
                f"Recompute stats IN-PLACE for {spec.repo_id} "
                "(overwrites meta/stats.json)? [y/N] "
            )
            if answer.strip().lower() != "y":
                print("  skipped")
                continue
        import subprocess

        print("  running:", " ".join(cmd))
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"  recompute FAILED (exit {result.returncode})")
            continue
        problems = missing_quantiles(load_stats_json(spec.repo_id, spec.root))
        if problems:
            print(f"  recompute ran but quantiles still missing: {problems}")
        else:
            print(f"{spec.repo_id}: ok after recompute")
            verdicts[spec.repo_id] = True
    return 0 if all(verdicts.values()) else 1


# ---------------------------------------------------------------------- #
# Stage-YAML-driven registry building                                     #
# ---------------------------------------------------------------------- #

CAMERA_RENAME_PRESETS = {
    "droid": lambda: dict(DROID_IMAGE_KEY_MAP),
    "pi05_base": lambda: dict(PI05_BASE_IMAGE_KEY_MAP),
}


def specs_from_stage_config(stage: dict) -> tuple[list[DatasetSpec], dict]:
    """Translate a stage YAML into build_task_registry inputs.

    Reads ``dataset.repo_id`` (or ``repo_ids``), ``root``, ``grouping_key``,
    ``camera_rename`` (preset name or explicit map), ``alias_map``,
    ``holdout_groups``/``holdout_tasks``, ``task_registry`` (output path) and
    ``train.seed`` — the stage configs stay the single source of truth.
    """
    dataset_cfg = stage["dataset"]
    repo_ids = dataset_cfg.get("repo_ids") or [dataset_cfg["repo_id"]]
    root = dataset_cfg.get("root", "~/.cache/huggingface/lerobot")

    rename = dataset_cfg.get("camera_rename", "pi05_base")
    if isinstance(rename, str):
        if rename not in CAMERA_RENAME_PRESETS:
            raise ValueError(
                f"unknown camera_rename preset {rename!r}; expected one of "
                f"{sorted(CAMERA_RENAME_PRESETS)} or an explicit map"
            )
        rename_map = CAMERA_RENAME_PRESETS[rename]()
    else:
        rename_map = dict(rename)

    specs = [
        DatasetSpec(
            repo_id=repo_id,
            root=root,
            camera_rename=dict(rename_map),
            grouping_key=dataset_cfg.get("grouping_key", "task"),
        )
        for repo_id in repo_ids
    ]
    holdout = dataset_cfg.get("holdout_groups") or dataset_cfg.get("holdout_tasks") or {}
    kwargs = {
        "alias_map": dataset_cfg.get("alias_map") or {},
        "holdout": {k: int(v) for k, v in holdout.items()},
        "seed": stage.get("train", {}).get("seed", 42),
        "out_path": dataset_cfg["task_registry"],
    }
    return specs, kwargs


# ---------------------------------------------------------------------- #
# Shared normalization / preprocessing                                   #
# ---------------------------------------------------------------------- #


def preprocess_demo_frames(frames_chw: torch.Tensor) -> torch.Tensor:
    """Apply the policy-camera transform to demo keyframes.

    Mirrors ``PI05Policy._preprocess_images`` exactly (resize-with-pad to
    the SigLIP resolution, [0,1] -> [-1,1]) so there is a single image code
    path for cameras and keyframes — no train/serve skew. VISUAL stats are
    IDENTITY, so dataset stats never touch this.

    Args:
        frames_chw: ``[F, 3, H, W]`` float in [0, 1].
    Returns:
        ``[F, 3, 224, 224]`` float in [-1, 1].
    """
    from lerobot.policies.common.vla_utils import resize_with_pad_torch

    x = frames_chw.permute(0, 2, 3, 1)  # [F, H, W, 3]
    if x.shape[1:3] != (224, 224):
        x = resize_with_pad_torch(x, 224, 224)
    x = x * 2.0 - 1.0
    return x.permute(0, 3, 1, 2).float()


class TrajNormalizer:
    """Quantile state/action normalization matching the pi05 processor.

    Wraps lerobot's ``NormalizerProcessorStep._apply_transform`` (pinned
    lerobot 0.6.1) with the ACTIVE dataset's stats; the DemoEncoder consumes
    already-normalized trajectories padded to ``max_state_dim +
    max_action_dim`` (zero padding in normalized space, consistent across
    stages).
    """

    def __init__(self, stats: dict, d_state: int, d_action: int,
                 max_state_dim: int = 32, max_action_dim: int = 32):
        from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
        from lerobot.processor.normalize_processor import NormalizerProcessorStep

        features = {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(d_state,)),
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(d_action,)),
        }
        norm_map = {
            FeatureType.STATE: NormalizationMode.QUANTILES,
            FeatureType.ACTION: NormalizationMode.QUANTILES,
        }
        self._step = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)
        self._FeatureType = FeatureType
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim

    @torch.no_grad()
    def __call__(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """``[S, d_state]`` + ``[S, d_action]`` -> normalized, padded ``[S, 64]``."""
        s = self._step._apply_transform(states, "observation.state", self._FeatureType.STATE)
        a = self._step._apply_transform(actions, "action", self._FeatureType.ACTION)
        s = torch.nn.functional.pad(s, (0, self.max_state_dim - s.shape[-1]))
        a = torch.nn.functional.pad(a, (0, self.max_action_dim - a.shape[-1]))
        return torch.cat([s, a], dim=-1).float()


# ---------------------------------------------------------------------- #
# Support/query sampler                                                  #
# ---------------------------------------------------------------------- #


class _DatasetBundle:
    """One wrapped LeRobotDataset with its stats, normalizer and episodes."""

    def __init__(self, spec_dict: dict, config):
        self.repo_id = spec_dict["repo_id"]
        self.camera_rename = dict(spec_dict["camera_rename"])
        self.reverse_rename = {v: k for k, v in self.camera_rename.items()}
        # Pinned to the on-disk episodes, hub disabled: a subset download
        # must never trigger lerobot's re-download of the missing files.
        self.dataset = open_local_dataset(self.repo_id, spec_dict["root"])
        info_features = self.dataset.meta.info["features"]
        self.d_state = info_features["observation.state"]["shape"][0]
        self.d_action = info_features["action"]["shape"][0]
        self.normalizer = TrajNormalizer(
            self.dataset.meta.stats, self.d_state, self.d_action,
            max_state_dim=config.max_state_dim, max_action_dim=config.max_action_dim,
        )
        eps = self.dataset.meta.episodes.to_pandas()
        self.episodes = {
            int(row["episode_index"]): (
                int(row["dataset_from_index"]),
                int(row["dataset_to_index"]),
                int(row["length"]),
            )
            for _, row in eps.iterrows()
        }
        # Column-projected, memory-mapped table for trajectory reads.
        self.traj_table = self.dataset.hf_dataset.select_columns(
            ["observation.state", "action"]
        )


def keypoint_cache_meta_path(cache_path: Path | str) -> Path:
    """Companion metadata file for a keypoint-cache npz."""
    return Path(str(cache_path) + ".meta.json")


def write_keypoint_cache_meta(
    cache_path: Path | str,
    demo_camera: str,
    frames_per_demo: int,
    resolved: dict[int, str],
) -> None:
    """Record which demo camera / frame budget a cache was built for.

    The npz itself is keyed only by ``<ds_idx>/<episode>``, so without this
    sidecar a cache built for a different ``demo_camera`` matches silently
    (only the frame count was checked).
    """
    import json

    meta = {
        "demo_camera": demo_camera,
        "frames_per_demo": int(frames_per_demo),
        "resolved": {str(k): v for k, v in resolved.items()},
    }
    keypoint_cache_meta_path(cache_path).write_text(json.dumps(meta, indent=2))


def validate_keypoint_cache_camera(
    cache_path: Path | str, demo_camera: str
) -> bool:
    """Refuse a cache built for a different demo camera.

    Returns True when camera identity was verified; warns (and returns
    False) for legacy caches with no metadata rather than failing, so
    pre-rev caches keep loading.
    """
    import json

    meta_path = keypoint_cache_meta_path(cache_path)
    if not meta_path.exists():
        logger.warning(
            "keypoint cache %s has no %s sidecar — cannot verify it was "
            "built for demo camera %r; rebuild it with `icl_data "
            "precompute-keypoints --from-config <stage.yaml>` to record "
            "camera identity",
            cache_path, meta_path.name, demo_camera,
        )
        return False
    meta = json.loads(meta_path.read_text())
    cached = meta.get("demo_camera")
    if cached != demo_camera:
        raise ValueError(
            f"keypoint cache {cache_path} was built for demo camera "
            f"{cached!r} but the stage uses {demo_camera!r} — keypoints "
            "would silently mismatch the demo frames. Rebuild it: "
            "`icl_data precompute-keypoints --from-config <stage.yaml>`"
        )
    return True


class ICLDataset(Dataset):
    """Task-grouped support/query sampling over one or more LeRobot datasets.

    Item = one query transition + a variable-size support pack:

    - query: random timestep of an episode, all cameras (renamed to policy
      keys, RAW [0,1] — the policy preprocessor owns the rest), raw state,
      ``chunk_size`` raw actions (edge-padded), task string;
    - support: ``k`` OTHER episodes sharing the query's task string
      (preferred) or task group (fallback), ``F`` keyframes each (uniform
      stride incl. first and last), trajectory downsampled to ``S`` steps,
      normalized + padded, all padded to ``k_max`` slots with a presence
      mask. Each episode is a query once per epoch and support for its
      same-task siblings on other draws — the roles alternate.
    """

    def __init__(
        self,
        registry_path: Path | str,
        config,
        *,
        split: str = "train",
        demo_camera: str = "observation.images.left_wrist_0_rgb",
        seed: int = 0,
        keypoint_cache: Path | str | None = None,
        cluster_cache: Path | str | None = None,
    ):
        self.registry = load_task_registry(registry_path)
        self.config = config
        self.split = split
        self.demo_camera = demo_camera
        self.seed = seed
        self.k_max = config.demo_encoder.k_max
        self.frames_per_demo = config.demo_encoder.frames_per_demo
        self.traj_steps = config.demo_encoder.traj_steps
        self.chunk_size = config.chunk_size
        # Allowed support sizes for the current curriculum phase; the train
        # loop updates this per epoch (ICL §6.3 k-curriculum).
        self.k_choices: list[int] = list(range(1, self.k_max + 1))

        # Keypoint features (rev 5): required when the encoder's keypoint
        # branch is enabled; optional (unused) otherwise.
        kp_cfg = config.demo_encoder.keypoints
        self._kp_cache = None
        self._kp_cache_keys: frozenset = frozenset()
        if kp_cfg.enabled:
            if keypoint_cache is None:
                raise ValueError(
                    "demo_encoder.keypoints.enabled=true but no keypoint_cache given "
                    "— run `pixi run icl_data precompute-keypoints --registry ...` "
                    "and pass dataset.keypoint_cache in the stage YAML."
                )
            cache = np.load(Path(keypoint_cache).expanduser())
            self._kp_cache = {k: cache[k] for k in cache.files}
            self._kp_cache_keys = frozenset(self._kp_cache.keys())
            if self._kp_cache:
                first = next(iter(self._kp_cache.values()))
                if first.shape[0] != self.frames_per_demo:
                    raise ValueError(
                        f"keypoint cache {keypoint_cache} holds {first.shape[0]} "
                        f"frames/demo but demo_encoder.frames_per_demo is "
                        f"{self.frames_per_demo} — rebuild it: "
                        "`icl_data precompute-keypoints --from-config <stage.yaml>`"
                    )
            validate_keypoint_cache_camera(keypoint_cache, demo_camera)

        self.bundles = [ _DatasetBundle(d, config) for d in self.registry["datasets"] ]
        group_eps = self.registry["splits"][split]
        self._index: list[tuple[int, str, int]] = []  # (ds_idx, group, query_episode)
        self._group_members: dict[str, list[tuple[int, int]]] = {}
        for group in group_eps:
            for ds_idx, ep, _length in self.registry["groups"][group]:
                self._group_members.setdefault(group, []).append((ds_idx, ep))
        for group, members in self._group_members.items():
            if len(members) < 2:
                continue  # need a support episode disjoint from the query
            for ds_idx, ep in members:
                self._index.append((ds_idx, group, ep))
        if not self._index:
            raise ValueError(
                f"split '{split}' has no groups with >= 2 episodes; registry {registry_path}"
            )
        # Demo camera resolved per bundle (dataset-side key).
        for b in self.bundles:
            key = b.reverse_rename.get(demo_camera, demo_camera)
            cam_keys = [
                k for k in b.dataset.meta.info["features"]
                if k.startswith("observation.images.")
            ]
            if key not in cam_keys:
                raise ValueError(
                    f"demo camera {demo_camera!r} (resolved to {key!r} for dataset "
                    f"{b.repo_id}) is not in the dataset's image features "
                    f"{cam_keys} — demos must come from the camera the policy "
                    "was conditioned on during training"
                )
            b.demo_camera_key = key

        # Same-task support preference (rev 6): a demo pack must show the
        # QUERY'S task, not merely its category — with language dropped on
        # half the batches the demos are the only task signal, and category
        # grouping (DROID `task_category` values are collection LOCATIONS,
        # and exact task strings are unique per episode) carries no task
        # alignment. Task key resolution order: task-cluster id
        # (`cluster_cache` sidecar from `icl_data cluster-tasks`) >
        # normalized task string > none (category-wide fallback).
        self._ep_task: dict[tuple[int, int], str] = {}
        cluster_of: dict[str, int] = {}
        if cluster_cache is not None:
            cluster_of = {
                k: int(v) for k, v in json.loads(
                    Path(cluster_cache).expanduser().read_text()
                ).items()
            }
        for ds_idx, b in enumerate(self.bundles):
            for ep, task in _episode_task_strings(b.dataset.meta).items():
                self._ep_task[(ds_idx, ep)] = task
        if cluster_of:
            hit = miss = 0
            for key, cluster in cluster_of.items():
                ds_idx_s, ep_s = key.split("/")
                ds_idx, ep = int(ds_idx_s), int(ep_s)
                if (ds_idx, ep) in self._ep_task:
                    self._ep_task[(ds_idx, ep)] = f"#{cluster}"  # "#" avoids string collisions
                    hit += 1
                else:
                    miss += 1
            logger.info(
                "task-cluster cache: %d episodes keyed by cluster, %d ignored "
                "(no task string on disk)", hit, miss,
            )

    def __len__(self):
        return len(self._index)

    @staticmethod
    def _sample_keyframe_indices(length: int, f: int) -> np.ndarray:
        """Exactly ``f`` indices spanning ``[0, length - 1]``, first/last pinned.

        Episodes shorter than ``f`` repeat timesteps instead of shrinking:
        the traj slot must stay ``[traj_steps, d]`` for the broadcast in
        ``__getitem__``, and repeated keyframes are harmless.
        """
        idx = np.linspace(0, length - 1, f).round().astype(int)
        idx[0], idx[-1] = 0, length - 1  # start and goal emphasis (ICL §4.4)
        return idx

    def _select_supports(
        self,
        members: list[tuple[int, int]],
        q_task: str | None,
        k: int,
        rng: np.random.Generator,
    ) -> list[tuple[int, int]]:
        """Pick ``k`` support episodes: same task string as the query first,
        remainder from the rest of the category (rev 6).

        ``q_task is None`` (episode has no resolvable task string) or a
        category-wide tie keeps the original uniform selection.
        """
        same = [m for m in members if q_task and self._ep_task.get(m) == q_task]
        if not same or len(same) == len(members):
            if k < len(members):
                return [members[j] for j in rng.choice(len(members), size=k, replace=False)]
            return list(members)
        rest = [m for m in members if m not in set(same)]
        take_same = min(k, len(same))
        picks = (
            list(same)
            if take_same == len(same)
            else [same[j] for j in rng.choice(len(same), size=take_same, replace=False)]
        )
        fill = k - len(picks)
        if fill:
            picks += [rest[j] for j in rng.choice(len(rest), size=fill, replace=False)]
        return picks

    def __getitem__(self, i: int) -> dict:
        rng = np.random.default_rng([self.seed, i, self._epoch])
        q_ds, group, q_ep = self._index[i]
        members = [m for m in self._group_members[group] if m != (q_ds, q_ep)]
        k = int(rng.choice(self.k_choices))
        k = min(k, len(members))
        members = self._select_supports(members, self._ep_task.get((q_ds, q_ep)), k, rng)

        demo_frames = torch.zeros(
            self.k_max, self.frames_per_demo, 3, 224, 224, dtype=torch.float32
        )
        demo_mask = torch.zeros(self.k_max, dtype=torch.bool)
        demo_traj = torch.zeros(
            self.k_max, self.traj_steps, self.config.max_state_dim + self.config.max_action_dim,
            dtype=torch.float32,
        )
        demo_traj_ok = torch.zeros(self.k_max, dtype=torch.float32)
        kp_cfg = self.config.demo_encoder.keypoints
        emit_kp = kp_cfg.enabled
        if emit_kp:
            demo_kp = torch.zeros(
                self.k_max, self.frames_per_demo, kp_cfg.n_kp, kp_cfg.kp_dim,
                dtype=torch.float32,
            )
            demo_kp_ok = torch.zeros(self.k_max, dtype=torch.float32)
        for slot, (ds_idx, ep) in enumerate(members[: self.k_max]):
            bundle = self.bundles[ds_idx]
            start, _end, length = bundle.episodes[ep]
            kf = self._sample_keyframe_indices(length, self.frames_per_demo)
            frames = torch.stack(
                [bundle.dataset[start + int(j)][bundle.demo_camera_key] for j in kf]
            )
            demo_frames[slot, : len(kf)] = preprocess_demo_frames(frames)
            demo_mask[slot] = True
            # Demos are video-only (rev 8): demo_traj stays zeros and
            # demo_traj_ok stays 0 — the encoder ignores the fields.
            if emit_kp:
                kp, kp_ok = self._load_demo_keypoints(ds_idx, ep, length, kf)
                demo_kp[slot] = kp
                demo_kp_ok[slot] = kp_ok

        qbundle = self.bundles[q_ds]
        start, _end, length = qbundle.episodes[q_ep]
        t_max = max(0, length - self.chunk_size)
        t = int(rng.integers(0, t_max + 1))
        item = qbundle.dataset[start + t]

        actions = qbundle.traj_table[start + t : start + t + self.chunk_size]["action"]
        actions = torch.as_tensor(np.asarray(actions), dtype=torch.float32)
        if actions.shape[0] < self.chunk_size:  # edge-pad short tails
            pad = actions[-1:].expand(self.chunk_size - actions.shape[0], -1)
            actions = torch.cat([actions, pad], dim=0)

        batch = {}
        for ds_key, policy_key in qbundle.camera_rename.items():
            batch[policy_key] = item[ds_key].float()
        batch["observation.state"] = item["observation.state"].float()
        batch["action"] = actions
        batch["task"] = str(item["task"])
        batch["icl.demo_frames"] = demo_frames
        batch["icl.demo_mask"] = demo_mask
        batch["icl.demo_traj"] = demo_traj
        batch["icl.demo_traj_ok"] = demo_traj_ok
        if emit_kp:
            batch["icl.demo_kp"] = demo_kp
            batch["icl.demo_kp_ok"] = demo_kp_ok
        return batch

    _epoch: int = 0

    def set_epoch(self, epoch: int) -> None:
        """Refreshes per-item RNG streams (call from the train loop)."""
        self._epoch = int(epoch)

    # ------------------------------------------------------------------ #
    # Keypoint demo features (rev 5, Keypoint Action Tokens style)        #
    # ------------------------------------------------------------------ #

    def _load_demo_keypoints(
        self, ds_idx: int, ep: int, length: int | None = None,
        kf: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, float]:
        """Cached SIFT keypoints for one demo episode: ``[F, K, kp_dim]``.

        Returns zeros + ok=0 when the episode is absent from the cache —
        the DemoEncoder masks that slot's keypoint branch; other demos in
        the pack are unaffected. When ``kf`` is given and the cache holds a
        different frame count (offline eval frames ablations), each wanted
        keyframe index is mapped to the nearest cached row via the shared
        deterministic ``_sample_keyframe_indices`` geometry — identity when
        the counts match.
        """
        kp_cfg = self.config.demo_encoder.keypoints
        key = f"{ds_idx}/{ep}"
        if self._kp_cache is None or key not in self._kp_cache_keys:
            return (
                torch.zeros(
                    self.frames_per_demo, kp_cfg.n_kp, kp_cfg.kp_dim,
                    dtype=torch.float32,
                ),
                0.0,
            )
        arr = torch.from_numpy(self._kp_cache[key].astype(np.float32, copy=False))
        if kf is not None and arr.shape[0] != self.frames_per_demo:
            if length is None:
                raise ValueError(
                    "keypoint subsampling needs the episode length "
                    "(frames_per_demo changed after cache build)"
                )
            cache_idx = self._sample_keyframe_indices(length, arr.shape[0])
            rows = np.abs(cache_idx[:, None] - np.asarray(kf)[None, :]).argmin(axis=0)
            arr = arr[rows]
        return arr, 1.0


def extract_keypoints(frames_chw: torch.Tensor, max_kp: int = 16) -> np.ndarray:
    """SIFT keypoints for demo keyframes -> ``[F, max_kp, 2 + 128 + 1]``.

    Per frame: up to ``max_kp`` keypoints by response, features = (x/w, y/h
    normalized coords, L2-normalized 128-d descriptor, 1.0 valid flag);
    absent slots are zeros (flag 0). Shared by the offline cache CLI and the
    bridge's live pack builder — one extraction path, no drift.
    """
    import cv2

    sift = cv2.SIFT_create(nfeatures=max_kp)
    f_total = frames_chw.shape[0]
    out = np.zeros((f_total, max_kp, 131), dtype=np.float32)
    for f in range(f_total):
        rgb = (frames_chw[f].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        kps, desc = sift.detectAndCompute(gray, None)
        if not kps or desc is None:
            continue
        order = np.argsort([-k.response for k in kps])[:max_kp]
        n = len(order)
        h, w = gray.shape
        coords = np.empty((n, 2), np.float32)
        for j, i in enumerate(order):
            px, py = kps[i].pt
            coords[j] = (px / max(1, w), py / max(1, h))
        d = desc[order].astype(np.float32)
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-8
        out[f, :n, 0:2] = coords
        out[f, :n, 2:130] = d
        out[f, :n, 130] = 1.0
    return out


class BurstyGroupBatchSampler:
    """Task-structured batch sampler: tasks arrive in bursts (rev 5).

    GEN-1.5 / Chan et al. 2022: in-context learning emerges when the
    training distribution is "bursty" — a few tasks dominate and recur in
    contiguous runs — rather than uniformly shuffled. Each epoch this
    sampler draws tasks with Zipfian popularity weights and emits
    ``burst_length`` consecutive query episodes of the drawn task before
    switching. Batches are task-coherent when ``burst_length == batch_size``
    (the default pairing).

    Rev 6: the burst unit is the FINEST task identity available — the
    episode's task cluster (``ICLDataset._ep_task``, from the
    ``cluster-tasks`` sidecar) when present, else the registry group. On
    DROID this matters: groups are collection locations, so group-bursts
    were kitchen-coherent but not task-coherent.

    Regenerates on every ``__iter__`` from ``dataset._epoch`` so the train
    loop's existing ``set_epoch`` call drives re-randomization; the
    per-item support/query RNG inside :class:`ICLDataset` is untouched.
    """

    def __init__(
        self,
        dataset: ICLDataset,
        batch_size: int,
        *,
        burst_length: int = 4,
        seed: int = 0,
        drop_last: bool = True,
        zipf_exponent: float = 1.0,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.burst_length = max(1, int(burst_length))
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        ep_task = getattr(dataset, "_ep_task", {})
        self.groups: dict[str, list[int]] = {}
        for i, (ds_idx, group, ep) in enumerate(dataset._index):
            task = ep_task.get((ds_idx, ep), group)
            self.groups.setdefault(task, []).append(i)
        names = sorted(self.groups)
        ranks = np.arange(1, len(names) + 1, dtype=np.float64)
        weights = 1.0 / np.power(ranks, float(zipf_exponent))
        self._group_names = names
        self._group_probs = weights / weights.sum()

    def _epoch_order(self, epoch: int) -> list[int]:
        rng = np.random.default_rng([self.seed, epoch])
        pools = {
            g: list(rng.permutation(indices))
            for g, indices in self.groups.items()
        }
        cursors = {g: 0 for g in pools}
        order: list[int] = []
        emitted: set[int] = set()
        target = len(self.dataset)
        guard = 0
        while len(emitted) < target:
            guard += 1
            if guard > 100 * target + 10_000:  # pragma: no cover - safety net
                raise RuntimeError("bursty sampler failed to cover the dataset")
            group = self._group_names[rng.choice(len(self._group_names), p=self._group_probs)]
            pool, cur = pools[group], cursors[group]
            if cur >= len(pool):
                # group exhausted this epoch: refill from its not-yet-emitted
                # members only (never re-emit an index within an epoch)
                remaining = [i for i in self.groups[group] if i not in emitted]
                if not remaining:
                    continue
                pool = list(rng.permutation(remaining))
                pools[group], cur = pool, 0
            take = min(self.burst_length, len(pool) - cur, target - len(emitted))
            chunk = pool[cur : cur + take]
            order.extend(chunk)
            emitted.update(chunk)
            cursors[group] = cur + take
        return order

    def __iter__(self):
        order = self._epoch_order(getattr(self.dataset, "_epoch", 0))
        for start in range(0, len(order), self.batch_size):
            chunk = order[start : start + self.batch_size]
            if len(chunk) < self.batch_size and self.drop_last:
                break
            yield chunk

    def __len__(self) -> int:
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)


# ---------------------------------------------------------------------- #
# CLI                                                                    #
# ---------------------------------------------------------------------- #


def _parse_dataset_tokens(tokens: list[str]) -> list[tuple[str, str]]:
    """``--datasets`` tokens -> (repo_id, root) pairs.

    Comma separates DATASETS (the README form: ``a,b,c``). A single token
    may still pin a custom root — ``repo,/abs/path``, ``repo,~/path`` or
    ``repo,rel`` (a part with no ``/`` is a root, never a repo id).
    """
    parts = [p for token in tokens for p in token.split(",") if p]
    if len(parts) == 1:
        return [(parts[0], "~/.cache/huggingface/lerobot")]
    first, second = parts[0], parts[1]
    looks_like_root = (
        not second.strip() or second.startswith(("/", "~", ".")) or "/" not in second
    )
    if len(parts) == 2 and looks_like_root:
        return [(first, second)]
    return [(p, "~/.cache/huggingface/lerobot") for p in parts]


def _cmd_build_registry(args) -> int:
    if not args.from_config and not args.datasets:
        raise SystemExit(
            "build-registry: either --from-config <stage.yaml> or --datasets <repo[,root]> "
            "(with --out) is required"
        )
    if args.from_config:
        import yaml

        stage = yaml.safe_load(Path(args.from_config).read_text())
        specs, kwargs = specs_from_stage_config(stage)
        build_task_registry(specs, **kwargs)
        return 0

    if not args.out:
        raise SystemExit("build-registry: --out is required with --datasets")
    specs = [
        DatasetSpec(
            repo_id=repo_id,
            root=root,
            camera_rename=(
                DROID_IMAGE_KEY_MAP if args.droid else dict(PI05_BASE_IMAGE_KEY_MAP)
            ),
            grouping_key=args.grouping_key,
        )
        for repo_id, root in _parse_dataset_tokens(args.datasets)
    ]
    alias = json.loads(args.alias_map) if args.alias_map else {}
    build_task_registry(
        specs,
        alias_map=alias,
        holdout={"eval": args.holdout_eval, "test": args.holdout_test},
        seed=args.seed,
        out_path=args.out,
    )
    return 0


def _cmd_check_stats(args) -> int:
    specs = [
        DatasetSpec(repo_id=repo_id, root=root)
        for repo_id, root in _parse_dataset_tokens(args.datasets)
    ]
    return check_stats(specs, fix=args.fix, assume_yes=args.yes)


# ---------------------------------------------------------------------- #
# Task-intent clustering (rev 6: NLP grouping for stage-1 sampling)      #
# ---------------------------------------------------------------------- #

CLUSTER_MODEL = "BAAI/bge-large-en-v1.5"  # already in the HF cache
CLUSTER_THRESHOLD = 0.85                  # min cosine sim for same-cluster


def _embed_strings(strings: list[str], model_name: str = CLUSTER_MODEL,
                   batch_size: int = 256) -> "np.ndarray":
    """L2-normalized CLS embeddings for task strings (bge via transformers).

    Uses the HF-cached model — no new dependency beyond transformers, and
    CUDA when available (a few thousand strings embed in seconds).
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    out = np.zeros((len(strings), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(strings), batch_size):
            batch = strings[start : start + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=64,
                      return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state[:, 0]  # CLS pooling (bge)
            out[start : start + len(batch)] = (
                torch.nn.functional.normalize(hidden, dim=-1).cpu().numpy()
            )
    return out


def _cluster_strings(strings: list[str], embeddings: "np.ndarray",
                     threshold: float = CLUSTER_THRESHOLD) -> list[int]:
    """Average-linkage agglomerative clustering on cosine distance.

    Returns one integer label per string (labels are arbitrary but
    deterministic for a given input order). Strings closer than
    ``1 - threshold`` cosine end up together; singletons are fine — the
    sampler falls back to the category for them.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    if len(strings) <= 1:
        return [0] * len(strings)
    dist = pdist(embeddings, metric="cosine")
    link = linkage(dist, method="average")
    return fcluster(link, t=1.0 - threshold, criterion="distance").tolist()


def _cmd_cluster_tasks(args) -> int:
    """`icl_data cluster-tasks`: episode -> task-cluster id sidecar.

    Clusters span collection locations (task intent, not scene), fixing the
    two measured failures of DROID grouping: exact task strings are
    essentially unique per episode (0.2% full-pack coverage) and
    `task_category` values are addresses/labs, not tasks. Keys match the
    keypoint-cache convention ``<ds_idx>/<ep>``.
    """
    if args.from_config:
        import yaml

        stage = yaml.safe_load(Path(args.from_config).read_text())
        dataset_cfg = stage.get("dataset", {})
        args.registry = args.registry or dataset_cfg.get("task_registry")
        args.out = args.out or dataset_cfg.get(
            "task_cluster_cache",
            str(args.registry).rsplit(".", 1)[0] + "_clusters.json" if args.registry else None,
        )
        if not args.registry:
            raise SystemExit("cluster-tasks: config has no dataset.task_registry")
    reg = load_task_registry(args.registry)
    out_path = Path(args.out) if args.out else (
        Path(args.registry).with_name(Path(args.registry).stem + "_clusters.json")
    )

    # Collect strings via the on-disk datasets behind the registry.
    ep_tasks: dict[str, str] = {}  # "ds_idx/ep" -> normalized task string
    for ds_idx, ds_info in enumerate(reg["datasets"]):
        ds = open_local_dataset(ds_info["repo_id"], ds_info.get("root", "~/.cache/huggingface/lerobot"))
        for ep, task in _episode_task_strings(ds.meta).items():
            ep_tasks[f"{ds_idx}/{ep}"] = task
    if not ep_tasks:
        raise SystemExit("cluster-tasks: no episode task strings resolved — nothing to cluster")

    uniq = sorted(set(ep_tasks.values()))
    logger.info("embedding %d unique task strings with %s", len(uniq), args.model)
    emb = _embed_strings(uniq, args.model, args.batch_size)
    labels = _cluster_strings(uniq, emb, args.threshold)
    label_of = dict(zip(uniq, labels))

    ep_cluster = {key: label_of[task] for key, task in ep_tasks.items()}
    out_path.write_text(json.dumps(ep_cluster))

    # Audit: what the sampler can now do with these clusters.
    by_cluster = Counter(ep_cluster.values())
    sizes = Counter(by_cluster.values())
    n = len(ep_cluster)
    full = sum(cnt for cnt in by_cluster.values() if cnt >= 5) / n  # k_max=4 + query
    any_sibling = sum(cnt for cnt in by_cluster.values() if cnt >= 2) / n
    inv = defaultdict(list)
    for s, lab in label_of.items():
        if len(inv[lab]) < 3:
            inv[lab].append(s)
    top = sorted(by_cluster.items(), key=lambda kv: -kv[1])[:8]
    print(f"wrote {out_path}: {len(uniq)} strings -> {len(by_cluster)} clusters "
          f"(threshold {args.threshold})")
    print(f"episode coverage: full same-cluster pack {full:.1%} | >=1 sibling {any_sibling:.1%} "
          f"(exact strings were 0.2% / 3.2%)")
    print("cluster size histogram (size: count):",
          dict(sorted(sizes.items())[:12]))
    print("largest clusters:")
    for lab, cnt in top:
        print(f"  #{lab} ({cnt} eps): {inv[lab]}")
    return 0



DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


# ---------------------------------------------------------------------- #
# Episode-range subset download                                          #
# ---------------------------------------------------------------------- #


def _video_keys_of(episode: dict) -> list[str]:
    """Camera keys referenced by one episode row (columns videos/<key>/...)."""
    return sorted(
        col[len("videos/") : -len("/chunk_index")]
        for col in episode
        if col.startswith("videos/") and col.endswith("/chunk_index")
    )


def build_subset_file_list(
    episodes: list[dict], info: dict, start: int, end: int
) -> tuple[set[str], int]:
    """Exact repo file set covering episodes ``[start, end]`` (inclusive).

    ``episodes`` are rows of ``meta/episodes/*/*.parquet`` (v3.0 layout: every
    row carries ``data/chunk_index`` + ``data/file_index`` and one
    ``videos/<key>/chunk_index`` + ``file_index`` pair per camera). Shards are
    size-rolled, so a shard picked at a range boundary also contains a few
    neighbouring episodes — that is inherent to the published layout.
    """
    data_tpl = info.get("data_path", DEFAULT_DATA_PATH)
    video_tpl = info.get("video_path", DEFAULT_VIDEO_PATH)
    files: set[str] = set()
    n_episodes = 0
    for ep in episodes:
        if not start <= int(ep["episode_index"]) <= end:
            continue
        n_episodes += 1
        if ep.get("data/chunk_index") is not None:
            files.add(
                data_tpl.format(
                    chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"]
                )
            )
        for key in _video_keys_of(ep):
            chunk_idx = ep.get(f"videos/{key}/chunk_index")
            file_idx = ep.get(f"videos/{key}/file_index")
            if chunk_idx is not None and file_idx is not None:
                files.add(
                    video_tpl.format(video_key=key, chunk_index=chunk_idx, file_index=file_idx)
                )
    return files, n_episodes


def _load_episode_map(meta_dir: Path) -> tuple[list[dict], dict]:
    """Read ``meta/info.json`` + all ``meta/episodes`` shards from disk."""
    import pyarrow.parquet as pq

    info = json.loads((meta_dir / "info.json").read_text())
    episodes: list[dict] = []
    for shard in sorted((meta_dir / "episodes").glob("*/*.parquet")):
        episodes.extend(pq.read_table(shard).to_pylist())
    episodes.sort(key=lambda ep: int(ep["episode_index"]))
    return episodes, info


def _subset_download_bytes(repo_id: str, wanted: set[str]) -> int:
    """Exact byte total of the wanted files (plus all of meta/) on the Hub."""
    from huggingface_hub import HfApi

    total = 0
    for entry in HfApi().list_repo_tree(repo_id, repo_type="dataset", recursive=True):
        path = getattr(entry, "path", None)
        size = getattr(entry, "size", None)
        if size is None or path is None:
            continue
        if path in wanted or path.startswith("meta/"):
            total += size
    return total


def _report_stray_files(local_dir: Path, wanted: set[str]) -> None:
    """List on-disk data/video files outside the requested subset (no delete)."""
    strays = []
    for pattern in ("data/**/*.parquet", "videos/**/*.mp4"):
        for path in local_dir.glob(pattern):
            rel = path.relative_to(local_dir).as_posix()
            if rel not in wanted:
                strays.append((rel, path.stat().st_size))
    if not strays:
        return
    gib = sum(size for _, size in strays) / 1024**3
    print(f"note: {len(strays)} on-disk files outside the requested range ({gib:.1f} GiB);")
    print("      remove them manually if unwanted:")
    for rel, size in sorted(strays)[:10]:
        print(f"        {rel} ({size / 1024**3:.2f} GiB)")
    if len(strays) > 10:
        print(f"        ... and {len(strays) - 10} more")


def _cmd_download_subset(args) -> int:
    """Episode-range DROID download (ICL §4.4).

    ``lerobot/droid_1.0.1`` is v3.0 file-sharded inside a single
    ``chunk-000`` (156 data shards, 812 video shards over 3 cameras), so
    chunk filtering is meaningless — episodes map to shard files via
    ``meta/episodes`` parquet and that map drives the exact file list.
    """
    from huggingface_hub import snapshot_download

    local_dir = Path(args.root).expanduser() / args.repo_id

    print(f"fetching meta for {args.repo_id} ...")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        allow_patterns=["meta/*"],
        local_dir=str(local_dir),
    )
    episodes, info = _load_episode_map(local_dir / "meta")
    total_episodes = int(info.get("total_episodes", len(episodes)))
    if not episodes:
        print(f"error: no episode metadata found under {local_dir}/meta")
        return 1

    start, end = args.start, args.end
    if end >= total_episodes:
        print(f"warning: --end {end} clamped to {total_episodes - 1} (total episodes)")
        end = total_episodes - 1
    if start > end or start >= total_episodes:
        print(f"error: empty episode range {start}-{end} (dataset has {total_episodes} episodes)")
        return 1

    wanted, n_episodes = build_subset_file_list(episodes, info, start, end)
    size_bytes = _subset_download_bytes(args.repo_id, wanted)
    print(
        f"episodes {start}-{end}: {n_episodes} episodes -> {len(wanted)} data/video shards "
        f"({size_bytes / 1024**3:.1f} GiB) + meta/"
    )
    if not args.yes:
        answer = input(f"download into {local_dir}? [y/N] ")
        if answer.strip().lower() != "y":
            print("aborted")
            return 1

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        allow_patterns=sorted(wanted) + ["meta/*"],
        local_dir=str(local_dir),
    )
    _report_stray_files(local_dir, wanted)
    print(
        "done. Now build the registry from the downloaded episodes only:\n"
        f"  pixi run -e lerobot icl_data build-registry --datasets {args.repo_id},{args.root} "
        f"--grouping-key task_category --droid --out so101_icl/configs/task_registry_droid.json"
    )
    return 0


def _cmd_precompute_keypoints(args) -> int:
    """SIFT keypoint cache for every registered episode (rev 5).

    Extracts keypoints for the SAME uniform-stride demo keyframes
    :class:`ICLDataset` samples, per episode, into one compressed npz keyed
    ``<ds_idx>/<episode>``. ``--from-config <stage.yaml>`` derives registry,
    demo camera, frame count and output path from the stage YAML (single
    source of truth — the cache frame count must equal
    ``demo_encoder.frames_per_demo`` or dataset loading refuses).
    """
    if args.from_config:
        import yaml

        stage = yaml.safe_load(Path(args.from_config).read_text())
        dataset_cfg = stage.get("dataset", {})
        args.registry = args.registry or dataset_cfg.get("task_registry")
        args.demo_camera = args.demo_camera or dataset_cfg.get(
            "demo_camera", "observation.images.left_wrist_0_rgb"
        )
        args.frames_per_demo = stage.get("demo_encoder", {}).get(
            "frames_per_demo", args.frames_per_demo
        )
        args.out = args.out or dataset_cfg.get("keypoint_cache")
        if not args.registry:
            raise SystemExit("precompute-keypoints: config has no dataset.task_registry")
    registry = load_task_registry(args.registry)
    frames_per_demo = int(args.frames_per_demo)
    max_kp = int(args.max_kp)

    datasets: dict[int, dict] = {}
    out: dict[str, np.ndarray] = {}
    for group, members in registry["groups"].items():
        for ds_idx, ep, _length in members:
            if ds_idx not in datasets:
                spec = registry["datasets"][ds_idx]
                ds = open_local_dataset(spec["repo_id"], spec["root"])
                rename = {v: k for k, v in dict(spec.get("camera_rename") or {}).items()}
                camera = rename.get(args.demo_camera, args.demo_camera)
                eps = ds.meta.episodes.to_pandas()
                bounds = {
                    int(row["episode_index"]): (
                        int(row["dataset_from_index"]), int(row["dataset_to_index"]),
                    )
                    for _, row in eps.iterrows()
                }
                datasets[ds_idx] = {"ds": ds, "camera": camera, "bounds": bounds}
            entry = datasets[ds_idx]
            key = f"{ds_idx}/{ep}"
            if key in out:
                continue
            start, _end = entry["bounds"][ep]
            length = entry["bounds"][ep][1] - start
            kf = ICLDataset._sample_keyframe_indices(length, frames_per_demo)
            frames = torch.stack(
                [entry["ds"][start + int(j)][entry["camera"]] for j in kf]
            )
            out[key] = extract_keypoints(preprocess_demo_frames(frames), max_kp=max_kp)
    out_path = Path(args.out or str(args.registry).rsplit(".", 1)[0] + "_kp.npz")
    np.savez_compressed(out_path, **out)
    write_keypoint_cache_meta(
        out_path, args.demo_camera, frames_per_demo,
        {idx: entry["camera"] for idx, entry in datasets.items()},
    )
    logger.info(
        "keypoint cache: %d episodes x %d frames x %d kp -> %s",
        len(out), frames_per_demo, max_kp, out_path,
    )
    return 0


def _cmd_smoke_local(args) -> int:
    reg = build_task_registry(
        [
            DatasetSpec("local/so101_test"),
            DatasetSpec("algorithmtheworld/so101-pick-and-place"),
        ],
        holdout={"eval": 0, "test": 0},
        seed=args.seed,
        out_path=args.out,
    )
    print(json.dumps({g: len(v) for g, v in reg["groups"].items()}, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="so101_icl data tooling")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-registry", help="scan on-disk datasets into task_registry.json")
    p.add_argument("--datasets", nargs="+",
                   help="repo_id[,root] entries (default root ~/.cache/huggingface/lerobot)")
    p.add_argument("--from-config", default=None,
                   help="stage YAML — reads dataset/holdout/seed/registry path from it "
                        "(mutually exclusive with the manual flags)")
    p.add_argument("--grouping-key", default="task",
                   help='"task" (default) or a column name such as task_category')
    p.add_argument("--droid", action="store_true", help="use the DROID camera rename map")
    p.add_argument("--alias-map", default=None, help="JSON {raw_group: curated_group}")
    p.add_argument("--holdout-eval", type=int, default=3)
    p.add_argument("--holdout-test", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_build_registry)

    p = sub.add_parser("check-stats", help="verify q01/q99 quantile stats (pi05 needs them)")
    p.add_argument("--datasets", nargs="+", required=True,
                   help="repo_id[,root] entries (default root ~/.cache/huggingface/lerobot)")
    p.add_argument("--fix", action="store_true",
                   help="run lerobot-edit-dataset recompute_stats IN-PLACE where needed")
    p.add_argument("--yes", action="store_true", help="skip the in-place confirmation prompt")
    p.set_defaults(func=_cmd_check_stats)

    p = sub.add_parser(
        "download-subset",
        help="episode-range droid_1.0.1 download (exact shard file list from meta)",
    )
    p.add_argument("--repo-id", default="lerobot/droid_1.0.1")
    p.add_argument("--root", default="~/.cache/huggingface/lerobot")
    p.add_argument("--start", type=int, default=0, help="first episode index (inclusive)")
    p.add_argument("--end", type=int, default=9, help="last episode index (inclusive)")
    p.add_argument("--yes", action="store_true", help="skip the size confirmation")
    p.set_defaults(func=_cmd_download_subset)

    p = sub.add_parser("smoke-local", help="build a registry for the on-disk SO-101 datasets")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="so101_icl/configs/task_registry_smoke_local.json")
    p.set_defaults(func=_cmd_smoke_local)

    p = sub.add_parser(
        "cluster-tasks",
        help="embed episode task strings and write a task-cluster sidecar "
             "(rev 6: task-intent grouping for stage-1 sampling)",
    )
    p.add_argument("--from-config", default=None,
                   help="stage YAML — derives registry and output sidecar from it "
                        "(dataset.task_registry, dataset.task_cluster_cache)")
    p.add_argument("--registry", default=None, help="task registry JSON")
    p.add_argument("--out", default=None,
                   help="output json (default: <registry stem>_clusters.json)")
    p.add_argument("--model", default=CLUSTER_MODEL,
                   help="HF sentence-embedding model (must be cached; default BAAI/bge-large-en-v1.5)")
    p.add_argument("--threshold", type=float, default=CLUSTER_THRESHOLD,
                   help="min cosine similarity for same-cluster (default 0.85)")
    p.add_argument("--batch-size", type=int, default=256)
    p.set_defaults(func=_cmd_cluster_tasks)

    p = sub.add_parser(
        "precompute-keypoints",
        help="SIFT keypoint cache for registered demo episodes (keypoint branch, rev 5)",
    )
    p.add_argument("--from-config", default=None,
                   help="stage YAML — derives registry/demo camera/frames/output "
                        "from it (dataset.task_registry, dataset.demo_camera, "
                        "demo_encoder.frames_per_demo, dataset.keypoint_cache)")
    p.add_argument("--registry", default=None, help="task registry JSON")
    p.add_argument("--out", default=None,
                   help="output npz (default: <registry stem>_kp.npz)")
    p.add_argument("--frames-per-demo", type=int, default=6,
                   help="must match demo_encoder.frames_per_demo of the stage config")
    p.add_argument("--max-kp", type=int, default=16, help="keypoints per frame")
    p.add_argument("--demo-camera", default="observation.images.left_wrist_0_rgb",
                   help="policy-side camera key (renamed back to the dataset key)")
    p.set_defaults(func=_cmd_precompute_keypoints)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
