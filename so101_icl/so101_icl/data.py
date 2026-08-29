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


def _episode_group_keys(spec: DatasetSpec, meta) -> dict[int, str]:
    """Map episode_index -> group key, from on-disk metadata only."""
    eps = meta.episodes.to_pandas()
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
    root = Path(spec.root).expanduser() / spec.repo_id
    keys: dict[int, str] = {}
    for _, row in eps.iterrows():
        chunk = int(row["data/chunk_index"])
        file_idx = int(row["data/file_index"])
        path = root / "data" / f"chunk-{chunk:03d}" / f"file-{file_idx:03d}.parquet"
        import pyarrow.parquet as pq

        table = pq.read_table(
            path, columns=["episode_index", spec.grouping_key]
        ).to_pydict()
        ep = int(row["episode_index"])
        for e, g in zip(table["episode_index"], table[spec.grouping_key]):
            if int(e) == ep:
                keys[ep] = str(g)
                break
    return keys


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
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

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
        dataset_root = str(Path(spec.root).expanduser() / spec.repo_id)
        ds = LeRobotDataset(spec.repo_id, root=dataset_root)
        registry["datasets"].append(
            {
                "repo_id": spec.repo_id,
                "root": spec.root,
                "camera_rename": spec.camera_rename,
                "grouping_key": spec.grouping_key,
            }
        )
        eps = ds.meta.episodes.to_pandas()
        keys = _episode_group_keys(spec, ds.meta)
        for _, row in eps.iterrows():
            ep = int(row["episode_index"])
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
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.repo_id = spec_dict["repo_id"]
        self.camera_rename = dict(spec_dict["camera_rename"])
        self.reverse_rename = {v: k for k, v in self.camera_rename.items()}
        self.dataset = LeRobotDataset(
            self.repo_id,
            root=str(Path(spec_dict["root"]).expanduser() / self.repo_id),  # root includes repo_id
        )
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


class ICLDataset(Dataset):
    """Task-grouped support/query sampling over one or more LeRobot datasets.

    Item = one query transition + a variable-size support pack:

    - query: random timestep of an episode, all cameras (renamed to policy
      keys, RAW [0,1] — the policy preprocessor owns the rest), raw state,
      ``chunk_size`` raw actions (edge-padded), task string;
    - support: ``k`` OTHER episodes of the same task group, ``F``
      keyframes each (uniform stride incl. first and last), trajectory
      downsampled to ``S`` steps, normalized + padded, all padded to
      ``k_max`` slots with a presence mask.
    """

    def __init__(
        self,
        registry_path: Path | str,
        config,
        *,
        split: str = "train",
        demo_camera: str = "observation.images.base_0_rgb",
        seed: int = 0,
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
            b.demo_camera_key = key

    def __len__(self):
        return len(self._index)

    def _sample_keyframe_indices(self, length: int, f: int) -> np.ndarray:
        idx = np.linspace(0, length - 1, f).round().astype(int)
        idx[0], idx[-1] = 0, length - 1  # start and goal emphasis (ICL §4.4)
        return np.unique(idx)

    def _load_traj(self, bundle: _DatasetBundle, ep: int) -> tuple[torch.Tensor, bool]:
        start, end, _ = bundle.episodes[ep]
        rows = bundle.traj_table[start:end]
        states = torch.as_tensor(np.asarray(rows["observation.state"]), dtype=torch.float32)
        actions = torch.as_tensor(np.asarray(rows["action"]), dtype=torch.float32)
        s_idx = self._sample_keyframe_indices(states.shape[0], self.traj_steps)
        traj = bundle.normalizer(states[s_idx], actions[s_idx])
        return traj, True

    def __getitem__(self, i: int) -> dict:
        rng = np.random.default_rng([self.seed, i, self._epoch])
        q_ds, group, q_ep = self._index[i]
        members = [m for m in self._group_members[group] if m != (q_ds, q_ep)]
        k = int(rng.choice(self.k_choices))
        k = min(k, len(members))
        if k < len(members):
            members = [
                members[j] for j in rng.choice(len(members), size=k, replace=False)
            ]

        demo_frames = torch.zeros(
            self.k_max, self.frames_per_demo, 3, 224, 224, dtype=torch.float32
        )
        demo_mask = torch.zeros(self.k_max, dtype=torch.bool)
        demo_traj = torch.zeros(
            self.k_max, self.traj_steps, self.config.max_state_dim + self.config.max_action_dim,
            dtype=torch.float32,
        )
        demo_traj_ok = torch.zeros(self.k_max, dtype=torch.float32)
        for slot, (ds_idx, ep) in enumerate(members[: self.k_max]):
            bundle = self.bundles[ds_idx]
            start, _end, length = bundle.episodes[ep]
            kf = self._sample_keyframe_indices(length, self.frames_per_demo)
            frames = torch.stack(
                [bundle.dataset[start + int(j)][bundle.demo_camera_key] for j in kf]
            )
            demo_frames[slot, : len(kf)] = preprocess_demo_frames(frames)
            demo_mask[slot] = True
            traj, ok = self._load_traj(bundle, ep)
            demo_traj[slot] = traj
            demo_traj_ok[slot] = float(ok)

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
        return batch

    _epoch: int = 0

    def set_epoch(self, epoch: int) -> None:
        """Refreshes per-item RNG streams (call from the train loop)."""
        self._epoch = int(epoch)


# ---------------------------------------------------------------------- #
# CLI                                                                    #
# ---------------------------------------------------------------------- #


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
            repo_id=spec.split(",")[0],
            root=(spec.split(",")[1] if "," in spec else "~/.cache/huggingface/lerobot"),
            camera_rename=(
                DROID_IMAGE_KEY_MAP if args.droid else dict(PI05_BASE_IMAGE_KEY_MAP)
            ),
            grouping_key=args.grouping_key,
        )
        for spec in args.datasets
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
        DatasetSpec(
            repo_id=spec.split(",")[0],
            root=(spec.split(",")[1] if "," in spec else "~/.cache/huggingface/lerobot"),
        )
        for spec in args.datasets
    ]
    return check_stats(specs, fix=args.fix, assume_yes=args.yes)


# ---------------------------------------------------------------------- #
# Episode-range subset download                                          #
# ---------------------------------------------------------------------- #

DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


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
        f"  python -m so101_icl.data build-registry --datasets {args.repo_id},{args.root} "
        f"--grouping-key task_category --droid --out so101_icl/configs/task_registry_droid.json"
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

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
