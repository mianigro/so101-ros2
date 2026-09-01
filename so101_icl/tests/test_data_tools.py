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

"""Stats quantile check + stage-YAML registry translation tests.

The end-to-end recompute test copies the 25 MB local dataset to tmp, strips
its q01/q99 entries, and runs the exact in-place command the droid session
will use — de-risking `check-stats --fix` against real lerobot behavior.
"""

import json
import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from so101_icl.data import (  # noqa: E402
    DatasetSpec,
    _load_episode_map,
    _subset_download_bytes,
    build_subset_file_list,
    build_task_registry,
    check_stats,
    keypoint_cache_meta_path,
    load_stats_json,
    missing_quantiles,
    on_disk_episodes,
    open_local_dataset,
    specs_from_stage_config,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
LOCAL_ROOT = Path.home() / ".cache/huggingface/lerobot"
LOCAL_PRESENT = (LOCAL_ROOT / "local/so101_test/meta/info.json").exists()


def _stats(state_q=True, action_q=True, null_q01=False):
    def entry(quantiles):
        e = {"min": [0.0], "max": [1.0], "mean": [0.5], "std": [0.1]}
        if quantiles:
            e.update({"q01": [0.1], "q10": [0.2], "q50": [0.5], "q90": [0.8], "q99": [0.9]})
        return e

    stats = {"observation.images.wrist": entry(False)}  # VISUAL: ignored
    stats["observation.state"] = entry(state_q)
    stats["action"] = entry(action_q)
    if null_q01:
        stats["observation.state"]["q01"] = None
    return stats


class TestMissingQuantiles(unittest.TestCase):
    def test_complete(self):
        self.assertEqual(missing_quantiles(_stats()), [])

    def test_min_max_only(self):
        problems = missing_quantiles(_stats(state_q=False))
        self.assertEqual(len(problems), 2)  # q01 + q99
        self.assertIn("observation.state: q01 is null", problems)

    def test_null_q01(self):
        problems = missing_quantiles(_stats(null_q01=True))
        self.assertEqual(problems, ["observation.state: q01 is null"])

    def test_feature_missing_entirely(self):
        self.assertIn("action: missing entirely", missing_quantiles({"observation.state": {}}))


class TestSpecsFromStageConfig(unittest.TestCase):
    def test_smoke_yaml_translates(self):
        import yaml

        stage = yaml.safe_load((CONFIGS / "icl_smoke_local_v1.yaml").read_text())
        specs, kwargs = specs_from_stage_config(stage)
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].repo_id, "local/so101_test")
        self.assertEqual(specs[0].grouping_key, "task")
        self.assertEqual(
            specs[0].camera_rename["observation.images.wrist"],
            "observation.images.left_wrist_0_rgb",
        )
        self.assertEqual(kwargs["seed"], 42)
        self.assertEqual(kwargs["out_path"], "so101_icl/configs/task_registry_smoke_local.json")
        self.assertEqual(kwargs["holdout"], {"eval": 0, "test": 0})

    def test_droid_yaml_translates(self):
        import yaml

        stage = yaml.safe_load((CONFIGS / "icl_pretrain_droid_v1.yaml").read_text())
        specs, kwargs = specs_from_stage_config(stage)
        self.assertEqual([s.repo_id for s in specs], ["lerobot/droid_1.0.1"])
        self.assertEqual(specs[0].grouping_key, "task_category")
        self.assertEqual(
            specs[0].camera_rename["observation.images.exterior_1_left"],
            "observation.images.base_0_rgb",
        )
        self.assertEqual(kwargs["holdout"], {"eval": 6, "test": 6})

    def test_multi_repo_stage(self):
        stage = {
            "dataset": {
                "repo_ids": ["local/a", "local/b"],
                "grouping_key": "task",
                "camera_rename": "pi05_base",
                "task_registry": "out.json",
            },
            "train": {"seed": 7},
        }
        specs, kwargs = specs_from_stage_config(stage)
        self.assertEqual([s.repo_id for s in specs], ["local/a", "local/b"])
        self.assertEqual(kwargs["seed"], 7)

    def test_unknown_preset_rejected(self):
        with self.assertRaises(ValueError):
            specs_from_stage_config(
                {"dataset": {"repo_id": "x", "camera_rename": "nope", "task_registry": "o.json"}}
            )

    @unittest.skipUnless(LOCAL_PRESENT, "local so101_test dataset not on disk")
    def test_from_config_matches_manual_build(self):
        """--from-config produces the same registry as the manual flags.

        The smoke YAML lists only local/so101_test, so the manual side is a
        single dataset too; cross-dataset merging is covered in
        test_sampler.test_groups_merge_across_datasets.
        """
        import tempfile
        import yaml

        stage = yaml.safe_load((CONFIGS / "icl_smoke_local_v1.yaml").read_text())
        specs, kwargs = specs_from_stage_config(stage)
        with tempfile.TemporaryDirectory() as tmp:
            kwargs["out_path"] = Path(tmp) / "from_config.json"
            from_config = build_task_registry(specs, **kwargs)
            manual = build_task_registry(
                [DatasetSpec("local/so101_test")],
                alias_map={},
                holdout={"eval": 0, "test": 0},
                seed=42,
                out_path=Path(tmp) / "manual.json",
            )
            self.assertEqual(from_config["splits"], manual["splits"])
            self.assertEqual(
                {g: sorted(map(tuple, eps)) for g, eps in from_config["groups"].items()},
                {g: sorted(map(tuple, eps)) for g, eps in manual["groups"].items()},
            )


class TestClusterStrings(unittest.TestCase):
    """Deterministic clustering core (rev 6) — synthetic embeddings, no model."""

    def test_two_groups_and_singleton(self):
        import numpy as np

        from so101_icl.data import _cluster_strings

        strings = ["put a in b", "put c in d", "grab e", "take f", "wave"]
        emb = np.array([
            [1.0, 0.0], [0.99, 0.14],   # group A (near-identical direction)
            [0.0, 1.0], [0.14, 0.99],   # group B
            [-1.0, 0.0],                # singleton, far from both
        ], dtype=np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        labels = _cluster_strings(strings, emb, threshold=0.85)
        self.assertEqual(labels[0], labels[1])       # A together
        self.assertEqual(labels[2], labels[3])       # B together
        self.assertNotEqual(labels[0], labels[2])    # A vs B apart
        self.assertNotIn(labels[4], (labels[0], labels[2]))  # singleton alone

    def test_single_string_is_one_cluster(self):
        import numpy as np

        from so101_icl.data import _cluster_strings

        self.assertEqual(_cluster_strings(["x"], np.array([[1.0, 0.0]])), [0])


@unittest.skipUnless(LOCAL_PRESENT, "local so101_test dataset not on disk")
class TestCheckStatsEndToEnd(unittest.TestCase):
    """Strip q01/q99 on a tmp copy, then fix with the real lerobot CLI."""

    def test_detect_and_fix(self):
        import tempfile

        with tempfile.TemporaryDirectory(prefix="icl_stats_") as tmp:
            root = Path(tmp)
            shutil.copytree(LOCAL_ROOT / "local/so101_test", root / "so101_test")
            stats_path = root / "so101_test/meta/stats.json"
            stats = json.loads(stats_path.read_text())
            for feature in ("observation.state", "action"):
                for key in ("q01", "q10", "q50", "q90", "q99"):
                    stats[feature][key] = None
            stats_path.write_text(json.dumps(stats))

            spec = DatasetSpec("so101_test", root=str(root))
            problems = missing_quantiles(load_stats_json(spec.repo_id, spec.root))
            self.assertEqual(len(problems), 4)  # q01+q99 for state and action
            self.assertEqual(check_stats([spec]), 1)  # report mode flags it

            rc = check_stats([spec], fix=True, assume_yes=True)
            self.assertEqual(rc, 0)
            self.assertEqual(missing_quantiles(load_stats_json(spec.repo_id, spec.root)), [])


class TestBuildSubsetFileList(unittest.TestCase):
    """Episode -> shard-file resolution for download-subset (v3.0 layout).

    droid_1.0.1 keeps everything in chunk-000 with per-episode chunk/file
    indices in meta/episodes; the helper must turn an episode range into the
    exact deduped shard list (per-camera file indices differ from data's).
    """

    INFO = {
        "total_episodes": 4,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }

    @staticmethod
    def _ep(idx, data_chunk=0, data_file=0, cam_files=(("cam_a", 0, 0), ("cam_b", 0, 5))):
        row = {
            "episode_index": idx,
            "data/chunk_index": data_chunk,
            "data/file_index": data_file,
        }
        for key, c, f in cam_files:
            row[f"videos/{key}/chunk_index"] = c
            row[f"videos/{key}/file_index"] = f
        return row

    def test_exact_file_set_per_camera(self):
        files, n = build_subset_file_list([self._ep(0)], self.INFO, 0, 0)
        self.assertEqual(n, 1)
        # cam_b sits at file 005 even though data/cam_a are at file 000
        self.assertEqual(
            files,
            {
                "data/chunk-000/file-000.parquet",
                "videos/cam_a/chunk-000/file-000.mp4",
                "videos/cam_b/chunk-000/file-005.mp4",
            },
        )

    def test_shard_dedup_across_episodes(self):
        # episodes 0 and 1 share every shard; episode 2 rolls to a new file
        rows = [self._ep(0), self._ep(1), self._ep(2, data_file=1)]
        files, n = build_subset_file_list(rows, self.INFO, 0, 2)
        self.assertEqual(n, 3)
        self.assertEqual(
            files,
            {
                "data/chunk-000/file-000.parquet",
                "data/chunk-000/file-001.parquet",
                "videos/cam_a/chunk-000/file-000.mp4",
                "videos/cam_b/chunk-000/file-005.mp4",
            },
        )

    def test_range_is_inclusive_and_filters(self):
        rows = [self._ep(i, data_file=i) for i in range(4)]
        files, n = build_subset_file_list(rows, self.INFO, 1, 2)
        self.assertEqual(n, 2)
        self.assertIn("data/chunk-000/file-001.parquet", files)
        self.assertIn("data/chunk-000/file-002.parquet", files)
        self.assertNotIn("data/chunk-000/file-000.parquet", files)
        self.assertNotIn("data/chunk-000/file-003.parquet", files)

    def test_custom_templates_from_info(self):
        info = {
            "data_path": "d/{chunk_index}/f-{file_index}.parq",
            "video_path": "v/{video_key}/{chunk_index}-{file_index}.mp4",
        }
        files, _ = build_subset_file_list([self._ep(0)], info, 0, 0)
        self.assertEqual(
            files, {"d/0/f-0.parq", "v/cam_a/0-0.mp4", "v/cam_b/0-5.mp4"}
        )

    def test_default_templates_used_when_info_lacks_paths(self):
        files, _ = build_subset_file_list([self._ep(0)], {}, 0, 0)
        self.assertIn("data/chunk-000/file-000.parquet", files)
        self.assertIn("videos/cam_a/chunk-000/file-000.mp4", files)

    def test_multi_chunk_indices_honoured(self):
        row = self._ep(7, data_chunk=3, data_file=12, cam_files=(("cam_a", 4, 9),))
        files, _ = build_subset_file_list([row], self.INFO, 0, 7)
        self.assertEqual(
            files,
            {
                "data/chunk-003/file-012.parquet",
                "videos/cam_a/chunk-004/file-009.mp4",
            },
        )


class TestLoadEpisodeMapAndSize(unittest.TestCase):
    def test_load_episode_map_reads_shards_in_order(self):
        import tempfile

        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            meta = Path(tmp) / "meta"
            (meta / "episodes/chunk-000").mkdir(parents=True)
            (meta / "info.json").write_text(
                json.dumps({"total_episodes": 2, "data_path": "d.parquet"})
            )
            table = pa.table(
                {
                    "episode_index": [1, 0],  # shard rows out of order
                    "data/chunk_index": [0, 0],
                    "data/file_index": [1, 0],
                }
            )
            pq.write_table(table, meta / "episodes/chunk-000/file-000.parquet")

            episodes, info = _load_episode_map(meta)
            self.assertEqual(info["total_episodes"], 2)
            self.assertEqual([e["episode_index"] for e in episodes], [0, 1])

    def test_subset_download_bytes_sums_wanted_and_meta_only(self):
        import types

        import huggingface_hub

        entries = [
            types.SimpleNamespace(path="meta/info.json", size=10),
            types.SimpleNamespace(path="meta/episodes/chunk-000/file-000.parquet", size=20),
            types.SimpleNamespace(path="data/chunk-000/file-000.parquet", size=100),
            types.SimpleNamespace(path="data/chunk-000/file-001.parquet", size=1000),
            types.SimpleNamespace(path="videos/cam/chunk-000/file-000.mp4", size=500),
        ]

        class FakeApi:
            def list_repo_tree(self, repo_id, repo_type="dataset", recursive=False):
                return iter(entries)

        original = huggingface_hub.HfApi
        huggingface_hub.HfApi = FakeApi
        try:
            wanted = {
                "data/chunk-000/file-000.parquet",
                "videos/cam/chunk-000/file-000.mp4",
            }
            # wanted data (100) + wanted video (500) + all meta/ (30) = 630
            self.assertEqual(_subset_download_bytes("x/y", wanted), 630)
        finally:
            huggingface_hub.HfApi = original


class TestOnDiskEpisodes(unittest.TestCase):
    """Subset downloads ship FULL meta/ + partial files; everything must be
    pinned to the episodes whose files actually exist (otherwise lerobot
    re-downloads the rest of the dataset)."""

    @staticmethod
    def _synthetic(tmp: Path, data_eps: tuple[int, ...], meta_eps: int = 4) -> Path:
        """meta/ lists ``meta_eps`` episodes; data shards exist only for
        ``data_eps`` (2 episodes per data shard)."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        base = tmp / "local/sub"
        (base / "meta/episodes/chunk-000").mkdir(parents=True)
        (base / "data/chunk-000").mkdir(parents=True)
        (base / "meta/info.json").write_text(json.dumps({"total_episodes": meta_eps}))
        pq.write_table(
            pa.table({
                "episode_index": list(range(meta_eps)),
                "data/chunk_index": [0] * meta_eps,
                "data/file_index": [i // 2 for i in range(meta_eps)],
            }),
            base / "meta/episodes/chunk-000/file-000.parquet",
        )
        for file_idx in sorted({e // 2 for e in data_eps}):
            eps_in_file = [e for e in data_eps if e // 2 == file_idx]
            pq.write_table(
                pa.table({"episode_index": eps_in_file}),
                base / f"data/chunk-000/file-{file_idx:03d}.parquet",
            )
        return base

    def test_only_downloaded_episodes_registered(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self._synthetic(Path(tmp), data_eps=(0, 1))  # meta says 0..3
            self.assertEqual(on_disk_episodes("local/sub", tmp), {0, 1})

    def test_empty_subset_raises(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self._synthetic(Path(tmp), data_eps=())
            with self.assertRaises(FileNotFoundError):
                open_local_dataset("local/sub", tmp)

    def test_non_contiguous_prefix_rejected(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self._synthetic(Path(tmp), data_eps=(1, 2))  # episode 0 missing
            with self.assertRaises(ValueError):
                open_local_dataset("local/sub", tmp)


@unittest.skipUnless(LOCAL_PRESENT, "local so101_test dataset not on disk")
class TestNoHubDownload(unittest.TestCase):
    """Opening the local dataset must never reach the hub — any
    snapshot_download call raises (stricter than HF_HUB_OFFLINE)."""

    def test_open_local_dataset_never_downloads(self):
        from unittest.mock import patch

        with (
            patch("lerobot.datasets.lerobot_dataset.snapshot_download",
                  side_effect=AssertionError("hub download attempted")),
            patch("lerobot.datasets.dataset_metadata.snapshot_download",
                  side_effect=AssertionError("hub download attempted")),
        ):
            ds = open_local_dataset("local/so101_test", str(LOCAL_ROOT))
        self.assertGreater(len(ds.episodes), 0)


class TestKeypointCacheMeta(unittest.TestCase):
    """Camera-identity sidecar: refuse mismatched caches, warn on legacy."""

    def _cache(self, tmp, meta=None):
        import numpy as np

        path = Path(tmp) / "kp_cache.npz"
        np.savez_compressed(path, **{"0/0": np.zeros((6, 16, 131), np.float32)})
        if meta is not None:
            keypoint_cache_meta_path(path).write_text(json.dumps(meta))
        return path

    def test_mismatched_camera_refused(self):
        import tempfile

        from so101_icl.data import validate_keypoint_cache_camera

        with tempfile.TemporaryDirectory() as tmp:
            path = self._cache(tmp, meta={
                "demo_camera": "observation.images.base_0_rgb",
                "frames_per_demo": 6, "resolved": {},
            })
            with self.assertRaises(ValueError) as ctx:
                validate_keypoint_cache_camera(
                    path, "observation.images.left_wrist_0_rgb")
            self.assertIn("built for demo camera", str(ctx.exception))

    def test_matching_camera_passes(self):
        import tempfile

        from so101_icl.data import validate_keypoint_cache_camera

        with tempfile.TemporaryDirectory() as tmp:
            path = self._cache(tmp, meta={
                "demo_camera": "observation.images.left_wrist_0_rgb",
                "frames_per_demo": 6, "resolved": {},
            })
            self.assertTrue(validate_keypoint_cache_camera(
                path, "observation.images.left_wrist_0_rgb"))

    def test_legacy_cache_warns_but_loads(self):
        import logging
        import tempfile

        from so101_icl.data import validate_keypoint_cache_camera

        with tempfile.TemporaryDirectory() as tmp:
            path = self._cache(tmp)  # no .meta.json sidecar
            with self.assertLogs("so101_icl.data", level=logging.WARNING) as logs:
                ok = validate_keypoint_cache_camera(
                    path, "observation.images.left_wrist_0_rgb")
            self.assertFalse(ok)
            self.assertTrue(any("cannot verify" in m for m in logs.output))

    def test_meta_written_by_precompute_helper(self):
        import tempfile

        from so101_icl.data import write_keypoint_cache_meta

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kp.npz"
            write_keypoint_cache_meta(
                path, "observation.images.left_wrist_0_rgb", 12,
                {0: "observation.images.wrist"})
            meta = json.loads(keypoint_cache_meta_path(path).read_text())
            self.assertEqual(
                meta["demo_camera"], "observation.images.left_wrist_0_rgb")
            self.assertEqual(meta["frames_per_demo"], 12)
            self.assertEqual(meta["resolved"], {"0": "observation.images.wrist"})


if __name__ == "__main__":
    unittest.main()
