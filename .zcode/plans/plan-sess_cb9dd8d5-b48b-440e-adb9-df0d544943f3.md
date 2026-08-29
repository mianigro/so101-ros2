## Problem
`download-subset --start 0 --end 2` downloads ~everything (~810 GB total repo). Cause: `lerobot/droid_1.0.1` v3.0 layout has ONE chunk dir (`chunk-000`) holding all shards (156 data parquet + 812 video mp4 across 3 cameras). The code builds patterns from chunk indices, so every `--start/--end` collapses to `chunk-000/*` = whole repo. HF has no episode-level download API — the documented approach is exact file lists passed to `snapshot_download(allow_patterns=[...])` / `hf download <files>`.

## Fix — `so101_icl/so101_icl/data.py:654-682` (`_cmd_download_subset`) + subparser help (:727-733)

`--start/--end` become inclusive **episode indices** (user-approved). Steps:

1. **Bootstrap meta from the Hub** (assumes nothing on disk): `snapshot_download(repo_id, repo_type="dataset", allow_patterns=["meta/*"], local_dir=root/repo_id)` — a few MB.
2. **Map episodes → shard files**: read `meta/episodes/*/*.parquet` (pyarrow, columns `episode_index`, `data/chunk_index`, `data/file_index`, `videos/{cam}/chunk_index`, `videos/{cam}/file_index` — verified schema from lerobot 0.6.1 source); read `data_path`/`video_path` templates + `total_episodes` from `meta/info.json`.
3. **Slice** episodes in `[start, end]`; clamp end to `total_episodes - 1` with a printed warning.
4. **Exact deduped file set**: format the templates per selected episode (data shards + per-camera video shards) + `meta/*`. Boundary shards inherently include a few neighboring episodes (size-rolled ~1 GB shards) — print episode count and file count.
5. **Preview + confirm**: use `snapshot_download(..., dry_run=True)` → exact total GB from `DryRunFileInfo`; show prompt unless `--yes` (print the total even with `--yes`). Fall back to `HfApi().list_repo_tree` size-summing if the installed `huggingface_hub` lacks `dry_run`.
6. **Download**: `snapshot_download(..., allow_patterns=sorted(file_set), local_dir=...)`.
7. **Report stray on-disk files** not in the requested set (informational only — per user's choice, no deletion).
8. Keep the trailing `build-registry` hint.

## Tests — `so101_icl/tests/test_data_tools.py`
Offline tests for the pure helper (e.g. `build_subset_file_list`): pattern set from mixed chunk/file indices, dedup across episodes sharing a shard, per-camera differing file indices, clamping. Mock for the dry-run size preview.

## Docs — `ICL_IMPLEMENTATION.md` §4.4 (~line 324)
Correct the wrong "1,000 episodes/chunk" claim: actual repo = single `chunk-000`, file-sharded, episode→file map lives in `meta/episodes/` parquet; document new episode-range semantics.

## Verify
`pixi run -e lerobot python -m pytest so101_icl/tests -q`, then a real smoke: `pixi run -e lerobot icl_data download-subset --start 0 --end 2` (expect a few GB, exact size shown up front).