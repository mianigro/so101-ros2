Review verdict: the so101_icl implementation is correct in architecture and matches ICL_IMPLEMENTATION.md Rev 2 (deviations are real but documented in the README). Five concrete bugs plus stale docs/configs need fixing.

## 1. Code fixes

**`so101_icl/so101_icl/bridge_icl_node.py`**
- Add `import sys` to the stdlib import block (fixes `NameError` in the `--stdin-events` reader thread, line 353).
- `build_demo_pack` (lines 265-268): always stack trajectories per-asset, zero-filling missing ones, instead of nulling the whole pack when `assets[0]` lacks one — keeps per-asset `traj_ok` flags consistent with the data actually sent:
  ```python
  traj = np.stack([a.traj if a.traj is not None else np.zeros((16, 64), np.float32) for a in assets])
  traj_ok = np.asarray([a.traj_ok for a in assets], dtype=np.float32)
  ```

**`so101_icl/so101_icl/data.py`**
- `_sample_keyframe_indices` (line 524): drop `np.unique` so the function always returns exactly `f` indices (short episodes repeat timesteps); update its docstring. This fixes the `demo_traj[slot] = traj` broadcast crash for episodes shorter than `traj_steps=16`. Make it a `@staticmethod` so it is unit-testable without a registry.

**`so101_icl/so101_icl/train_loop.py`**
- Fix curriculum timing in `run_training` (lines 248-260): set `train_loader.dataset.k_choices = k_choices_for_step(settings.curriculum, step)` *before* each `iter(train_loader)` (initial creation and every epoch rollover), instead of after `next()`. Restructure to `data_iter = None` + `continue` on `StopIteration`. Update the comment to state plainly that curriculum applies at epoch granularity (workers fork per `iter()`).

**`so101_icl/so101_icl/eval_icl.py`**
- `load_eval_policy`: accept `demo_encoder`/`lora` config overrides and pass them to `PI05ICLPolicy.from_base(...)` so the module is *built* from the stage config.
- `run_offline` (lines 87-90): construct `DemoEncoderConfig`/`LoRAConfig` from the stage dict and pass them into `load_eval_policy`; delete the post-hoc `policy.config.demo_encoder = ...` replacement.
- `main` offline branch: when `--trials`/`--k` are not given on the CLI, default them from the stage config's `eval.offline_trials` and `eval.ablations.k` (makes the §7 config schema keys real; argparse defaults become `None`).

## 2. Config fixes
- `configs/icl_finetune_so101_v1.yaml`: remove the dead `dataset.camera_keys` line (nothing consumes it).
- `configs/icl_pretrain_droid_v1.yaml`: fix the header comment command to `pixi run -e lerobot icl_data download-subset ...` (the bare `python -m` form doesn't run from the repo root); annotate both configs' `eval:` blocks (offline_trials/ablations.k now consumed; sim/real keys deferred to the M3/M4 campaigns).

## 3. README fixes (`so101_icl/README.md`)
- Replace the bare `python -m so101_icl.data ...` commands (lines 38-45) with `pixi run -e lerobot icl_data ...` variants.
- Fix lines 85-86 (`pixi run -e lerobot python -m so101_icl.data ...` → `pixi run -e lerobot icl_data ...`).
- Bridge block (lines 69-73): prefix `PYTHONPATH=so101_icl` for the `python -m so101_icl.bridge_icl_node` invocation; reword the "ros2 run via BridgeICLNode" note to say it's not yet an installable ROS package (no package.xml/setup.py) — run the rclpy class manually.
- Close the unclosed parenthesis in the line-82 comment.
- Correct the intro claim: the repo edits outside the package are the `"pi05_icl"` line in `policy_server/inference_engine.py` plus pixi.toml (peft and tensorboard deps, icl_server/icl_data tasks).
- Update the layout-tree test comment to list the actual coverage (package, zero-init, prefix shapes, sampler, data tools, bridge, transport, serving e2e).

## 4. ICL_IMPLEMENTATION.md — surgical factual corrections only
- §4.3/§9: adapter sidecar is `icl_adapter_config.json` (distinct from the serving checkpoint's `icl_config.json`).
- §3 layout: registry artifacts are `task_registry_*.json`; add the inner `eval_icl.py`, `configs/icl_smoke_local_v1.yaml`, and the full test file list.
- §4.1: align `set_demo_pack` signature with the implementation `(frames, mask, traj, traj_ok)` and note the splice is implemented as the unit-tested `splice_demo_tokens`.

## 5. New tests
- `tests/test_sampler.py`: add a non-skipped `TestKeyframeIndices` class for `ICLDataset._sample_keyframe_indices` — long case (strictly increasing, first/last pinned) and short case (length < f returns exactly f indices).
- `tests/test_bridge.py`: extend `_StubTransport` to record `traj`/`traj_ok`; add a `build_demo_pack` test where the first selected demo has no trajectory and the second does (fake normalizer) — asserts traj is stacked with zeros and `traj_ok == [0, 1]`.

## 6. Verification
- Baseline first, then after changes: `pixi run -e lerobot python -m unittest discover -s so101_icl/tests -p "test_*.py"` (GPU tests need CUDA + cached `lerobot/pi05_base`, which this machine has per the existing smoke run).
- Sanity-grep the README/config commands for the `python -m so101_icl.*` pattern to confirm none remain outside PYTHONPATH/pixi wrappers.

Not changing (reviewed, judged fine): the `--resume-from` path not enforcing `expected_init_adapter` fingerprints (guard metadata only, stage chaining is by explicit path), the O(episodes×shards) parquet fallback in `_episode_group_keys` (DROID perf only), and the hardcoded `(16, 64)` traj shape constants shared by bridge/transport (consistent with both stage configs today).