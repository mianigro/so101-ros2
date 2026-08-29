# Close the remaining so101_icl code gaps

Three targeted additions inside `so101_icl/` (no changes outside the package). Each is independently testable without droid data or ROS.

## 1. `recompute_stats: auto` automation (`so101_icl/data.py`)

Closes the "config key documented but nothing acts on it" gap (ICL §4.4 stats check).

- `stats_need_quantiles(stats: dict) -> bool | list` — reads a dataset's `meta/stats.json` directly (no dataset load); reports whether `observation.state`/`action` carry non-null `q01`/`q99` (pi05 quantile normalization needs them; images are VISUAL=IDENTITY, ignored).
- CLI subcommand `check-stats`:
  - `--datasets repo[,root] ...` → per-dataset verdict (ok / missing-quantiles).
  - `--fix --yes` → runs `lerobot-edit-dataset --repo-id <id> --root <parent> --operation.type recompute_stats --operation.overwrite true` via subprocess — **in-place** (verified semantics: `lerobot_edit_dataset.py:674-710` — without `overwrite` it `copytree`s the whole 85 GB; with it, only stats are regenerated). Note their `--root` is the PARENT dir (their default is `$HF_LEROBOT_HOME/repo_id`), unlike `LeRobotDataset(root=...)` which is the full dataset dir — the test proves the right convention. `--fix` without `--yes` prompts with the in-place-overwrite warning.
- `train_icl.py` preflight: when the stage YAML has `dataset.recompute_stats: auto`, run the check before training; on missing quantiles print the exact fix command and abort (never auto-mutate from the trainer).

**Tests** (`tests/test_data_tools.py`):
- Detection unit tests on synthetic stats dicts (complete / min-max-only / null-q01).
- End-to-end on a tmp copy of `local/so101_test` (25 MB): strip q01/q99 from the copy's `meta/stats.json` → `check-stats` flags it → `--fix` subprocess restores them → verdict flips to ok. This also de-risks the exact command the droid session will run.

## 2. Bridge terminal-condition wiring (`so101_icl/bridge_icl_node.py`)

Closes the "terminal predicates parsed but nothing subscribes" gap (ICL §4.7). The CLI currently advances through all subtasks per poll regardless of terminal events.

- ROS-free core — `TerminalMonitor`:
  - wraps `BridgeState`; `set_active(subtask)` records the active subtask's `terminal_topic`;
  - `record_event(topic)` marks a topic fired (idempotent);
  - `should_advance()` — True when the active subtask's terminal topic fired; subtasks with no predicate advance immediately (their terminal is "pack pushed").
- Rework the existing `main()` loop to drive through the monitor: poll dispatch → build state → per active subtask: push pack (existing `build_demo_pack`) → wait for `should_advance()` (poll loop with `--poll-s`) → `transport.clear()` → advance. Add `--advance-mode {terminal, auto}` (`auto` = legacy immediate-advance behavior, kept for smoke use).
- rclpy wrapper — `BridgeICLNode(rclpy.Node)` (import-guarded, in the same module): parameters for api/dispatch, demo transport endpoint, k / frames-per-demo / stats path, `--terminal-msg-type` (default `std_msgs/Empty`); timer polls dispatch; on active-subtask change (re)creates a subscription to its terminal topic; subscription callback → `monitor.record_event(topic)`; spin loop advances on `should_advance()` exactly like the CLI. `conditioning: prompt` subtasks keep the clear-only baseline path.
- README: document the live-ROS caveat (wrapper untested against a real Bridge Robot; core is covered by tests).

**Tests** (extend `tests/test_bridge.py`):
- predicate subtask does not advance until `record_event` on its own topic; unrelated topics don't trigger;
- no-predicate subtask advances after pack push;
- full lifecycle: open (terminal) → place (prompt mode) with a stub transport asserting the set/clear sequence and completion ordering.

## 3. Stage-YAML-driven registry build (`so101_icl/data.py`)

Closes the "holdout counts live in CLI flags, YAML keys ignored" inconsistency.

- `specs_from_stage_config(stage: dict) -> (list[DatasetSpec], build_kwargs)` — pure function: reads `dataset.repo_id` **or** `repo_ids`, `root`, `grouping_key`, `camera_rename` (`droid` / `pi05_base` preset name or explicit map), `alias_map`, `holdout_groups`/`holdout_tasks` (`{eval, test}`), `task_registry` (output path), plus `train.seed`.
- CLI: `build-registry --from-config <stage.yaml>` (mutually exclusive with the manual flags).

**Tests**: `specs_from_stage_config` on the smoke YAML reproduces exactly what the current manual CLI invocation produces; the droid YAML translates to one DatasetSpec with `task_category` grouping + DROID camera preset (no data needed — translation only).

## Wrap-up
- Update `so101_icl/README.md`: new commands (`check-stats --fix`, `build-registry --from-config`, bridge `--advance-mode`) and remove the corresponding "small gaps" caveats.
- Full light-test sweep + one M0 spot-run (`test_zero_init.py TestRegistry` only — heavy parity is unaffected by these changes) to confirm nothing regressed.
- No commits unless you ask; changes stay on `dev/icl`.

Not in scope (unchanged deferrals): droid downloads and stage-1 training (joint session), multi-camera demo keyframes, grad-accum fallback testing, live Bridge-Robot run.