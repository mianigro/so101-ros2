# so101_icl — demo-conditioned π0.5 on SO-101

In-context demonstration conditioning for π0.5: a frozen
`lerobot/pi05_base` plus a small `DemoEncoder` (attention-pooled demo
keyframes + trajectory tokens) and LoRA adapters on the VLM attention. At
inference, up to `k≤4` demonstrations of the current task are spliced
into the PaliGemma prefix as extra tokens — the robot "learns" the task
from the demos in-context, once per subtask, with no weight updates.

Design, theory and as-built deviations: [`ICL_IMPLEMENTATION.md`](ICL_IMPLEMENTATION.md)
(Rev 3 — documents the implementation as built, incl. the §11 known-gaps list).

**Status**: M0 (zero-init gate) and M1 (overfit smoke) done; a stage-1
training run exists (`runs/icl_pretrain_droid_v1/` metrics + tensorboard)
but the M2 gate has not been formally evaluated; the stage-2 sim/real
campaigns (M3/M4) and the ROS bridge (M5) are code-complete and unrun.
The blocking prerequisite is the stage-2 teleop data campaign.

Integration is subclassing + runtime registration only — the only repo
edits outside this package are `"pi05_icl"` in
`policy_server/inference_engine.py` (`SUPPORTED_POLICIES`) and
`pixi.toml` (the `peft`/`tensorboard` deps and the `icl_data` /
`icl_server` / `icl_sim_server` / `icl_bridge` tasks).

## Layout

```
so101_icl/
├── train_icl.py / serve_icl.py / eval_icl.py   # entry wrappers (pixi lerobot env)
├── serve_rollout_icl.py          # rollout-wire ICL server (M3/M4 campaigns)
├── eval_sim_campaign.py          # Isaac Sim campaign (M3; bootstraps Isaac itself)
├── configs/                      # stage 1 (droid), stage 2 (so101), smoke + registries
├── runs/<run>/                   # metrics.jsonl + tensorboard + adapter checkpoints
├── tests/                        # package, zero-init gate, prefix shapes, sampler,
│                                 # data tools, conditions, latency, offline report,
│                                 # bridge, transport, serving e2e
├── so101_icl_bridge/             # M5: ament_python ROS 2 package (colcon-buildable)
└── so101_icl/                    # the python package
    ├── configuration_pi05_icl.py # ICLConfig(PI05Config), registered "pi05_icl"
    ├── modeling_pi05_icl.py      # PI05ICLCore (embed_prefix splice) + PI05ICLPolicy
    │                             #   + merged-LoRA serving checkpoints
    ├── processor_pi05_icl.py     # delegates to pi05 (registry convention)
    ├── demo_encoder.py           # attention pool + traj MLP; only gates zero-init
    ├── lora.py                   # manual peft injection, adapter save/load
    ├── data.py                   # task registry + support/query sampler + CLI
    ├── train_loop.py             # Accelerate loop + CLI main
    ├── registration.py           # import to register the policy type
    ├── demo_transport.py         # ZMQ side channel (set/clear/status, :8661)
    ├── conditions.py             # full_icl / prompt_enriched / bare_prompt arms
    ├── latency.py                # latency recorder, p95 gate, server probe
    ├── bridge_icl_node.py        # dispatch-package consumer (ROS-free core)
    └── eval_icl.py               # offline 3-condition eval + stage-2 drivers
```

## Tests

```bash
pixi run -e lerobot python so101_icl/tests/test_zero_init.py   # M0 needs CUDA + cached pi05_base
for t in so101_icl/tests/test_*.py; do PYTHONPATH=so101_icl pixi run -e lerobot python $t; done
```

## Data (icl_data CLI — always run from the repo root; never downloads implicitly)

```bash
# registry for the on-disk local SO-101 datasets (smoke)
pixi run -e lerobot icl_data smoke-local
# registry from a stage YAML (single source of truth: datasets, holdouts, seed, path)
pixi run -e lerobot icl_data build-registry --from-config so101_icl/configs/icl_smoke_local_v1.yaml
pixi run -e lerobot icl_data build-registry --from-config so101_icl/configs/icl_pretrain_droid_v1.yaml
# manual: --datasets <repo[,root]> --grouping-key task [--droid] --out <reg.json>

# stats preflight for pi05 quantile normalization (q01/q99)
pixi run -e lerobot icl_data check-stats --datasets <repo[,root]>
pixi run -e lerobot icl_data check-stats --datasets <repo[,root]> --fix --yes   # in-place recompute

# DROID episode-range download (exact shard file list from meta; GiB preview)
pixi run -e lerobot icl_data download-subset --start 0 --end 9999 --yes   # ~10k episodes, ~85 GB
```

Every dataset open is pinned to the on-disk episodes with
`HF_HUB_OFFLINE=1` (`data.open_local_dataset`) — a partial subset can
never trigger lerobot's full-dataset re-download.

## Training

```bash
# smoke on local SO-101 data first (no downloads)
python so101_icl/train_icl.py --config so101_icl/configs/icl_smoke_local_v1.yaml

# stage 1 on both 4080s (after download-subset + build-registry + check-stats;
# batch 4/GPU + grad_accum 2 — 16 GB cards OOM at 8)
pixi run -e lerobot accelerate launch --num_processes 2 --multi_gpu \
  so101_icl/train_icl.py --config so101_icl/configs/icl_pretrain_droid_v1.yaml

# live dashboards: every run writes metrics.jsonl AND tensorboard events
tail -f so101_icl/runs/<run>/metrics.jsonl | jq .
pixi run -e lerobot tensorboard --logdir so101_icl/runs --port 6006
```

## Serving

```bash
# policy-server wire: registers pi05_icl, demo side channel :8661, ZMQ main
pixi run -e lerobot icl_server --host 0.0.0.0 --port 8090

# rollout wire (self-improve rollout server, :8660) for the M3/M4 campaigns
pixi run -e lerobot icl_sim_server --repo-id so101_icl/runs/icl_finetune_so101_v1/serving \
    --policy-type pi05_icl --host 0.0.0.0 --port 8660
```

Serving checkpoints ship the LoRA **merged** into the base weights with
ROS camera-key/state-dim renames (`save_serving_checkpoint`) — a fresh
policy strict-loads them directly.

## Evaluation

```bash
# offline (stage 1): demo vs demo-zeroed vs bare-prompt query loss,
# k/F/traj ablations + per-group tables; defaults from the stage YAML
# `eval:` block; CLI flags override
python so101_icl/eval_icl.py offline --adapter <run>/best --registry <reg.json> \
    --config <stage.yaml> --k 1 2 4 --frames 4 6 8 --traj on off --output report.md
# M2 gate: median demo/zeroed <= 0.90 on held-out groups (per-group table
# in the report; a train-split fallback stamps a WARNING into it)

# sim campaign (M3) — against the rollout-wire server above
PYTHONPATH=so101_icl python so101_icl/eval_sim_campaign.py \
    --registry so101_icl/configs/task_registry_so101.json \
    --stats so101_icl/runs/icl_finetune_so101_v1/final/stage_stats.json \
    --trials 20 --output m3_report.md
# or via the harness: PYTHONPATH=so101_icl python -m so101_icl.eval_icl sim --registry ... --trials 20

# real campaign (M4) — per-condition supervised sessions on the robot
# (pushes/clears the demo pack per condition; --post converts + judges)
PYTHONPATH=so101_icl python -m so101_icl.eval_icl real \
    --registry so101_icl/configs/task_registry_so101.json \
    --stats .../stage_stats.json --repo-id .../serving \
    --server-address 127.0.0.1:8090 --post --output m4_report.md
```

## Bridge (M5)

```bash
# ROS-free CLI; --stdin-events feeds terminal topics for smoke runs
# (PYTHONPATH because the package lives under so101_icl/so101_icl)
PYTHONPATH=so101_icl python -m so101_icl.bridge_icl_node --api-base ... --workspace ... \
    --mission-id ... --k 4 --k-max 4 --advance-mode terminal --stdin-events

# live ROS operation: the so101_icl_bridge ament package
pixi run icl_bridge            # rclpy main in a ROS 2 shell (same entry point)
ros2 launch so101_icl_bridge bridge_icl.launch.py api_base:=... workspace:=... \
    mission_id:=... stats_path:=... k:=4 k_max:=4    # after colcon build
```

Core logic (ordering, selection, state machine, terminal monitor) is
unit-tested; the live-robot bring-up itself is a manual session.

## Design notes & deviations

Full detail in `ICL_IMPLEMENTATION.md` §2.1/§4.0/§11. The load-bearing
items:

- **Zero-init recipe (dead-saddle fix).** Only the demo gates (and order
  embedding) start at zero; the doc's original "zero the out-projections
  too" makes `gate * proj(x)` a saddle with zero gradient in both factors
  (observed: gates pinned at 0 for 200 steps). Bit-parity with pi05_base
  holds and is asserted for the **no-pack** path only — present-but-gated
  demo tokens perturb attention at init, and training learns to
  down-weight them (`loss_demo_zeroed` tracks it).
- **The lerobot processor drops unknown batch keys** — the `icl.*`
  demo-pack fields bypass it and re-merge after preprocessing
  (`train_loop._split_icl_fields`); otherwise training silently runs
  bare-prompt.
- **Serving checkpoints ship LoRA merged into the base weights** — a
  LoRA-injected state dict has `q_proj.base_layer.*` names a fresh policy
  cannot strict-load; adapter-only artifacts keep the unmerged LoRA for
  training.
- **Hub-completion pinning**: `LeRobotDataset` re-downloads whatever the
  full `meta/` lists but the disk lacks — every open goes through
  `data.open_local_dataset` (pinned `on_disk_episodes`, contiguous-prefix
  enforced, `HF_HUB_OFFLINE=1`).
- **peft freezes the DemoEncoder during injection** — `lora.inject_lora`
  re-enables `demo_encoder.*` afterwards.
- **`lerobot-edit-dataset` in-place recompute quirks** (needs
  `--new_root=<same dir>`, full-dataset `--root`, draccus underscores) —
  `icl_data check-stats --fix` wraps all of it.
