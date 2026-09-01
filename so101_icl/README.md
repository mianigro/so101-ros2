# so101_icl — demo-conditioned π0.5 on SO-101

Show the robot a few demonstrations of a task and it does the task — no
weight updates at deployment.

A frozen `lerobot/pi05_base` VLA is conditioned on up to `k ≤ 4`
demonstrations per subtask. A small trainable side (`DemoEncoder` + LoRA
adapters on the VLM attention) is meta-trained across many tasks so that
demos in context improve action prediction. Stage 1 pretrains that skill
on DROID, stage 2 fine-tunes it on SO-101 data, and at runtime the demo
pack is swapped over a side channel.

Full design and as-built detail: [`ICL_IMPLEMENTATION.md`](ICL_IMPLEMENTATION.md).

**Status**

| Milestone | What | State |
|---|---|---|
| M0 | zero-init / base-parity property | done, tested |
| M1 | overfit smoke on local data | done |
| Stage 1 | DROID ICL pretraining | stopped early; re-run with rev-6 training loop (ratio-based selection, resumable checkpoints) |
| M2 | offline gate (demos must beat demo-zeroed) | not yet evaluated |
| Stage 2 | SO-101 fine-tune | blocked on teleop data (≥20 tasks × ≥20 demos) |
| M3/M4 | sim/real campaigns | code complete, unrun |
| M5 | ROS bridge | code complete, unit-tested; never against a live dispatch API |

## The flow

```
data ──► stage 1 (DROID) ──► M2 gate ──► stage 2 (SO-101) ──► serve ──► bridge ──► robot
```

Each phase below is one section with the exact commands.

## 1. Data

Everything runs from the repo root. Nothing downloads implicitly — every
dataset open is pinned to on-disk episodes (`HF_HUB_OFFLINE=1`).

**Sampling contract (rev 6):** support demos are drawn from episodes in the
query's **task cluster** — embedded task-string clusters from
`icl_data cluster-tasks` (see stage 1, step 3) — falling back to exact
task string, then to category-wide selection, when no cluster cache or
task strings exist. This matches deployment, where the bridge always
pushes demos of the subtask being executed. Each episode alternates roles
across training — query (the prediction target) once per epoch, support
(the example) for its same-cluster siblings on other draws.

Run only the block for what you're doing:

**Smoke test (local SO-101 data only)**

```bash
pixi run -e lerobot icl_data smoke-local
```

**Stage 1 — DROID pretraining**

```bash
# 1. download the subset (~10k episodes, ~85 GB)
pixi run -e lerobot icl_data download-subset --start 0 --end 9999 --yes

# 2. build the registry from what's now on disk
pixi run -e lerobot icl_data build-registry --from-config so101_icl/configs/icl_pretrain_droid_v1.yaml

# 3. cluster task strings
pixi run -e lerobot icl_data cluster-tasks --from-config so101_icl/configs/icl_pretrain_droid_v1.yaml

# 4. check pi05 quantile stats; add --fix --yes if it reports missing q01/q99
pixi run -e lerobot icl_data check-stats --datasets lerobot/droid_1.0.1

# 5. keypoint cache for the demo branch
pixi run icl_data precompute-keypoints --from-config so101_icl/configs/icl_pretrain_droid_v1.yaml
```

**Stage 2 — SO-101 fine-tuning (own teleop + sim data)**

```bash
# 1. registry over the curated SO-101 datasets
pixi run -e lerobot icl_data build-registry --from-config so101_icl/configs/icl_finetune_so101_v1.yaml

# 2. stats check; add --fix --yes if q01/q99 are missing
pixi run -e lerobot icl_data check-stats --datasets local/so101_teleop_v1,local/so101_sim_round2

# 3. keypoint cache (frames from the stage config's demo_encoder)
pixi run icl_data precompute-keypoints --from-config so101_icl/configs/icl_finetune_so101_v1.yaml
```

## 2. Stage 1 — pretrain ICL on DROID

```bash
pixi run -e lerobot accelerate launch --num_processes 2 --multi_gpu \
  so101_icl/train_icl.py --config so101_icl/configs/icl_pretrain_droid_v1.yaml
```

Prerequisites: `download-subset`, `build-registry`, `check-stats`,
`precompute-keypoints`.

**What training does (rev 6)**

1. **Demo-usage hinge loss.** Every step, the batch is replayed with demo
   gates forced to 0 (identical flow-matching noise) and the loss gains
   `w · relu(loss_demo − loss_zeroed + margin)`. Gradient fires exactly
   when demos are not helping — the optimizer cannot profit from ignoring
   them. Knobs: `train.demo_usage_weight`, `train.demo_usage_margin`.
   Cost: ~1.4x step time.
2. **Bursty task sampling.** Task groups are drawn with Zipfian popularity
   and arrive in contiguous bursts (`group_sampling: bursty`,
   `burst_length`), the data regime in which in-context learning emerges.
3. **Keypoint demo tokens.** Demos carry SIFT correspondences
   (`demo_encoder.keypoints`) in addition to pooled SigLIP features and
   trajectories — each branch gated, spliced into the PaliGemma prefix.
4. **Ratio-based `best/` selection (rev 6).** At each validation pass the
   held-out batches run twice — demos on and gates zeroed (same noise) —
   and `best/` is saved on the held-out `demo/zeroed` ratio (the §6.4/M2
   acceptance signal), not on bare-prompt loss, which a demo-ignoring
   checkpoint would win. Bare-prompt `val/loss` stays logged as a drift bound.
5. **Resumable checkpoints (rev 6).** Every `best/`, `step_N/` and
   `final/` dir also carries `trainer_state.pt` (step, optimizer,
   scheduler, RNG, dataset epoch). `--resume-from <dir>` continues the LR
   schedule where it stopped; without trainer state it is a warm start
   (stage-2 semantics).
6. Carried over: k-curriculum, `gate_floor`, `language_dropout`.

Watch `metrics.jsonl`: `usage_loss` → 0 means demos reliably help;
`demo_zeroed_ratio` and `val/demo_ratio` < 1 are the same signal as a
ratio. The logged `gate_vis/gate_traj/gate_kp` are the EFFECTIVE gates
(rev 7: `gate_budget * softmax(logits)`); they sum to the budget (default
`n_branches * gate_floor`), so one rising means the others fall — watch
the allocation, not absolute growth. All three near `floor/1` uniformity
while `lora_norm` races up means the LoRA is compensating for a
demo branch the gates never learned to use.

## 3. M2 gate — did demos learn to matter?

```bash
python so101_icl/eval_icl.py offline \
    --adapter so101_icl/runs/icl_pretrain_droid_v1/best \
    --registry so101_icl/configs/task_registry_droid.json \
    --config so101_icl/configs/icl_pretrain_droid_v1.yaml --output m2_report.md
```

Gate: median demo/zeroed ≤ 0.90 **on held-out task groups** (the report
stamps a warning if it fell back to train groups). Do not start stage 2
before this passes.

## 4. Stage 2 — fine-tune on SO-101 (real + sim data)

```bash
python so101_icl/train_icl.py --config so101_icl/configs/icl_finetune_so101_v1.yaml
```

Short-hot per GEN-1.5's few-step evidence: 500 steps, lr 1e-4,
checkpoint/validation every 50, initialized from stage-1 `best/`.
`best/` is tracked on the held-out demo/zeroed ratio (same selection rule
as stage 1); extend past 500 only while that ratio is still falling.
Blocked on the teleop campaign (≥20 tasks × ≥20 demos).

## 5. Serving

```bash
# main policy-server wire (ZMQ :8090) + demo side channel (:8661)
pixi run -e lerobot icl_server --host 0.0.0.0 --port 8090

# rollout wire (:8660) for the M3/M4 campaigns
pixi run -e lerobot icl_sim_server \
    --repo-id so101_icl/runs/icl_finetune_so101_v1/serving \
    --policy-type pi05_icl --host 0.0.0.0 --port 8660
```

Serving checkpoints ship the LoRA **merged** into the base weights with
camera-key/state-dim renames — a fresh policy strict-loads them.

## 6. Evaluation campaigns

```bash
# M3 sim: per-condition success-rate table against the rollout-wire server
PYTHONPATH=so101_icl python -m so101_icl.eval_icl sim \
    --registry so101_icl/configs/task_registry_so101.json \
    --stats so101_icl/runs/icl_finetune_so101_v1/final/stage_stats.json \
    --trials 20 --output m3_report.md

# M4 real: supervised per-condition sessions on the robot (--post judges)
PYTHONPATH=so101_icl python -m so101_icl.eval_icl real \
    --registry so101_icl/configs/task_registry_so101.json \
    --stats so101_icl/runs/icl_finetune_so101_v1/final/stage_stats.json \
    --repo-id so101_icl/runs/icl_finetune_so101_v1/serving \
    --server-address 127.0.0.1:8090 --post --output m4_report.md
```

Conditions per campaign: `full_icl` (demo pack), `prompt_enriched`,
`bare_prompt`. M4 gate: full_icl ≥ prompt_enriched ≥ bare_prompt success
trend, p95 latency within +20 %.

## 7. Bridge (M5) — dispatch missions → demo packs

```bash
# ROS-free CLI smoke (terminal events fed on stdin)
PYTHONPATH=so101_icl python -m so101_icl.bridge_icl_node --api-base <url> \
    --workspace <ws> --mission-id <id> --k 4 --advance-mode terminal --stdin-events

# live ROS 2 (after colcon build of so101_icl_bridge)
ros2 launch so101_icl_bridge bridge_icl.launch.py api_base:=<url> workspace:=<ws> \
    mission_id:=<id> stats_path:=<stats.json> k:=4 k_max:=4
```

Per subtask: top-k demo selection → keyframe fetch → shared image
transform → `set_demo_pack` on the side channel → terminal predicate gates
advancement → `clear` on switch. Add `--keypoints` / `keypoints:=true`
when serving a keypoint-enabled checkpoint.

## Tests

```bash
cd so101_icl/tests && PYTHONPATH=so101_icl:<repo_root> \
  <pixi lerobot env python> -m unittest discover -p "test_*.py"
```

101 tests, CPU-only except `test_zero_init.py` (needs CUDA + cached
`pi05_base`). Covers the rev-5/6 pieces (hinge math, bursty sampler,
keypoint branch, zero-init, bit-identity when the branch is off) and the
full bridge path (stub dispatch → real ZMQ side channel).

## Package layout

```
so101_icl/
├── train_icl.py / serve_icl.py / eval_icl.py   entry wrappers
├── serve_rollout_icl.py / eval_sim_campaign.py rollout-wire server, M3 campaign
├── configs/                       stage 1 / stage 2 / smoke YAMLs + registries
├── runs/<run>/                    metrics.jsonl, tensorboard, adapter checkpoints
├── tests/                         101 unit + integration tests
├── so101_icl_bridge/              ament_python ROS 2 package (M5)
└── so101_icl/
    ├── configuration_pi05_icl.py  ICLConfig: demo_encoder (vis/kp/traj), lora
    ├── modeling_pi05_icl.py       PI05ICLCore prefix splice; serving checkpoints
    ├── demo_encoder.py            gated vis + keypoint + trajectory branches
    ├── lora.py                    structural target discovery, peft injection, validated adapter save/load
    ├── data.py                    registries, sampler, bursty sampler, kp CLI
    ├── train_loop.py              Accelerate loop, usage-hinge loss, ratio-based selection, resumable state
    ├── demo_transport.py          ZMQ side channel :8661
    ├── conditions.py              campaign arms (full_icl / enriched / bare)
    ├── bridge_icl_node.py         dispatch consumer (ROS-free core + rclpy shell)
    └── eval_icl.py                offline M2 eval + sim/real drivers
```

## Gotchas that have bitten before

Full detail in `ICL_IMPLEMENTATION.md` §2.1/§4.0/§11/§13. The short list:

- **The lerobot processor drops unknown batch keys** — `icl.*` fields
  bypass it and re-merge after preprocessing (`train_loop._split_icl_fields`).
- **`from_pretrained` swallows strict-load failures** — loading is
  verified by a frozen-weight probe (`assert_weights_loaded`).
- **peft freezes the DemoEncoder during injection** — `lora.inject_lora`
  re-enables `demo_encoder.*` afterwards.
- **`LeRobotDataset` re-downloads whatever `meta/` lists but disk lacks**
  — every open goes through `data.open_local_dataset`.
- **Zero-init only the gates** — zeroing the out-projections too creates a
  dead saddle (observed: gates pinned at 0 for 200 steps).
- **Stage-1 `final/` does not exist** (run stopped early) — stage 2 loads
  `best/`; stage 2 cannot run before stage 1 produces one. Re-running
  stage 1 is safe: every checkpoint carries `trainer_state.pt`, so
  `--resume-from <step_dir>` continues the LR schedule mid-run.
- **Adapter loading is strict about structure** — rank/alpha/demo_encoder
  must match the recorded meta; targets may only widen
  (`vlm_attention` → `vlm_attention+expert`, the §9 expert-LoRA
  contingency — fresh expert adapters start at B=0). Dropout may differ.
