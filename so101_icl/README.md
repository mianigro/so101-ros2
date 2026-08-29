# so101_icl — demo-conditioned π0.5 on SO-101

In-context demonstration conditioning for π0.5 (`VLA_PLAN.md` Path 2,
built per `ICL_IMPLEMENTATION.md` Rev 2): frozen `lerobot/pi05_base` +
DemoEncoder + LoRA adapters on the VLM attention. Integration is
subclassing + runtime registration only — the single repo edit outside this
package is `"pi05_icl"` in `policy_server/inference_engine.py`
(`SUPPORTED_POLICIES`) plus the `peft` dependency in `pixi.toml`.

## Layout

```
so101_icl/
├── train_icl.py / serve_icl.py / eval_icl.py   # entry wrappers (pixi lerobot env)
├── configs/          # stage 1 (droid), stage 2 (so101), local smoke + registries
├── tests/            # M0 zero-init gate, prefix shapes, sampler, bridge, transport
└── so101_icl/
    ├── configuration_pi05_icl.py  # ICLConfig(PI05Config), registered "pi05_icl"
    ├── modeling_pi05_icl.py       # PI05ICLCore (embed_prefix splice) + PI05ICLPolicy
    ├── processor_pi05_icl.py      # delegates to pi05 (registry convention)
    ├── demo_encoder.py            # attention pool + traj MLP, all zero-init
    ├── lora.py                    # manual peft injection, adapter save/load
    ├── data.py                    # task registry + support/query sampler + CLI
    ├── train_loop.py              # Accelerate loop + CLI main
    ├── registration.py            # import to register the policy type
    ├── demo_transport.py          # ZMQ side channel (set/clear/status)
    ├── bridge_icl_node.py         # dispatch-package consumer (ROS-free core)
    └── eval_icl.py                # offline 3-condition eval + stage-2 drivers
```

## Commands

```bash
# tests (M0 needs CUDA + the cached lerobot/pi05_base)
pixi run -e lerobot python so101_icl/tests/test_zero_init.py
pixi run -e lerobot python -m unittest discover -s so101_icl/tests -p "test_*.py"  # or per-file

# registry from on-disk datasets (never downloads)
python -m so101_icl.data smoke-local                       # local SO-101 datasets
python -m so101_icl.data build-registry --from-config so101_icl/configs/icl_smoke_local_v1.yaml
python -m so101_icl.data build-registry --datasets <repo[,root]> --grouping-key task --out ...

# stats preflight for pi05 quantile normalization (q01/q99)
python -m so101_icl.data check-stats --datasets <repo[,root]>
python -m so101_icl.data check-stats --datasets <repo[,root]> --fix --yes   # in-place recompute

# training (single GPU; 2x4080: accelerate launch --num_processes 2 --multi_gpu ...)
python so101_icl/train_icl.py --config so101_icl/configs/icl_smoke_local_v1.yaml
python so101_icl/train_icl.py --config so101_icl/configs/icl_finetune_so101_v1.yaml

# live dashboards: every run writes metrics.jsonl AND tensorboard event
# files under <output_dir>/tensorboard (set train.tensorboard: false to disable)
tail -f so101_icl/runs/<run>/metrics.jsonl | jq .
pixi run -e lerobot tensorboard --logdir so101_icl/runs --port 6006

# data tooling (pixi task sets PYTHONPATH; always run from the repo root)
pixi run -e lerobot icl_data smoke-local
pixi run -e lerobot icl_data build-registry --from-config so101_icl/configs/icl_pretrain_droid_v1.yaml
pixi run -e lerobot icl_data check-stats --datasets lerobot/droid_1.0.1 --fix --yes

# serving (registers pi05_icl, starts demo side channel :8661, runs ZMQ main)
pixi run -e lerobot icl_server --host 0.0.0.0 --port 8090

# offline eval (stage 1): demo vs demo-zeroed vs bare-prompt query loss
python so101_icl/eval_icl.py offline --adapter <run>/best --registry <reg.json> \
    --config <stage.yaml> --k 1 2 4 --output report.md

# bridge (ROS-free CLI; --stdin-events feeds terminal topics for smoke runs)
python -m so101_icl.bridge_icl_node --api-base ... --workspace ... --mission-id ... \
    --advance-mode terminal --stdin-events
# live ROS operation: ros2 run via BridgeICLNode (rclpy wrapper; core logic
# is unit-tested, the wrapper itself is not yet exercised against a live
# Bridge Robot)
```

## Training
```bash
# 1. smoke on the 464 MB dataset first (recommended before the big download)
pixi run -e lerobot python -c "from huggingface_hub import snapshot_download; snapshot_download('lerobot/droid_100', repo_type='dataset')"

# 2. the ~85 GB subset (chunk-filtered, confirmation prompt)
pixi run -e lerobot python -m so101_icl.data download-subset --start 0 --end 9 --yes

# 3. registry from the downloaded chunks only + stats check
pixi run -e lerobot python -m so101_icl.data build-registry --from-config so101_icl/configs/icl_pretrain_droid_v1.yaml
pixi run -e lerobot python -m so101_icl.data check-stats --datasets lerobot/droid_1.0.1 --fix --yes

# 4. stage-1 training on both 4080s (~half a day for 10k steps)
pixi run -e lerobot accelerate launch --num_processes 2 --multi_gpu \
  so101_icl/train_icl.py --config so101_icl/configs/icl_pretrain_droid_v1.yaml

# 5. M2 gate: demo vs demo-zeroed vs bare on held-out task_category groups
pixi run -e lerobot python so101_icl/eval_icl.py offline \
  --adapter so101_icl/runs/icl_pretrain_droid_v1/final \
  --registry so101_icl/configs/task_registry_droid.json \
  --config so101_icl/configs/icl_pretrain_droid_v1.yaml --k 1 2 4 --output m2_report.md
  ```

## Deferred (planned joint session)

- DROID downloads: `python -m so101_icl.data download-subset --start 0 --end 9 --yes`
  then `build-registry --datasets lerobot/droid_1.0.1,<root> --grouping-key
  task_category --droid --out so101_icl/configs/task_registry_droid.json`
  (implemented, deliberately not executed in the initial build).
- Stage-1 pre-training run (M2), stage-2 data collection + fine-tune (M3),
  real-robot eval (M4), live bridge integration (M5).

## Honest deviations from ICL_IMPLEMENTATION.md

- **Bit-identity with a demo pack at gate 0 does not hold.** Present-but-zero
  demo tokens carry attention keys of exactly 0, which after softmax steal a
  large fraction of every query's attention (~0.8 chunk delta at init). Exact
  base parity holds for the **no-pack** path (asserted bit-identical, M0
  test_03); with a pack the model must learn to down-weight demo keys via
  their embeddings, and `loss_demo_zeroed` tracks that during training.
- **Zero-init recipe fixed (dead saddle).** The doc zeroes BOTH the demo
  out-projections and the gates; `gate * proj(x)` with both factors zero has
  zero gradient w.r.t. both — the demo branch can never leave zero (observed:
  gates pinned at 0.0 for 200 steps). Only the gates (and order embedding)
  start at zero; the out-projections keep their default init so the gate
  gets gradient immediately.
- **peft freezes the DemoEncoder during injection**
  (`peft/tuners/tuners_utils.py:1057` marks only adapters trainable);
  `lora.inject_lora` re-enables `demo_encoder.*` afterwards.
- **The lerobot processor pipeline drops unknown batch keys** — the `icl.*`
  demo-pack fields must bypass it and be re-merged after preprocessing
  (`train_loop._split_icl_fields`); otherwise training silently runs
  bare-prompt.
- **Serving checkpoints ship LoRA merged into the base weights**
  (`modeling_pi05_icl._merge_lora_state_dict`): a LoRA-injected state dict
  has `q_proj.base_layer.*` names that a freshly constructed policy cannot
  strict-load. Adapter-only artifacts keep the unmerged LoRA for training.
- **`lerobot-edit-dataset` in-place recompute quirks**: without
  `--new_root=<same dir>` the handler copytrees the whole dataset, its
  `--root`/`--new_root` are FULL dataset dirs (unlike the parent-dir style
  some tools use), and draccus wants underscores (`--repo_id`, not
  `--repo-id`). `so101_icl.data check-stats --fix` wraps all of this.
- **`lerobot` root semantics**: the dataset `root` argument replaces the
  whole dataset directory (must include `repo_id`).
