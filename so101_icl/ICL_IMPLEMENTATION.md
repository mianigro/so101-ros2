# ICL Implementation Plan — demo-conditioned π0.5 on SO-101

Build plan for `so101_icl`: in-context demonstration conditioning for π0.5
(frozen base + demo encoder + LoRA adapters) — the "demo tokens in the
prefix" path (in-context conditioning proper, as opposed to
language-enrichment-only prompting). Integration happens through
subclassing, runtime registration, and composition — the only repo edits
outside the new package are `"pi05_icl"` in `policy_server`'s
`SUPPORTED_POLICIES` and pixi tasks/deps (§1, §10).

**Rev 3 (2026-08-30).** This revision documents the implementation **as
built**: the design below now matches the code in `so101_icl/`, with the
deviations discovered during implementation folded in (zero-init recipe,
checkpoint layout, serving/campaign machinery) and remaining gaps collected
in §11. Rev 2 (2026-08-29) introduced the foundational base
(`lerobot/pi05_base`) and two-stage training:

- **Stage 1 — ICL pre-training** on the open-source DROID dataset
  (`lerobot/droid_1.0.1`), teaching the adapter *how to use demos*
  in-context across many tasks.
- **Stage 2 — fine-tuning with ICL** on our own SO-101 data: the *same*
  LoRA + DemoEncoder + gates continue training at lower LR on robot
  exports, adapting the in-context mechanism to our embodiment and tasks.

Its verification pass re-checked every §1 anchor against installed
LeRobot 0.6.1 (correcting the strict-load trap of §4.0, the
`so101_30hz.yaml` data-features attribution, PEFT anchor line numbers and
registry mechanics).

---

## 1. Verified integration surface (LeRobot 0.6.1)

| Anchor | Location | What it gives us |
|---|---|---|
| `PI05Policy(PreTrainedPolicy)` | `modeling_pi05.py:711` | Policy wrapper: `from_pretrained`, `select_action`, `predict_action_chunk`, `forward` (loss). Subclassable — must define both `config_class` and `name` (`__init_subclass__` enforces it, `policies/pretrained.py:123-128`). |
| `PI05Pytorch(nn.Module)` | `modeling_pi05.py:401` | Core model; `self.model = PI05Pytorch(...)` lives in `PI05Policy.__init__` (:736). VLM inside is `self.paligemma_with_expert` (:417). **No `state_proj` exists** — state is discretized into the PaliGemma prompt (`max_state_dim=32` is prompt-tokenization padding, not a projection). |
| `embed_prefix(images, img_masks, tokens, masks)` | `modeling_pi05.py:483` | Builds the prefix (SigLIP image tokens per camera + language tokens) and returns `(embs, pad_masks, att_masks)`. **The single injection point for demo tokens.** Called by both training `forward` (:570) and inference `sample_actions` (:640) — one override covers both. |
| Attention semantics | `embed_prefix` / `embed_suffix`; `policies/common/vla_utils.py:61-90` | `att_masks` entry `0` = bidirectional prefix token, `1` = causal suffix token (exact openpi `make_att_2d_masks` copy). Demo tokens get `0` → they join the bidirectional prefix block. `position_ids = cumsum(pad_masks) - 1` is computed in `PI05Pytorch.forward` (:584), `sample_actions` (:642), and `denoise_step` (:689-690, via `prefix_offsets = sum(prefix_pad_masks)`) — inserting tokens *before the suffix* extends everything downstream automatically. Note `embed_suffix` (:557-558) marks only the *first* action token `1`, so prefix tokens can never attend to actions. |
| `sample_actions` KV cache | `modeling_pi05.py:640-669` | Prefix is encoded **once per action-chunk query** with `use_cache=True` (:647-653); the 10 denoising steps run expert-only forwards over cloned `past_key_values` (:671-703). Demo tokens in the prefix are therefore paid once per chunk, not per denoise step. |
| `PaliGemmaWithExpertModel.embed_image` / `embed_language_tokens` | `modeling_pi05.py:284/295` | Reusable frozen SigLIP + vision→Gemma projection for demo keyframes — same embedding space as the prefix. `embed_image` takes `[B,3,224,224]` in [-1,1], returns projected features `[B,256,2048]`, callable standalone and detachable. |
| Flow-matching loss | `modeling_pi05.py:564-611` | `PI05Policy.forward` (`:1058`) computes it; inherited untouched. |
| Gradient checkpointing | `gradient_checkpointing_enable` `:443` | Already implemented (non-reentrant, `use_reentrant=False`). |
| PEFT support | `lerobot/configs/policies.py:68` (`use_peft`), `lerobot/policies/pretrained.py:381` (`wrap_with_peft`), `:493` (`_build_peft_config`), pi05 `_get_default_peft_targets` `modeling_pi05.py:1098` | LeRobot 0.6.1 ships LoRA wiring, **but** (a) `wrap_with_peft` freezes *all* parameters before wrapping — it would freeze our DemoEncoder too, and (b) the default pi05 target regex matches modules that don't exist in pi05 (`state_proj`, `action_time_mlp_*`). We therefore inject adapters manually (§4.3) with explicit targets — the overrides are mandatory, not polish. |
| VLM attention module path | `modeling_pi05.py:574` | `paligemma_with_expert.paligemma.model.language_model.layers[i].self_attn.q_proj` — exact peft target string (also `k/v/o_proj`). |
| Image transform | `PI05Policy._preprocess_images` `modeling_pi05.py:952-1016` | The resize-with-pad + [-1,1] SigLIP scaling lives **here**, not in the processor (`make_pre_post_processors` only renames/normalizes/tokenizes). Demo keyframes reuse `_preprocess_images` → zero train/serve skew. |
| Policy registry | `lerobot/policies/factory.py:389-435` | Convention-based resolution over the draccus choice registry: config class `<X>Config` in a `configuration_*` module → policy class `<X>Policy` in the sibling `modeling_*` module. `@PreTrainedConfig.register_subclass("pi05_icl")` at import time works from user code — no lerobot edits. Additionally `make_pre_post_processors` (factory.py:438-480) looks for `make_pi05_icl_pre_post_processors` in a `processor_pi05_icl` module — we must provide it (delegating to pi05's) or processor creation raises. |
| Weight loading | `PI05Policy.from_pretrained` `modeling_pi05.py:746-862` | Base weights load with the inherited `model.*` remapping — **but** the signature defaults `strict=True` (:759) and wraps `load_state_dict` in a try/except that only prints a warning (:859-861). See §4.0 gotcha 1. |
| Policy loading in server | `policy_server/inference_engine.py:305-307` | `get_policy_class(cfg.policy_type)` + `.from_pretrained(path)` — a runtime-registered `"pi05_icl"` class loads here with no changes. **However** `load_policy` (:290-292) first checks a hardcoded `SUPPORTED_POLICIES` list (:33-43) and raises `ValueError` for unknown names → **one-line edit required** to add `"pi05_icl"` (precedent: `"xvla"` was added the same way, see comment at :29-32). The policy type arrives at runtime in the client handshake (`RemotePolicyConfig`, built by `so101_inference/async_client.py:141-147` from ROS parameters) — no server-side config file changes. |
| Feature validation | `policy_server/inference_engine.py:88-128` | Validates only `observation.images.*` / `observation.state` against the setup schema. Demo keyframes are **not** observation keys → no schema conflict. |
| Serving pattern | `self-improve/serve_policy.py` (self-improve rollout server, port 8660); `policy_server/zmq_server.py` (real-robot path) | Standalone entry script in pixi `lerobot` env, `sys.path` insert, run server main. Mirrored for `so101_icl`. |
| Data features | `rosbag_to_lerobot/config/so101_30hz.yaml` + `--setup` (cameras) | Timing yaml defines only robot type, fps 30, `observation.state` (6) / `action` (6) topics. **Camera features come exclusively from `--setup`** (`so101_bringup/config/setups/*.yaml`; `monomanual_dual_overhead` → wrist + overhead_1 + overhead_2), shape [480,640,3]. Datasets are **AV1 video** (`libsvtav1`), not JPEG — JPEG is only the ROS transport format. `chunk_size=50` / `max_state_dim=max_action_dim=32` are **pi05_base policy-config values**, not converter settings. LeRobot v3.0 exports. |
| Base checkpoint | `lerobot/pi05_base` | **Foundational** openpi π0.5 in LeRobot format (config `type: pi05`; inputs `observation.images.{base_0_rgb,left_wrist_0_rgb,right_wrist_0_rgb}`, state/action padded to 32, chunk 50, quantile normalization, tokenizer `google/paligemma-3b-pt-224`). Cached at `~/.cache/huggingface/hub/models--lerobot--pi05_base`. fp32 safetensors (16.6 GB) → **load with `dtype=bfloat16`**. |

---

## 2. Architecture

### 2.1 What is frozen / trainable

| Component | Params | State |
|---|---|---|
| PaliGemma VLM (`paligemma_with_expert.paligemma`) | ~2.9 B | **Frozen**, except LoRA adapters below |
| SigLIP vision tower (inside above) | ~0.4 B | **Frozen** (reused by demo encoder, features detached) |
| Action expert (`gemma_expert`, ~311 M) + `action_in/out_proj`, `time_mlp` | ~320 M | **Frozen** |
| LoRA on VLM attention (`q,k,v,o_proj`, r=32) | ~10 M | **Trained**, zero-init |
| `DemoEncoder` (pool + trajectory MLP) | ~2 M | **Trained**, out-projections at default init |
| Demo gates `g` (scalar per token block) + order embedding | 3 | **Trained**, init 0 |

Zero-init property, **as built** (rev 3 — differs from the original recipe):
LoRA `B` matrices are 0 by construction (peft default), and only the
**gates and the per-slot order embedding** start at zero
(`DemoEncoder.reset_icl_parameters`). The demo out-projections keep their
default init. The original plan also zeroed the out-projections, but
`gate * proj(x)` with **both** factors zero is a dead saddle — the gradient
is zero w.r.t. both, and the demo branch can never leave zero (observed:
gates pinned at 0.0 for 200 steps before the fix). With default-init
projections the gate receives gradient immediately while the model output
at init is still gated off.

Consequences for the M0 gate: with **no demo pack** the modified model is
bit-identical to `lerobot/pi05_base` (asserted, `tests/test_zero_init.py`).
With a pack, present-but-gated-off demo tokens carry attention keys of
exactly 0, which after softmax still perturb every query's attention
(~0.8 loss delta at init) — exact base parity with a pack does **not**
hold; training must learn to down-weight the demo keys via their
embeddings, and `loss_demo_zeroed` tracks that progress.

**Null-out failure mode + gate floor (run 3, stage 1).** With descriptive
task prompts (DROID `task_category`), language+observation suffice for
most batches, so the cheapest loss minimum is "make the demos irrelevant":
the optimizer drives the single scalar gates toward zero (observed:
`gate_vis` 0.05 → 0.027 with `demo_zeroed_ratio` climbing back above 1.1).
Two mitigations, both config knobs:

- `demo_encoder.gate_floor` (e.g. 0.1): the gates are initialized AT the
  floor and clamped below it (`clamp(gate, min=floor)` in `forward`) — the
  demo branch cannot be silenced through one scalar, while the gate still
  grows freely upward. `gate_scale=0` (the `loss_demo_zeroed` replay)
  multiplies AFTER the clamp, so the metric keeps working. With
  `gate_floor > 0` the zero-token-at-init property is intentionally
  given up; no-pack parity still holds (no pack → no demo tokens).
- `train.language_dropout` (e.g. 0.3): on that fraction of micro-batches
  the task prompt is replaced by a vague stand-in (`"do the task."`),
  making the demo pack the only task signal — direct gradient demand for
  the demo path. Whole-batch granularity, a dedicated RNG (the
  flow-matching noise stream is untouched), and the running application
  rate is logged as `lang_dropout_rate`.

### 2.2 Token flow

```
                         ┌──────────────────────── PI05ICLCore ──────────────────────┐
 cam images ────────────►│ embed_image (frozen SigLIP)  ── 256 tok/cam               │
 instruction ───────────►│ embed_language_tokens         ── ≤128 tok                 │
 demo keyframes ────────►│ DemoEncoder:                                              │
   (k≤4 demos ×          │   frames → embed_image (frozen, detached)                 │
    F=6 frames)          │   → per-demo attention pool (T=64 tok/demo, learned)      │
 demo trajectory ───────►│   states+actions (S=16, downsampled) → MLP (32 tok/demo)  │
   (optional, masked)    │   + presence mask + per-demo order embedding              │
                         │ prefix = [cams | demo tokens | language]  (all att=0)     │
                         │ suffix  = flow-matching action tokens (att=1, unchanged)  │
                         └───────────────────────────────────────────────────────────┘
```

Demo tokens are gated: `demo_tok' = g · pool_out(...)`, `g` init 0.
Training moves `g` and the adapters; inference uses the cached prefix KV
exactly as `sample_actions` does today.

### 2.3 Token and memory budget

| Item | Count / size |
|---|---|
| Camera prefix (2–3 cams × 256) | 512–768 tok |
| Language prefix | ≤128 tok |
| Demo tokens (k=4 × 64) | ≤256 tok |
| Demo trajectory tokens (k=4 × 32) | ≤128 tok |
| **Added prefix length** | **≤384 tok (~+50 %)** — paid once per chunk query (KV cache), denoise steps unchanged |
| Frozen base bf16 per card (pi05_base, 3.62 B params) | ~7.0–7.3 GB |
| Trainables + AdamW states | <0.5 GB |
| Activations (grad ckpt, batch 8/card, seq ≈1.6k) | ~5–7 GB |
| **Total VRAM** | ~13–15 GB |

### 2.4 Demo prefix lifecycle (runtime)

1. Bridge receives subtask S with conditioning pack → selects top-k demos
   (success outcomes first).
2. `set_demo_pack(keyframes, trajectory?, mask)` → DemoEncoder runs **once**
   (no grad) → demo embeddings cached on the policy object.
3. Every `predict_action_chunk` for subtask S reuses the cached demo tokens
   inside `embed_prefix` — per-tick cost is only the extra attention columns.
4. Subtask switch → `clear_demo_pack()` → re-encode for the next pack.

---

## 3. Package layout

```
so101_icl/
├── train_icl.py                  # entry wrapper --config <stage.yaml> → train_loop.main (mirrors self-improve/train_bc.py pattern)
├── serve_icl.py                  # policy-server wire: registers pi05_icl, demo side channel, ZMQ main
├── serve_rollout_icl.py          # rollout wire (self-improve rollout server, port 8660) variant for M3/M4
├── eval_icl.py                   # entry wrapper: offline (stage 1) + sim/real (stage 2) eval harness
├── eval_sim_campaign.py          # M3 Isaac Sim campaign (bootstraps itself into Isaac Sim's Python)
├── configs/
│   ├── icl_pretrain_droid_v1.yaml   # stage 1 (see §7)
│   ├── icl_finetune_so101_v1.yaml   # stage 2 (see §7)
│   ├── icl_smoke_local_v1.yaml      # M1 local smoke run
│   └── task_registry_*.json         # built by data.py CLI, per stage (droid / smoke_local; so101 pending)
├── runs/<run>/                   # metrics.jsonl + tensorboard events + adapter ckpts (best/final/step)
├── so101_icl_bridge/             # M5: ament_python ROS 2 package wrapping the bridge node
│   ├── launch/bridge_icl.launch.py
│   └── so101_icl_bridge/bridge.py # rclpy main; resolves so101_icl via PYTHONPATH/env/sibling
├── tests/
│   ├── test_package.py           # config registration, adapter save/load round-trip
│   ├── test_zero_init.py         # M0 gate (incl. weights-actually-loaded assertion, §4.0)
│   ├── test_prefix_shapes.py     # mask/position-id correctness with demo tokens
│   ├── test_sampler.py           # support/query sampler invariants
│   ├── test_data_tools.py        # registry building, stats checks, subset file lists
│   ├── test_conditions.py        # the three campaign arms + DemoPackBuilder
│   ├── test_latency.py           # latency recorder + gate + policy-server probe plumbing
│   ├── test_eval_offline.py      # offline report rendering + ablation-flag plumbing
│   ├── test_bridge.py            # dispatch parsing, ordering, state machine, demo packs
│   ├── test_bridge_package.py    # the ament package wiring
│   ├── test_serve.py             # demo transport wire format + SUPPORTED_POLICIES gate
│   └── test_serving_e2e.py       # serving checkpoint end-to-end
└── so101_icl/
    ├── __init__.py               # BASE_CHECKPOINT + camera-rename maps (PI05_BASE / DROID)
    ├── configuration_pi05_icl.py # ICLConfig(PI05Config): encoder + LoRA knobs; registered as "pi05_icl"
    ├── modeling_pi05_icl.py      # PI05ICLCore(PI05Pytorch), PI05ICLPolicy(PI05Policy),
    │                             #   splice_demo_tokens, merged-LoRA serving checkpoints (§4.1)
    ├── processor_pi05_icl.py     # make_pi05_icl_pre_post_processors → delegates to pi05's (registry requirement, §1)
    ├── demo_encoder.py           # DemoEncoder (attention pool + traj MLP + mask + gate_scale hook)
    ├── lora.py                   # manual peft injection, adapter save/load, setup_trainable_policy
    ├── data.py                   # ICLDataset, on-disk registry building, stats tools, download-subset CLI
    ├── train_loop.py             # Accelerate DDP training loop + CLI main
    ├── conditions.py             # full_icl / prompt_enriched / bare_prompt arms + DemoPackBuilder
    ├── latency.py                # LatencyRecorder, p95 latency gate, policy-server probe
    ├── eval_icl.py               # offline 3-condition eval (M2 gate, k/F/traj ablations) + sim/real drivers
    ├── registration.py           # runtime policy-registry insertion ("pi05_icl")
    ├── demo_transport.py         # ZMQ side channel: set/clear/status + engine monkey-patch + stage-stats I/O
    └── bridge_icl_node.py        # ROS-free core + optional rclpy shell: dispatch pack → top-k → transport
```

Plus: `policy_server/inference_engine.py` — `"pi05_icl"` added to
`SUPPORTED_POLICIES` — and `pixi.toml` (`peft`/`tensorboard` deps, the
`icl_data`/`icl_server`/`icl_sim_server`/`icl_bridge` tasks).

---

## 4. Module designs

### 4.0 Load-bearing gotchas (verified in the code, several found the hard way)

1. **`from_pretrained` swallows load failures.** `PI05Policy.from_pretrained`
   defaults `strict=True` (:759) and wraps `load_state_dict` in a broad
   try/except that only *prints* a warning (:859-861). Loading pi05_base
   weights into `PI05ICLPolicy` — which adds `demo_encoder`/gate params that
   are legitimately missing from the base checkpoint — would raise inside
   the try and return a model with **no weights loaded at all**, silently.
   → The ICL loader **must pass `strict=False`** (missing demo keys are then
   only logged, :836-844), and `tests/test_zero_init.py` must assert a
   frozen parameter actually equals its checkpoint tensor (e.g. one
   `paligemma` embedding row) before running the parity check.
2. **Registry naming is convention-based.** Config class ends in `Config`
   and lives in a `configuration_*` module; policy class is `<Name>Policy`
   in the sibling `modeling_*` module; a `processor_<name>` module with
   `make_<name>_pre_post_processors` must exist (ours delegates to
   `make_pi05_pre_post_processors`, `processor_pi05.py:91`). Subclasses must
   define both `config_class` and `name`.
3. **peft wiring.** `wrap_with_peft` (`pretrained.py:381`) freezes *all*
   parameters before wrapping — it would freeze the DemoEncoder; and the
   default pi05 target regex (`modeling_pi05.py:1098`) matches modules that
   don't exist (`state_proj`, `action_time_mlp_*`). We therefore inject
   adapters manually with `inject_adapter_in_model` and an explicit target
   regex (§4.3), keeping the `PreTrainedPolicy` outer type intact for the
   server. Additionally — found during implementation — **peft's tuner
   re-freezes everything except adapters at injection time**
   (`peft/tuners/tuners_utils.py:1057`), so `lora.inject_lora` must
   re-enable `demo_encoder.*` **after** `inject_adapter_in_model` returns.
4. **State is not projected.** π0.5 has no `state_proj`; state is
   discretized into the PaliGemma prompt (`max_state_dim=32` is prompt
   padding). The demo trajectory branch is therefore the *only* new consumer
   of raw state/action values, and it consumes them already normalized (its
   stats come from the active stage's policy preprocessor — see §6.5).
5. **Insertion mechanics.** Extend all three `embed_prefix` outputs
   consistently: `[B, n_demo, D]` embeddings (cast to the prefix dtype —
   training `forward` casts the concatenated prefix to bf16 when expert
   weights are bf16, :573-578; width 2048 for `gemma_2b`), a **bool**
   `[B, n_demo]` pad mask, and `0` entries in `att_masks` so demo tokens
   stay in the bidirectional prefix block. `pad_masks` drive `position_ids`
   and the KV-cache `prefix_offsets` — everything adapts as long as tokens
   are inserted before the suffix. Camera order is
   `config.image_features` order (`_preprocess_images` fills missing
   cameras with −1 images + zero masks).
6. **The lerobot processor pipeline drops unknown batch keys.** The `icl.*`
   demo-pack fields must bypass the pre/post processor and be re-merged
   after preprocessing (`train_loop._split_icl_fields` /
   `_move_icl_fields`); otherwise training **silently runs bare-prompt** —
   no error, the keys are simply gone.
7. **LeRobot hub-completion on partial subsets.** `LeRobotDataset` treats
   every episode listed in `meta/` as required and re-downloads whatever
   the disk lacks (`dataset_reader.try_load` → `_download`) — a
   `download-subset` range (full meta, partial files) therefore triggers a
   full-dataset pull. Every dataset open goes through
   `data.open_local_dataset`: pinned to `on_disk_episodes` (with
   contiguous-prefix enforcement, the only layout frame indexing supports)
   and `HF_HUB_OFFLINE=1`, so residual hub calls fail loudly instead.

### 4.1 `modeling_pi05_icl.py`

```python
class PI05ICLCore(PI05Pytorch):
    """Overrides embed_prefix to append cached demo tokens."""
    def __init__(self, config, rtc_processor=None):
        super().__init__(config, rtc_processor)
        self.demo_encoder = DemoEncoder(config)          # new module
        self._demo_cache = None                          # (embs, pad_masks) or None

    def set_demo_pack(self, frames, demo_mask, traj, traj_ok): ...
        # implemented signature; runs DemoEncoder under no_grad,
        # stores (embs [1,T,D], pad_masks [1,T])

    def clear_demo_pack(self): self._demo_cache = None

    def embed_prefix(self, images, img_masks, tokens, masks):
        embs, pad_masks, att_masks = super().embed_prefix(images, img_masks, tokens, masks)
        if self._demo_cache is None:
            return embs, pad_masks, att_masks
        d_embs, d_pad = self._demo_cache
        d_embs = d_embs.to(embs.dtype).expand(embs.shape[0], -1, -1)
        d_pad  = d_pad.expand(embs.shape[0], -1)
        embs     = torch.cat([embs[:, :n_img], d_embs, embs[:, n_img:]], dim=1)  # cams | demo | lang
        pad_masks = torch.cat([pad_masks[:, :n_img], d_pad, pad_masks[:, n_img:]], dim=1)
        att_masks = insert_zeros(att_masks, n_demo)       # demo tokens are prefix (0) tokens
        return embs, pad_masks, att_masks
```

(The pseudocode above sketches the shape contract; the implementation is
`splice_demo_tokens` — a pure, unit-tested function in
`modeling_pi05_icl.py` that also handles batch expansion and the
`pad_masks`-driven position-id/KV-cache offsets.)

- Ordering: `[cameras | demo | language]` keeps language last (matches
  PaliGemma pretraining habit of images-then-text) and keeps demo tokens
  adjacent to the visual stream they extend.
- Training path: demo tokens come from the batch (not the cache) —
  `PI05ICLPolicy.forward` calls `self.model.set_demo_pack(...)` from batch
  tensors **with gradients enabled for the encoder** (cache path is
  inference-only; in training the encoder runs inside the graph).
- `PI05ICLPolicy(PI05Policy)`: `config_class = ICLConfig`,
  `name = "pi05_icl"`; constructor builds `PI05ICLCore` instead of
  `PI05Pytorch`; everything else (`from_pretrained` remapping,
  `predict_action_chunk`, `forward` loss, processors) inherited unchanged.
- Loading the base: `PI05ICLPolicy.from_pretrained("lerobot/pi05_base",
  config=ICLConfig(...), strict=False, dtype=torch.bfloat16)` — passing our
  config explicitly matters because the checkpoint's `config.json` has
  `type: "pi05"` and would resolve a plain `PI05Config`; `strict=False` per
  §4.0.1.
- `from_pretrained` checkpoint layout (as built): base weights come from
  `lerobot/pi05_base` (stage 1) or the stage-1 serving dir (stage 2, base
  unchanged either way); adapter + encoder weights are a separate
  `icl_adapter.safetensors` in the same dir (plus
  `icl_adapter_config.json` with base/init fingerprints — the loader
  refuses mismatched pairs), loaded by `serve_icl`/`train_icl` resume.
  **The base checkpoint is never rewritten.**
- **Serving artifacts ship LoRA merged into the base weights**
  (`modeling_pi05_icl._merge_lora_state_dict` +
  `save_serving_checkpoint` + `so101_serving_config`): a LoRA-injected
  state dict has `q_proj.base_layer.*` key names that a freshly constructed
  policy cannot strict-load on the server, so serving dirs contain a merged
  `W + (α/r)·BA` checkpoint under plain key names, along with a config
  whose image keys are renamed to the ROS camera names
  (`wrist`/`overhead_1`/`overhead_2`) and state-dim set to 6 — the
  policy-server feature validation then passes without edits.
  Adapter-only artifacts keep the unmerged LoRA for training.

### 4.2 `demo_encoder.py`

```python
class DemoEncoder(nn.Module):
    """One demo pack -> T_vis + T_traj tokens in the PaliGemma embedding space."""
    # inputs, per demo i in k:
    #   frames:  [F, C, 224, 224]  (already through the SAME transform as policy cams)
    #   traj:    [S, d_state+d_action] normalized with the SAME dataset stats, S=16 stride-downsampled
    #   traj_ok: 0/1
    #
    # vision branch:
    #   frame_embs = paligemma.embed_image(frames).detach()        # [F, 256, D], frozen
    #   pooled = MultiHeadAttention(q=T learned queries, k=v=frame_embs)  # [T, D], T=64
    #   out_vis = LayerNorm(vis_proj(pooled)) * gate              # default-init proj, gate init 0
    # trajectory branch:
    #   out_traj = LayerNorm(traj_proj(MLP(traj))) * traj_ok * gate2  # [32, D]
    # outputs: concat over demos + order embedding (learned, k≤4, init 0) + pad mask
    # reset_icl_parameters(): zeroes ONLY gates + order embedding (§2.1 dead-saddle fix)
```

- Image transform reuse: demo frames go through the same code path as
  policy cameras — `PI05Policy._preprocess_images` (`modeling_pi05.py:952-1016`,
  resize-with-pad to 224 + [-1,1] SigLIP scaling; VISUAL normalization is
  IDENTITY so dataset stats never touch images). One code path for cams and
  keyframes, no train/serve skew.
- Normalization reuse: trajectory states/actions are normalized with the
  **active stage's** dataset stats (loaded from the policy preprocessor;
  DROID-subset stats in stage 1, SO-101 stats in stage 2 — see §6.5), padded
  to `max_state_dim`/`max_action_dim` (32) — same convention as the prompt
  tokenization.
- `embed_image` detached → no gradients into SigLIP, negligible extra
  forward cost (k×F ≤ 24 images, once per subtask).

### 4.3 `lora.py`

- peft `LoraConfig`:
  - `target_modules = r"paligemma_with_expert\.paligemma\.model\.language_model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj"`
  - `r=32, alpha=64, dropout=0.05`, bias none.
- Applied to the frozen `PI05ICLCore` **after** base weights load (with
  `strict=False`, §4.0.1); base params set `requires_grad=False` before
  adapter injection. We wrap manually with `inject_adapter_in_model` —
  not lerobot's `wrap_with_peft`, which freezes everything including the
  DemoEncoder (§4.0.3) — keeping the `PreTrainedPolicy` outer type intact
  for the server.
- Zero-init checks: peft initializes LoRA `B=0` and
  `lora.assert_zero_init` verifies the structure (LoRA `B` zeros, gates
  zero); `tests/test_zero_init.py` then asserts the modified model is
  **bit-identical to pi05_base on a fixed no-pack batch** — after first
  asserting base weights actually loaded (§4.0.1). Parity with a demo pack
  deliberately does not hold (§2.1).
- Optional knob (config): additionally adapt the expert attention
  (`gemma_expert.*.self_attn.(q|v)_proj`) — off in v1.
- Save/load: `save_icl_adapter(dir)` writes `icl_adapter.safetensors`
  (LoRA + DemoEncoder + gates, <100 MB) + `icl_adapter_config.json`;
  `load_icl_adapter(policy, dir)` restores. Stage 2 initializes from the
  stage-1 artifact via the same loader.

### 4.4 `data.py` — support/query sampling (two dataset regimes)

Wraps `LeRobotDataset` (v3.0). Task grouping: episodes grouped by a
**grouping key** — the episode `task` string for our own data, the
`task_category` column for DROID (see below). A `task_registry.json` maps
group → episode ids; built by the `data.py` CLI from what is actually on
disk.

#### Stage 1 — `lerobot/droid_1.0.1` (open-source ICL pre-training)

- **Why DROID**: LeRobot **v3.0 native**, real
  multi-episode-per-task structure, camera/state/action layout matching
  pi05_base's openpi DROID schema, Apache-2.0.
- Stats: 95,658 episodes / 27.6M frames at 15 fps; **49,630 raw task
  strings but only 86 `task_category` values (~1,100 episodes each)** →
  group by `task_category`, never by raw string. 3 cameras
  (`observation.images.{exterior_1_left, exterior_2_left, wrist_left}`,
  180×320 AV1), `observation.state` [8], `action` [8] → padded to 32.
- **Camera rename** (policy expects openpi names; applied in our
  collation — we emit policy key names directly, no dataset rewrite):
  `exterior_1_left → base_0_rgb`, `wrist_left → left_wrist_0_rgb`,
  `exterior_2_left → right_wrist_0_rgb` (the second external occupies the
  right-wrist slot — the same pattern our SO-101 data already uses for
  `overhead_2`).
- **Subset download strategy**: full set is 809.9 GB (980 files) — the
  actual v3.0 repo is **file-sharded inside a single `chunk-000`** (156
  `data/chunk-000/file-XXX.parquet`, 812
  `videos/<cam>/chunk-000/file-XXX.mp4` over 3 cameras, plus
  `meta/episodes/chunk-000/file-000..006.parquet`); there are no chunk-001+
  directories, so chunk filtering is meaningless. The unit is the
  **episode**: `meta/episodes` parquet maps every episode to its data shard
  (`data/chunk_index` + `data/file_index`) and per-camera video shards
  (`videos/<cam>/chunk_index` + `file_index`). `data.py`'s `download-subset
  --start E0 --end E1` subcommand fetches `meta/` first, resolves episodes
  `E0–E1` to the exact deduped shard file list, shows the exact GiB (via
  `list_repo_tree` sizes) before downloading, and passes that list as
  `snapshot_download(allow_patterns=...)`. Shards are size-rolled
  (~100 MB data / ~1 GB video), so a range boundary pulls a few neighbouring
  episodes along — unavoidable in the published layout. Budget guideline:
  ~8.5 MB/episode → ~85 GB holds ~10,000 episodes; size the range to free
  disk (the full download needs ~810 GB).
- **Registry must follow the disk**: DROID's global meta covers all 95k
  episodes, so `task_registry.json` is built by scanning *downloaded* shard
  files, never from global meta.
- **Stats check**: pi05 normalization is quantile (q01/q99); if the
  downloaded subset ships only min/max/mean/std, recompute with
  `lerobot-edit-dataset --operation.type recompute_stats` before training.
- **Smoke test**: `lerobot/droid_100` (464 MB, 100 episodes, 47 tasks;
  v2.0 — loads with a version notice; group by task string there). Used by
  M1 before any large download.
- Documented fallback: `nvidia/BridgeData2_LeRobot_v3` (90.5 GB, v3.0,
  50,415 episodes, 4 cams, state[7]/action[7], 5 fps WidowX) — fits wholly
  on disk but its 22,199 task strings require environment+skill grouping.

#### Stage 2 — own SO-101 data (fine-tuning with ICL)

- Sources: teleop MCAP → `rosbag_to_lerobot` exports (`--setup`
  cameras, 30 Hz, state/action 6 → padded 32), `self-improve` mixed
  rounds, the teleop anchor (`algorithmtheworld/so101-pick-and-place`).
- Camera rename via the existing map (`self-improve/self_improve/contract.py:75-79`):
  `wrist → left_wrist_0_rgb`, `overhead_1 → base_0_rgb`,
  `overhead_2 → right_wrist_0_rgb`.
- Grouping by `task` string with the alias-map curation step (as rev 1).
- **Embodiment gap is expected and handled**: stage 1 is Franka 8-dim at
  15 fps; stage 2 is SO-101 6-dim at 30 fps. Both pad to 32 and state is
  prompt-tokenized, so there is no shape incompatibility — the stage-2
  continued training adapts the in-context mechanism to our embodiment
  (RICL's per-task finetune variant is the precedent for short continued
  training on top of an ICL-capable model).
- **Current data reality (honest note)**: on-disk own data is ~1 task × 4
  episodes (anchor + `local/so101_test`); the per-session
  `algorithmtheworld/*` caches are mostly empty or use incompatible camera
  keys. Stage 2's gate requires the curation target (≥20 tasks × ≥20
  demos, §6.1) — a teleop collection campaign through the dataset builder
  quality gates is a prerequisite, in parallel with stage-1 training.

#### Sampler (both stages)

1. task t ~ p(task) (uniform over train groups, temperature knob for
   rebalancing scarce tasks);
2. support: k ~ U{1..4} episodes of t, disjoint from the query episode;
3. keyframes: F=6 frames per support episode — indices = uniform stride
   over episode length, always including first and last frame (start and
   goal emphasis);
4. trajectory: `observation.state` + `action` downsampled to S=16
   (uniform stride), `traj_ok` from availability flag in the episode
   metadata (video-only exports → `traj_ok=0`);
5. query: random timestep from a *different* episode of t; batch fields
   identical to standard pi05 training (`delta_timestamps` for action
   chunk, camera keys from policy `input_features`).
- Variable-k handling: per-sample demo tokens are padded to k_max=4 with
  zero pad-mask entries; batch tensors are
  `[B, k_max, F, C, 224, 224]` (frames) and `[B, k_max, S, d]` (traj).
- Split: **whole task groups held out** — stage 1: whole `task_category`
  values; stage 2: whole tasks. The sampler never mixes held-out groups
  into training support/query.
- Curriculum sampling knobs: `k_dist`, `frames_per_demo_dist` — v1 schedule
  in §6.3.

### 4.5 `train_loop.py` + `train_icl.py`

- Accelerate, 2 processes (one per 4080), `torchrun`/`accelerate launch`;
  bf16 mixed precision, `model.gradient_checkpointing_enable()`.
- Step: batch → `policy.model.set_demo_pack_train(batch)` (encoder inside
  graph) → `loss, logs = policy.forward(query_batch)` (inherited
  flow-matching loss) → backward (grads flow only into LoRA + DemoEncoder +
  gates) → clip 1.0 → AdamW.
- Stages share the loop; they differ only in config: stage 2 sets
  `init_adapter_from:` (loads the stage-1 `icl_adapter.safetensors` before
  training) and a lower LR / shorter schedule (§6.3).
- Gradient accumulation knob (`train.grad_accum`) keeps the effective batch
  at 16 when per-GPU batches must shrink for VRAM (§6.3); the k-curriculum
  is applied at **epoch granularity** via `CurriculumPhase`.
- Logged metrics (per step, to `metrics.jsonl` AND tensorboard events under
  `<output_dir>/tensorboard`):
  - `loss` (query action loss), `val/loss` on fixed support/query sets;
  - `loss_demo_zeroed`: same query batch replayed under the *same RNG
    state* with the demo gate forced 0 (`demo_zeroed_every` knob) — the
    in-training ICL signal — together with
    `demo_zeroed_ratio = loss / loss_demo_zeroed` (< 1 means the demos are
    helping; hovering at 1.0 means the demos are being nulled, §2.1/§6.4);
  - `lang_dropout_rate`: running fraction of batches that received the
    vague prompt (sanity check on `train.language_dropout`);
  - `lr`, gate values, `sec_per_step`.
  - (LoRA-norm and token/dropout stats from the original plan are **not**
    implemented — deferred, §11.)
- Param groups: the demo gates get `gate_lr_mult` (default 10×) the base LR.
- Checkpointing: `ckpt_every` steps + best-val + final; adapter-only
  artifacts; resume flag.
- Determinism/seed in config; `WANDB_MODE=offline` default.

### 4.6 `registration.py` + `serve_icl.py` — serving with one server edit

- `registration.py`: registers `ICLConfig` with
  `@PreTrainedConfig.register_subclass("pi05_icl")` at import time (draccus
  choice registry — no lerobot file edits; `so101_icl` is imported before
  the server starts).
- **The one repo edit**: add `"pi05_icl"` to `SUPPORTED_POLICIES` in
  `policy_server/inference_engine.py:33-43` (precedent: `"xvla"`, comment
  at :29-32). Without it `load_policy` (:290-292) rejects the name before
  `get_policy_class` is ever reached. Everything else about the engine —
  `get_policy_class`/`from_pretrained` loading, camera/state schema
  validation, chunking — is untouched.
- `serve_icl.py` entry (pixi task `icl_server`):
  1. `import so101_icl.registration` (registers class),
  2. `from policy_server import zmq_server; zmq_server.main()` — the
     client handshake supplies `policy_type: "pi05_icl"` at runtime
     (`async_client.py:141-147`), camera/state schema validation unchanged
     (demo keyframes are not observation keys).
- Demo transport (`demo_transport.py`): a small ZMQ REP socket on the
  server host (port 8661, msgpack header + raw f32 frames;
  `set_demo_pack` / `clear` / `status`, with shape validation and encode
  timing), calling the loaded policy's cache API — installed into the
  engine by monkey-patching `InferenceEngine.load_policy`/`unload_policy`
  (`install_into_engine`). It also carries `save_stage_stats` /
  `load_stage_stats` for the stage-2 trajectory normalizer. The async
  inference client and the 30 Hz contract are untouched; the **bridge
  node** drives the side channel.
- Rollout wire (M3/M4): `serve_rollout_icl.py` extends the self-improve
  rollout server's `SUPPORTED_POLICIES` at runtime, defaults to
  `--policy-type pi05_icl`, and publishes the loaded policy into the demo
  transport so `set_demo_pack` reaches the rollout-serving model.
- Latency: `set_demo_pack` ≈ k×F ≤ 24 SigLIP forwards + pool ≪ 1 s on the
  GPU server; per-tick inference adds only the extra prefix attention
  columns (≤384 tok) in the already-cached prefix pass.

### 4.7 `bridge_icl_node.py` — ROS 2 node (dispatch-package consumer)

The core (parsing, ordering, selection, state machine, terminal monitor,
asset resolution) is ROS-free and unit-tested; the optional rclpy shell
wires parameters, a dispatch-poll timer, and dynamic terminal-topic
subscriptions. The ament packaging lives in `so101_icl_bridge/`
(colcon-buildable, `ros2 launch so101_icl_bridge bridge_icl.launch.py`).
A ROS-free CLI (`--stdin-events`, `--once`) drives the same core for
smoke runs. **Not yet exercised against a live Bridge Robot.**

- Inputs: mission id; polls `GET /api/mission/<ws>/<id>/dispatch-package`
  (`ros2_action_goal:v1`).
- State machine: subtasks ordered topologically over `depends_on` edges
  (cycles rejected); one active subtask; only its conditioning pack is in
  context. Terminal predicates (`action_spec.terminal.topic`) gate
  advancement via `TerminalMonitor`; `--advance-mode auto` skips the gate.
- Demo selection: top-k from `conditioning_pack.demonstrations`, ranked
  success-first, then recency (`prov_id`), then ranker score. `k`,
  `frames_per_demo` and `k_max` are CLI flags / ROS parameters (rev 3 —
  previously hardcoded).
- Asset resolution: keyframes via presigned URL or `file://` → decode →
  resample to exactly `frames_per_demo` → shared policy image transform
  (§4.2) → transport `set_demo_pack`; trajectory slice when
  `trajectory.available`, else `traj_ok=0` (a trajectory-less demo never
  masks the trajectories of the other selected demos).
- Subtask terminal condition → `clear_demo_pack()` → advance.
- Fallback mode `conditioning: prompt`: the node **clears the pack** —
  no demo tokens. There is deliberately no language-enrichment logic here
  (§10); the enriched-prompt arm of the campaigns is produced by
  `conditions.py` on the eval side, not by the bridge.

### 4.8 `eval_icl.py` + the campaign machinery (as built)

- **Stage 1 (offline, no robot)**: held-out DROID `task_category` groups —
  demo-conditioned vs demo-zeroed vs bare-prompt query loss on fixed
  support/query sets; ablations `k ∈ {1,2,4}`, `F ∈ {4,6,8}` (evaluated at
  the anchor k), trajectory on/off, driven by CLI flags or the stage YAML
  `eval.ablations` block. Output: aggregate settings table + **per-group
  table** (the M2 gate reads per held-out group) + median demo/zeroed
  ratio. If the requested split is empty the harness falls back to train
  groups **and stamps a WARNING into the report** — the gate does not
  apply to such a run.
- **Stage 2 sim (M3)**: `eval_sim_campaign.py` bootstraps itself into
  Isaac Sim's Python (`launch_isaac_sim_before_task_imports`), runs
  `sim_rollout.run_rollouts` per condition against the rollout-wire ICL
  server (`serve_rollout_icl.py`, port 8660), and writes a markdown report
  with the M3 gate line (≥ +10 pts, ≥ 20 trials). `eval_icl.py sim`
  re-execs it.
- **Stage 2 real (M4)**: `eval_icl.py real` drives the proven
  `self_improve.real_rollout` session stack per condition
  (`run_session`/`run_post` with the VLM judge), pushes/clears the demo
  pack per condition via the side channel, probes chunk latency per
  condition (`latency.probe_policy_server`), and applies the latency gate
  (p95 ≤ +20 % of the unconditioned arm).
- **Conditions** (`conditions.py`): `full_icl` (demo pack),
  `prompt_enriched` (pack cleared, full prompt), `bare_prompt` (first
  sentence only). `DemoPackBuilder` deterministically picks the first k
  episodes of a registry group, mirroring the training sampler.
- Protocol: ≥20 sim trials / ≥10 real trials per condition per task;
  metrics: success@1, chunk latency p50/p95, per-condition delta.

---

## 5. Serving/deployment picture (unchanged parts)

```
[lerobot/droid_1.0.1 subset] ──► train_icl --stage pretrain (2×4080) ──► icl_adapter (stage 1)
                                                                                  │
[teleop/sim] → MCAP → rosbag_to_lerobot → LeRobot v3 datasets ───────────────────┤
[self-improve rounds] ───────────────────────────────────────────────────────────┘
                                                                                  ▼
                                              train_icl --stage finetune (own data)
                                                                                  │
                                                                  icl_adapter (stage 2)
                                                                                  ▼
cameras/state @30Hz → async_inference_node ──ZMQ──► serve_icl (pi05_icl) ──► action chunks (16)
                                                        ▲
bridge_icl_node ── dispatch pack → top-k → demo_transport ┘ (once per subtask)
```

The async node, transports, `InferenceEngine` chunking logic, setup YAMLs,
and control stack are used exactly as-is.

---

## 6. Training methodology

### 6.1 Data

- **Stage 1 (open-source)**: `lerobot/droid_1.0.1` subset, 10 chunks
  (~10k episodes / ~85 GB) recommended; grouping by `task_category`
  (86 groups); hold out whole categories (e.g. 6 eval + 6 test) before
  sampling. Quality: DROID episodes are filtered by the dataset's own
  success annotations where present.
- **Stage 2 (own robot)**: existing exports — teleop MCAP →
  `rosbag_to_lerobot` (`--setup` cameras, 30 Hz), Isaac Sim rollouts from
  `self-improve` rounds, plus the teleop anchor. Curation target: **≥20
  tasks × ≥20 demos** (teleop first; sim rollouts pad scarce tasks);
  quality gate: outcome/object criteria from the dataset builder / VLM
  judge, as in `self-improve`. Current on-disk reality is ~1 task × 4
  episodes → collection campaign runs in parallel with stage 1.
- Task identity = the episode `task` string, normalized (lowercase, strip
  object counts) — except stage 1 DROID, where it is `task_category`. The
  curation script (part of `data.py` CLI) produces `task_registry.json` +
  split file with held-out group lists.

### 6.2 Objective

Given k support demos of task t and a query observation from a *different*
episode of t, predict the query's action chunk with π0.5's flow-matching
loss (inherited, unmodified). Gradients reach only: VLM LoRA, DemoEncoder,
demo gates. Identical in both stages; only the data and the LR schedule
differ.

### 6.3 Schedule and hyperparameters (v1)

| Knob | Stage 1 — DROID pre-train | Stage 2 — SO-101 fine-tune |
|---|---|---|
| Init | `lerobot/pi05_base` + fresh zero-init adapters | stage-1 `icl_adapter.safetensors` |
| LoRA rank / alpha / dropout | 32 / 64 / 0.05 | unchanged (adapters continue training) |
| peft targets | VLM attention q,k,v,o (§4.3) | unchanged |
| Tokens per demo (vis / traj) | 64 / 32 | unchanged |
| k (support demos) | curriculum: 1k steps @ k=1 → k ~ U{1,2} → k ~ U{1,4} | k ~ U{1,4} (no curriculum) |
| Frames per demo F | 6 (ablate 4/8) | 6 |
| Trajectory branch | on when available, `traj_ok` mask | on when available |
| Batch | 4/GPU × grad_accum 2 × 2 (DDP) — 16 GB 4080s OOM at 8 during backward | 8/GPU × 2 |
| Optimizer | AdamW, lr 1e-4, wd 0.01, betas default | AdamW, lr 2.5e-5 (RICL finetune analog) |
| Schedule | cosine, warmup 500 steps | cosine, warmup 100 steps |
| Steps | 10k; extend to 20k if val still falling | 1k–3k; early-stop on val |
| Precision / memory | bf16 mixed, gradient checkpointing on | same |
| Grad clip | 1.0 | 1.0 |
| Seed | 42 (config) | 43 (config) |

### 6.4 Acceptance signals during training

- `loss_demo_zeroed` > `loss` by a clear margin on *held-out groups* by
  mid-training (demos are being used; if the two curves coincide from
  start to end, the gate stayed shut — raise gate lr or lower LoRA dropout
  before touching anything else).
- No catastrophic drift on seen groups: bare-prompt val loss on seen groups
  within ~10 % of its value at step 0 (zero-init should keep this
  automatic; it is asserted, not assumed).

### 6.5 Stage-boundary mechanics

- **Pre/post processors are rebuilt per stage** from that stage's dataset
  stats (quantile normalization): stage 1 serves DROID stats, stage 2
  serves SO-101 stats. The DemoEncoder consumes *already-normalized*
  trajectories, so no encoder weights depend on which stats are active —
  the boundary is processor-only.
- Demo-frame transform is stats-free (VISUAL=IDENTITY, §4.2) → no image
  skew across stages.
- The stage-2 artifact records the base checkpoint hash (pi05_base) *and*
  the stage-1 adapter hash; the loader refuses mismatched pairs.

---

## 7. Configs (schema)

### 7.1 `configs/icl_pretrain_droid_v1.yaml` (stage 1)

```yaml
base_checkpoint: lerobot/pi05_base          # FOUNDATION model (openpi), cached locally; load bf16
dtype: bfloat16
output_dir: so101_icl/runs/icl_pretrain_droid_v1
dataset:
  repo_id: lerobot/droid_1.0.1
  root: ~/.cache/huggingface/lerobot        # subset lives here, episode-range filtered
                                             # (range = whatever download-subset fetched)
  grouping_key: task_category               # 86 groups; NOT the 49,630 raw task strings
  camera_rename: droid                      # PRESET name (exterior_1->base_0_rgb, wrist->left_wrist_0_rgb,
                                             # exterior_2->right_wrist_0_rgb); explicit maps also accepted
  demo_camera: observation.images.left_wrist_0_rgb   # which camera feeds demo keyframes
  task_registry: so101_icl/configs/task_registry_droid.json   # built from downloaded chunks only
  holdout_groups: {eval: 3, test: 3}        # whole task_category values, names frozen in registry
  recompute_stats: auto                     # recompute quantiles if subset ships min/max only
demo_encoder: {k_max: 4, frames_per_demo: 6, tokens_vis: 64, tokens_traj: 32, traj_steps: 16, n_heads: 8,
               gate_floor: 0.1}    # null-out mitigation (§2.1); 0.0 = original zero-init
lora: {rank: 32, alpha: 64, dropout: 0.01, targets: vlm_attention, gate_lr_mult: 50.0}
train:
  steps: 10000
  batch_per_gpu: 4                # 16 GB 4080s OOM at 8 during backward
  grad_accum: 2                   # effective batch 4*2*2 GPUs = 16
  lr: 1.0e-4
  warmup: 500
  grad_clip: 1.0
  curriculum: [[1000, [1]], [3000, [1, 2]], [100000000, [1, 2, 3, 4]]]   # epoch-granular k phases
  ckpt_every: 2000
  seed: 42
  demo_zeroed_every: 100
  language_dropout: 0.3           # vague-prompt batches: demos become the only task signal (§2.1)
  val_every: 1000
eval:                             # offline defaults for `eval_icl.py offline`
  offline_trials: 500             # query samples (CLI flags override)
  ablations: {k: [1, 2, 4], frames: [4, 6, 8], traj: [on, off]}
```

### 7.2 `configs/icl_finetune_so101_v1.yaml` (stage 2)

```yaml
base_checkpoint: lerobot/pi05_base            # unchanged — the base is never rewritten
dtype: bfloat16
output_dir: so101_icl/runs/icl_finetune_so101_v1
init_adapter_from: so101_icl/runs/icl_pretrain_droid_v1/final
dataset:
  repo_ids: [local/so101_teleop_v1, local/so101_sim_round2]   # rosbag_to_lerobot + self-improve exports (to be curated; §6.1)
  grouping_key: task
  camera_rename: pi05_base       # PRESET (wrist->left_wrist_0_rgb, overhead_1->base_0_rgb, overhead_2->right_wrist_0_rgb)
  demo_camera: observation.images.base_0_rgb
  task_registry: so101_icl/configs/task_registry_so101.json
  holdout_tasks: {eval: 3, test: 3}
demo_encoder: {k_max: 4, frames_per_demo: 6, tokens_vis: 64, tokens_traj: 32, traj_steps: 16, n_heads: 8}
lora: {rank: 32, alpha: 64, dropout: 0.05, targets: vlm_attention, gate_lr_mult: 10.0}
train: {steps: 2000, batch_per_gpu: 8, lr: 2.5e-5, warmup: 100, grad_clip: 1.0, ckpt_every: 500, seed: 43,
        curriculum: [[100000000, [1, 2, 3, 4]]], demo_zeroed_every: 100, val_every: 250}
eval:                           # sim/real campaign knobs — consumed by eval_icl.py sim|real
  sim_trials: 20
  real_trials: 10
  conditions: [bare_prompt, prompt_enriched, full_icl]
```

---

## 8. Milestones and gates — with as-built status (rev 3)

| M | Scope | Gate (objective, binary) | Status |
|---|---|---|---|
| **M0** | `PI05ICLCore` + DemoEncoder skeleton + LoRA + zero-init test; no training | Loaded model (pi05_base, `strict=False`, weights-load assertion per §4.0.1) with adapters+gate at init produces **bit-identical** action chunks to `lerobot/pi05_base` on a fixed **no-pack** obs batch (`tests/test_zero_init.py`; pack parity deliberately not asserted, §2.1) | **Done** |
| **M1** | `data.py` sampler + overfit smoke run | Overfit **one** task (support/query from its episodes): query loss ↓ ≥80 % vs step-0, and demo-zeroed loss stays high → demo tokens steer actions (`tests/test_sampler.py` invariants pass) | **Done** — on the local SO-101 datasets, not `droid_100` (§11) |
| **M2** | **Stage 1**: DROID subset download + full ICL pre-training + offline eval | On held-out `task_category` groups: median demo-conditioned query loss ≤ 0.9× demo-zeroed loss **and** ≤ bare-prompt loss; seen-group bare-prompt regression <10 % | **Partial**: a stage-1 training run happened (`runs/icl_pretrain_droid_v1/` metrics + tensorboard) but the DROID subset download is deferred and no `final/` adapter artifact exists on disk — the gate has **not** been formally evaluated (§11) |
| **M3** | **Stage 2**: fine-tune with ICL on own data (after curation target met) + sim eval | In sim on the SO-101 workcell: full_icl success > bare_prompt success (≥10 pts absolute, ≥20 trials each) on held-out tasks; seen-task bare-prompt regression <10 % | **Code complete, not run** (`eval_sim_campaign.py`, `serve_rollout_icl.py`); blocked on stage-2 data + stage-1 final adapter |
| **M4** | `serve_icl` + real-robot eval | Real arm, ≥10 trials/condition: full_icl ≥ prompt_enriched ≥ bare_prompt trend on instruction-following details; p95 chunk latency within +20 % of unconditioned | **Code complete, not run** (`eval_icl.py real`, `latency.py`) |
| **M5** | `bridge_icl_node` on dispatch packages | End-to-end: objective → dispatch pack → bridge → conditioned rollouts; subtask switch demo-encode ≤1 S; same `ros2_action_goal:v1` contract | **Core unit-tested + ament-packaged; live-robot bring-up pending** |

M0 and M1 needed no large downloads. Stage-2 data collection runs in
parallel with stage 1 (§11).

---

## 9. Risks and engineering mitigations

| Risk | Mitigation |
|---|---|
| **Silent weight-load failure** (`from_pretrained` strict-default swallows errors, §4.0.1) | Loader passes `strict=False`; M0 asserts a frozen param equals its checkpoint tensor before parity |
| Prefix attention mask/position-id errors with inserted tokens | `tests/test_prefix_shapes.py`: assert mask symmetry, position-id monotonicity, and loss parity between demo-at-end vs demo-mid insertion on a tiny batch |
| Demo tokens ignored (gate stays ~0, `loss_demo_zeroed` ≈ `loss`) | Curriculum on k; gate lr multiplier; if still shut: lower LoRA dropout, raise tokens/demo, extend steps — in that order |
| **Embodiment gap stage 1 → 2** (Franka 8-dim/15 fps → SO-101 6-dim/30 fps) | Both pad to 32 + prompt-tokenized state (no shape mismatch); stage-2 continued training adapts; optional expert-LoRA knob if transfer is weak; RICL per-task finetune is the precedent for short continued training |
| **DROID disk/subset constraints** (~810 GB full, ~225 GB free) | Chunk-subset strategy (§4.4); registry built from downloaded chunks only; droid_100 smoke first; BridgeData2 documented fallback |
| Stage-1/stage-2 processor-stats skew | Processors rebuilt per stage (§6.5); DemoEncoder consumes already-normalized inputs; image path stats-free (VISUAL=IDENTITY) |
| DROID task-string noise (49,630 strings) fragments groups | Group by `task_category` (86 values); registry alias map for our own data |
| Train/serve skew in demo-frame preprocessing | Single shared path: `PI05Policy._preprocess_images` reused by `data.py`, `demo_transport.py`, `bridge_icl_node.py` (§4.2) |
| VRAM overshoot on 16 GB | Knobs: batch 8→4/GPU (+grad accum 2), F 6→4, tokens 96→64/demo; activation memory is the only variable term |
| LeRobot version drift (installed 0.6.1 pinned in pixi) | Anchors in §1 are line-checked against 0.6.1; `so101_icl` imports only public API (`PI05Policy`, `PI05Pytorch`, registry) + one internal path constant (`paligemma_with_expert...`) asserted at import with a clear error message |
| Adapter checkpoint vs base mismatch | `icl_adapter_config.json` stores base checkpoint hash **and** (stage 2) init-adapter hash; loader refuses mismatched pairs |

---

## 10. What is deliberately NOT built here

- Edits outside the new package: `"pi05_icl"` added to
  `SUPPORTED_POLICIES` in `policy_server/inference_engine.py` (required —
  the name is gate-checked before the registry is consulted; precedent
  `"xvla"`), plus `pixi.toml` deps/tasks. No other changes to
  `policy_server`, `so101_inference`, `rosbag_to_lerobot`, `self-improve`,
  or lerobot source. Registration and subclassing only.
- No language-enrichment (Path 1) logic: the bridge's prompt mode simply
  clears the demo pack; the `prompt_enriched` campaign arm reuses the
  dispatch prompt verbatim. Enriching prompts from demo metadata was
  considered and dropped.
- No upstream Bridge Robot changes: the node consumes the documented
  `ros2_action_goal:v1` dispatch-package contract as-is.
- No full-DROID download and no dataset re-encoding: the subset strategy
  and collation-time renames (§4.4) avoid touching the published dataset.

---

## 11. Known gaps and deferred work (rev 3)

Everything below is honestly open; nothing here is silently missing from
the code — if it isn't listed as done in §8, it isn't done.

1. **Stage-1 completion**: the DROID subset download is deferred (config
   comment); `runs/icl_pretrain_droid_v1/` holds metrics/tensorboard but
   **no `final/` or `best/` adapter directories**, while the stage-2
   config's `init_adapter_from` points at `.../final`. Stage 2 cannot run
   end-to-end from what is on disk until stage 1 is re-run/finished.
2. **Stage-2 dataset**: on-disk own data is ~1 task × 4 episodes (the
   smoke registry lists the same episodes under both datasets). The ≥20
   tasks × ≥20 demos curation target requires the teleop collection
   campaign — the blocking prerequisite for M3/M4.
3. **M3/M4/M5 execution**: campaigns and bridge are code-complete but
   unrun (§8); the bridge has never talked to a live Bridge Robot.
4. **Deferred loss-metric logging**: LoRA-norm and token/dropout stats
   from the original §4.5 plan are not implemented (loss, lr, gates,
   sec_per_step, loss_demo_zeroed, val/loss are).
5. **M1 smoke ran on local SO-101 data**, not `lerobot/droid_100` as
   originally planned (no droid download at the time). Same gate
   semantics, different data.
6. **Offline eval granularity**: the per-group table attributes
   batch-level means to each group a batch touches (a group's rows are
   its batches); the original plan's "500 query episodes per group"
   budget is approximated by `offline_trials` over the whole split.
   Successive-batch sampling within a group would tighten this.
7. **`droid_100` v2.0 vs v3.0**: if the droid smoke is ever run, it loads
   with a version notice; only the local smoke is exercised.
