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

"""Accelerate training loop for pi05_icl (ICL §4.5).

Both stages share this loop; they differ only in config. Each step:

1. batch -> policy preprocessor (quantile normalization with the ACTIVE
   stage's dataset stats, state discretization into the prompt, tokenization)
2. ``PI05ICLPolicy.forward`` — which first pushes the batch's support pack
   through the DemoEncoder INSIDE the autograd graph
3. inherited flow-matching loss -> backward -> grads reach only LoRA +
   DemoEncoder + gates -> clip 1.0 -> AdamW (gates get ``gate_lr_mult``).

Logged: ``loss``, ``val/loss`` (bare-prompt), ``loss_demo_zeroed`` and
``demo_zeroed_ratio = loss / loss_demo_zeroed`` (same
batch with gates forced to 0 — the in-training ICL signal), gate values and
LoRA norms. Artifacts are adapter-only (``icl_adapter.safetensors``); the
base checkpoint is never rewritten.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .lora import save_icl_adapter, trainable_parameters
from .modeling_pi05_icl import DEMO_FRAMES, DEMO_MASK, DEMO_TRAJ, DEMO_TRAJ_OK

logger = logging.getLogger(__name__)


@dataclass
class CurriculumPhase:
    until_step: int  # phase active while step < until_step (last phase: inf)
    k: list[int] = field(default_factory=lambda: [1, 2, 3, 4])


@dataclass
class TrainSettings:
    steps: int = 10_000
    batch_per_gpu: int = 8
    grad_accum: int = 1  # micro-batches per optimizer step; effective batch = batch_per_gpu * grad_accum * n_gpus
    lr: float = 1e-4
    warmup: int = 500
    grad_clip: float = 1.0
    ckpt_every: int = 2_000
    seed: int = 42
    num_workers: int = 4
    curriculum: list[CurriculumPhase] = field(
        default_factory=lambda: [
            CurriculumPhase(1_000, [1]),
            CurriculumPhase(3_000, [1, 2]),
            CurriculumPhase(math.inf, [1, 2, 3, 4]),
        ]
    )
    log_every: int = 20
    demo_zeroed_every: int = 200
    val_every: int = 1_000
    val_batches: int = 8
    gate_lr_mult: float = 10.0
    # Language dropout (null-out mitigation, ICL §4.5): on this fraction of
    # micro-batches the task prompt is replaced by a vague stand-in, making
    # the demo pack the only task signal — gradient pressure to USE demos.
    language_dropout: float = 0.0
    tensorboard: bool = True  # event files under <output_dir>/tensorboard (needs the tensorboard package)


def k_choices_for_step(phases: list[CurriculumPhase], step: int) -> list[int]:
    for phase in phases:
        if step < phase.until_step:
            return phase.k
    return phases[-1].k


# Vague stand-in prompt used by language dropout: carries no task information,
# so the demo pack is the only task signal on dropped batches.
LANGUAGE_DROPOUT_PROMPT = "do the task."


def _apply_language_dropout(batch: dict, rng: random.Random, p: float) -> bool:
    """Replace the task prompt of a whole micro-batch in place.

    Whole-batch granularity (not per-sample) on purpose: the processor
    tokenizes one prompt list per batch and partial drops would blur the
    signal. Uses a dedicated RNG so the torch noise stream (flow-matching
    noise/time replay in ``_demo_zeroed_loss``) is untouched.
    Returns whether the drop was applied (for the logged rate).
    """
    if p <= 0 or "task" not in batch or rng.random() >= p:
        return False
    tasks = batch["task"]
    batch["task"] = [LANGUAGE_DROPOUT_PROMPT] * len(tasks)
    return True


def build_stage_preprocessor(config, stats: dict):
    """Pre/post processors for the ACTIVE stage (ICL §6.5).

    Built from code with the stage's dataset stats — never from the base
    checkpoint's processor jsons (those carry DROID stats). Demo keyframes
    are stats-free (VISUAL=IDENTITY) and never pass through this pipeline.
    """
    from .processor_pi05_icl import make_pi05_icl_pre_post_processors

    return make_pi05_icl_pre_post_processors(config, dataset_stats=stats)


def _split_icl_fields(batch: dict) -> tuple[dict, dict]:
    """Separate icl.* tensors: the lerobot processor pipeline DROPS unknown
    keys, so demo-pack fields must bypass it and be re-merged afterwards."""
    icl = {k: v for k, v in batch.items() if k.startswith("icl.")}
    rest = {k: v for k, v in batch.items() if not k.startswith("icl.")}
    return icl, rest


def _move_icl_fields(batch: dict, device: str) -> dict:
    out = dict(batch)
    for key in (DEMO_FRAMES, DEMO_MASK, DEMO_TRAJ, DEMO_TRAJ_OK):
        if key in out:
            out[key] = out[key].to(device, non_blocking=True)
    return out


def _strip_icl_fields(batch: dict) -> dict:
    """Bare-prompt batch: no demo keys -> forward clears the pack."""
    return {k: v for k, v in batch.items() if not k.startswith("icl.")}


@torch.no_grad()
def evaluate_loss(policy, preprocessor, batches, device, bare_prompt: bool) -> float:
    """Mean flow-matching loss over pre-fetched (already collated) batches."""
    policy.eval()
    losses = []
    for batch in batches:
        icl, rest = _split_icl_fields(batch)
        rest = preprocessor(rest)
        if bare_prompt:
            batch = rest
        else:
            batch = {**rest, **_move_icl_fields(icl, device)}
        loss, _ = policy.forward(batch)
        losses.append(loss.item())
    policy.train()
    return sum(losses) / max(1, len(losses))


def _capture_rng_state() -> tuple:
    """Snapshot CPU + CUDA RNG (the flow-matching noise/time come from here)."""
    states = (torch.get_rng_state(),)
    if torch.cuda.is_available():
        states = states + (torch.cuda.get_rng_state_all(),)
    return states


def _restore_rng_state(states: tuple) -> None:
    torch.set_rng_state(states[0])
    if len(states) > 1 and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states[1])


def _demo_zeroed_loss(policy, batch, rng_state: tuple | None = None) -> float:
    """Same batch, demo gates forced to 0 — how much the demos are used.

    ``rng_state`` replays the SAME flow-matching noise/time as the reference
    forward; without it the metric is confounded by resampled noise.
    """
    model = policy.model
    if rng_state is not None:
        _restore_rng_state(rng_state)
    old = model._demo_gate_scale
    model._demo_gate_scale = 0.0
    try:
        with torch.no_grad():
            loss, _ = policy.forward(batch)
        return loss.item()
    finally:
        model._demo_gate_scale = old


def _param_groups(policy, settings: TrainSettings):
    gates, others = [], []
    for name, param in trainable_parameters(policy):
        (gates if "gate_" in name else others).append(param)
    groups = [{"params": others, "lr": settings.lr}]
    if gates:
        groups.append({"params": gates, "lr": settings.lr * settings.gate_lr_mult})
    return groups


def _lr_lambda(step: int, settings: TrainSettings) -> float:
    if step < settings.warmup:
        return (step + 1) / max(1, settings.warmup)
    progress = (step - settings.warmup) / max(1, settings.steps - settings.warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def run_training(
    policy,
    preprocessor,
    train_loader,
    val_batches: list[dict] | None,
    settings: TrainSettings,
    output_dir: Path | str,
    *,
    accelerator=None,
    init_adapter_path: Path | str | None = None,
) -> Path:
    """Main loop; called by train_icl.py on each accelerate process."""
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    if accelerator is None:
        # find_unused_parameters: the demo encoder gates/queries may not all
        # participate in the loss on every step (e.g. bare-prompt batches).
        accelerator = Accelerator(
            mixed_precision="bf16",
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        )
    output_dir = Path(output_dir)
    is_main = accelerator.is_main_process
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.jsonl").touch(exist_ok=True)
    device = accelerator.device

    writer = None
    if is_main and settings.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"), flush_secs=10)
        except ImportError:
            logger.warning("tensorboard not installed; falling back to metrics.jsonl only")

    torch.manual_seed(settings.seed)

    policy.model.gradient_checkpointing_enable()
    policy.train()

    optimizer = torch.optim.AdamW(
        _param_groups(policy, settings), weight_decay=0.01, betas=(0.9, 0.95)
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr_lambda(step, settings)
    )
    base_policy = policy  # unwrapped reference; `policy` becomes a DDP module below
    policy, optimizer, train_loader = accelerator.prepare(policy, optimizer, train_loader)

    metrics_file = output_dir / "metrics.jsonl"
    step = 0
    micro_step = 0
    best_val = float("inf")
    start = time.time()
    data_iter = None
    lang_rng = random.Random(settings.seed + 1)
    dropped = seen = 0  # language-dropout application counters

    while step < settings.steps:
        # k-curriculum applies at epoch granularity: dataloader workers fork
        # per iter(), so k_choices must be set BEFORE the iterator is created
        # (mid-epoch mutation never reaches already-forked workers).
        if data_iter is None:
            train_loader.dataset.k_choices = k_choices_for_step(settings.curriculum, step)
            data_iter = iter(train_loader)
        try:
            batch = next(data_iter)
        except StopIteration as exc:
            if len(train_loader) == 0:
                raise RuntimeError(
                    "train loader yields no batches — dataset smaller than "
                    "batch_per_gpu with drop_last=True?"
                ) from exc
            epoch = getattr(train_loader.dataset, "_epoch", 0) + 1
            train_loader.dataset.set_epoch(epoch)
            data_iter = None
            continue

        rng_state = _capture_rng_state()
        icl_fields, query_batch = _split_icl_fields(batch)
        seen += 1
        if _apply_language_dropout(query_batch, lang_rng, settings.language_dropout):
            dropped += 1
        query_batch = preprocessor(query_batch)
        batch = {**query_batch, **_move_icl_fields(icl_fields, device)}
        loss, _ = policy.forward(batch)
        accelerator.backward(loss / settings.grad_accum)

        micro_step += 1
        if micro_step % settings.grad_accum != 0:
            continue
        torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad], settings.grad_clip
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if not is_main:
            continue

        record = {
            "step": step,
            "loss": loss.item(),
            "lr": scheduler.get_last_lr()[0],
            "gate_vis": base_policy.model.demo_encoder.gate_vis.item(),
            "gate_traj": base_policy.model.demo_encoder.gate_traj.item(),
            "sec_per_step": (time.time() - start) / step,
        }
        if settings.language_dropout > 0:
            record["lang_dropout_rate"] = dropped / max(1, seen)
        if step % settings.demo_zeroed_every == 0 or step == 1:
            policy.eval()
            record["loss_demo_zeroed"] = _demo_zeroed_loss(base_policy, batch, rng_state)
            record["demo_zeroed_ratio"] = record["loss"] / record["loss_demo_zeroed"]
            policy.train()
        if val_batches and (step % settings.val_every == 0 or step == settings.steps):
            record["val/loss"] = evaluate_loss(
                policy, preprocessor, val_batches, str(device), bare_prompt=True
            )
            if record["val/loss"] < best_val:
                best_val = record["val/loss"]
                save_icl_adapter(
                    base_policy, output_dir / "best", init_adapter_path=init_adapter_path
                )
        print(f"step {step}/{settings.steps} {record}", flush=True)
        if writer is not None:
            for key, value in record.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(key, value, global_step=step)
        with open(metrics_file, "a") as f:
            f.write(json.dumps(record) + "\n")
        if step % settings.ckpt_every == 0 or step == settings.steps:
            save_icl_adapter(
                base_policy, output_dir / f"step_{step}", init_adapter_path=init_adapter_path
            )

    final = save_icl_adapter(
        base_policy, output_dir / "final", init_adapter_path=init_adapter_path
    )
    if writer is not None:
        if is_main:
            writer.add_hparams(
                {
                    "lr": settings.lr,
                    "steps": settings.steps,
                    "batch_per_gpu": settings.batch_per_gpu,
                    "grad_accum": settings.grad_accum,
                    "warmup": settings.warmup,
                    "seed": settings.seed,
                },
                {"hparam/final_loss": record["loss"]},
            )
        writer.close()
    if is_main:
        logger.info("training done in %.0f s; adapter at %s", time.time() - start, final)
    return final


# ---------------------------------------------------------------------- #
# CLI entry (train_icl.py wrapper calls this)                             #
# ---------------------------------------------------------------------- #


def _load_stage_config(path: Path | str) -> dict:
    import yaml

    cfg = yaml.safe_load(Path(path).read_text())
    for required in ("base_checkpoint", "dataset", "train"):
        if required not in cfg:
            raise ValueError(f"stage config {path} missing '{required}'")
    return cfg


def _val_batches(loader: DataLoader, n: int) -> list[dict]:
    batches = []
    for batch in loader:
        batches.append({k: v for k, v in batch.items()})
        if len(batches) >= n:
            break
    return batches


def main(argv=None) -> int:
    """Entry for the ``train_icl.py`` wrapper (ICL §4.5).

    ``--stage`` selects semantics only through the config file contents
    (stage 2 passes ``init_adapter_from``); the loop is shared.
    """
    import argparse

    from .configuration_pi05_icl import DemoEncoderConfig, LoRAConfig, icl_config_from_base
    from .data import ICLDataset
    from .lora import setup_trainable_policy
    from .modeling_pi05_icl import PI05ICLPolicy

    parser = argparse.ArgumentParser(description="Train pi05_icl (stage 1 or 2)")
    parser.add_argument("--config", required=True, help="stage YAML (configs/*.yaml)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume-from", default=None,
                        help="adapter dir to resume from (overrides config)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    stage = _load_stage_config(args.config)

    demo_cfg = DemoEncoderConfig(**stage.get("demo_encoder", {}))
    lora_cfg = LoRAConfig(**stage.get("lora", {}))
    # The policy is built before Accelerator exists, so honor LOCAL_RANK here;
    # otherwise every rank puts its copy on cuda:0 and multi-GPU runs OOM.
    device = f"cuda:{os.environ.get('LOCAL_RANK', '0')}" if torch.cuda.is_available() else "cpu"
    policy = PI05ICLPolicy.from_base(
        stage["base_checkpoint"], dtype=stage.get("dtype", "bfloat16"),
        device=device,
        demo_encoder=demo_cfg, lora=lora_cfg,
        gradient_checkpointing=True,
    )
    policy = setup_trainable_policy(policy, policy.config)

    dataset_cfg = stage["dataset"]
    if str(dataset_cfg.get("recompute_stats", "")).lower() == "auto":
        from .data import load_stats_json, missing_quantiles, recompute_stats_command, specs_from_stage_config

        specs, _ = specs_from_stage_config(stage)
        for spec in specs:
            problems = missing_quantiles(load_stats_json(spec.repo_id, spec.root))
            if problems:
                raise RuntimeError(
                    f"{spec.repo_id} lacks quantile stats required by pi05 normalization "
                    f"({'; '.join(problems)}). Fix before training:\n  "
                    + " ".join(recompute_stats_command(spec.repo_id, spec.root))
                )
        logger.info("stats preflight ok (q01/q99 present for state/action in all stage datasets)")
    registry_path = Path(dataset_cfg["task_registry"])
    init_adapter = args.resume_from or stage.get("init_adapter_from")
    if init_adapter:
        from .lora import load_icl_adapter

        load_icl_adapter(policy, init_adapter, base_name_or_path=stage["base_checkpoint"])
        logger.info("initialized adapters from %s", init_adapter)
    if not registry_path.exists():
        if registry_path.name.endswith("_droid.json"):
            raise FileNotFoundError(
                f"{registry_path} missing — run `pixi run -e lerobot icl_data download-subset` "
                "and `build-registry` first (the droid download is a separate step)"
            )
        raise FileNotFoundError(
            f"{registry_path} missing — run `pixi run -e lerobot icl_data build-registry ...`"
        )

    train_ds = ICLDataset(registry_path, policy.config, split="train",
                          demo_camera=dataset_cfg.get("demo_camera", "observation.images.base_0_rgb"),
                          seed=stage["train"].get("seed", 42))
    train_settings = TrainSettings(
        gate_lr_mult=lora_cfg.gate_lr_mult, **{
            k: v for k, v in stage["train"].items()
            if k in ("steps", "batch_per_gpu", "grad_accum", "lr", "warmup", "grad_clip",
                     "ckpt_every", "seed", "num_workers", "log_every",
                     "demo_zeroed_every", "val_every", "val_batches", "tensorboard",
                     "language_dropout")
        }
    )
    if "curriculum" in stage["train"]:
        train_settings.curriculum = [
            CurriculumPhase(p[0], p[1]) for p in stage["train"]["curriculum"]
        ]
    train_loader = DataLoader(
        train_ds, batch_size=train_settings.batch_per_gpu, shuffle=True,
        num_workers=train_settings.num_workers, pin_memory=True, drop_last=True,
    )

    # Fixed validation batches (bare-prompt); falls back to train samples
    # when the registry holds out no eval groups (e.g. the local smoke run).
    try:
        val_ds = ICLDataset(registry_path, policy.config, split="eval",
                            demo_camera=dataset_cfg.get("demo_camera", "observation.images.base_0_rgb"),
                            seed=train_settings.seed)
    except ValueError:
        val_ds = None
    val_batches = None
    if val_ds is not None or len(train_ds) >= train_settings.batch_per_gpu:
        val_batches = _val_batches(
            DataLoader(
                (val_ds or train_ds), batch_size=train_settings.batch_per_gpu, shuffle=False,
                num_workers=2,
            ),
            train_settings.val_batches,
        )

    from .data import load_task_registry, open_local_dataset

    reg = load_task_registry(registry_path)
    primary = reg["datasets"][0]
    primary_ds = open_local_dataset(primary["repo_id"], primary["root"])
    preprocessor, _post = build_stage_preprocessor(policy.config, primary_ds.meta.stats)

    output_dir = Path(args.output_dir or stage.get("output_dir", "so101_icl/runs/run"))
    run_training(
        policy, preprocessor, train_loader, val_batches, train_settings,
        output_dir, init_adapter_path=Path(init_adapter) if init_adapter else None,
    )
    return 0
