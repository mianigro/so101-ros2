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

"""Evaluation harness for pi05_icl (ICL §4.8).

``offline`` (stage 1, no robot): fixed support/query sets over held-out
registry groups; three conditions per group — demo-conditioned,
demo-zeroed (gates forced 0) and bare-prompt — plus k/F ablations; markdown
report. The M2 gate reads directly off this table: median
demo-conditioned loss <= 0.9x demo-zeroed AND <= bare-prompt.

``sim``/``real`` (stage 2): thin drivers into the self-improve rollout
machinery; the actual campaigns run when the stage-2 data exists.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ICLDataset, load_task_registry
from .lora import load_icl_adapter, setup_trainable_policy
from .modeling_pi05_icl import PI05ICLPolicy
from .train_loop import (
    _demo_zeroed_loss,
    _move_icl_fields,
    _split_icl_fields,
    _strip_icl_fields,
    build_stage_preprocessor,
    evaluate_loss,
)

logger = logging.getLogger(__name__)


def load_eval_policy(adapter_dir: Path | str, base_checkpoint: str = "lerobot/pi05_base",
                     device: str = "cuda", *, demo_encoder=None, lora=None):
    """Build the policy FROM the stage config so module shapes match it.

    Replacing ``policy.config.demo_encoder`` after construction would leave
    the built DemoEncoder at the default shapes — a latent crash whenever a
    stage YAML deviates from the defaults.
    """
    from .configuration_pi05_icl import DemoEncoderConfig, LoRAConfig

    policy = PI05ICLPolicy.from_base(
        base_checkpoint, device=device, dtype="bfloat16",
        demo_encoder=demo_encoder or DemoEncoderConfig(), lora=lora or LoRAConfig(),
    )
    setup_trainable_policy(policy, policy.config)
    adapter_dir = Path(adapter_dir)
    if (adapter_dir / "icl_adapter.safetensors").exists():
        load_icl_adapter(policy, adapter_dir, base_name_or_path=base_checkpoint)
    else:
        logger.warning("no adapter found in %s; evaluating the zero-init policy", adapter_dir)
    policy.eval()
    return policy


def _fixed_batches(dataset: ICLDataset, batch_size: int, n_batches: int, device: str):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    batches = []
    for batch in loader:
        batches.append(batch)
        if len(batches) >= n_batches:
            break
    return batches


def run_offline(
    adapter_dir: Path | str,
    registry_path: Path | str,
    stage_config: dict,
    *,
    split: str = "test",
    trials: int = 8,
    batch_size: int = 4,
    ablate_k: tuple[int, ...] = (1, 2, 4),
) -> str:
    """Three-condition query-loss table over held-out groups (markdown)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from .configuration_pi05_icl import DemoEncoderConfig, LoRAConfig

    policy = load_eval_policy(
        adapter_dir, stage_config["base_checkpoint"], device,
        demo_encoder=DemoEncoderConfig(**stage_config.get("demo_encoder", {})),
        lora=LoRAConfig(**stage_config.get("lora", {})),
    )

    try:
        dataset = ICLDataset(registry_path, policy.config, split=split,
                             demo_camera=stage_config["dataset"].get(
                                 "demo_camera", "observation.images.base_0_rgb"))
    except ValueError:
        logger.warning("split '%s' is empty; falling back to train (smoke registries "
                       "hold out no groups)", split)
        dataset = ICLDataset(registry_path, policy.config, split="train",
                             demo_camera=stage_config["dataset"].get(
                                 "demo_camera", "observation.images.base_0_rgb"))
    reg = load_task_registry(registry_path)
    primary = reg["datasets"][0]
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    primary_ds = LeRobotDataset(
        primary["repo_id"], root=str(Path(primary["root"]).expanduser() / primary["repo_id"])
    )
    preprocessor, _ = build_stage_preprocessor(policy.config, primary_ds.meta.stats)

    rows = []
    for k in ablate_k:
        dataset.k_choices = [k]
        dataset.set_epoch(0)  # fixed support/query sets across runs
        batches = _fixed_batches(dataset, batch_size, max(1, trials // batch_size), device)
        processed = []
        for raw in batches:
            icl, rest = _split_icl_fields(raw)
            rest = preprocessor(rest)
            processed.append({**rest, **_move_icl_fields(icl, device)})

        losses_demo, losses_zeroed, losses_bare = [], [], []
        for batch in processed:
            with torch.no_grad():
                loss, _ = policy.forward(batch)
                losses_demo.append(loss.item())
                losses_zeroed.append(_demo_zeroed_loss(policy, batch))
                bare = _strip_icl_fields(batch)
                loss_bare, _ = policy.forward(bare)
                losses_bare.append(loss_bare.item())
        rows.append(
            (k, float(np.mean(losses_demo)), float(np.mean(losses_zeroed)),
             float(np.mean(losses_bare)))
        )
        logger.info("k=%d demo=%.4f zeroed=%.4f bare=%.4f", *rows[-1])

    lines = [
        "| k | loss (demo) | loss (demo-zeroed) | loss (bare) | demo/zeroed |",
        "|---|---|---|---|---|",
    ]
    for k, d, z, b in rows:
        lines.append(f"| {k} | {d:.4f} | {z:.4f} | {b:.4f} | {d / z:.2f} |")
    med = np.median([r[1] / r[2] for r in rows])
    lines.append("")
    lines.append(f"median demo/zeroed ratio: {med:.2f} "
                 f"(M2 gate: <= 0.90 on held-out groups)")
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="pi05_icl evaluation harness")
    sub = parser.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("offline", help="stage-1 offline eval (no robot)")
    p.add_argument("--adapter", required=True)
    p.add_argument("--registry", required=True)
    p.add_argument("--config", required=True, help="stage YAML")
    p.add_argument("--split", default="test", choices=["eval", "test"])
    p.add_argument("--trials", type=int, default=None,
                   help="query samples (default: eval.offline_trials from the stage YAML, else 8)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--k", type=int, nargs="+", default=None,
                   help="k ablations (default: eval.ablations.k from the stage YAML, else 1 2 4)")
    p.add_argument("--output", default=None, help="write the markdown report here")

    p = sub.add_parser("sim", help="stage-2 sim rollout campaign (deferred)")
    p.add_argument("--trials", type=int, default=20)
    p = sub.add_parser("real", help="stage-2 real-robot campaign (deferred)")
    p.add_argument("--trials", type=int, default=10)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if args.mode == "offline":
        import yaml

        stage = yaml.safe_load(Path(args.config).read_text())
        eval_cfg = stage.get("eval") or {}
        trials = args.trials if args.trials is not None else int(eval_cfg.get("offline_trials", 8))
        if args.k is not None:
            ablate_k = tuple(args.k)
        else:
            ablate_k = tuple(eval_cfg.get("ablations", {}).get("k", [1, 2, 4]))
        report = run_offline(
            args.adapter, args.registry, stage, split=args.split,
            trials=trials, batch_size=args.batch_size, ablate_k=ablate_k,
        )
        print(report)
        if args.output:
            Path(args.output).write_text(report + "\n")
        return 0

    if args.mode in ("sim", "real"):
        print(
            f"stage-2 {args.mode} campaign runs through the self-improve machinery "
            f"({args.trials} trials/condition); deferred until the stage-2 data "
            "curation target is met (ICL §6.1). Planned commands:\n"
            "  python self-improve/rollout_sim.py --config <round.yaml> --eval "
            "--image-key-map pi05_base   # sim arm\n"
            "  python self-improve/real_rollout.py session --setup monomanual_dual_overhead"
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
