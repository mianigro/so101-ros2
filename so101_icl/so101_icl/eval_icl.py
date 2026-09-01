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

``sim`` (stage 2, M3): re-executes ``so101_icl/eval_sim_campaign.py``
(Isaac Sim bootstraps itself), which runs the per-condition campaign
against ``so101_icl/serve_rollout_icl.py`` — success-rate table, chunk
latency, M3 gate line.

``real`` (stage 2, M4): supervised per-condition campaign on the robot —
pushes/clears the demo pack per condition, drives the proven
``self_improve.real_rollout`` session stack, optionally runs the
post-session judge, and writes the M4 report (success trend + chunk
latency via the policy-server probe).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ICLDataset, load_task_registry
from .lora import setup_trainable_policy
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
    adapter_dir = Path(adapter_dir)
    init_adapter = adapter_dir if (adapter_dir / "icl_adapter.safetensors").exists() else None
    if init_adapter is None:
        logger.warning("no adapter found in %s; evaluating the zero-init policy", adapter_dir)
    setup_trainable_policy(
        policy, policy.config, init_adapter_from=init_adapter,
        base_name_or_path=base_checkpoint,
    )
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
    ablate_frames: tuple[int, ...] = (),
) -> str:
    """Three-condition query-loss table over held-out groups (markdown).

    Ablation dimensions (ICL §4.8): ``ablate_k`` (support size) and
    ``ablate_frames`` (keyframes per demo, evaluated at the largest k). The
    per-group table uses the largest-k run — the M2 gate reads per held-out
    group, not just the aggregate.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from .configuration_pi05_icl import DemoEncoderConfig, LoRAConfig

    policy = load_eval_policy(
        adapter_dir, stage_config["base_checkpoint"], device,
        demo_encoder=DemoEncoderConfig(**stage_config.get("demo_encoder", {})),
        lora=LoRAConfig(**stage_config.get("lora", {})),
    )

    split_note = None
    try:
        dataset = ICLDataset(registry_path, policy.config, split=split,
                             demo_camera=stage_config["dataset"].get(
                                 "demo_camera", "observation.images.left_wrist_0_rgb"),
                             keypoint_cache=stage_config["dataset"].get("keypoint_cache"))
    except ValueError:
        # Loud in the log AND in the report: the M2 gate is only meaningful
        # on held-out groups (smoke registries hold out none).
        split_note = (f"WARNING: split '{split}' is empty — evaluated on TRAIN "
                      f"groups; the M2 gate does not apply to this report.")
        logger.warning(split_note)
        dataset = ICLDataset(registry_path, policy.config, split="train",
                             demo_camera=stage_config["dataset"].get(
                                 "demo_camera", "observation.images.left_wrist_0_rgb"),
                             keypoint_cache=stage_config["dataset"].get("keypoint_cache"))
    reg = load_task_registry(registry_path)
    primary = reg["datasets"][0]
    from .data import open_local_dataset

    primary_ds = open_local_dataset(primary["repo_id"], primary["root"])
    preprocessor, _ = build_stage_preprocessor(policy.config, primary_ds.meta.stats)

    default_frames = dataset.frames_per_demo

    def _run(label: str, *, k: int, frames: int):
        dataset.k_choices = [k]
        dataset.frames_per_demo = frames
        dataset.set_epoch(0)  # fixed support/query sets across runs
        batches = _fixed_batches(dataset, batch_size, max(1, trials // batch_size), device)
        per_group: dict[str, dict[str, list[float]]] = {}
        losses_demo, losses_zeroed, losses_bare = [], [], []
        for j, raw in enumerate(batches):
            # shuffle=False: batch j covers index j*batch_size .. — group per row
            groups = [dataset._index[i][1]
                      for i in range(j * batch_size, min((j + 1) * batch_size, len(dataset)))]
            icl, rest = _split_icl_fields(raw)
            rest = preprocessor(rest)
            batch = {**rest, **_move_icl_fields(icl, device)}
            with torch.no_grad():
                loss, _ = policy.forward(batch)
                zeroed = _demo_zeroed_loss(policy, batch)
                loss_bare, _ = policy.forward(_strip_icl_fields(batch))
            losses_demo.append(loss.item())
            losses_zeroed.append(zeroed)
            losses_bare.append(loss_bare.item())
            # attribute the batch means to every group it touches (batches can
            # mix groups at boundaries; per-batch granularity is what we have)
            for g in set(groups):
                entry = per_group.setdefault(g, {"demo": [], "zeroed": []})
                entry["demo"].append(loss.item())
                entry["zeroed"].append(zeroed)
        dataset.frames_per_demo = default_frames
        row = (label, float(np.mean(losses_demo)), float(np.mean(losses_zeroed)),
               float(np.mean(losses_bare)))
        logger.info("%s demo=%.4f zeroed=%.4f bare=%.4f", *row)
        return row, per_group

    k_anchor = max(ablate_k)
    runs: list[tuple[tuple, dict]] = []
    for k in ablate_k:
        runs.append(_run(f"k={k}", k=k, frames=default_frames))
    for f in ablate_frames:
        runs.append(_run(f"F={f}", k=k_anchor, frames=f))

    return _offline_report(
        [r[0] for r in runs], n_k_runs=len(ablate_k), per_group=runs[min(len(ablate_k), len(runs)) - 1][1],
        split_note=split_note,
    )


def _offline_report(rows, *, n_k_runs: int, per_group: dict, split_note: str | None) -> str:
    """Assemble the offline markdown report (pure — unit-tested)."""
    lines = [
        "| setting | loss (demo) | loss (demo-zeroed) | loss (bare) | demo/zeroed |",
        "|---|---|---|---|---|",
    ]
    for label, d, z, b in rows:
        lines.append(f"| {label} | {d:.4f} | {z:.4f} | {b:.4f} | {d / z:.2f} |")
    med = np.median([r[1] / r[2] for r in rows[:n_k_runs]])
    lines.append("")
    lines.append(f"median demo/zeroed ratio (k ablation): {med:.2f} "
                 f"(M2 gate: <= 0.90 on held-out groups)")

    # per held-out group at the anchor k — the gate is per group, not aggregate
    if len(per_group) > 1:
        lines += ["", f"### per-group ({rows[min(n_k_runs, len(rows)) - 1][0]})", "",
                  "| group | loss (demo) | loss (zeroed) | demo/zeroed |",
                  "|---|---|---|---|"]
        for g in sorted(per_group):
            d = float(np.mean(per_group[g]["demo"]))
            z = float(np.mean(per_group[g]["zeroed"]))
            lines.append(f"| {g} | {d:.4f} | {z:.4f} | {d / z:.2f} |")

    if split_note:
        lines += ["", split_note]
    return "\n".join(lines)


def run_sim(args) -> int:
    """Delegate to the standalone campaign script (it re-execs into Isaac Sim)."""
    import os
    import sys

    script = Path(__file__).resolve().parents[1] / "eval_sim_campaign.py"
    if not script.is_file():
        raise SystemExit(f"campaign script missing: {script}")
    fwd = [
        "--registry", args.registry, "--trials", str(args.trials),
        "--conditions", args.conditions,
    ]
    for flag, value in (
        ("--group", getattr(args, "group", None)),
        ("--stats", getattr(args, "stats", None)),
        ("--k", getattr(args, "k", None)),
        ("--frames-per-demo", getattr(args, "frames_per_demo", None)),
        ("--demo-host", args.demo_host),
        ("--demo-port", args.demo_port),
        ("--output", args.output),
        ("--config", getattr(args, "config", None)),
    ):
        if value is not None:
            fwd += [flag, str(value)]
    if args.pass_through:
        fwd += args.pass_through  # AppLauncher flags (--headless, --viz kit, ...)
    logger.info("re-exec: python %s %s", script, " ".join(fwd))
    os.execv(sys.executable, [sys.executable, str(script), *fwd])
    return 0  # unreachable


def run_real(args) -> int:
    """Per-condition supervised campaign on the robot (ICL §8 M4).

    For each condition: push/clear the demo pack, run the proven
    ``self_improve.real_rollout`` session stack (follower + cameras +
    async ICL inference + human supervisor r/s/d), optionally convert and
    judge afterwards. The operator ends a condition's session with Ctrl-C.
    """
    import yaml

    repo_root = Path(__file__).resolve().parents[2]
    sys_path_setup = [str(repo_root / "self-improve"), str(repo_root)]
    import json
    import sys

    for p in sys_path_setup:
        if p not in sys.path:
            sys.path.insert(0, p)

    from self_improve.config import RoundConfig
    from self_improve.real_rollout import run_post, run_session

    from .conditions import DemoPackBuilder, apply_condition
    from .demo_transport import DemoTransportClient
    from .latency import LatencyRecorder, latency_gate, probe_policy_server

    config = RoundConfig.load(args.round_config) if args.round_config else RoundConfig()
    config.train.policy_type = "pi05_icl"
    if args.repo_id:
        config.train.init_from = "base"
        config.train.base_repo_id = args.repo_id
    if args.task_prompt:
        config.rollout.task_prompt = args.task_prompt

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    builder = None
    if "full_icl" in conditions:
        builder = DemoPackBuilder(
            args.registry, k=args.k, frames_per_demo=args.frames_per_demo,
            k_max=args.k_max, stats_path=args.stats,
            demo_camera=args.demo_camera,
        )
    group = builder.resolve_group(args.group) if builder is not None else None

    verdicts: dict[str, dict] = {}
    latencies: dict[str, LatencyRecorder] = {}
    for condition in conditions:
        print(f"\n=== condition: {condition} ({args.trials} episodes, supervisor r/s/d) ===")
        transport = DemoTransportClient(args.demo_host, args.demo_port)
        reply = apply_condition(condition, transport, builder, group=group)
        print(f"demo side channel: {reply}")
        if not args.dry_run:
            run_session(
                config, setup=args.setup, experiment=f"icl_{condition}",
                policy_server_address=args.server_address, dry_run=False,
            )
            if args.post:
                run_post(
                    config, Path(args.rounds_root),
                    input_dir=Path(args.input_dir),
                    repo_id=f"{args.repo_id or 'so101_icl_eval'}/{condition}",
                    vlm_model_id=args.vlm_model_id,
                    interactive=False, dry_run=False,
                )
                verdicts_path = config.round_dir(Path(args.rounds_root)) / "verdicts.jsonl"
                if verdicts_path.exists():
                    entries = [json.loads(l) for l in verdicts_path.read_text().splitlines() if l.strip()]
                    verdicts[condition] = {
                        "episodes": len(entries),
                        "approved": sum(1 for e in entries if e.get("approved")),
                    }
        transport.clear()

    if not args.skip_probe and not args.dry_run:
        print("\n=== chunk latency probe (per condition) ===")
        for condition in conditions:
            transport = DemoTransportClient(args.demo_host, args.demo_port)
            apply_condition(condition, transport, builder, group=group)
            try:
                latencies[condition] = probe_policy_server(
                    args.server_address,
                    repo_id=args.repo_id or config.train.base_repo_id,
                    n=args.probe_n,
                    task=config.rollout.task_prompt,
                )
                print(latencies[condition].summary_line())
            except RuntimeError as e:
                print(f"probe failed for {condition}: {e}")
            transport.clear()

    report = _real_report(conditions, verdicts, latencies, args)
    print(report)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report + "\n")
    return 0


def _real_report(conditions: list[str], verdicts: dict, latencies: dict, args) -> str:
    lines = ["# M4 real-robot campaign", ""]
    if verdicts:
        lines += [
            "| condition | episodes | approved | approval rate |",
            "|---|---|---|---|",
        ]
        for condition in conditions:
            v = verdicts.get(condition)
            if v is None:
                lines.append(f"| {condition} | - | - | - |")
                continue
            rate = v["approved"] / v["episodes"] if v["episodes"] else 0.0
            lines.append(f"| {condition} | {v['episodes']} | {v['approved']} | {rate:.1%} |")
        lines.append("")
    else:
        lines.append("_no judge verdicts recorded (run with --post for success counts)_")
        lines.append("")
    for condition in conditions:
        lat = latencies.get(condition)
        if lat is not None:
            lines.append(lat.summary_line())
    if latencies:
        lines.append("")
        lines.append(latency_gate(latencies))
    lines.append("")
    lines.append(f"M4 gate: full_icl >= prompt_enriched >= bare_prompt success trend "
                 f"over >= {args.trials} trials/condition, p95 within +20%.")
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
    p.add_argument("--frames", type=int, nargs="+", default=None,
                   help="frames-per-demo ablations (default: eval.ablations.frames, else none)")
    p.add_argument("--output", default=None, help="write the markdown report here")

    p = sub.add_parser(
        "sim", help="stage-2 sim rollout campaign (M3; re-execs eval_sim_campaign.py)"
    )
    p.add_argument("--registry", required=True,
                   help="stage-2 task registry (demo source for full_icl)")
    p.add_argument("--trials", type=int, default=None,
                   help="episodes/condition (default: eval.sim_trials from the stage YAML, else 20)")
    p.add_argument("--conditions", default=None,
                   help="comma-separated conditions (default: eval.conditions, else all three)")
    p.add_argument("--group", default=None, help="registry task group")
    p.add_argument("--stats", default=None, help="stage_stats.json for traj normalization")
    p.add_argument("--demo-camera", default="observation.images.left_wrist_0_rgb",
                   help="policy-side camera key demos are drawn from "
                        "(must match the stage YAML the policy was trained with)")
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--frames-per-demo", type=int, default=None)
    p.add_argument("--demo-host", default="127.0.0.1")
    p.add_argument("--demo-port", type=int, default=8661)
    p.add_argument("--output", default=None, help="write the markdown report here")
    p.add_argument("--config", default=None,
                   help="self-improve round YAML for rollout defaults")
    p.add_argument("pass_through", nargs="*", default=None,
                   help=argparse.SUPPRESS)  # AppLauncher flags pass through

    p = sub.add_parser("real", help="stage-2 real-robot campaign (M4)")
    p.add_argument("--trials", type=int, default=10,
                   help="episodes/condition (M4 gate: >= 10)")
    p.add_argument("--conditions", default=None,
                   help="comma-separated conditions (default: eval.conditions, else all three)")
    p.add_argument("--registry", default=None, help="stage-2 registry (full_icl)")
    p.add_argument("--group", default=None)
    p.add_argument("--stats", default=None)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--frames-per-demo", type=int, default=6)
    p.add_argument("--k-max", type=int, default=4,
                   help="DemoEncoder slot count (config.demo_encoder.k_max)")
    p.add_argument("--demo-host", default="127.0.0.1")
    p.add_argument("--demo-port", type=int, default=8661)
    p.add_argument("--round-config", default=None, help="self-improve round YAML")
    p.add_argument("--rounds-root", default="self-improve/rounds")
    p.add_argument("--input-dir", default=None,
                   help="kept-episode MCAP dir for --post conversion")
    p.add_argument("--repo-id", default=None,
                   help="serving checkpoint served on the policy server")
    p.add_argument("--task-prompt", default=None)
    p.add_argument("--setup", default="monomanual_dual_overhead")
    p.add_argument("--server-address", default="127.0.0.1:8090")
    p.add_argument("--vlm-model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--post", action="store_true",
                   help="after each condition's session: convert + judge kept episodes")
    p.add_argument("--skip-probe", action="store_true",
                   help="skip the chunk-latency probe against the policy server")
    p.add_argument("--probe-n", type=int, default=30)
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan only; launch nothing")
    p.add_argument("--output", default=None, help="write the markdown report here")

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
        abl = eval_cfg.get("ablations", {})
        ablate_frames = (tuple(args.frames) if args.frames is not None
                         else tuple(abl.get("frames", [])))
        report = run_offline(
            args.adapter, args.registry, stage, split=args.split,
            trials=trials, batch_size=args.batch_size, ablate_k=ablate_k,
            ablate_frames=ablate_frames,
        )
        print(report)
        if args.output:
            Path(args.output).write_text(report + "\n")
        return 0

    if args.mode == "sim":
        eval_cfg = {}
        if args.config and str(args.config).endswith((".yaml", ".yml")):
            import yaml

            stage = yaml.safe_load(Path(args.config).read_text())
            if "rollout" not in stage:  # a stage YAML, not a round YAML
                eval_cfg = stage.get("eval") or {}
                args.config = None
        args.trials = args.trials if args.trials is not None else int(eval_cfg.get("sim_trials", 20))
        if args.conditions is None:
            args.conditions = ",".join(eval_cfg.get("conditions") or
                                       ["full_icl", "prompt_enriched", "bare_prompt"])
        if args.k is None:
            args.k = 2
        if args.frames_per_demo is None:
            args.frames_per_demo = 6
        return run_sim(args)

    if args.mode == "real":
        if args.conditions is None:
            args.conditions = "full_icl,prompt_enriched,bare_prompt"
        if "full_icl" in args.conditions and not args.registry:
            raise SystemExit("real: --registry is required when full_icl is a condition")
        return run_real(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
