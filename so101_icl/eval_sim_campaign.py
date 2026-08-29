#!/usr/bin/env python3
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

"""M3 sim campaign: per-condition success-rate table in Isaac Sim.

Re-executes itself into Isaac Sim's Python exactly like ``rollout_sim.py``
(the bootstrap re-execs this file, hence a standalone script rather than a
module under ``so101_icl/``), then for each condition in
``full_icl / prompt_enriched / bare_prompt``:

1. pushes/clears the demo pack over the demo side channel
   (``so101_icl/serve_rollout_icl.py`` must be the rollout server);
2. runs ``trials`` episodes through the shared ``run_rollouts`` loop
   (evaluation mode, oracle success verdicts);
3. times every action-chunk round trip (:class:`TimedRolloutClient`).

Start the server first::

    pixi run -e lerobot python so101_icl/serve_rollout_icl.py \\
        --repo-id so101_icl/runs/icl_finetune_so101_v1/serving \\
        --policy-type pi05_icl --host 0.0.0.0 --port 8660

Then (from the repo root)::

    python so101_icl/eval_sim_campaign.py --registry \\
        so101_icl/configs/task_registry_so101.json --stats \\
        so101_icl/runs/icl_finetune_so101_v1/final/stage_stats.json \\
        --trials 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent          # so101_icl/
REPOSITORY_ROOT = PROJECT_ROOT.parent                    # repo root
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "self-improve"))
sys.path.insert(0, str(REPOSITORY_ROOT / "isaaclab" / "source" / "so101_rl"))

from so101_rl.runtime import launch_isaac_sim_before_task_imports  # noqa: E402

launch_isaac_sim_before_task_imports(Path(__file__))

from self_improve import contract, sim_rollout  # noqa: E402
from self_improve.config import RoundConfig  # noqa: E402
from self_improve.wire import RolloutClient  # noqa: E402

from so101_icl.conditions import apply_condition, condition_prompt  # noqa: E402
from so101_icl.demo_transport import DemoTransportClient  # noqa: E402
from so101_icl.latency import LatencyRecorder, TimedRolloutClient  # noqa: E402

logger = logging.getLogger("eval_sim_campaign")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--registry", type=Path, required=True,
                        help="stage-2 task registry (demo source for full_icl)")
    parser.add_argument("--group", default=None,
                        help="registry task group (default: the registry's first)")
    parser.add_argument("--stats", type=Path, default=None,
                        help="stage_stats.json for trajectory normalization")
    parser.add_argument("--conditions", default="full_icl,prompt_enriched,bare_prompt",
                        help="comma-separated subset of full_icl,prompt_enriched,bare_prompt")
    parser.add_argument("--trials", type=int, default=20,
                        help="episodes per condition (M3 gate: >= 20)")
    parser.add_argument("--task", default=None, help="Isaac Sim task id "
                        "(default: SO101-Object-In-Cup-Vision-Fixed-v0)")
    parser.add_argument("--task-prompt", default=None,
                        help="enriched language instruction (default: the round config's)")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--frames-per-demo", type=int, default=6)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--actions-per-chunk", type=int, default=None)
    parser.add_argument("--server-host", default=None)
    parser.add_argument("--server-port", type=int, default=None)
    parser.add_argument("--demo-host", default="127.0.0.1")
    parser.add_argument("--demo-port", type=int, default=8661)
    parser.add_argument("--config", type=Path, default=None,
                        help="round YAML for rollout defaults (like rollout_sim)")
    parser.add_argument("--output", type=Path, default=None,
                        help="write the markdown report here")
    parser.add_argument("--metrics-out", type=Path, default=None,
                        help="write the summary JSON here")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-wall-time-min", type=float, default=240.0)
    parser.add_argument("--visualizer", "--viz", default=None)
    parser.add_argument("--headless", action="store_true", default=False,
                        help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    from so101_icl.conditions import DemoPackBuilder

    config = RoundConfig.load(args.config) if args.config else RoundConfig()
    rollout = config.rollout
    if args.task:
        rollout.task_id = args.task
    if args.task_prompt:
        rollout.task_prompt = args.task_prompt
    if args.num_envs:
        rollout.num_envs = args.num_envs
    if args.max_episode_steps:
        rollout.max_episode_steps = args.max_episode_steps
    if args.actions_per_chunk:
        rollout.actions_per_chunk = args.actions_per_chunk
    if args.server_host:
        rollout.server_host = args.server_host
    if args.server_port:
        rollout.server_port = args.server_port
    if args.seed is not None:
        rollout.seed = args.seed

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    builder = DemoPackBuilder(
        args.registry, k=args.k, frames_per_demo=args.frames_per_demo,
        k_max=4, stats_path=args.stats,
    )
    transport = DemoTransportClient(args.demo_host, args.demo_port)
    group = builder.resolve_group(args.group)

    client = TimedRolloutClient(
        RolloutClient(rollout.server_host, rollout.server_port, timeout_s=600.0)
    )
    client.hello(_build_features(), rollout.actions_per_chunk, rollout.task_prompt,
                 action_dim=contract.NUM_JOINTS)
    logger.info("policy server handshake ok: chunk=%d action_dim=%d",
                client.chunk_size, client.action_dim)

    env = sim_rollout.make_env(
        rollout.task_id, rollout.num_envs,
        image_width=contract.IMAGE_WIDTH, image_height=contract.IMAGE_HEIGHT,
        seed=rollout.seed, device=args.device,
    )

    results: dict[str, dict] = {}
    latencies: dict[str, LatencyRecorder] = {}
    try:
        for condition in conditions:
            reply = apply_condition(condition, transport, builder, group=group)
            logger.info("condition %s: %s", condition, reply)
            client.recorder = LatencyRecorder(f"chunk:{condition}")
            summary = sim_rollout.run_rollouts(
                env, client,
                task=condition_prompt(condition, rollout.task_prompt),
                joint_map=config.resolve_joint_map(),
                num_episodes=args.trials,
                max_episode_steps=rollout.max_episode_steps,
                actions_per_chunk=rollout.actions_per_chunk,
                chunk_size_threshold=rollout.chunk_size_threshold,
                aggregate_fn_name=rollout.aggregate,
                round_dir=None,  # evaluation mode: oracle verdicts only
                max_wall_time_min=args.max_wall_time_min,
            )
            results[condition] = summary
            latencies[condition] = client.recorder
            logger.info("condition %s summary: %s", condition, json.dumps(summary))
    finally:
        env.close()
        client.close()
        transport.clear()

    report = build_report(results, latencies, trials=args.trials)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps({
            condition: {
                "rollout": summary,
                "latency": latencies[condition].stats(),
            } for condition, summary in results.items()
        }, indent=2) + "\n")
    return 0


def _build_features() -> dict:
    """Camera/state schema advertised to the rollout server (mirrors rollout_sim)."""
    features = {
        contract.STATE_FEATURE_KEY: {
            "dtype": "float32",
            "shape": (contract.NUM_JOINTS,),
            "names": [f"{joint}.pos" for joint in contract.SO101_JOINT_NAMES],
        },
    }
    for camera in contract.CAMERA_KEYS:
        features[contract.camera_feature_key(camera)] = {
            "dtype": "image",
            "shape": (contract.IMAGE_HEIGHT, contract.IMAGE_WIDTH,
                      contract.IMAGE_CHANNELS),
            "names": ["height", "width", "channels"],
        }
    return features


def build_report(results: dict[str, dict], latencies: dict[str, LatencyRecorder],
                 *, trials: int) -> str:
    lines = [
        "| condition | episodes | successes | success rate | chunk p50 (s) | chunk p95 (s) |",
        "|---|---|---|---|---|---|",
    ]
    rates = {}
    for condition in ("full_icl", "prompt_enriched", "bare_prompt"):
        if condition not in results:
            continue
        summary = results[condition]
        eps = int(summary.get("episodes", 0))
        succ = int(summary.get("successes", 0))
        rates[condition] = succ / eps if eps else 0.0
        lat = latencies.get(condition)
        st = lat.stats() if lat else {"n": 0}
        p50 = f"{st['p50_s']:.3f}" if st["n"] else "-"
        p95 = f"{st['p95_s']:.3f}" if st["n"] else "-"
        lines.append(f"| {condition} | {eps} | {succ} | {rates[condition]:.1%} | {p50} | {p95} |")

    lines.append("")
    if "full_icl" in rates and "bare_prompt" in rates:
        delta = (rates["full_icl"] - rates["bare_prompt"]) * 100.0
        gate_ok = delta >= 10.0 and trials >= 20
        lines.append(
            f"full_icl - bare_prompt = {delta:+.1f} pts over {trials} trials/condition "
            f"(M3 gate: >= +10 pts, >= 20 trials): {'PASS' if gate_ok else 'FAIL'}"
        )
    else:
        lines.append("M3 gate: not evaluable (needs full_icl and bare_prompt)")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
