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

"""Supervised-assist real-robot rollouts: the real-world "generate" step.

The robot-side machinery is exactly the proven recording stack — nothing new
runs on the robot:

1. ``follower_recording.launch.py`` — follower + cameras + MCAP recorder;
2. ``async_infer.launch.py policy_type:=pi05`` — the checkpoint drives the
   arm through the async chunked-inference nodes (policy served by
   ``policy_server`` on the GPU machine);
3. ``teleop_episode_keyboard`` — the human supervisor starts/stops episodes
   (r/s/d) and can stop the arm at any moment.

This wrapper spawns (1) and (2), reminds the human of (3), and afterwards
converts the kept episodes with the ``autonomous`` provenance tag, runs the
Qwen3-VL judge on final frames, and writes ``verdicts.jsonl`` into the round
directory for the same filter -> mix -> train stages used by sim rounds.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from . import contract
from .config import RoundConfig
from .train_bc import init_weights_source

logger = logging.getLogger(__name__)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETUP = "monomanual_dual_overhead"
DEFAULT_EXPERIMENT = "pi05_selfimprove"


def session_commands(config: RoundConfig, *, setup: str,
                     experiment: str, policy_server_address: str) -> List[List[str]]:
    """The exact robot-side launch commands for a supervised session."""
    checkpoint = init_weights_source(config)
    return [
        ["ros2", "launch", "so101_bringup", "follower_recording.launch.py",
         f"setup:={setup}",
         f"experiment_name:={experiment}",
         f"task:={config.rollout.task_prompt}"],
        ["ros2", "launch", "so101_inference", "async_infer.launch.py",
         f"setup:={setup}",
         "policy_type:=pi05",
         f"repo_id:={checkpoint}",
         f"server_address:={policy_server_address}",
         f"task:={config.rollout.task_prompt}",
         "actions_per_chunk:=16"],
        ["ros2", "run", "episode_recorder", "teleop_episode_keyboard"],
    ]


def run_session(config: RoundConfig, *, setup: str, experiment: str,
                policy_server_address: str, dry_run: bool) -> int:
    input_policy = init_weights_source(config)
    commands = session_commands(config, setup=setup, experiment=experiment,
                                policy_server_address=policy_server_address)
    print("\n=== Supervised self-improvement session ===")
    print("Start the policy server on the GPU machine first, e.g.:")
    print(f"  pixi run -e lerobot python policy_server/zmq_server.py "
          f"(serving {input_policy})\n")
    if dry_run:
        for command in commands:
            print("  " + " ".join(command))
        print("\n(dry run: nothing launched)")
        return 0

    processes = []
    try:
        for command in commands[:2]:
            print("launching:", " ".join(command))
            processes.append(subprocess.Popen(command))
            time.sleep(3.0)
        print("\nRobot is live. In another terminal run:")
        print("  " + " ".join(commands[2]))
        print("\nSupervisor loop: press r to record an autonomous episode,")
        print("s to keep it, d to discard; watch the arm — Ctrl-C here to end")
        print("the session. Converted data: run the 'post' subcommand next.\n")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nending session...")
        return 0
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:  # pragma: no cover
                process.kill()


def run_post(config: RoundConfig, rounds_root: Path, *, input_dir: Path,
             repo_id: str, vlm_model_id: str, interactive: bool,
             dry_run: bool) -> int:
    """Convert kept episodes -> tag autonomous -> VLM judge -> verdicts."""
    round_dir = config.round_dir(rounds_root)
    round_dir.mkdir(parents=True, exist_ok=True)

    convert_cmd = [
        "pixi", "run", "-e", "lerobot", "convert", "--",
        "--input-dir", str(input_dir),
        "--config", str(REPOSITORY_ROOT / "rosbag_to_lerobot" / "config" /
                        "so101_30hz.yaml"),
        "--setup", DEFAULT_SETUP,
        "--repo-id", repo_id,
        "--dataset-source", "autonomous",
    ]
    print("\n=== Converting kept episodes ===\n  " + " ".join(convert_cmd))
    if not dry_run:
        subprocess.run(convert_cmd, check=True)

    judge_cmd = [
        sys.executable, str(REPOSITORY_ROOT / "pi05-self-improve" / "judge_rollouts"),
        "--round-dir", str(round_dir),
        "--dataset-repo-id", repo_id,
        "--model-id", vlm_model_id,
    ]
    if interactive:
        judge_cmd.append("--interactive")
    print("\n=== VLM judge ===\n  " + " ".join(judge_cmd))
    if not dry_run:
        subprocess.run(judge_cmd, check=True)

    verdicts = contract.read_jsonl(round_dir / contract.VERDICT_FILENAME)
    approved = sum(1 for entry in verdicts if entry.get("approved"))
    print(f"\n{approved}/{len(verdicts)} episodes approved. Build the round "
          f"dataset next:")
    print(f"  python pi05-self-improve/build_round_dataset "
          f"--config {round_dir / 'config.yaml'} "
          f"--real-dataset-repo-id {repo_id}")
    meta_path = round_dir / contract.META_FILENAME
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    meta["real"] = {
        "input_dir": str(input_dir),
        "dataset_repo_id": repo_id,
        "episodes": len(verdicts),
        "approved": approved,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return 0


def main(argv: Optional[list] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    session = sub.add_parser("session", help="spawn launch files + supervise")
    session.add_argument("--config", type=Path, required=True)
    session.add_argument("--rounds-root", type=Path,
                         default=Path(__file__).resolve().parents[1] / "rounds")
    session.add_argument("--setup", default=DEFAULT_SETUP)
    session.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    session.add_argument("--policy-server-address", default="127.0.0.1:8090")
    session.add_argument("--dry-run", action="store_true")

    post = sub.add_parser("post", help="convert + judge a finished session")
    post.add_argument("--config", type=Path, required=True)
    post.add_argument("--rounds-root", type=Path,
                      default=Path(__file__).resolve().parents[1] / "rounds")
    post.add_argument("--input-dir", type=Path, default=Path.home() /
                      ".ros" / "so101_episodes" / DEFAULT_EXPERIMENT)
    post.add_argument("--repo-id", default=None,
                      help="converted dataset repo id "
                           "(default: local/so101_pi05_round<N>_real)")
    post.add_argument("--vlm-model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    post.add_argument("--interactive", action="store_true",
                      help="confirm each verdict before it counts")
    post.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if args.command == "session":
        config = RoundConfig.load(args.config)
        return run_session(config, setup=args.setup,
                           experiment=args.experiment,
                           policy_server_address=args.policy_server_address,
                           dry_run=args.dry_run)
    config = RoundConfig.load(args.config)
    repo_id = args.repo_id or f"local/so101_pi05_round{config.round_index}_real"
    return run_post(config, args.rounds_root, input_dir=args.input_dir,
                    repo_id=repo_id, vlm_model_id=args.vlm_model_id,
                    interactive=args.interactive, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
