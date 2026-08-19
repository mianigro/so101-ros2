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

"""Round orchestration: generate -> build -> train -> eval, one command.

The rollout policy server and the training run both want the whole GPU, so
``run_round`` owns the server's lifecycle: it spawns ``serve_policy`` with the
round's input checkpoint for the rollout stage, stops it before training, and
respawns it with the fresh checkpoint for the eval stage.  The sim rollout
itself runs as a subprocess of the ``rollout_sim`` entry, which re-executes
into Isaac Sim's Python exactly like the isaaclab/ entry points.
"""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import RoundConfig
from .train_bc import init_weights_source

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGES = ("rollout", "build", "train", "eval")
ALL_STAGES = ("rollout", "judge", "build", "train", "eval")


class PolicyServerProcess:
    """Managed ``serve_policy`` subprocess (same interpreter = pixi lerobot)."""

    def __init__(self, repo_id: str, *, host: str = "127.0.0.1",
                 port: int = 8660, policy_type: str = "pi05",
                 actions_per_chunk: int = 16,
                 device: str = "cuda",
                 dtype: str = "bfloat16") -> None:
        self.repo_id = repo_id
        self.host = host
        self.port = port
        self._argv = [
            str(PROJECT_ROOT / "serve_policy.py"),
            "--repo-id", repo_id,
            "--host", host,
            "--port", str(port),
            "--policy-type", policy_type,
            "--actions-per-chunk", str(actions_per_chunk),
            "--device", device,
            "--dtype", dtype,
        ]
        self._process: Optional[subprocess.Popen] = None

    def start(self, timeout_s: float = 600.0) -> None:
        if self._process is not None:
            raise RuntimeError("policy server already running")
        logger.info("starting policy server: %s", " ".join(self._argv))
        self._process = subprocess.Popen(self._argv)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"serve_policy exited early with code "
                    f"{self._process.returncode}")
            try:
                with socket.create_connection(
                        (self.host, self.port), timeout=1.0):
                    return
            except OSError:
                time.sleep(2.0)
        self.stop()
        raise RuntimeError(
            f"policy server did not open {self.host}:{self.port} within "
            f"{timeout_s:.0f}s (3B checkpoints take a while to load)")

    def stop(self) -> None:
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:  # pragma: no cover - stubborn server
            self._process.kill()
            self._process.wait(timeout=10.0)
        self._process = None
        # Give the GPU a beat to release memory before training starts.
        time.sleep(5.0)
        logger.info("policy server stopped")


def _update_round_meta(round_dir: Path, key: str, value: Any) -> None:
    meta_path = round_dir / "round_meta.json"
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    meta[key] = value
    round_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def _run_sim_rollout(config: RoundConfig, rounds_root: Path,
                     *, eval_mode: bool,
                     metrics_out: Optional[Path] = None,
                     visualizer: Optional[str] = None,
                     device: str = "cuda:0",
                     num_envs: Optional[int] = None) -> None:
    argv = [
        str(PROJECT_ROOT / "rollout_sim.py"),
        "--config", str(_config_cache_path(config, rounds_root)),
        "--rounds-root", str(rounds_root),
        "--device", device,
    ]
    if num_envs is not None:
        argv += ["--num-envs", str(num_envs)]
    if eval_mode:
        argv.append("--eval")
        argv += ["--num-episodes", str(config.eval.num_episodes)]
    if metrics_out is not None:
        argv += ["--metrics-out", str(metrics_out)]
    if visualizer is not None:
        argv += ["--visualizer", visualizer]
    logger.info("launching: %s", " ".join(argv))
    subprocess.run(argv, check=True)


def _config_cache_path(config: RoundConfig, rounds_root: Path) -> Path:
    """Persist the effective config next to the round artifacts for reuse."""
    path = config.round_dir(rounds_root) / "config.yaml"
    if not path.exists():
        config.save(path)
    return path


def run_round(config: RoundConfig, rounds_root: Path,
              stages: List[str], *,
              visualizer: Optional[str] = None,
              policy_device: str = "cuda",
              policy_dtype: str = "bfloat16",
              sim_device: str = "cuda:0",
              num_envs: Optional[int] = None) -> Dict[str, Any]:
    """Execute the requested stages of one self-improvement round."""
    unknown = [stage for stage in stages if stage not in ALL_STAGES]
    if unknown:
        raise ValueError(f"unknown stages {unknown}; expected subset of "
                         f"{ALL_STAGES}")

    input_policy = None
    if any(stage in {"rollout", "train", "eval"} for stage in stages):
        # Resolve before creating artifacts or starting the expensive policy
        # server so an invalid previous-checkpoint config fails immediately.
        input_policy = init_weights_source(config)

    round_dir = config.round_dir(rounds_root)
    round_dir.mkdir(parents=True, exist_ok=True)
    config.save(round_dir / "config.yaml")

    if "rollout" in stages:
        assert input_policy is not None
        server = PolicyServerProcess(
            input_policy,
            host=config.rollout.server_host,
            port=config.rollout.server_port,
            policy_type=config.train.policy_type,
            actions_per_chunk=config.rollout.actions_per_chunk,
            device=policy_device,
            dtype=policy_dtype,
        )
        server.start()
        try:
            _run_sim_rollout(config, rounds_root, eval_mode=False,
                             visualizer=visualizer, device=sim_device,
                             num_envs=num_envs)
        finally:
            server.stop()
        manifest = (round_dir / "rollouts.jsonl")
        if manifest.exists():
            entries = [json.loads(line) for line in
                       manifest.read_text().splitlines() if line.strip()]
            _update_round_meta(round_dir, "rollout", {
                "episodes": len(entries),
                "successes": sum(1 for e in entries if e.get("success")),
            })

    if "judge" in stages:
        # Sim rounds are already labeled by the scripted oracle at rollout
        # time; the VLM judge stage matters for real rounds (judge_rollouts).
        logger.info("judge stage: sim rollouts carry scripted verdicts in "
                    "rollouts.jsonl; nothing to do (use judge_rollouts for "
                    "real episodes)")

    if "build" in stages:
        from .dataset_tools.build_round_dataset import build_from_config

        summary = build_from_config(config, rounds_root)
        _update_round_meta(round_dir, "dataset", summary)

    if "train" in stages:
        from .train_bc import run_training

        record = run_training(config, rounds_root)
        _update_round_meta(round_dir, "train", record)

    if "eval" in stages:
        from .train_bc import find_latest_checkpoint

        train_dir = round_dir / "checkpoint" / "run"
        latest = find_latest_checkpoint(train_dir)
        if latest is None:
            assert input_policy is not None
            latest = Path(input_policy)
        server = PolicyServerProcess(
            str(latest),
            host=config.rollout.server_host,
            port=config.rollout.server_port,
            policy_type=config.train.policy_type,
            actions_per_chunk=config.rollout.actions_per_chunk,
            device=policy_device,
            dtype=policy_dtype,
        )
        metrics_path = round_dir / "metrics_eval.json"
        server.start()
        try:
            _run_sim_rollout(config, rounds_root, eval_mode=True,
                             metrics_out=metrics_path,
                             visualizer=visualizer, device=sim_device,
                             num_envs=num_envs)
        finally:
            server.stop()
        if metrics_path.exists():
            _update_round_meta(round_dir, "eval",
                               json.loads(metrics_path.read_text()))

    return json.loads((round_dir / "round_meta.json").read_text()) \
        if (round_dir / "round_meta.json").exists() else {}


def main(argv: Optional[list] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run one pi0.5 self-improvement round (sim phase).",
    )
    parser.add_argument("--config", type=Path, required=True,
                        help="round YAML")
    parser.add_argument("--rounds-root", type=Path,
                        default=PROJECT_ROOT / "rounds")
    parser.add_argument("--stages", default="rollout,build,train,eval",
                        help=f"comma-separated subset of {ALL_STAGES}")
    parser.add_argument(
        "--visualizer", default=None,
        help="Isaac Lab visualizer for rollout/eval (use 'kit' for the "
             "native Isaac Sim window; omitted is headless)",
    )
    parser.add_argument(
        "--policy-device", default="cuda",
        help="PyTorch device for the rollout/eval policy server",
    )
    parser.add_argument(
        "--policy-dtype", default="bfloat16",
        choices=("auto", "bfloat16", "float32"),
        help="policy inference precision; bfloat16 (default) halves the 3B "
             "footprint — pi05_base is stored float32 (~12.4 GB) which OOMs "
             "16 GB cards alongside batched sim rollouts",
    )
    parser.add_argument(
        "--sim-device", default="cuda:0",
        help="Isaac Sim physics and rendering device",
    )
    parser.add_argument(
        "--num-envs", type=int, default=None,
        help="one-off override of the config's rollout.num_envs (e.g. 4 "
             "when the kit viewport pushes the sim GPU's memory)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config = RoundConfig.load(args.config)
    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    meta = run_round(config, args.rounds_root, stages,
                     visualizer=args.visualizer,
                     policy_device=args.policy_device,
                     policy_dtype=args.policy_dtype,
                     sim_device=args.sim_device,
                     num_envs=args.num_envs)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
