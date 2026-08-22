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

"""Round configuration for the VLA self-improvement loop.

One round = generate rollouts -> judge -> filter+mix -> BC retrain -> eval.
The same RoundConfig drives both phases; ``phase`` selects where rollouts run
(``sim`` fully autonomous, ``real`` supervised-assist on the robot).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from . import contract


@dataclass
class RolloutCfg:
    """Autonomous rollout generation settings."""

    task_id: str = "SO101-Object-In-Cup-Vision-Fixed-v0"
    task_prompt: str = "pick up the cube and place it in the cup"
    num_envs: int = 8
    num_episodes: int = 48
    max_episode_steps: int = 450  # 15 s at 30 Hz
    actions_per_chunk: int = 16
    aggregate: str = "weighted_average"  # temporal chunk aggregation
    chunk_size_threshold: float = 0.3  # refill when queue <= 30% of chunk
    seed: int = 0
    server_host: str = "127.0.0.1"
    server_port: int = 8660
    joint_map: Optional[Dict[str, List[float]]] = None  # None -> default map
    # Dataset/client->policy camera key remap for zero-shot base checkpoints.
    # Used by both rollout inference and lerobot-train; None = identity.
    image_key_map: Optional[Dict[str, str]] = None


@dataclass
class JudgeCfg:
    """Success evaluation settings."""

    name: str = "scripted"  # "scripted" (sim) | "vlm" | "human"
    vlm_model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    confirm_human: bool = True  # real phase: human confirms verdicts
    min_successes: int = 8  # refuse to build a dataset below this


@dataclass
class DatasetCfg:
    """Success filtering + teleop mixing settings."""

    repo_prefix: str = "local/so101_self_improve_round"  # + round index
    teleop_repo_ids: List[str] = field(
        default_factory=lambda: ["algorithmtheworld/so101-pick-and-place"]
    )
    teleop_episode_cap: Optional[int] = None  # None = all episodes
    keep_failures_on_disk: bool = True


@dataclass
class TrainCfg:
    """BC retraining (wraps lerobot-train)."""

    steps: int = 20_000
    batch_size: int = 8
    save_steps: int = 5_000
    policy_type: str = "pi05"
    init_from: str = "base"  # "previous" | "base"
    base_repo_id: str = "lerobot/pi05_base"
    freeze_vision_encoder: bool = True
    num_workers: int = 4
    extra_args: List[str] = field(default_factory=list)


@dataclass
class EvalCfg:
    num_episodes: int = 24


@dataclass
class RoundConfig:
    round_index: int = 0
    phase: str = "sim"  # "sim" | "real"
    checkpoint_in: Optional[str] = None  # required when train.init_from=previous
    rollout: RolloutCfg = field(default_factory=RolloutCfg)
    judge: JudgeCfg = field(default_factory=JudgeCfg)
    dataset: DatasetCfg = field(default_factory=DatasetCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)

    # ------------------------------------------------------------------
    # Round directory layout
    # ------------------------------------------------------------------

    @property
    def round_name(self) -> str:
        return f"round_{self.round_index}"

    def round_dir(self, rounds_root: Path) -> Path:
        return Path(rounds_root) / self.round_name

    def rollouts_raw_dir(self, rounds_root: Path) -> Path:
        return self.round_dir(rounds_root) / "rollouts_raw"

    def checkpoint_dir(self, rounds_root: Path) -> Path:
        return self.round_dir(rounds_root) / "checkpoint"

    def mixed_repo_id(self) -> str:
        return f"{self.dataset.repo_prefix}{self.round_index}_mixed"

    def resolve_joint_map(self) -> contract.JointMap:
        if self.rollout.joint_map is None:
            return contract.default_joint_map()
        return contract.JointMap.from_dict(self.rollout.joint_map)

    # ------------------------------------------------------------------
    # (De)serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def load(cls, path: Path) -> "RoundConfig":
        data = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoundConfig":
        rollout = RolloutCfg(**{**data.get("rollout", {})})
        judge = JudgeCfg(**{**data.get("judge", {})})
        dataset = DatasetCfg(**{**data.get("dataset", {})})
        train = TrainCfg(**{**data.get("train", {})})
        evaluation = EvalCfg(**{**data.get("eval", {})})
        return cls(
            round_index=int(data.get("round_index", 0)),
            phase=str(data.get("phase", "sim")),
            checkpoint_in=data.get("checkpoint_in"),
            rollout=rollout,
            judge=judge,
            dataset=dataset,
            train=train,
            eval=evaluation,
        )


__all__ = [
    "DatasetCfg",
    "EvalCfg",
    "JudgeCfg",
    "RolloutCfg",
    "RoundConfig",
    "TrainCfg",
]
