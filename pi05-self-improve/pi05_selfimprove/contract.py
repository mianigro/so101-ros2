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

"""Data / sim / wire contract shared by every stage of the self-improvement loop.

This module MUST stay importable from Isaac Sim's own Python (stdlib + numpy
only): the sim rollout process imports it, and so do the pixi-side tools.
It duplicates the small set of constants from
``isaaclab/source/so101_rl/so101_rl/visual_contract.py`` on purpose — that
package cannot be imported without Isaac Lab.  ``tests/test_contract.py``
cross-checks the duplicates so they cannot drift.

Two unit spaces meet here:

* **dataset units** — what LeRobot datasets and the trained policy speak:
  absolute joint positions in the *robot's calibrated* radians
  (``observation.state`` / ``action`` from ``/follower/joint_states`` and
  ``/follower/forward_controller/commands``);
* **sim units** — the Isaac Sim articulation's joint positions in URDF
  radians, where the calibrated zero/range of the gripper differs.

``JointMap`` is the single, configurable bridge between them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# SO-101 contract (mirror of so101_rl.visual_contract; cross-checked in tests)
# ---------------------------------------------------------------------------

SO101_JOINT_NAMES: tuple = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
NUM_JOINTS = len(SO101_JOINT_NAMES)

#: Dataset camera feature names, in the policy's expected feature order.
CAMERA_KEYS: tuple = ("wrist", "overhead_1", "overhead_2")


def camera_feature_key(camera: str) -> str:
    return f"observation.images.{camera}"


STATE_FEATURE_KEY = "observation.state"
ACTION_FEATURE_KEY = "action"
TASK_FIELD = "task"

CONTROL_FREQUENCY_HZ = 30.0
CONTROL_PERIOD_S = 1.0 / CONTROL_FREQUENCY_HZ

#: Resolution recorded in LeRobot datasets from this rig (HWC).
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
IMAGE_CHANNELS = 3

#: URDF (== Isaac Sim) joint limits in radians, per joint, in SO101 order.
SIM_JOINT_LIMITS_RAD: Dict[str, tuple] = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.84121),
    "gripper": (-0.174533, 1.74533),
}

# ---------------------------------------------------------------------------
# Dataset <-> sim joint mapping
# ---------------------------------------------------------------------------

#: Gripper range observed in the teleop datasets (calibrated robot radians).
DATASET_GRIPPER_RANGE = (-0.5567, 0.5492)
#: Matching nominal gripper range in the sim articulation (URDF radians):
#: 0.0 jaw closed on the task cube, 1.5 the arm's nominal open pose.
SIM_GRIPPER_RANGE = (0.0, 1.5)


@dataclass
class JointMap:
    """Per-joint affine map between dataset units and sim radians.

    ``sim = scale * dataset + offset``.  Identity for the five arm joints;
    the gripper needs a remap because the Feetech calibration zero differs
    from the URDF zero (dataset [-0.56, 0.55] rad vs sim [0.0, 1.5] rad).
    Values are clamped to the sim URDF limits when converted to sim units.
    """

    scale: np.ndarray  # shape (6,), dataset -> sim
    offset: np.ndarray  # shape (6,), dataset -> sim

    def to_sim(self, dataset_positions: np.ndarray) -> np.ndarray:
        values = np.asarray(dataset_positions, dtype=np.float64)
        sim = values * self.scale + self.offset
        return self.clamp_to_sim(sim)

    def to_dataset(self, sim_positions: np.ndarray) -> np.ndarray:
        values = np.asarray(sim_positions, dtype=np.float64)
        return (values - self.offset) / self.scale

    def clamp_to_sim(self, sim_positions: np.ndarray) -> np.ndarray:
        values = np.asarray(sim_positions, dtype=np.float64)
        low = np.array([SIM_JOINT_LIMITS_RAD[j][0] for j in SO101_JOINT_NAMES])
        high = np.array([SIM_JOINT_LIMITS_RAD[j][1] for j in SO101_JOINT_NAMES])
        return np.clip(values, low, high)

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "scale": [float(v) for v in self.scale],
            "offset": [float(v) for v in self.offset],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Iterable[float]]) -> "JointMap":
        return cls(
            scale=np.asarray(data["scale"], dtype=np.float64),
            offset=np.asarray(data["offset"], dtype=np.float64),
        )


def default_joint_map() -> JointMap:
    """Identity for arm joints; dataset gripper range mapped onto sim range."""
    scale = np.ones(NUM_JOINTS)
    offset = np.zeros(NUM_JOINTS)
    d_lo, d_hi = DATASET_GRIPPER_RANGE
    s_lo, s_hi = SIM_GRIPPER_RANGE
    scale[-1] = (s_hi - s_lo) / (d_hi - d_lo)
    offset[-1] = s_lo - d_lo * scale[-1]
    return JointMap(scale=scale, offset=offset)


def identity_joint_map() -> JointMap:
    return JointMap(scale=np.ones(NUM_JOINTS), offset=np.zeros(NUM_JOINTS))


# ---------------------------------------------------------------------------
# Raw rollout episodes (npz) and manifests
# ---------------------------------------------------------------------------

#: npz array keys inside one ``episode_XXXXXX.npz``.  ``state``/``action`` are
#: in *dataset* units; ``sim_state``/``sim_action`` keep the raw sim radians
#: for debugging and mapping validation.
NPZ_STATE = "state"
NPZ_ACTION = "action"
NPZ_SIM_STATE = "sim_state"
NPZ_SIM_ACTION = "sim_action"
NPZ_TASK = "task"
NPZ_SUCCESS = "success"
NPZ_REASON = "reason"
NPZ_SOURCE = "source"
NPZ_FPS = "fps"
NPZ_JOINT_NAMES = "joint_names"
NPZ_JOINT_MAP = "joint_map"

ROLL_FILENAME = "rollouts.jsonl"
VERDICT_FILENAME = "verdicts.jsonl"
META_FILENAME = "round_meta.json"


@dataclass
class RolloutManifestEntry:
    """One line of ``rollouts.jsonl`` — the raw-rollout ground truth."""

    episode_index: int
    file: str  # npz path relative to the round's rollouts_raw/ dir
    source: str  # "sim" | "real"
    task: str
    success: bool
    reason: str  # "success" | "timeout" | "dropped" | "invalid" | ...
    steps: int
    fps: float


def save_episode_npz(
    path: Path,
    *,
    images: Dict[str, np.ndarray],
    state: np.ndarray,
    action: np.ndarray,
    sim_state: Optional[np.ndarray] = None,
    sim_action: Optional[np.ndarray] = None,
    task: str,
    success: bool,
    reason: str,
    source: str,
    joint_map: JointMap,
) -> None:
    """Write one recorded episode (images, state, action, labels) to disk.

    ``images`` maps camera name -> uint8 array of shape (T, H, W, 3), stacked
    over the episode's timesteps.  ``state``/``action`` are (T, 6) in dataset
    units; the optional ``sim_*`` twins keep raw sim radians for debugging.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        NPZ_STATE: np.asarray(state, dtype=np.float32),
        NPZ_ACTION: np.asarray(action, dtype=np.float32),
        NPZ_TASK: task,
        NPZ_SUCCESS: np.bool_(success),
        NPZ_REASON: reason,
        NPZ_SOURCE: source,
        NPZ_FPS: CONTROL_FREQUENCY_HZ,
        NPZ_JOINT_NAMES: np.array(SO101_JOINT_NAMES),
        NPZ_JOINT_MAP: json.dumps(joint_map.to_dict()),
    }
    if sim_state is not None:
        payload[NPZ_SIM_STATE] = np.asarray(sim_state, dtype=np.float32)
    if sim_action is not None:
        payload[NPZ_SIM_ACTION] = np.asarray(sim_action, dtype=np.float32)
    for camera in CAMERA_KEYS:
        if camera not in images:
            raise ValueError(f"missing camera '{camera}' in episode images")
        frames = np.asarray(images[camera])
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(
                f"camera '{camera}' frames must be uint8 (T, H, W, 3), got "
                f"{frames.dtype} {frames.shape}"
            )
        payload[camera_feature_key(camera)] = frames
    np.savez_compressed(path, **payload)


def append_jsonl(path: Path, entry: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    entries = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def iter_episode_files(round_dir: Path) -> Iterator[Path]:
    raw_dir = round_dir / "rollouts_raw"
    if not raw_dir.exists():
        return iter(())
    return iter(sorted(raw_dir.glob("episode_*.npz")))


__all__ = [
    "ACTION_FEATURE_KEY",
    "CAMERA_KEYS",
    "CONTROL_FREQUENCY_HZ",
    "CONTROL_PERIOD_S",
    "DATASET_GRIPPER_RANGE",
    "IMAGE_CHANNELS",
    "IMAGE_HEIGHT",
    "IMAGE_WIDTH",
    "JointMap",
    "META_FILENAME",
    "NPZ_ACTION",
    "NPZ_FPS",
    "NPZ_JOINT_MAP",
    "NPZ_JOINT_NAMES",
    "NPZ_REASON",
    "NPZ_SIM_ACTION",
    "NPZ_SIM_STATE",
    "NPZ_SOURCE",
    "NPZ_STATE",
    "NPZ_SUCCESS",
    "NPZ_TASK",
    "NUM_JOINTS",
    "ROLL_FILENAME",
    "SIM_GRIPPER_RANGE",
    "SIM_JOINT_LIMITS_RAD",
    "SO101_JOINT_NAMES",
    "STATE_FEATURE_KEY",
    "TASK_FIELD",
    "VERDICT_FILENAME",
    "append_jsonl",
    "camera_feature_key",
    "default_joint_map",
    "identity_joint_map",
    "iter_episode_files",
    "read_jsonl",
    "save_episode_npz",
]
