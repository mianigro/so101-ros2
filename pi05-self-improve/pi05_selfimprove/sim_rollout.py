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

"""Autonomous VLA rollouts in Isaac Sim — the "generate" step of the
pi0.5 self-improvement loop.

This module runs inside Isaac Sim's Python (bootstrapped by the
``rollout_sim`` entry script).  The policy itself lives in the pixi
``lerobot`` env behind a :class:`pi05_selfimprove.wire.RolloutClient`; here we
only step the environment, execute action chunks with the same temporal
aggregation as the real async inference client, record every episode, and read
the scripted success oracle from the termination manager.

Isaac Lab imports are deliberately deferred into functions so that the pure
pieces (chunk buffer, episode recorder, aggregation) stay importable — and
testable — without Isaac Sim.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from . import contract
from .wire import RolloutClient

logger = logging.getLogger(__name__)

# Same aggregation registry as so101_inference.async_client (lerobot-style).
AGGREGATE_FUNCTIONS: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "latest_only": lambda old, new: new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
}

#: Scene sensor names for the three dataset cameras, keyed by camera name.
CAMERA_SENSORS = {
    "wrist": "wrist_camera",
    "overhead_1": "overhead_1_camera",
    "overhead_2": "overhead_2_camera",
}

#: Termination-term names of the object-in-cup task, in precedence order.
SUCCESS_TERM = "success"
FAILURE_TERMS = ("dropped", "invalid")
TIMEOUT_TERM = "time_out"


class EnvActionBuffer:
    """Per-env action chunk queue with timestep-keyed aggregation.

    Mirrors ``so101_inference.async_client._aggregate_actions``: an incoming
    chunk starts at the observation timestep; actions for timesteps already
    queued are merged with the aggregate function; actions for past timesteps
    are dropped.  ``pop`` returns None when starved, and the caller holds the
    last command (exactly what the real controller does).
    """

    def __init__(self, aggregate_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        self._aggregate_fn = aggregate_fn
        self._queue: Dict[int, np.ndarray] = {}
        self._latest_executed = -1
        self.last_action: Optional[np.ndarray] = None

    def submit(self, chunk: np.ndarray, start_timestep: int) -> None:
        for index in range(chunk.shape[0]):
            timestep = start_timestep + index
            if timestep <= self._latest_executed:
                continue
            existing = self._queue.get(timestep)
            if existing is None:
                self._queue[timestep] = np.array(chunk[index], dtype=np.float64)
            else:
                self._queue[timestep] = self._aggregate_fn(existing, chunk[index])

    def pending_from(self, timestep: int) -> int:
        return sum(1 for key in self._queue if key >= timestep)

    def pop(self, timestep: int) -> Optional[np.ndarray]:
        action = self._queue.pop(timestep, None)
        if action is not None:
            self._latest_executed = max(self._latest_executed, timestep)
            self.last_action = action
        return action

    def reset(self) -> None:
        self._queue.clear()
        self._latest_executed = -1
        self.last_action = None


class EpisodeRecorder:
    """Buffers one episode; camera frames go straight to disk memmaps.

    A 15 s episode is 3 cameras x 450 frames x 480x640x3 B ~= 1.2 GiB raw, so
    frames stream into per-camera ``.npy`` memmaps under a temp dir and are
    folded into the compressed episode npz when the episode finishes.
    """

    def __init__(self, tmp_dir: Path, env_id: int, episode_index: int,
                 max_steps: int, image_height: int = contract.IMAGE_HEIGHT,
                 image_width: int = contract.IMAGE_WIDTH) -> None:
        self._tmp_dir = Path(tmp_dir)
        self._env_id = env_id
        self._episode_index = episode_index
        self._max_steps = max_steps
        self._image_height = image_height
        self._image_width = image_width
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.sim_states: List[np.ndarray] = []
        self.sim_actions: List[np.ndarray] = []
        self._frames: Dict[str, Any] = {}
        self._paths: List[Path] = []
        self._steps = 0

    @property
    def episode_index(self) -> int:
        return self._episode_index

    def start(self) -> None:
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        for camera in contract.CAMERA_KEYS:
            path = (self._tmp_dir /
                    f"env{self._env_id}_ep{self._episode_index}_{camera}.npy")
            self._frames[camera] = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.uint8,
                shape=(self._max_steps, self._image_height,
                       self._image_width, contract.IMAGE_CHANNELS),
            )
            self._paths.append(path)

    def add(self, images: Dict[str, np.ndarray], state: np.ndarray,
            action: np.ndarray, sim_state: np.ndarray,
            sim_action: np.ndarray) -> None:
        if self._steps >= self._max_steps:
            raise IndexError("episode exceeded max steps")
        for camera, frames in self._frames.items():
            frames[self._steps] = images[camera]
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.sim_states.append(np.asarray(sim_state, dtype=np.float32))
        self.sim_actions.append(np.asarray(sim_action, dtype=np.float32))
        self._steps += 1

    @property
    def steps(self) -> int:
        return self._steps

    def save(self, rollouts_raw_dir: Path, *, task: str, success: bool,
             reason: str, source: str, joint_map: contract.JointMap) -> Path:
        if self._steps == 0:
            raise ValueError("cannot save an empty episode")
        path = (Path(rollouts_raw_dir) /
                f"episode_{self._episode_index:06d}.npz")
        images = {camera: np.asarray(frames[: self._steps])
                  for camera, frames in self._frames.items()}
        contract.save_episode_npz(
            path,
            images=images,
            state=np.stack(self.states),
            action=np.stack(self.actions),
            sim_state=np.stack(self.sim_states),
            sim_action=np.stack(self.sim_actions),
            task=task,
            success=success,
            reason=reason,
            source=source,
            joint_map=joint_map,
        )
        return path

    def discard(self) -> None:
        for frames in self._frames.values():
            del frames
        self._frames = {}
        for path in self._paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:  # pragma: no cover - best-effort cleanup
                pass
        self._paths = []
        if self._tmp_dir is not None:
            try:
                self._tmp_dir.rmdir()  # succeeds only when the last env cleaned up
            except OSError:
                pass


def make_env(task_id: str, num_envs: int, *, image_width: int,
             image_height: int, seed: int, device: str = "cuda:0"):
    """Build the registered task retooled for absolute-position VLA control.

    Overrides applied on top of the task's env cfg (nothing in ``so101_rl``
    is modified):

    * cameras render at the dataset resolution (480x640) instead of the
      RL-policy resolution (120x160) — same aspect ratio, so the calibrated
      intrinsics stay valid;
    * the delta-action term is replaced by an absolute ``JointPositionAction``
      in SO101 joint order, which is what VLA action chunks command;
    * a fixed seed for reproducible round configs.
    """
    import gymnasium as gym  # noqa: PLC0415 - must import after app launch
    from isaaclab.envs.mdp import JointPositionActionCfg  # noqa: PLC0415
    from isaaclab.utils.configclass import configclass  # noqa: PLC0415
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: PLC0415

    import so101_rl.tasks  # noqa: F401,PLC0415 - registers the task ids

    env_cfg = parse_env_cfg(task_id, device=device, num_envs=num_envs)
    for sensor in CAMERA_SENSORS.values():
        camera_cfg = getattr(env_cfg.scene, sensor)
        camera_cfg.width = image_width
        camera_cfg.height = image_height

    @configclass
    class AbsoluteJointActionsCfg:
        joint_position = JointPositionActionCfg(
            asset_name="robot",
            joint_names=list(contract.SO101_JOINT_NAMES),
            preserve_order=True,
            scale=1.0,
            use_default_offset=False,
        )

    env_cfg.actions = AbsoluteJointActionsCfg()
    env_cfg.seed = seed
    env = gym.make(task_id, cfg=env_cfg).unwrapped
    return env


def read_cameras(env, env_ids: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
    """Return {camera: uint8 [B, H, W, 3]} frames for the selected envs."""
    selector = slice(None) if env_ids is None else list(env_ids)
    images = {}
    for camera, sensor in CAMERA_SENSORS.items():
        rgb = env.scene[sensor].data.output["rgb"].torch  # [N, H, W, 4]
        images[camera] = rgb[selector, ..., :3].cpu().numpy()
    return images


def termination_reasons(env) -> Dict[str, np.ndarray]:
    """Read the scripted oracle: success term + failure terms per env."""
    manager = env.termination_manager
    return {
        "dones": manager.dones.cpu().numpy(),
        SUCCESS_TERM: manager.get_term(SUCCESS_TERM).cpu().numpy(),
        TIMEOUT_TERM: manager.get_term(TIMEOUT_TERM).cpu().numpy(),
        **{name: manager.get_term(name).cpu().numpy() for name in FAILURE_TERMS},
    }


def classify(reason_terms: Dict[str, np.ndarray], env_id: int) -> tuple:
    """(success, reason) for one env from the per-term arrays."""
    if reason_terms[SUCCESS_TERM][env_id]:
        return True, "success"
    for name in FAILURE_TERMS:
        if reason_terms[name][env_id]:
            return False, name
    if reason_terms[TIMEOUT_TERM][env_id]:
        return False, "timeout"
    return False, "unknown"


class RolloutSummary:
    def __init__(self) -> None:
        self.episodes = 0
        self.successes = 0
        self.reasons: Dict[str, int] = {}
        self.starved_steps = 0
        self.start_time = time.time()

    def add(self, success: bool, reason: str) -> None:
        self.episodes += 1
        if success:
            self.successes += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        elapsed = time.time() - self.start_time
        return {
            "episodes": self.episodes,
            "successes": self.successes,
            "success_rate": (self.successes / self.episodes
                             if self.episodes else 0.0),
            "reasons": dict(self.reasons),
            "starved_steps": self.starved_steps,
            "wall_time_s": round(elapsed, 1),
        }


def run_rollouts(
    env,
    client: RolloutClient,
    *,
    task: str,
    joint_map: contract.JointMap,
    num_episodes: int,
    max_episode_steps: int,
    actions_per_chunk: int,
    chunk_size_threshold: float,
    aggregate_fn_name: str = "weighted_average",
    round_dir: Optional[Path] = None,
    max_wall_time_min: float = 240.0,
) -> Dict[str, Any]:
    """Generate ``num_episodes`` autonomous episodes across all envs.

    With ``round_dir`` set, every completed episode is written to
    ``round_dir/rollouts_raw`` and logged to ``rollouts.jsonl``; without it,
    the loop only measures success (evaluation mode).
    """
    import torch  # noqa: PLC0415 - Isaac Sim side only

    aggregate_fn = AGGREGATE_FUNCTIONS[aggregate_fn_name]
    num_envs = env.num_envs
    chunk = client.chunk_size or actions_per_chunk
    record = round_dir is not None
    rollouts_raw = Path(round_dir) / "rollouts_raw" if record else None
    manifest_path = (Path(round_dir) / contract.ROLL_FILENAME
                     if record else None)

    buffers = [EnvActionBuffer(aggregate_fn) for _ in range(num_envs)]
    summary = RolloutSummary()
    episode_counter = 0
    recorders: List[Optional[EpisodeRecorder]] = [None] * num_envs

    obs, _ = env.reset()
    timestep = 0
    while summary.episodes < num_episodes:
        if time.time() - summary.start_time > max_wall_time_min * 60.0:
            logger.warning("wall-time budget exhausted; stopping early")
            break

        state_sim = obs["joint_state"].cpu().numpy().astype(np.float64)
        state_dataset = joint_map.to_dataset(state_sim)

        # --- refill chunks for envs whose queues run low -------------------
        need_refill = [
            env_id for env_id in range(num_envs)
            if buffers[env_id].pending_from(timestep)
            <= chunk_size_threshold * chunk
        ]
        if need_refill:
            images = read_cameras(env, need_refill)
            wire_images = {
                contract.camera_feature_key(camera): frames
                for camera, frames in images.items()
            }
            chunks = client.infer(
                wire_images, state_dataset[need_refill], task,
            )
            for row, env_id in enumerate(need_refill):
                buffers[env_id].submit(chunks[row], timestep)

        # --- pop the command for this step (dataset units) -----------------
        commands = np.empty_like(state_dataset)
        starved = []
        for env_id in range(num_envs):
            action = buffers[env_id].pop(timestep)
            if action is None:
                starved.append(env_id)
                hold = buffers[env_id].last_action
                commands[env_id] = hold if hold is not None else state_dataset[env_id]
            else:
                commands[env_id] = action
        if starved:
            summary.starved_steps += len(starved)
            if timestep % 30 == 0:
                logger.debug("starved envs at t=%d: %s", timestep, starved)

        # --- record the frame before stepping ------------------------------
        if record:
            all_images = read_cameras(env)
            for env_id in range(num_envs):
                if recorders[env_id] is None:
                    recorder = EpisodeRecorder(
                        rollouts_raw / "tmp", env_id, episode_counter,
                        max_episode_steps,
                        image_height=contract.IMAGE_HEIGHT,
                        image_width=contract.IMAGE_WIDTH,
                    )
                    recorder.start()
                    recorders[env_id] = recorder
                    episode_counter += 1
                recorders[env_id].add(
                    {camera: frames[env_id]
                     for camera, frames in all_images.items()},
                    state_dataset[env_id],
                    commands[env_id],
                    state_sim[env_id],
                    joint_map.to_sim(commands[env_id]),
                )

        # --- step with absolute sim-radian targets -------------------------
        targets_sim = joint_map.to_sim(commands)
        step_return = env.step(
            torch.tensor(targets_sim, dtype=torch.float32, device=env.device))
        # ManagerBasedRLEnv.step returns (obs, reward, terminated,
        # truncated, extras); older builds expose .obs instead.
        obs = getattr(step_return, "obs", step_return[0])

        # --- finalize finished episodes ------------------------------------
        reason_terms = termination_reasons(env)
        dones = np.nonzero(reason_terms["dones"])[0]
        for env_id in dones:
            success, reason = classify(reason_terms, env_id)
            recorder = recorders[env_id]
            if record and recorder is not None and recorder.steps > 0:
                path = recorder.save(
                    rollouts_raw,
                    task=task,
                    success=success,
                    reason=reason,
                    source="sim",
                    joint_map=joint_map,
                )
                contract.append_jsonl(manifest_path, {
                    "episode_index": recorder.episode_index,
                    "file": str(path.relative_to(Path(round_dir))),
                    "source": "sim",
                    "task": task,
                    "success": bool(success),
                    "reason": reason,
                    "steps": recorder.steps,
                    "fps": contract.CONTROL_FREQUENCY_HZ,
                })
                recorder.discard()
            recorders[env_id] = None
            buffers[env_id].reset()
            summary.add(success, reason)
            logger.info(
                "episode %d done: success=%s reason=%s (%d/%d episodes, "
                "%d successes)", summary.episodes, success, reason,
                summary.episodes, num_episodes, summary.successes,
            )
            if summary.episodes >= num_episodes:
                break
        timestep += 1

    for recorder in recorders:
        if recorder is not None:
            recorder.discard()
    return summary.to_dict()
