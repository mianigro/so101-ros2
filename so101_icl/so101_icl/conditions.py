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

"""Condition driver shared by the M3 (sim) and M4 (real) campaigns.

Three comparison arms (ICL §8 M3/M4 gates):

- ``full_icl``       — top-k demos from the stage-2 registry pushed through
                       the demo side channel (``set_demo_pack``);
- ``prompt_enriched`` — language only, any previous pack cleared (the
                       bridge's ``conditioning: prompt`` baseline arm);
- ``bare_prompt``     — language only AND the enrichment stripped
                       (:func:`bare_prompt`, mirroring the offline
                       harness's bare-prompt condition).

The pack-building path reuses the single shared image/normalization code
(:func:`so101_icl.data.preprocess_demo_frames`, :class:`TrajNormalizer`)
— same as training and the bridge, no third code path.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import torch

from .data import (
    ICLDataset,
    TrajNormalizer,
    load_task_registry,
    open_local_dataset,
    preprocess_demo_frames,
)

logger = logging.getLogger(__name__)

CONDITIONS = ("full_icl", "prompt_enriched", "bare_prompt")

_SENTENCE_END = re.compile(r"^(.*?[.!?])(?:\s|$)", re.DOTALL)


def bare_prompt(prompt: str) -> str:
    """Strip prompt enrichment down to the first sentence.

    The offline harness's bare condition drops everything the enrichment
    added; for campaign prompts (one instruction sentence plus enrichment)
    the first sentence IS the bare instruction.
    """
    text = prompt.strip()
    if not text:
        return text
    match = _SENTENCE_END.match(text)
    if match is None:
        return text  # no terminal punctuation: take it as-is
    return match.group(1).strip()


def condition_prompt(condition: str, prompt: str) -> str:
    """The task string a given campaign condition sends to the policy."""
    if condition == "bare_prompt":
        return bare_prompt(prompt)
    return prompt


class DemoPackBuilder:
    """Resolve a task group's demo pack from the stage-2 registry.

    Mirrors :class:`ICLDataset`'s support-pack construction (uniform-stride
    keyframes with start/goal pinned, trajectory downsampled to
    ``traj_steps`` and normalized with the ACTIVE stage's stats) but
    deterministically: the first ``k`` registered episodes of the group,
    so every episode of a condition sees the same pack.
    """

    def __init__(
        self,
        registry_path: Path | str,
        *,
        k: int = 2,
        frames_per_demo: int = 6,
        k_max: int = 4,
        traj_steps: int = 16,
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        stats_path: Path | str | None = None,
        demo_camera: str | None = None,
        keypoint_cache: Path | str | None = None,
        n_kp: int = 16,
        kp_dim: int = 131,
    ):
        self.registry = load_task_registry(registry_path)
        self.k = int(k)
        self.frames_per_demo = int(frames_per_demo)
        self.k_max = int(k_max)
        self.traj_steps = int(traj_steps)
        self.k = min(self.k, self.k_max)
        self._bundles: dict[int, dict] = {}
        self._kp_cache = None
        self._kp_keys = frozenset()
        if keypoint_cache is not None:
            import numpy as np

            cache = np.load(Path(keypoint_cache).expanduser())
            self._kp_cache = {name: cache[name] for name in cache.files}
            self._kp_keys = frozenset(self._kp_cache.keys())
        self._n_kp = int(n_kp)
        self._kp_dim = int(kp_dim)

        self._stats_override = None
        if stats_path is not None:
            from .demo_transport import load_stage_stats

            self._stats_override = load_stage_stats(stats_path)
        self._demo_camera = demo_camera
        self._max_state_dim = max_state_dim
        self._max_action_dim = max_action_dim

    def groups(self) -> list[str]:
        return sorted(self.registry["groups"].keys())

    def resolve_group(self, group: str | None) -> str:
        """The named group, or the single/first group of the registry."""
        if group is not None:
            if group not in self.registry["groups"]:
                raise ValueError(
                    f"group {group!r} not in registry; available: {self.groups()[:10]}..."
                )
            return group
        return self.groups()[0]

    def _bundle(self, ds_idx: int) -> dict:
        if ds_idx in self._bundles:
            return self._bundles[ds_idx]
        spec = self.registry["datasets"][ds_idx]
        dataset = open_local_dataset(spec["repo_id"], spec["root"])
        info_features = dataset.meta.info["features"]
        d_state = info_features["observation.state"]["shape"][0]
        d_action = info_features["action"]["shape"][0]
        stats = self._stats_override or dataset.meta.stats
        normalizer = TrajNormalizer(
            stats, d_state, d_action,
            max_state_dim=self._max_state_dim, max_action_dim=self._max_action_dim,
        )
        camera = self._demo_camera
        if camera is None:
            keys = [k for k in info_features if k.startswith("observation.images.")]
            camera = keys[0] if keys else "observation.images.base_0_rgb"
        # registry datasets carry a camera_rename map (dataset -> policy keys)
        reverse = {v: k for k, v in dict(spec.get("camera_rename") or {}).items()}
        camera = reverse.get(camera, camera)
        eps = dataset.meta.episodes.to_pandas()
        episodes = {
            int(row["episode_index"]): (
                int(row["dataset_from_index"]), int(row["dataset_to_index"]),
                int(row["length"]),
            )
            for _, row in eps.iterrows()
        }
        traj_table = dataset.hf_dataset.select_columns(["observation.state", "action"])
        bundle = {
            "dataset": dataset, "episodes": episodes, "traj_table": traj_table,
            "normalizer": normalizer, "camera": camera,
        }
        self._bundles[ds_idx] = bundle
        return bundle

    def build(self, group: str | None = None) -> dict:
        """Build the pack for ``group``: numpy arrays ready for the wire.

        Returns ``{"frames": [k, F, 3, 224, 224], "traj": [k, S, d],
        "traj_ok": [k], "episodes": [...], "group": str}``.
        """
        group = self.resolve_group(group)
        members = self.registry["groups"][group]
        if not members:
            raise ValueError(f"group {group!r} has no episodes")
        selected = members[: self.k]

        frames = np.zeros(
            (self.k_max, self.frames_per_demo, 3, 224, 224), dtype=np.float32
        )
        traj = np.zeros(
            (self.k_max, self.traj_steps, self._max_state_dim + self._max_action_dim),
            dtype=np.float32,
        )
        traj_ok = np.zeros(self.k_max, dtype=np.float32)
        kp = np.zeros(
            (self.k_max, self.frames_per_demo, self._n_kp, self._kp_dim),
            dtype=np.float32,
        )
        kp_ok = np.zeros(self.k_max, dtype=np.float32)
        used = []
        for slot, (ds_idx, ep, _length) in enumerate(selected[: self.k_max]):
            bundle = self._bundle(ds_idx)
            start, _end, length = bundle["episodes"][ep]
            kf = ICLDataset._sample_keyframe_indices(length, self.frames_per_demo)
            raw = torch.stack(
                [bundle["dataset"][start + int(j)][bundle["camera"]] for j in kf]
            )
            frames[slot] = preprocess_demo_frames(raw).numpy()

            rows = bundle["traj_table"][start:end]
            states = torch.as_tensor(np.asarray(rows["observation.state"]), dtype=torch.float32)
            actions = torch.as_tensor(np.asarray(rows["action"]), dtype=torch.float32)
            s_idx = ICLDataset._sample_keyframe_indices(states.shape[0], self.traj_steps)
            traj[slot] = bundle["normalizer"](states[s_idx], actions[s_idx]).numpy()
            traj_ok[slot] = 1.0
            if self._kp_cache is not None:
                key = f"{int(ds_idx)}/{int(ep)}"
                if key in self._kp_cache:
                    kp[slot] = self._kp_cache[key].astype(np.float32, copy=False)
                    kp_ok[slot] = 1.0
            used.append({"ds": int(ds_idx), "episode": int(ep)})

        logger.info("demo pack for %r: %d demo(s) %s", group, len(used), used)
        return {"frames": frames, "traj": traj, "traj_ok": traj_ok,
                "kp": kp, "kp_ok": kp_ok, "kp_enabled": self._kp_cache is not None,
                "episodes": used, "group": group}


def apply_condition(
    condition: str,
    transport,
    builder: DemoPackBuilder | None = None,
    *,
    group: str | None = None,
) -> dict:
    """Push/clear the demo pack for one campaign condition.

    Returns the transport reply; ``full_icl`` requires a ``builder``.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")
    if condition == "full_icl":
        if builder is None:
            raise ValueError("full_icl needs a DemoPackBuilder (registry + stats)")
        pack = builder.build(group)
        return transport.set_demo_pack(
            pack["frames"][: builder.k], traj=pack["traj"][: builder.k],
            traj_ok=pack["traj_ok"][: builder.k], k_max=builder.k_max,
            **({"kp": pack["kp"][: builder.k], "kp_ok": pack["kp_ok"][: builder.k]}
               if pack.get("kp_enabled") else {}),
        )
    transport.clear()
    return {"status": "ok", "cleared": True, "condition": condition}
