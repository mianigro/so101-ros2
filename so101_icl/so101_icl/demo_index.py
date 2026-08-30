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

"""Training-free demo retrieval + output-side action fusion.

Strategy (Retrieval-VLA / RICL analog, ICL_IMPLEMENTATION.md §12):

- demo keyframes are embedded with a CHEAP deterministic embedding
  (8x8 adaptive-average-pool + L2 norm) — no model download, no GPU
  dependency, good enough to rank same-workspace observations. The
  embedding function is a parameter everywhere, so SigLIP/DINO-v2 can be
  plugged in at the serving boundary without touching the fusion math.
- at inference the live camera frame is embedded the same way, the nearest
  demo keyframe is retrieved, and the base policy's action chunk is blended
  with the demo action at the retrieved keyframe's progress:
  ``a = e^{-lam*d} * a_demo + (1 - e^{-lam*d}) * a_base``
  (distance-dependent Retrieval-VLA interpolation). Far from any demo the
  output degenerates to exactly the base policy — base parity for free.

Everything here is pure tensor/numpy logic and unit-tested (no lerobot,
no model). The serving-side hook lives in
:meth:`so101_icl.modeling_pi05_icl.PI05ICLPolicy.predict_action_chunk`.
"""

from __future__ import annotations

import numpy as np
import torch


def embed_frames(frames: torch.Tensor, grid: int = 8) -> torch.Tensor:
    """Deterministic frame embedding: avg-pool to ``grid x grid`` + L2 norm.

    Args:
        frames: ``[N, 3, H, W]`` float in [-1, 1] (policy-transform space)
            or ``[3, H, W]`` / ``[H, W, 3]`` numpy arrays (converted here).
        grid: spatial pooling resolution (8 -> 192-dim embeddings).
    Returns:
        ``[N, 3 * grid * grid]`` float32, L2-normalized on dim -1.
    """
    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(np.ascontiguousarray(frames))
    if frames.ndim == 3:
        frames = frames[None]
    if frames.ndim != 4:
        raise ValueError(f"expected [N, 3, H, W], got shape {tuple(frames.shape)}")
    pooled = torch.nn.functional.adaptive_avg_pool2d(frames.float(), (grid, grid))
    emb = pooled.reshape(frames.shape[0], -1)
    return torch.nn.functional.normalize(emb, dim=-1)


def nearest_keyframe(query: torch.Tensor, bank: torch.Tensor) -> tuple[int, float]:
    """L2-nearest bank entry to the query (both L2-normalized embeddings).

    Returns ``(index, l2_distance)``.
    """
    if bank.ndim != 2 or query.ndim != 1 or bank.shape[1] != query.shape[0]:
        raise ValueError(f"shape mismatch: query {tuple(query.shape)}, bank {tuple(bank.shape)}")
    dists = torch.norm(bank - query[None], dim=-1)
    idx = int(torch.argmin(dists))
    return idx, float(dists[idx])


def demo_action_at(actions: np.ndarray, progress: float) -> np.ndarray:
    """The demo action at ``progress`` in [0, 1] (linear interp between steps)."""
    if actions.ndim != 2 or actions.shape[0] == 0:
        raise ValueError(f"expected [T, d] actions, got {tuple(actions.shape)}")
    t = float(np.clip(progress, 0.0, 1.0)) * (actions.shape[0] - 1)
    lo, hi = int(np.floor(t)), int(np.ceil(t))
    if lo == hi:
        return actions[lo].copy()
    frac = t - lo
    return (1.0 - frac) * actions[lo] + frac * actions[hi]


def fuse_with_demo(
    base_action: np.ndarray,
    demo_action: np.ndarray,
    distance: float,
    lam: float = 5.0,
) -> np.ndarray:
    """Retrieval-VLA blend: near matches follow the demo, far ones the base."""
    if base_action.shape != demo_action.shape:
        raise ValueError(
            f"action dim mismatch: base {base_action.shape}, demo {demo_action.shape}"
        )
    w_demo = float(np.exp(-lam * distance))
    return w_demo * demo_action + (1.0 - w_demo) * base_action


class DemoRetrievalIndex:
    """Keyframe embeddings + (per-keyframe progress, demo action table).

    Built once per subtask from the SAME demo pack the bridge already ships
    (frames preprocessed to [-1, 1]; actions in the policy's NORMALIZED
    action space — the same space ``predict_action_chunk`` emits before the
    postprocessor, so blending there is consistent).
    """

    def __init__(
        self,
        frames: torch.Tensor | np.ndarray,   # [k, F, 3, H, W], [-1, 1]
        actions: torch.Tensor | np.ndarray,  # [k, T, d] normalized demo actions
        demo_mask: torch.Tensor | np.ndarray | None = None,  # [k] bool
        *,
        embed_fn=embed_frames,
    ):
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames)
        if isinstance(actions, np.ndarray):
            actions = torch.as_tensor(actions, dtype=torch.float32)
        k, f = frames.shape[:2]
        if actions.shape[0] != k:
            raise ValueError(f"frames carry {k} demos, actions carry {actions.shape[0]}")
        if demo_mask is None:
            demo_mask = torch.ones(k, dtype=torch.bool)
        elif isinstance(demo_mask, np.ndarray):
            demo_mask = torch.from_numpy(demo_mask)
        self.actions = actions.numpy().astype(np.float32)
        self.demo_mask = demo_mask.bool().numpy()
        self._embed_fn = embed_fn

        embs = embed_fn(frames.reshape(k * f, *frames.shape[2:]))   # [k*F, D]
        # per-keyframe progress = position of the keyframe within its demo
        progress = torch.tile(torch.arange(f, dtype=torch.float32) / max(1, f - 1), (k,))
        keep = torch.from_numpy(np.repeat(self.demo_mask, f))
        self.bank = embs[keep].numpy().astype(np.float32)           # [n_kept, D]
        self.bank_progress = progress[keep].numpy().astype(np.float32)
        # bank slot -> demo row (to index the action table)
        self.bank_demo = np.repeat(np.arange(k), f)[keep]
        if len(self.bank) == 0:
            raise ValueError("demo retrieval index is empty (all demos masked out)")

    def query(self, obs_frame: torch.Tensor | np.ndarray) -> tuple[np.ndarray, float]:
        """Nearest demo keyframe -> ``(demo_action_at_its_progress, distance)``."""
        emb = self._embed_fn(obs_frame)
        if emb.ndim == 2:
            emb = emb[0]
        idx, dist = nearest_keyframe(emb, torch.from_numpy(self.bank))
        demo_row = self.bank_demo[idx]
        return demo_action_at(self.actions[demo_row], self.bank_progress[idx]), dist

    @property
    def size(self) -> int:
        return len(self.bank)
