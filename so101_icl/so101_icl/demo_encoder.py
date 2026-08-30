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

"""DemoEncoder: one demonstration pack -> prefix tokens in the PaliGemma
embedding space (ICL_IMPLEMENTATION.md §4.2).

Inputs (per demo ``i`` in ``k <= k_max``):
  - ``frames``:  ``[F, 3, 224, 224]`` keyframes, already through the SAME
    transform as the policy cameras (``PI05Policy._preprocess_images`` —
    resize-with-pad + [-1, 1] SigLIP scaling); visual normalization is
    IDENTITY so dataset stats never touch images.
  - ``traj``:    ``[S, max_state_dim + max_action_dim]`` states+actions,
    downsampled to ``S = traj_steps`` and normalized with the ACTIVE stage's
    dataset stats (the encoder itself is stats-agnostic).
  - ``traj_ok``: 0/1 availability flag (video-only exports -> 0).

Branches:
  - vision: frozen SigLIP embeddings (detached) -> cross-attention pool with
    ``T = tokens_vis`` learned queries -> projection -> x gate.
  - trajectory: per-step state MLP + action MLP -> ``2 * S`` tokens
    (one state and one action token per step) -> x traj_ok x gate.

Zero-init property: by default the gates (and the order embedding) start at
exactly zero, so at initialization every demo token embedding is the zero
vector. The out-projections keep their default init — zeroing them TOO would
create a dead saddle (``gate * proj(x)`` with both factors zero has zero
gradient w.r.t. both). A positive ``gate_floor`` instead starts the gates AT
the floor and clamps them below it (null-out mitigation, see
``reset_icl_parameters``). Note that present-but-zero tokens still perturb
attention softmax normalization (they are attended with a zero key), so
exact base parity holds for the no-pack path and the structural zeros are
asserted separately (tests/test_zero_init.py, default config).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .configuration_pi05_icl import ICLConfig


class DemoEncoder(nn.Module):
    """Encode a support-demo pack into ``k * (T_vis + T_traj)`` prefix tokens."""

    def __init__(self, config: ICLConfig, vlm_width: int):
        super().__init__()
        de = config.demo_encoder
        self.config = de
        self.vlm_width = vlm_width
        self.k_max = de.k_max
        self.tokens_vis = de.tokens_vis
        self.tokens_traj = de.tokens_traj  # == 2 * traj_steps (validated)

        # --- vision branch: attention pool over SigLIP frame tokens ---
        self.queries = nn.Parameter(torch.randn(de.tokens_vis, vlm_width) * 0.02)
        self.pool = nn.MultiheadAttention(vlm_width, de.n_heads, batch_first=True)
        self.vis_proj = nn.Linear(vlm_width, vlm_width)
        self.vis_norm = nn.LayerNorm(vlm_width)
        self.gate_vis = nn.Parameter(torch.zeros(()))

        # --- trajectory branch: per-step state/action tokens ---
        d_state, d_action = config.max_state_dim, config.max_action_dim
        hidden = de.hidden_dim
        self.state_mlp = nn.Sequential(
            nn.Linear(d_state, hidden), nn.GELU(), nn.Linear(hidden, vlm_width)
        )
        self.action_mlp = nn.Sequential(
            nn.Linear(d_action, hidden), nn.GELU(), nn.Linear(hidden, vlm_width)
        )
        self.traj_norm = nn.LayerNorm(vlm_width)
        self.gate_traj = nn.Parameter(torch.zeros(()))

        # --- per-demo order embedding (which demo in the pack) ---
        self.order_emb = nn.Embedding(de.k_max, vlm_width)

        self._demo_dim = d_state + d_action
        self.reset_icl_parameters()

    @property
    def tokens_per_demo(self) -> int:
        return self.tokens_vis + self.tokens_traj

    def reset_icl_parameters(self) -> None:
        """Init the gates (at ``gate_floor``) / zero the order embedding.

        With the default ``gate_floor=0`` every demo token embedding is
        exactly the zero vector at initialization (the M0 property). The
        out-projections keep their DEFAULT init on purpose: zeroing them as
        well would make ``out = gate * proj(x)`` a dead saddle (both factors
        zero -> zero gradient to both -> the demo branch can never leave
        zero; observed empirically in the first smoke run: gates pinned at
        0.0 for 200 steps). With proj != 0 the gate gets a nonzero gradient
        immediately.

        With a positive ``gate_floor`` (null-out mitigation, Rev 3 §2.1) the
        gates START at the floor and are clamped below it in ``forward`` —
        the demo branch cannot be silenced through a single scalar, and
        no-pack parity still holds (no pack -> no demo tokens at all).
        """
        nn.init.zeros_(self.order_emb.weight)
        with torch.no_grad():
            self.gate_vis.fill_(self.config.gate_floor)
            self.gate_traj.fill_(self.config.gate_floor)

    def _embed_frames(self, frames: Tensor, embed_fn) -> Tensor:
        """SigLIP-embed demo frames, detached, chunked to bound peak memory.

        Args:
            frames: ``[N, 3, H, W]`` in [-1, 1].
            embed_fn: frozen ``PaliGemmaWithExpertModel.embed_image``.
        Returns:
            ``[N, n_tok, D]`` float32 features, detached from the graph.
        """
        outs = []
        chunk = 64
        with torch.no_grad():
            for start in range(0, frames.shape[0], chunk):
                emb = embed_fn(frames[start : start + chunk].to(torch.float32))
                outs.append(emb.detach().to(torch.float32))
        return torch.cat(outs, dim=0)

    def forward(
        self,
        frames: Tensor,
        demo_mask: Tensor,
        traj: Tensor,
        traj_ok: Tensor,
        embed_fn,
        gate_scale: float | Tensor = 1.0,
    ) -> tuple[Tensor, Tensor]:
        """Encode a batched demo pack.

        Args:
            frames: ``[B, k_max, F, 3, H, W]`` preprocessed keyframes
                ([-1, 1], policy image resolution); absent demo slots may be
                arbitrary (they are masked out via ``demo_mask``).
            demo_mask: ``[B, k_max]`` bool, True for present demos.
            traj: ``[B, k_max, S, d_state + d_action]`` normalized
                trajectories, zero-padded to ``max_state_dim``/``max_action_dim``.
            traj_ok: ``[B, k_max]`` 0/1 trajectory availability.
            embed_fn: frozen SigLIP embed function (see :meth:`_embed_frames`).
            gate_scale: extra multiplicative factor on the gates; the training
                loop sets 0.0 to compute ``loss_demo_zeroed``.
        Returns:
            ``(embs, pad)`` — ``[B, k_max * tokens_per_demo, D]`` token
            embeddings and ``[B, k_max * tokens_per_demo]`` bool pad mask.
        """
        B, K, F = frames.shape[:3]
        if K != self.k_max:
            raise ValueError(f"expected k_max={self.k_max} demo slots, got K={K}")

        # Vision branch: [B*K, F*256, D] keys/values, T learned queries.
        flat_frames = frames.reshape(B * K * F, *frames.shape[3:])
        frame_embs = self._embed_frames(flat_frames, embed_fn)  # [B*K*F, 256, D]
        n_tok, D = frame_embs.shape[1], frame_embs.shape[2]
        kv = frame_embs.reshape(B * K, F * n_tok, D)
        q = self.queries[None].expand(B * K, -1, -1)
        pooled, _ = self.pool(q, kv, kv, need_weights=False)  # [B*K, T, D]
        gate_vis = self.gate_vis.clamp(min=self.config.gate_floor)
        vis_tokens = self.vis_norm(self.vis_proj(pooled)) * gate_vis * gate_scale

        # Trajectory branch: one state + one action token per step -> [B, K, 2S, D].
        state_part = traj[..., : self._demo_dim // 2]
        action_part = traj[..., self._demo_dim // 2 :]
        state_tokens = self.state_mlp(state_part)  # [B, K, S, D]
        action_tokens = self.action_mlp(action_part)
        traj_tokens = torch.cat([state_tokens, action_tokens], dim=2)  # [B, K, 2S, D]
        gate_traj = self.gate_traj.clamp(min=self.config.gate_floor)
        traj_tokens = self.traj_norm(traj_tokens) * gate_traj * gate_scale
        traj_tokens = traj_tokens * traj_ok[..., None, None].to(traj_tokens.dtype)

        tokens = torch.cat(
            [vis_tokens.reshape(B, K, self.tokens_vis, D), traj_tokens], dim=2
        )  # [B, K, T_vis + 2S, D]
        tokens = tokens + self.order_emb.weight[None, :, None, :]  # order per demo slot

        per_demo = tokens.shape[2]
        embs = tokens.reshape(B, K * per_demo, D)
        pad = demo_mask[:, :, None].expand(B, K, per_demo).reshape(B, K * per_demo)
        return embs, pad

    @torch.no_grad()
    def encode_pack(
        self,
        frames: Tensor,
        demo_mask: Tensor,
        traj: Tensor,
        traj_ok: Tensor,
        embed_fn,
    ) -> tuple[Tensor, Tensor]:
        """Inference-side encoding of a single pack (batch dim 1, no grad)."""
        embs, pad = self.forward(frames, demo_mask, traj, traj_ok, embed_fn)
        return embs.detach(), pad
