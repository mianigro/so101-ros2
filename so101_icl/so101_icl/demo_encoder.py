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
  - ``traj``:    ``[S, max_state_dim + max_action_dim]`` states+actions.
    REV 8: demos are VIDEO-ONLY — the trajectory branch is removed and this
    field is accepted-and-ignored (zeros on the wire for format compat).
  - ``traj_ok``: accepted-and-ignored alongside ``traj``.

Branches (rev 8, video-only demos):
  - vision: frozen SigLIP embeddings (detached) -> cross-attention pool with
    ``T = tokens_vis`` learned queries -> projection -> x gate.
  - keypoints (optional, ``keypoints.enabled``): per-keypoint MLP ->
    attention pool -> ``tokens_kp`` tokens -> x kp_ok x gate.

Gates: the raw scalars are logits; the effective gates are
``gate_budget * softmax(logits)`` — the branches share a fixed loudness
budget, so one branch can only grow by taking share from the other
(``effective_gates``).

Zero-init property: by default the gate logits (and the order embedding)
start at exactly zero and the budget is ``n_branches * gate_floor``; with
``gate_floor=0`` every demo token embedding is the zero vector at init. The
out-projections keep their default init — zeroing them TOO would create a
dead saddle (``gate * proj(x)`` with both factors zero has zero gradient
w.r.t. both). Note that present-but-zero tokens still perturb attention
softmax normalization (they are attended with a zero key), so exact base
parity holds for the no-pack path and the structural zeros are asserted
separately (tests/test_zero_init.py, default config).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .configuration_pi05_icl import ICLConfig


class DemoEncoder(nn.Module):
    """Encode a support-demo pack (video-only) into ``k * (T_vis + T_kp)`` prefix tokens."""

    def __init__(self, config: ICLConfig, vlm_width: int):
        super().__init__()
        de = config.demo_encoder
        self.config = de
        self.vlm_width = vlm_width
        self.k_max = de.k_max
        self.tokens_vis = de.tokens_vis

        # --- vision branch: attention pool over SigLIP frame tokens ---
        self.queries = nn.Parameter(torch.randn(de.tokens_vis, vlm_width) * 0.02)
        self.pool = nn.MultiheadAttention(vlm_width, de.n_heads, batch_first=True)
        self.vis_proj = nn.Linear(vlm_width, vlm_width)
        self.vis_norm = nn.LayerNorm(vlm_width)
        self.gate_vis = nn.Parameter(torch.zeros(()))

        # --- keypoint branch (rev 5, Keypoint Action Tokens style) ---
        # per-keypoint MLP over (coords + SIFT descriptor + valid flag), then
        # attention-pool over F*K keypoints/demo -> tokens_kp gated tokens.
        # Disabled by default (config.demo_encoder.keypoints.enabled=false):
        # token layout and checkpoints are then bit-identical to rev 4.
        kp = de.keypoints
        self.keypoints_enabled = bool(kp.enabled)
        if self.keypoints_enabled:
            self.kp_mlp = nn.Sequential(
                nn.Linear(kp.kp_dim, de.hidden_dim), nn.GELU(),
                nn.Linear(de.hidden_dim, vlm_width),
            )
            self.kp_queries = nn.Parameter(torch.randn(kp.tokens_kp, vlm_width) * 0.02)
            self.kp_pool = nn.MultiheadAttention(vlm_width, de.n_heads, batch_first=True)
            self.kp_norm = nn.LayerNorm(vlm_width)
            self.gate_kp = nn.Parameter(torch.zeros(()))
            self._kp_shape = (de.frames_per_demo, kp.n_kp, kp.kp_dim)

        # --- trajectory branch: REMOVED (rev 8). Demos are video-only at
        # deployment (wrist-cam frames, no joint states); the model must not
        # learn to lean on a channel that will be empty in the field. The
        # traj/traj_ok pack fields survive on the wire and in batches but are
        # ignored here (all zeros). ---

        # --- per-demo order embedding (which demo in the pack) ---
        self.order_emb = nn.Embedding(de.k_max, vlm_width)

        self.reset_icl_parameters()

    @property
    def tokens_per_demo(self) -> int:
        per_demo = self.tokens_vis
        if self.keypoints_enabled:
            per_demo += self.config.keypoints.tokens_kp
        return per_demo

    def reset_icl_parameters(self) -> None:
        """Init the gate logits / zero the order embedding.

        The raw gate scalars are logits initialized at 0, so the softmax
        shares start uniform and each effective gate equals
        ``gate_budget / n_branches`` (= ``gate_floor`` with the default
        budget). The out-projections keep their DEFAULT init on purpose:
        zeroing them as well would make ``out = gate * proj(x)`` a dead
        saddle (both factors zero -> zero gradient to both -> the demo
        branch can never leave zero; observed empirically in the first
        smoke run: gates pinned at 0.0 for 200 steps). With proj != 0 the
        gate logits get a nonzero gradient immediately.
        """
        nn.init.zeros_(self.order_emb.weight)
        with torch.no_grad():
            self.gate_vis.zero_()
            if self.keypoints_enabled:
                self.gate_kp.zero_()

    def effective_gates(
        self, available: tuple[bool, ...] | None = None
    ) -> tuple[Tensor, ...]:
        """Effective (vis[, kp]) gates used to scale branch tokens.

        The raw scalars are logits and the branches share a fixed budget —
        ``gate_budget * softmax(logits)`` — so no branch can grow without
        taking share from the other (single-gate runaway and lockstep drift
        are both structurally impossible).

        ``available`` marks branches whose demo data is actually present
        (order vis[, kp]); unavailable branches get -inf logits, so their
        share — and its gradient — flows to the remaining branches instead
        of being lost (e.g. a demo whose keypoints missed the cache).
        """
        raw = [self.gate_vis]
        if self.keypoints_enabled:
            raw.append(self.gate_kp)
        n = len(raw)
        budget = self.config.gate_budget
        if budget is None:
            # max(n, 2) keeps the init invariant in EVERY branch layout:
            # 2+ branches — uniform softmax shares give each gate_floor;
            # 1 branch — sigmoid(0) = 0.5, so budget 2*floor also gives
            # gate_floor. A plain softmax over a single logit is a constant
            # (dead gate) — the sigmoid fallback keeps it learnable and
            # capped by the same budget.
            budget = max(n, 2) * self.config.gate_floor
        if n == 1:
            return (budget * torch.sigmoid(raw[0]),)
        logits = torch.stack(raw)
        if available is not None and len(available) != len(raw):
            raise ValueError(
                f"available has {len(available)} entries for {len(raw)} branches"
            )
        if available is not None and not all(available):
            mask = torch.tensor(
                available, dtype=torch.bool, device=logits.device
            )
            logits = logits.masked_fill(~mask, float("-inf"))
        shares = torch.softmax(logits, dim=0)
        return tuple(budget * s for s in shares)

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
        kp: Tensor | None = None,
        kp_ok: Tensor | None = None,
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
            kp: ``[B, k_max, F, K, kp_dim]`` cached keypoint features
                (required when the keypoint branch is enabled).
            kp_ok: ``[B, k_max]`` 0/1 keypoint availability.
        Returns:
            ``(embs, pad)`` — ``[B, k_max * tokens_per_demo, D]`` token
            embeddings and ``[B, k_max * tokens_per_demo]`` bool pad mask.
        """
        B, K, F = frames.shape[:3]
        if K != self.k_max:
            raise ValueError(f"expected k_max={self.k_max} demo slots, got K={K}")
        if self.keypoints_enabled and kp is None:
            raise ValueError(
                "demo_encoder.keypoints.enabled=true but no kp tensor given — "
                "the batch/pack must carry icl.demo_kp (see data.ICLDataset)."
            )

        # Vision branch: [B*K, F*256, D] keys/values, T learned queries.
        flat_frames = frames.reshape(B * K * F, *frames.shape[3:])
        frame_embs = self._embed_frames(flat_frames, embed_fn)  # [B*K*F, 256, D]
        n_tok, D = frame_embs.shape[1], frame_embs.shape[2]
        kv = frame_embs.reshape(B * K, F * n_tok, D)
        q = self.queries[None].expand(B * K, -1, -1)
        pooled, _ = self.pool(q, kv, kv, need_weights=False)  # [B*K, T, D]
        # Branch availability from the ok masks: a demo pack whose keypoints
        # missed the cache gets its softmax share moved to the vision branch.
        available = (True,)
        if self.keypoints_enabled:
            kp_avail = True if kp_ok is None else bool(kp_ok.any())
            available = available + (kp_avail,)
        gates = self.effective_gates(available)
        gate_vis = gates[0]
        vis_tokens = self.vis_norm(self.vis_proj(pooled)) * gate_vis * gate_scale
        branch_tokens = [vis_tokens.reshape(B, K, self.tokens_vis, D)]

        # Keypoint branch: [B*K, F*K_kp, D] keys/values, learned queries.
        if self.keypoints_enabled:
            kp_emb = self.kp_mlp(kp)                            # [B, K, F, K_kp, D]
            kp_kv = kp_emb.reshape(B * K, F * kp.shape[3], D)
            kp_q = self.kp_queries[None].expand(B * K, -1, -1)
            kp_pooled, _ = self.kp_pool(kp_q, kp_kv, kp_kv, need_weights=False)
            gate_kp = gates[-1]
            kp_tokens = (self.kp_norm(kp_pooled) * gate_kp * gate_scale).reshape(
                B, K, self.config.keypoints.tokens_kp, D
            )
            if kp_ok is not None:
                kp_tokens = kp_tokens * kp_ok[..., None, None].to(kp_tokens.dtype)
            branch_tokens.append(kp_tokens)

        # Trajectory branch removed (rev 8): demos are video-only; the
        # traj/traj_ok arguments are accepted and ignored.

        tokens = torch.cat(branch_tokens, dim=2)  # [B, K, T_vis + T_kp, D]
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
        kp: Tensor | None = None,
        kp_ok: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Inference-side encoding of a single pack (batch dim 1, no grad)."""
        embs, pad = self.forward(frames, demo_mask, traj, traj_ok, embed_fn,
                                 kp=kp, kp_ok=kp_ok)
        return embs.detach(), pad
