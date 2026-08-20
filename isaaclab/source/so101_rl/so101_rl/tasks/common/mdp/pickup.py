"""Pure tensor primitives for bounded, grasp-agnostic pickup rewards."""

from __future__ import annotations

import torch


def target_alignment_score(
    object_position_w: torch.Tensor,
    target_position_w: torch.Tensor,
    *,
    position_scale: float,
) -> torch.Tensor:
    """Return the bounded geometric alignment score used by pickup stages."""
    position_error = torch.linalg.vector_norm(
        object_position_w - target_position_w, dim=-1
    )
    return 1.0 - torch.tanh(position_error / position_scale)


def episode_best_increment(
    value: torch.Tensor,
    best: torch.Tensor,
    initialized: torch.Tensor,
    eligible: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance an episode-best score and return only its bounded positive delta."""
    if not (value.shape == best.shape == initialized.shape == eligible.shape):
        raise ValueError("progress tensors must have identical shapes")
    increase = torch.clamp(value - best, min=0.0)
    increment = torch.where(initialized & eligible, increase, 0.0)
    next_best = torch.where(eligible, torch.maximum(best, value), best)
    next_initialized = initialized | eligible
    return increment, next_best, next_initialized
