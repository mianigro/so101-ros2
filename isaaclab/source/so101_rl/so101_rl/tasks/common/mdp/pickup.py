"""Pure tensor primitives for bounded, contact-validated pickup rewards."""

from __future__ import annotations

import torch


def rotate_vectors_xyzw(quaternion: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by unit quaternions stored in Isaac Lab's ``xyzw`` order."""
    vector_part, vectors = torch.broadcast_tensors(quaternion[..., :3], vectors)
    scalar_part = torch.broadcast_tensors(quaternion[..., 3:4], vectors[..., :1])[0]
    twice_cross = 2.0 * torch.linalg.cross(vector_part, vectors, dim=-1)
    return vectors + scalar_part * twice_cross + torch.linalg.cross(
        vector_part, twice_cross, dim=-1
    )


def projected_half_extent(
    object_quaternion_w: torch.Tensor,
    normal_w: torch.Tensor,
    half_extents: tuple[float, float, float] | torch.Tensor,
) -> torch.Tensor:
    """Return an oriented box's support distance along a world-space normal."""
    inverse_quaternion = object_quaternion_w.clone()
    inverse_quaternion[..., :3].neg_()
    normal_object = rotate_vectors_xyzw(inverse_quaternion, normal_w)
    extents = torch.as_tensor(
        half_extents, dtype=normal_object.dtype, device=normal_object.device
    )
    return (normal_object.abs() * extents).sum(dim=-1)


def grasp_targets_from_fixed_pad(
    pad_center_w: torch.Tensor,
    pad_quaternion_w: torch.Tensor,
    object_quaternion_w: torch.Tensor,
    half_extents: tuple[float, float, float] | torch.Tensor,
    *,
    pad_thickness: float,
    clearance: float,
) -> torch.Tensor:
    """Compute attainable object centres from the fixed pad's inward (+X) normal.

    ``object_quaternion_w`` may be ``(N, 4)`` or ``(N, M, 4)``.  The result
    has the corresponding ``(N, 3)`` or ``(N, M, 3)`` shape.  Starting from
    the pad body centre also aligns both transverse coordinates to that centre.
    """
    local_inward = torch.zeros_like(pad_center_w)
    local_inward[..., 0] = 1.0
    inward_w = rotate_vectors_xyzw(pad_quaternion_w, local_inward)
    while inward_w.ndim < object_quaternion_w.ndim:
        inward_w = inward_w.unsqueeze(-2)
        pad_center_w = pad_center_w.unsqueeze(-2)
    support = projected_half_extent(object_quaternion_w, inward_w, half_extents)
    distance = support + clearance + 0.5 * pad_thickness
    return pad_center_w + inward_w * distance.unsqueeze(-1)


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


def open_gripper_gate(
    gripper_position: torch.Tensor,
    *,
    open_position_min: float,
    fully_open_position: float,
) -> torch.Tensor:
    """Return zero for a closed jaw and one for a fully open jaw."""
    return torch.clamp(
        (gripper_position - open_position_min)
        / (fully_open_position - open_position_min),
        0.0,
        1.0,
    )


def pickup_alignment_score(
    object_position_w: torch.Tensor,
    target_position_w: torch.Tensor,
    gripper_position: torch.Tensor,
    *,
    position_scale: float,
    open_position_min: float,
    fully_open_position: float,
) -> torch.Tensor:
    """Return geometric alignment gated continuously by actual jaw position."""
    alignment = target_alignment_score(
        object_position_w, target_position_w, position_scale=position_scale
    )
    open_gate = open_gripper_gate(
        gripper_position,
        open_position_min=open_position_min,
        fully_open_position=fully_open_position,
    )
    while open_gate.ndim < alignment.ndim:
        open_gate = open_gate.unsqueeze(-1)
    return alignment * open_gate


def bilateral_same_step_contact(
    fixed_force_history_w: torch.Tensor,
    moving_force_history_w: torch.Tensor,
    *,
    force_threshold: float,
) -> torch.Tensor:
    """Return filters contacted by both pads in at least one identical substep.

    Inputs follow the PhysX sensor layout ``(env, history, sensor, filter, xyz)``.
    Combining forces before the history/filter comparison would incorrectly
    accept asynchronous contacts or one pad touching a different object.
    """
    if fixed_force_history_w.shape != moving_force_history_w.shape:
        raise ValueError("fixed and moving contact histories must have identical shapes")
    if fixed_force_history_w.ndim != 5 or fixed_force_history_w.shape[-1] != 3:
        raise ValueError(
            "contact histories must have shape (env, history, sensor, filter, 3)"
        )
    fixed = torch.linalg.vector_norm(fixed_force_history_w, dim=-1) > force_threshold
    moving = torch.linalg.vector_norm(moving_force_history_w, dim=-1) > force_threshold
    same_substep = fixed.any(dim=2) & moving.any(dim=2)
    return same_substep.any(dim=1)


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


def bounded_closure_increment(
    previous_position: torch.Tensor,
    current_position: torch.Tensor,
    alignment: torch.Tensor,
    credited: torch.Tensor,
    eligible: torch.Tensor,
    initialized: torch.Tensor,
    *,
    closure_range: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Credit actual closing motion to the best-aligned eligible object once."""
    if alignment.shape != credited.shape or alignment.shape != eligible.shape:
        raise ValueError("per-object closure tensors must have identical shapes")
    candidate = alignment.masked_fill(~eligible, -1.0)
    selected_index = candidate.argmax(dim=-1)
    selected = torch.nn.functional.one_hot(
        selected_index, num_classes=alignment.shape[-1]
    ).to(dtype=torch.bool)
    selected &= eligible
    selected_alignment = (alignment * selected).sum(dim=-1)
    closing = torch.clamp(previous_position - current_position, min=0.0)
    raw = closing / closure_range * selected_alignment.pow(4)
    raw = torch.where(initialized, raw, 0.0)
    per_object = selected * raw.unsqueeze(-1)
    increment = torch.minimum(per_object, torch.clamp(1.0 - credited, min=0.0))
    return increment, credited + increment, torch.ones_like(initialized)


def first_event_increment(
    event: torch.Tensor, credited: torch.Tensor, eligible: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a one-shot per-object event impulse."""
    increment = event & eligible & ~credited
    return increment.to(dtype=torch.float32), credited | (event & eligible)
