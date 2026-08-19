"""Permutation-invariant tensor geometry for three boxes and three cups."""

from __future__ import annotations

import itertools

import torch

THREE_WAY_PERMUTATIONS = tuple(itertools.permutations(range(3)))


def best_assignment_values(scores: torch.Tensor) -> torch.Tensor:
    """Return the three pair scores from the highest-scoring one-to-one assignment."""
    if scores.ndim != 3 or scores.shape[1:] != (3, 3):
        raise ValueError(f"assignment scores must have shape [N, 3, 3], got {scores.shape}")
    candidates = torch.stack(
        [
            torch.stack(
                [
                    scores[:, box_index, cup_index]
                    for box_index, cup_index in enumerate(permutation)
                ],
                dim=-1,
            )
            for permutation in THREE_WAY_PERMUTATIONS
        ],
        dim=1,
    )
    best = candidates.to(dtype=torch.float32).sum(dim=-1).argmax(dim=-1)
    batch = torch.arange(scores.shape[0], device=scores.device)
    return candidates[batch, best]


def best_assignment_score(scores: torch.Tensor) -> torch.Tensor:
    """Return the normalized score of the best one-to-one assignment."""
    return best_assignment_values(scores).to(dtype=torch.float32).mean(dim=-1)


def complete_assignment(valid_pairs: torch.Tensor) -> torch.Tensor:
    """Return environments with three valid pairs using three distinct cups."""
    return best_assignment_values(valid_pairs).all(dim=-1)


def placement_matrix(
    box_positions: torch.Tensor,
    cup_positions: torch.Tensor,
    box_linear_velocities: torch.Tensor,
    box_angular_velocities: torch.Tensor,
    end_effector_position: torch.Tensor,
    gripper_position: torch.Tensor,
    *,
    xy_tolerance: float,
    center_z_min: float,
    center_z_max: float,
    linear_velocity_max: float,
    angular_velocity_max: float,
    released_position_min: float,
    release_distance_min: float,
    require_stable: bool,
) -> torch.Tensor:
    """Return valid box/cup pairs, preserving a one-to-one assignment boundary."""
    relative = box_positions[:, :, None, :] - cup_positions[:, None, :, :]
    radial = torch.linalg.vector_norm(relative[..., :2], dim=-1)
    inside = (
        (radial <= xy_tolerance)
        & (relative[..., 2] >= center_z_min)
        & (relative[..., 2] <= center_z_max)
    )
    gripper_distance = torch.linalg.vector_norm(
        box_positions - end_effector_position[:, None, :], dim=-1
    )
    released = (gripper_position >= released_position_min)[:, None] | (
        gripper_distance >= release_distance_min
    )
    valid = inside & released[:, :, None]
    if require_stable:
        linear_speed = torch.linalg.vector_norm(box_linear_velocities, dim=-1)
        angular_speed = torch.linalg.vector_norm(box_angular_velocities, dim=-1)
        stable = (linear_speed <= linear_velocity_max) & (
            angular_speed <= angular_velocity_max
        )
        valid &= stable[:, :, None]
    return valid


def matched_entities(valid_pairs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return box and cup masks selected by the best valid one-to-one assignment."""
    assigned = best_assignment_values(valid_pairs)
    permutations = torch.tensor(
        THREE_WAY_PERMUTATIONS, device=valid_pairs.device, dtype=torch.long
    )
    candidates = torch.stack(
        [
            torch.stack(
                [
                    valid_pairs[:, box_index, cup_index]
                    for box_index, cup_index in enumerate(permutation)
                ],
                dim=-1,
            )
            for permutation in THREE_WAY_PERMUTATIONS
        ],
        dim=1,
    )
    best = candidates.to(dtype=torch.float32).sum(dim=-1).argmax(dim=-1)
    selected_cups = permutations[best]
    cup_mask = torch.zeros_like(assigned, dtype=torch.bool)
    cup_mask.scatter_(1, selected_cups, assigned.to(dtype=torch.bool))
    return assigned.to(dtype=torch.bool), cup_mask


def update_settle_counter(counter: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Increment consecutive all-three placements and reset interrupted episodes."""
    return torch.where(valid, counter + 1, 0)
