"""Pure tensor geometry shared by rewards, terminations, and focused tests."""

from __future__ import annotations

import torch


def placement_mask(
    object_to_cup: torch.Tensor,
    object_linear_velocity: torch.Tensor,
    object_angular_velocity: torch.Tensor,
    gripper_position: torch.Tensor,
    *,
    xy_tolerance: float,
    center_z_min: float,
    center_z_max: float,
    linear_velocity_max: float,
    angular_velocity_max: float,
    released_position_min: float,
) -> torch.Tensor:
    """Return environments whose released object is settled inside the cup.

    ``object_to_cup`` is the object-center position minus the cup-bottom-center
    position. Boundary comparisons are inclusive so the contract is unambiguous.
    """
    radial_distance = torch.linalg.vector_norm(object_to_cup[:, :2], dim=-1)
    linear_speed = torch.linalg.vector_norm(object_linear_velocity, dim=-1)
    angular_speed = torch.linalg.vector_norm(object_angular_velocity, dim=-1)
    return (
        (radial_distance <= xy_tolerance)
        & (object_to_cup[:, 2] >= center_z_min)
        & (object_to_cup[:, 2] <= center_z_max)
        & (linear_speed <= linear_velocity_max)
        & (angular_speed <= angular_velocity_max)
        & (gripper_position >= released_position_min)
    )


def update_settle_counter(counter: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Increment consecutive valid placements and reset interrupted placements."""
    return torch.where(valid, counter + 1, 0)
