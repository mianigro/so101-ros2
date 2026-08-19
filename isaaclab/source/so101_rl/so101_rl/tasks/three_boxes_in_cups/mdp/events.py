"""Collision-safe curriculum resets for three boxes and three cups."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def sample_separated_positions(
    nominal_positions: torch.Tensor,
    *,
    count: int,
    progress: float,
    nominal_jitter: float,
    zone_low: tuple[float, float],
    zone_high: tuple[float, float],
    minimum_separation: float,
    max_attempts: int,
    device: str,
) -> torch.Tensor:
    """Sample three XY positions per environment without pairwise overlap."""
    if nominal_positions.shape != (3, 2):
        raise ValueError("nominal_positions must have shape [3, 2]")
    if not 0.0 <= progress <= 1.0:
        raise ValueError("curriculum progress must be within [0, 1]")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")

    nominal_positions = nominal_positions.to(device=device, dtype=torch.float32)
    full_low = torch.tensor(zone_low, device=device)
    full_high = torch.tensor(zone_high, device=device)
    low = (1.0 - progress) * (nominal_positions - nominal_jitter) + progress * full_low
    high = (1.0 - progress) * (nominal_positions + nominal_jitter) + progress * full_high
    result = torch.empty((count, 3, 2), device=device)
    unresolved = torch.ones(count, dtype=torch.bool, device=device)
    diagonal = torch.eye(3, dtype=torch.bool, device=device).unsqueeze(0)
    for _ in range(max_attempts):
        candidate = low.unsqueeze(0) + (high - low).unsqueeze(0) * torch.rand(
            (count, 3, 2), device=device
        )
        distances = torch.cdist(candidate, candidate).masked_fill(diagonal, 1.0)
        valid = (distances >= minimum_separation).all(dim=(1, 2))
        accepted = unresolved & valid
        result[accepted] = candidate[accepted]
        unresolved &= ~valid
        if not unresolved.any():
            break
    if unresolved.any():
        raise RuntimeError(
            "could not sample a collision-free three-asset layout after "
            f"{max_attempts} attempts; widen the zone or reduce spacing"
        )
    return result


def _yaw_quaternion(yaw: torch.Tensor) -> torch.Tensor:
    quaternion = torch.zeros((yaw.shape[0], 4), device=yaw.device)
    quaternion[:, 2] = torch.sin(0.5 * yaw)
    quaternion[:, 3] = torch.cos(0.5 * yaw)
    return quaternion


def reset_three_box_layout(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    curriculum_steps: int,
    nominal_jitter: float,
    box_zone_low: tuple[float, float],
    box_zone_high: tuple[float, float],
    cup_zone_low: tuple[float, float],
    cup_zone_high: tuple[float, float],
    box_minimum_separation: float,
    cup_minimum_separation: float,
    cross_minimum_separation: float,
    max_attempts: int,
    fixed_layout: bool = False,
    box_names: tuple[str, str, str] = ("box_1", "box_2", "box_3"),
    cup_names: tuple[str, str, str] = ("cup_1", "cup_2", "cup_3"),
) -> None:
    """Reset separated pickup/placement zones and grow them over training."""
    count = len(env_ids)
    progress = (
        0.0
        if fixed_layout
        else min(
            float(env.common_step_counter) / float(max(curriculum_steps, 1)), 1.0
        )
    )
    jitter = 0.0 if fixed_layout else nominal_jitter
    boxes = [env.scene[name] for name in box_names]
    cups = [env.scene[name] for name in cup_names]
    nominal_boxes = torch.stack(
        [asset.data.default_root_pose.torch[0, :2] for asset in boxes]
    )
    nominal_cups = torch.stack(
        [asset.data.default_root_pose.torch[0, :2] for asset in cups]
    )
    box_xy = sample_separated_positions(
        nominal_boxes,
        count=count,
        progress=progress,
        nominal_jitter=jitter,
        zone_low=box_zone_low,
        zone_high=box_zone_high,
        minimum_separation=box_minimum_separation,
        max_attempts=max_attempts,
        device=env.device,
    )
    cup_xy = sample_separated_positions(
        nominal_cups,
        count=count,
        progress=progress,
        nominal_jitter=jitter,
        zone_low=cup_zone_low,
        zone_high=cup_zone_high,
        minimum_separation=cup_minimum_separation,
        max_attempts=max_attempts,
        device=env.device,
    )
    cross_distances = torch.linalg.vector_norm(
        box_xy[:, :, None, :] - cup_xy[:, None, :, :], dim=-1
    )
    if (cross_distances < cross_minimum_separation).any():
        raise RuntimeError("configured pickup and placement zones violate cross spacing")

    origins = env.scene.env_origins[env_ids]
    zero_velocity = torch.zeros((count, 6), device=env.device)
    for index, asset in enumerate(boxes):
        pose = asset.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += origins
        pose[:, :2] = origins[:, :2] + box_xy[:, index]
        yaw = torch.empty(count, device=env.device).uniform_(
            -torch.pi * progress, torch.pi * progress
        )
        pose[:, 3:7] = _yaw_quaternion(yaw)
        asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim_index(
            root_velocity=zero_velocity, env_ids=env_ids
        )

    for index, asset in enumerate(cups):
        pose = asset.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += origins
        pose[:, :2] = origins[:, :2] + cup_xy[:, index]
        yaw = torch.empty(count, device=env.device).uniform_(
            -torch.pi * progress, torch.pi * progress
        )
        pose[:, 3:7] = _yaw_quaternion(yaw)
        asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
