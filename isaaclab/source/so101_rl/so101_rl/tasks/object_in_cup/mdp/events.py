"""Randomized reset layout for the cube and cup."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _uniform(count: int, low: float, high: float, device: str) -> torch.Tensor:
    return low + (high - low) * torch.rand(count, device=device)


def _yaw_quaternion(yaw: torch.Tensor) -> torch.Tensor:
    quaternion = torch.zeros((yaw.shape[0], 4), device=yaw.device)
    quaternion[:, 2] = torch.sin(0.5 * yaw)
    quaternion[:, 3] = torch.cos(0.5 * yaw)
    return quaternion


def reset_task_layout(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    curriculum_steps: int,
    object_xy_range_nominal: tuple[float, float],
    object_xy_range_full: tuple[float, float],
    cup_xy_range_nominal: tuple[float, float],
    cup_xy_range_full: tuple[float, float],
    object_yaw_range_full: tuple[float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    cup_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> None:
    """Reset separated object/cup regions and grow variation over training steps."""
    object_asset = env.scene[object_cfg.name]
    cup_asset = env.scene[cup_cfg.name]
    count = len(env_ids)
    progress = min(
        float(env.common_step_counter) / float(max(curriculum_steps, 1)), 1.0
    )

    object_xy = tuple(
        nominal + progress * (full - nominal)
        for nominal, full in zip(
            object_xy_range_nominal, object_xy_range_full, strict=True
        )
    )
    cup_xy = tuple(
        nominal + progress * (full - nominal)
        for nominal, full in zip(cup_xy_range_nominal, cup_xy_range_full, strict=True)
    )

    object_pose = object_asset.data.default_root_pose.torch[env_ids].clone()
    cup_pose = cup_asset.data.default_root_pose.torch[env_ids].clone()
    object_pose[:, :3] += env.scene.env_origins[env_ids]
    cup_pose[:, :3] += env.scene.env_origins[env_ids]
    object_pose[:, 0] += _uniform(count, -object_xy[0], object_xy[0], env.device)
    object_pose[:, 1] += _uniform(count, -object_xy[1], object_xy[1], env.device)
    cup_pose[:, 0] += _uniform(count, -cup_xy[0], cup_xy[0], env.device)
    cup_pose[:, 1] += _uniform(count, -cup_xy[1], cup_xy[1], env.device)

    yaw_low = object_yaw_range_full[0] * progress
    yaw_high = object_yaw_range_full[1] * progress
    object_pose[:, 3:7] = _yaw_quaternion(
        _uniform(count, yaw_low, yaw_high, env.device)
    )
    cup_pose[:, 3:7] = _yaw_quaternion(_uniform(count, -torch.pi, torch.pi, env.device))
    zero_velocity = torch.zeros((count, 6), device=env.device)

    object_asset.write_root_pose_to_sim_index(root_pose=object_pose, env_ids=env_ids)
    object_asset.write_root_velocity_to_sim_index(
        root_velocity=zero_velocity, env_ids=env_ids
    )
    cup_asset.write_root_pose_to_sim_index(root_pose=cup_pose, env_ids=env_ids)
