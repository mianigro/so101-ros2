"""Dense phase rewards for grasping and placing the object in the cup."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .geometry import placement_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reach_object(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    distance = torch.linalg.vector_norm(
        env.scene[object_cfg.name].data.root_pos_w.torch
        - env.scene[ee_frame_cfg.name].data.target_pos_w.torch[:, 0, :],
        dim=-1,
    )
    return 1.0 - torch.tanh(distance / std)


def grasp_object(
    env: ManagerBasedRLEnv,
    distance_threshold: float,
    closed_position_max: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
) -> torch.Tensor:
    object_pos = env.scene[object_cfg.name].data.root_pos_w.torch
    ee_pos = env.scene[ee_frame_cfg.name].data.target_pos_w.torch[:, 0, :]
    gripper_pos = (
        env.scene[robot_cfg.name]
        .data.joint_pos.torch[:, robot_cfg.joint_ids]
        .squeeze(-1)
    )
    close_enough = (
        torch.linalg.vector_norm(object_pos - ee_pos, dim=-1) <= distance_threshold
    )
    closed = gripper_pos <= closed_position_max
    return (close_enough & closed).float()


def lift_object(
    env: ManagerBasedRLEnv,
    lift_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    cup_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    object_height = env.scene[object_cfg.name].data.root_pos_w.torch[:, 2]
    cup_base_height = env.scene[cup_cfg.name].data.root_pos_w.torch[:, 2]
    return torch.clamp(
        (object_height - cup_base_height) / lift_height, min=0.0, max=1.0
    )


def transport_object(
    env: ManagerBasedRLEnv,
    std: float,
    minimum_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    cup_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    relative = (
        env.scene[object_cfg.name].data.root_pos_w.torch
        - env.scene[cup_cfg.name].data.root_pos_w.torch
    )
    radial_distance = torch.linalg.vector_norm(relative[:, :2], dim=-1)
    lifted = relative[:, 2] >= minimum_height
    return (1.0 - torch.tanh(radial_distance / std)) * lifted.float()


def insert_object(
    env: ManagerBasedRLEnv,
    xy_tolerance: float,
    center_z_max: float,
    approach_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    cup_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    relative = (
        env.scene[object_cfg.name].data.root_pos_w.torch
        - env.scene[cup_cfg.name].data.root_pos_w.torch
    )
    aligned = torch.linalg.vector_norm(relative[:, :2], dim=-1) <= xy_tolerance
    vertical_progress = torch.clamp(
        (approach_height - relative[:, 2]) / (approach_height - center_z_max), 0.0, 1.0
    )
    return vertical_progress * aligned.float()


def release_object(
    env: ManagerBasedRLEnv,
    xy_tolerance: float,
    center_z_min: float,
    center_z_max: float,
    released_position_min: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    cup_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
) -> torch.Tensor:
    relative = (
        env.scene[object_cfg.name].data.root_pos_w.torch
        - env.scene[cup_cfg.name].data.root_pos_w.torch
    )
    gripper_pos = (
        env.scene[robot_cfg.name]
        .data.joint_pos.torch[:, robot_cfg.joint_ids]
        .squeeze(-1)
    )
    inside = (
        (torch.linalg.vector_norm(relative[:, :2], dim=-1) <= xy_tolerance)
        & (relative[:, 2] >= center_z_min)
        & (relative[:, 2] <= center_z_max)
    )
    return (inside & (gripper_pos >= released_position_min)).float()


def stable_placement_reward(
    env: ManagerBasedRLEnv,
    xy_tolerance: float,
    center_z_min: float,
    center_z_max: float,
    linear_velocity_max: float,
    angular_velocity_max: float,
    released_position_min: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    cup_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
) -> torch.Tensor:
    object_asset = env.scene[object_cfg.name]
    relative = (
        object_asset.data.root_pos_w.torch
        - env.scene[cup_cfg.name].data.root_pos_w.torch
    )
    gripper_pos = (
        env.scene[robot_cfg.name]
        .data.joint_pos.torch[:, robot_cfg.joint_ids]
        .squeeze(-1)
    )
    return placement_mask(
        relative,
        object_asset.data.root_lin_vel_w.torch,
        object_asset.data.root_ang_vel_w.torch,
        gripper_pos,
        xy_tolerance=xy_tolerance,
        center_z_min=center_z_min,
        center_z_max=center_z_max,
        linear_velocity_max=linear_velocity_max,
        angular_velocity_max=angular_velocity_max,
        released_position_min=released_position_min,
    ).float()
