"""State observations for the object-in-cup policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_to_object(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    object_pos = env.scene[object_cfg.name].data.root_pos_w.torch
    ee_pos = env.scene[ee_frame_cfg.name].data.target_pos_w.torch[:, 0, :]
    return object_pos - ee_pos


def object_to_cup(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    cup_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    object_pos = env.scene[object_cfg.name].data.root_pos_w.torch
    cup_pos = env.scene[cup_cfg.name].data.root_pos_w.torch
    return object_pos - cup_pos


def object_orientation(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    return env.scene[object_cfg.name].data.root_quat_w.torch


def object_velocity(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    return env.scene[object_cfg.name].data.root_vel_w.torch


def gripper_position(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
) -> torch.Tensor:
    return env.scene[robot_cfg.name].data.joint_pos.torch[:, robot_cfg.joint_ids]
