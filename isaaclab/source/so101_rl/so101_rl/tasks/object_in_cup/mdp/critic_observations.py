"""Exact simulator observations for the training-only value function."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


CRITIC_STATE_COMPONENTS = (
    ("joint_position", 6),
    ("joint_velocity", 6),
    ("last_action", 6),
    ("gripper_to_object", 3),
    ("object_to_cup", 3),
    ("object_quaternion", 4),
    ("object_linear_velocity", 3),
    ("object_angular_velocity", 3),
)
CRITIC_STATE_DIM = sum(width for _, width in CRITIC_STATE_COMPONENTS)


def critic_task_state(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    cup_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Return the noiseless, task-complete 34-value critic observation."""
    robot = env.scene[robot_cfg.name]
    object_asset = env.scene[object_cfg.name]
    object_position = object_asset.data.root_pos_w.torch
    cup_position = env.scene[cup_cfg.name].data.root_pos_w.torch
    gripper_position = env.scene[ee_frame_cfg.name].data.target_pos_w.torch[:, 0, :]

    return torch.cat(
        (
            robot.data.joint_pos.torch[:, robot_cfg.joint_ids],
            robot.data.joint_vel.torch[:, robot_cfg.joint_ids],
            env.action_manager.action,
            object_position - gripper_position,
            object_position - cup_position,
            object_asset.data.root_quat_w.torch,
            object_asset.data.root_lin_vel_w.torch,
            object_asset.data.root_ang_vel_w.torch,
        ),
        dim=-1,
    )
