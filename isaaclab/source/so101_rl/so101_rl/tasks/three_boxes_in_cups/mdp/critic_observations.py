"""Exact simulator observation for the three-box training-only value function."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


THREE_BOX_CRITIC_STATE_COMPONENTS = (
    ("joint_position", 6),
    ("joint_velocity", 6),
    ("last_action", 6),
    ("gripper_to_boxes", 9),
    ("box_to_cup_pairs", 27),
    ("box_quaternions", 12),
    ("box_linear_velocities", 9),
    ("box_angular_velocities", 9),
)
THREE_BOX_CRITIC_STATE_DIM = sum(
    width for _, width in THREE_BOX_CRITIC_STATE_COMPONENTS
)


def critic_task_state(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    box_names: tuple[str, str, str] = ("box_1", "box_2", "box_3"),
    cup_names: tuple[str, str, str] = ("cup_1", "cup_2", "cup_3"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Return the noiseless 84-value state in documented box-major order."""
    robot = env.scene[robot_cfg.name]
    boxes = [env.scene[name] for name in box_names]
    box_positions = torch.stack(
        [box.data.root_pos_w.torch for box in boxes], dim=1
    )
    cup_positions = torch.stack(
        [env.scene[name].data.root_pos_w.torch for name in cup_names], dim=1
    )
    gripper_position = env.scene[ee_frame_cfg.name].data.target_pos_w.torch[:, 0, :]
    pairwise = box_positions[:, :, None, :] - cup_positions[:, None, :, :]
    batch = box_positions.shape[0]

    return torch.cat(
        (
            robot.data.joint_pos.torch[:, robot_cfg.joint_ids],
            robot.data.joint_vel.torch[:, robot_cfg.joint_ids],
            env.action_manager.action,
            (box_positions - gripper_position[:, None, :]).reshape(batch, -1),
            pairwise.reshape(batch, -1),
            torch.stack([box.data.root_quat_w.torch for box in boxes], dim=1).reshape(
                batch, -1
            ),
            torch.stack(
                [box.data.root_lin_vel_w.torch for box in boxes], dim=1
            ).reshape(batch, -1),
            torch.stack(
                [box.data.root_ang_vel_w.torch for box in boxes], dim=1
            ).reshape(batch, -1),
        ),
        dim=-1,
    )
