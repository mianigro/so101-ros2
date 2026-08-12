"""Terminal conditions for the object-in-cup task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .geometry import placement_mask, update_settle_counter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import TerminationTermCfg


def object_dropped(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    return env.scene[object_cfg.name].data.root_pos_w.torch[:, 2] < minimum_height


def invalid_state(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    object_state = env.scene[object_cfg.name].data.root_state_w.torch
    robot = env.scene[robot_cfg.name]
    robot_state = torch.cat(
        (robot.data.joint_pos.torch, robot.data.joint_vel.torch), dim=-1
    )
    return ~(
        torch.isfinite(object_state).all(dim=-1)
        & torch.isfinite(robot_state).all(dim=-1)
    )


class stable_placement(ManagerTermBase):
    """Terminate only after placement remains valid for a fixed settling window."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._consecutive_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._consecutive_steps.zero_()
        else:
            self._consecutive_steps[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        required_steps: int,
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
        valid = placement_mask(
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
        )
        self._consecutive_steps = update_settle_counter(self._consecutive_steps, valid)
        return self._consecutive_steps >= required_steps
