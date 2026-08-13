"""Bounded pickup stages and placement rewards for the object-in-cup task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from so101_rl.tasks.common.mdp.pickup import (
    bilateral_same_step_contact,
    bounded_closure_increment,
    episode_best_increment,
    first_event_increment,
    grasp_targets_from_fixed_pad,
    target_alignment_score,
)

from .geometry import placement_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


def _gripper_position(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    return (
        env.scene[robot_cfg.name]
        .data.joint_pos.torch[:, robot_cfg.joint_ids]
        .squeeze(-1)
    )


def _pickup_alignment(
    env: ManagerBasedRLEnv,
    half_extents: tuple[float, float, float],
    pad_thickness: float,
    clearance: float,
    position_scale: float,
    *,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    fixed_sensor_cfg: SceneEntityCfg = SceneEntityCfg("fixed_jaw_contact"),
) -> torch.Tensor:
    object_asset = env.scene[object_cfg.name]
    fixed_pad = env.scene[fixed_sensor_cfg.name]
    target = grasp_targets_from_fixed_pad(
        fixed_pad.data.pos_w.torch[:, 0, :],
        fixed_pad.data.quat_w.torch[:, 0, :],
        object_asset.data.root_quat_w.torch,
        half_extents,
        pad_thickness=pad_thickness,
        clearance=clearance,
    )
    return target_alignment_score(
        object_asset.data.root_pos_w.torch,
        target,
        position_scale=position_scale,
    ).unsqueeze(-1)


def _bilateral_contact(
    env: ManagerBasedRLEnv,
    force_threshold: float,
    fixed_sensor_cfg: SceneEntityCfg,
    moving_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    fixed = env.scene[fixed_sensor_cfg.name].data.force_matrix_w_history
    moving = env.scene[moving_sensor_cfg.name].data.force_matrix_w_history
    if fixed is None or moving is None:
        raise RuntimeError("pickup rewards require filtered contact-force history")
    return bilateral_same_step_contact(
        fixed.torch, moving.torch, force_threshold=force_threshold
    )


class approach_progress(ManagerTermBase):
    """Pay only new episode-best alignment, at most one per episode."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._best = torch.zeros((env.num_envs, 1), device=env.device)
        self._initialized = torch.ones(
            (env.num_envs, 1), dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self._best[env_ids] = 0.0
        self._initialized[env_ids] = True

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        half_extents: tuple[float, float, float],
        pad_thickness: float,
        clearance: float = 0.0005,
        position_scale: float = 0.04,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        fixed_sensor_cfg: SceneEntityCfg = SceneEntityCfg("fixed_jaw_contact"),
    ) -> torch.Tensor:
        # Pure alignment only: no open-jaw gate, so approach never rewards
        # being open and no longer competes with closure/lift at the cube.
        score = _pickup_alignment(
            env,
            half_extents,
            pad_thickness,
            clearance,
            position_scale,
            object_cfg=object_cfg,
            fixed_sensor_cfg=fixed_sensor_cfg,
        )
        eligible = torch.ones_like(score, dtype=torch.bool)
        increment, self._best, self._initialized = episode_best_increment(
            score, self._best, self._initialized, eligible
        )
        return increment.squeeze(-1) / env.step_dt


class closure_progress(ManagerTermBase):
    """Pay aligned physical jaw closure, capped to one per episode."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous = torch.zeros(env.num_envs, device=env.device)
        self._credited = torch.zeros((env.num_envs, 1), device=env.device)
        self._initialized = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self._previous[env_ids] = 0.0
        self._credited[env_ids] = 0.0
        self._initialized[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        half_extents: tuple[float, float, float],
        pad_thickness: float,
        clearance: float = 0.0005,
        position_scale: float = 0.04,
        closure_range: float = 1.2,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
        fixed_sensor_cfg: SceneEntityCfg = SceneEntityCfg("fixed_jaw_contact"),
    ) -> torch.Tensor:
        alignment = _pickup_alignment(
            env,
            half_extents,
            pad_thickness,
            clearance,
            position_scale,
            object_cfg=object_cfg,
            fixed_sensor_cfg=fixed_sensor_cfg,
        )
        current = _gripper_position(env, robot_cfg)
        increment, self._credited, self._initialized = bounded_closure_increment(
            self._previous,
            current,
            alignment,
            self._credited,
            torch.ones_like(alignment, dtype=torch.bool),
            self._initialized,
            closure_range=closure_range,
        )
        self._previous.copy_(current)
        return increment.sum(dim=-1) / env.step_dt


class grasp_acquired(ManagerTermBase):
    """Pay once when both pads contact the cube in the same physics substep."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._credited = torch.zeros(
            (env.num_envs, 1), dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self._credited[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        force_threshold: float = 0.1,
        fixed_sensor_cfg: SceneEntityCfg = SceneEntityCfg("fixed_jaw_contact"),
        moving_sensor_cfg: SceneEntityCfg = SceneEntityCfg("moving_jaw_contact"),
    ) -> torch.Tensor:
        contact = _bilateral_contact(
            env, force_threshold, fixed_sensor_cfg, moving_sensor_cfg
        )
        increment, self._credited = first_event_increment(
            contact,
            self._credited,
            torch.ones_like(contact, dtype=torch.bool),
        )
        return increment.sum(dim=-1) / env.step_dt


class lift_progress(ManagerTermBase):
    """Pay a constant reward while the gripped cube is off the table."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        lift_clearance: float,
        object_rest_height: float,
        force_threshold: float = 0.1,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        fixed_sensor_cfg: SceneEntityCfg = SceneEntityCfg("fixed_jaw_contact"),
        moving_sensor_cfg: SceneEntityCfg = SceneEntityCfg("moving_jaw_contact"),
    ) -> torch.Tensor:
        height = env.scene[object_cfg.name].data.root_pos_w.torch[:, 2:3]
        lifted = height >= object_rest_height + lift_clearance
        contact = _bilateral_contact(
            env, force_threshold, fixed_sensor_cfg, moving_sensor_cfg
        )
        return (lifted & contact).to(dtype=height.dtype).squeeze(-1)


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
