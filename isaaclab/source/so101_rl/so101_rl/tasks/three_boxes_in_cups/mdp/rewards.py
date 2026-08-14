"""Bounded pickup stages and permutation-invariant three-box placement rewards."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from so101_rl.tasks.common.mdp.pickup import (
    bilateral_same_step_contact,
    bounded_closure_increment,
    episode_best_increment,
    grasp_targets_from_fixed_pad,
    object_between_jaws,
    retryable_event_increment,
    target_alignment_score,
)

from .geometry import (
    best_assignment_score,
    matched_entities,
    placement_matrix,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


BOX_NAMES = ("box_1", "box_2", "box_3")
CUP_NAMES = ("cup_1", "cup_2", "cup_3")


def _task_tensors(
    env: ManagerBasedRLEnv,
    box_names: tuple[str, str, str],
    cup_names: tuple[str, str, str],
    robot_cfg: SceneEntityCfg,
    ee_frame_name: str,
) -> tuple[torch.Tensor, ...]:
    boxes = [env.scene[name] for name in box_names]
    box_positions = torch.stack(
        [box.data.root_pos_w.torch for box in boxes], dim=1
    )
    cup_positions = torch.stack(
        [env.scene[name].data.root_pos_w.torch for name in cup_names], dim=1
    )
    linear_velocities = torch.stack(
        [box.data.root_lin_vel_w.torch for box in boxes], dim=1
    )
    angular_velocities = torch.stack(
        [box.data.root_ang_vel_w.torch for box in boxes], dim=1
    )
    end_effector = env.scene[ee_frame_name].data.target_pos_w.torch[:, 0, :]
    gripper = (
        env.scene[robot_cfg.name]
        .data.joint_pos.torch[:, robot_cfg.joint_ids]
        .squeeze(-1)
    )
    return (
        box_positions,
        cup_positions,
        linear_velocities,
        angular_velocities,
        end_effector,
        gripper,
    )


def _placed_pairs(tensors: tuple[torch.Tensor, ...], placement: dict) -> torch.Tensor:
    return placement_matrix(*tensors, **placement, require_stable=False)


def _pickup_tensors(
    env: ManagerBasedRLEnv,
    placement: dict,
    box_names: tuple[str, str, str],
    cup_names: tuple[str, str, str],
    robot_cfg: SceneEntityCfg,
    ee_frame_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = _task_tensors(env, box_names, cup_names, robot_cfg, ee_frame_name)
    box_quaternions = torch.stack(
        [env.scene[name].data.root_quat_w.torch for name in box_names], dim=1
    )
    matched_boxes, _ = matched_entities(_placed_pairs(tensors, placement))
    return tensors[0], box_quaternions, tensors[-1], ~matched_boxes


def _pickup_alignment(
    env: ManagerBasedRLEnv,
    half_extents: tuple[float, float, float],
    pad_thickness: float,
    clearance: float,
    position_scale: float,
    placement: dict,
    *,
    box_names: tuple[str, str, str],
    cup_names: tuple[str, str, str],
    robot_cfg: SceneEntityCfg,
    ee_frame_name: str,
    fixed_sensor_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positions, quaternions, gripper, eligible = _pickup_tensors(
        env, placement, box_names, cup_names, robot_cfg, ee_frame_name
    )
    fixed_pad = env.scene[fixed_sensor_cfg.name]
    target = grasp_targets_from_fixed_pad(
        fixed_pad.data.pos_w.torch[:, 0, :],
        fixed_pad.data.quat_w.torch[:, 0, :],
        quaternions,
        half_extents,
        pad_thickness=pad_thickness,
        clearance=clearance,
    )
    score = target_alignment_score(positions, target, position_scale=position_scale)
    return score, gripper, eligible


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
    """Pay episode-best alignment independently for each unplaced box."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._best = torch.zeros((env.num_envs, len(BOX_NAMES)), device=env.device)
        self._initialized = torch.ones_like(self._best, dtype=torch.bool)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self._best[env_ids] = 0.0
        self._initialized[env_ids] = True

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        half_extents: tuple[float, float, float],
        pad_thickness: float,
        placement: dict,
        clearance: float = 0.0005,
        position_scale: float = 0.04,
        box_names: tuple[str, str, str] = BOX_NAMES,
        cup_names: tuple[str, str, str] = CUP_NAMES,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
        ee_frame_name: str = "ee_frame",
        fixed_sensor_cfg: SceneEntityCfg = SceneEntityCfg("fixed_jaw_contact"),
    ) -> torch.Tensor:
        # Pure alignment only: no open-jaw gate, so approach never rewards
        # being open and no longer competes with closure/lift at the box.
        score, _, eligible = _pickup_alignment(
            env,
            half_extents,
            pad_thickness,
            clearance,
            position_scale,
            placement,
            box_names=box_names,
            cup_names=cup_names,
            robot_cfg=robot_cfg,
            ee_frame_name=ee_frame_name,
            fixed_sensor_cfg=fixed_sensor_cfg,
        )
        increment, self._best, self._initialized = episode_best_increment(
            score, self._best, self._initialized, eligible
        )
        return increment.sum(dim=-1) / env.step_dt


class closure_progress(ManagerTermBase):
    """Pay closure after an unplaced box edge enters between the pads."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous = torch.zeros(env.num_envs, device=env.device)
        self._credited = torch.zeros((env.num_envs, len(BOX_NAMES)), device=env.device)
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
        placement: dict,
        clearance: float = 0.0005,
        position_scale: float = 0.04,
        closure_range: float = 1.2,
        fixed_pad_length: float = 0.025,
        minimum_insertion: float = 0.001,
        box_names: tuple[str, str, str] = BOX_NAMES,
        cup_names: tuple[str, str, str] = CUP_NAMES,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
        ee_frame_name: str = "ee_frame",
        fixed_sensor_cfg: SceneEntityCfg = SceneEntityCfg("fixed_jaw_contact"),
        moving_sensor_cfg: SceneEntityCfg = SceneEntityCfg("moving_jaw_contact"),
    ) -> torch.Tensor:
        alignment, current, eligible = _pickup_alignment(
            env,
            half_extents,
            pad_thickness,
            clearance,
            position_scale,
            placement,
            box_names=box_names,
            cup_names=cup_names,
            robot_cfg=robot_cfg,
            ee_frame_name=ee_frame_name,
            fixed_sensor_cfg=fixed_sensor_cfg,
        )
        positions = torch.stack(
            [env.scene[name].data.root_pos_w.torch for name in box_names], dim=1
        )
        quaternions = torch.stack(
            [env.scene[name].data.root_quat_w.torch for name in box_names], dim=1
        )
        fixed_pad = env.scene[fixed_sensor_cfg.name].data
        moving_pad = env.scene[moving_sensor_cfg.name].data
        between = object_between_jaws(
            positions,
            quaternions,
            fixed_pad.pos_w.torch[:, 0, :],
            fixed_pad.quat_w.torch[:, 0, :],
            moving_pad.pos_w.torch[:, 0, :],
            half_extents,
            fixed_pad_length=fixed_pad_length,
            minimum_insertion=minimum_insertion,
        )
        increment, self._credited, self._initialized = bounded_closure_increment(
            self._previous,
            current,
            alignment,
            self._credited,
            eligible & between,
            self._initialized,
            closure_range=closure_range,
        )
        self._previous.copy_(current)
        return increment.sum(dim=-1) / env.step_dt


class grasp_acquired(ManagerTermBase):
    """Pay once per grasp attempt for synchronous bilateral same-box contact.

    The per-box impulse re-arms whenever contact is lost, so re-grasping a box
    after a drop earns the reward again.  Within one continuous contact
    session it still fires at most once per box (rising edge).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._in_contact = torch.zeros(
            (env.num_envs, len(BOX_NAMES)), dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self._in_contact[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        placement: dict,
        force_threshold: float = 0.1,
        box_names: tuple[str, str, str] = BOX_NAMES,
        cup_names: tuple[str, str, str] = CUP_NAMES,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
        ee_frame_name: str = "ee_frame",
        fixed_sensor_cfg: SceneEntityCfg = SceneEntityCfg("fixed_jaw_contact"),
        moving_sensor_cfg: SceneEntityCfg = SceneEntityCfg("moving_jaw_contact"),
    ) -> torch.Tensor:
        contact = _bilateral_contact(
            env, force_threshold, fixed_sensor_cfg, moving_sensor_cfg
        )
        eligible = _pickup_tensors(
            env, placement, box_names, cup_names, robot_cfg, ee_frame_name
        )[-1]
        increment, self._in_contact = retryable_event_increment(
            contact, self._in_contact, eligible
        )
        return increment.sum(dim=-1) / env.step_dt


def grasp_held(
    env: ManagerBasedRLEnv,
    half_extents: tuple[float, float, float],
    placement: dict,
    *,
    closed_threshold: float = 0.15,
    fixed_pad_length: float = 0.025,
    minimum_insertion: float = 0.001,
    box_names: tuple[str, str, str] = BOX_NAMES,
    cup_names: tuple[str, str, str] = CUP_NAMES,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
    ee_frame_name: str = "ee_frame",
    fixed_sensor_cfg: SceneEntityCfg = SceneEntityCfg("fixed_jaw_contact"),
    moving_sensor_cfg: SceneEntityCfg = SceneEntityCfg("moving_jaw_contact"),
) -> torch.Tensor:
    """Dense per-box reward for holding an unplaced box clamped in the jaws.

    Geometry-based (box between the pads with the gripper closed) so the
    policy gets a continuous grip gradient regardless of contact-force
    margins, and gated to unplaced boxes so a placed box stops earning it.
    Bridges ``closure_progress`` and ``lift_progress`` for each box.
    """
    _, _, _, eligible = _pickup_tensors(
        env, placement, box_names, cup_names, robot_cfg, ee_frame_name
    )
    positions = torch.stack(
        [env.scene[name].data.root_pos_w.torch for name in box_names], dim=1
    )
    quaternions = torch.stack(
        [env.scene[name].data.root_quat_w.torch for name in box_names], dim=1
    )
    fixed_pad = env.scene[fixed_sensor_cfg.name].data
    moving_pad = env.scene[moving_sensor_cfg.name].data
    between = object_between_jaws(
        positions,
        quaternions,
        fixed_pad.pos_w.torch[:, 0, :],
        fixed_pad.quat_w.torch[:, 0, :],
        moving_pad.pos_w.torch[:, 0, :],
        half_extents,
        fixed_pad_length=fixed_pad_length,
        minimum_insertion=minimum_insertion,
    )
    gripper = (
        env.scene[robot_cfg.name]
        .data.joint_pos.torch[:, robot_cfg.joint_ids]
        .squeeze(-1)
    )
    closed = (gripper <= closed_threshold).unsqueeze(-1)
    held = between & eligible & closed
    return held.to(dtype=positions.dtype).sum(dim=-1)


class lift_progress(ManagerTermBase):
    """Pay a constant reward for each gripped, unplaced box off the table."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        lift_clearance: float,
        object_rest_height: float,
        placement: dict,
        force_threshold: float = 0.1,
        box_names: tuple[str, str, str] = BOX_NAMES,
        cup_names: tuple[str, str, str] = CUP_NAMES,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
        ee_frame_name: str = "ee_frame",
        fixed_sensor_cfg: SceneEntityCfg = SceneEntityCfg("fixed_jaw_contact"),
        moving_sensor_cfg: SceneEntityCfg = SceneEntityCfg("moving_jaw_contact"),
    ) -> torch.Tensor:
        positions, _, _, eligible = _pickup_tensors(
            env, placement, box_names, cup_names, robot_cfg, ee_frame_name
        )
        lifted = positions[..., 2] >= object_rest_height + lift_clearance
        contact = _bilateral_contact(
            env, force_threshold, fixed_sensor_cfg, moving_sensor_cfg
        )
        return (lifted & contact & eligible).to(dtype=positions.dtype).sum(dim=-1)


def transport_to_empty_cup(
    env: ManagerBasedRLEnv,
    std: float,
    minimum_height: float,
    placement: dict,
    box_names: tuple[str, str, str] = BOX_NAMES,
    cup_names: tuple[str, str, str] = CUP_NAMES,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
    ee_frame_name: str = "ee_frame",
) -> torch.Tensor:
    tensors = _task_tensors(env, box_names, cup_names, robot_cfg, ee_frame_name)
    box_positions, cup_positions = tensors[:2]
    matched_boxes, matched_cups = matched_entities(_placed_pairs(tensors, placement))
    relative = box_positions[:, :, None, :] - cup_positions[:, None, :, :]
    radial = torch.linalg.vector_norm(relative[..., :2], dim=-1)
    active = ~matched_boxes[:, :, None] & ~matched_cups[:, None, :]
    lifted = relative[..., 2] >= minimum_height
    scores = (1.0 - torch.tanh(radial / std)) * (active & lifted).float()
    return scores.flatten(start_dim=1).max(dim=-1).values


def insertion_progress(
    env: ManagerBasedRLEnv,
    xy_tolerance: float,
    center_z_max: float,
    approach_height: float,
    box_names: tuple[str, str, str] = BOX_NAMES,
    cup_names: tuple[str, str, str] = CUP_NAMES,
) -> torch.Tensor:
    box_positions = torch.stack(
        [env.scene[name].data.root_pos_w.torch for name in box_names], dim=1
    )
    cup_positions = torch.stack(
        [env.scene[name].data.root_pos_w.torch for name in cup_names], dim=1
    )
    relative = box_positions[:, :, None, :] - cup_positions[:, None, :, :]
    aligned = (
        torch.linalg.vector_norm(relative[..., :2], dim=-1) <= xy_tolerance
    )
    vertical = torch.clamp(
        (approach_height - relative[..., 2]) / (approach_height - center_z_max),
        0.0,
        1.0,
    )
    return best_assignment_score(vertical * aligned.float())


def released_placement_progress(
    env: ManagerBasedRLEnv,
    placement: dict,
    box_names: tuple[str, str, str] = BOX_NAMES,
    cup_names: tuple[str, str, str] = CUP_NAMES,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
    ee_frame_name: str = "ee_frame",
) -> torch.Tensor:
    tensors = _task_tensors(env, box_names, cup_names, robot_cfg, ee_frame_name)
    return best_assignment_score(_placed_pairs(tensors, placement))


def stable_placement_progress(
    env: ManagerBasedRLEnv,
    placement: dict,
    box_names: tuple[str, str, str] = BOX_NAMES,
    cup_names: tuple[str, str, str] = CUP_NAMES,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
    ee_frame_name: str = "ee_frame",
) -> torch.Tensor:
    tensors = _task_tensors(env, box_names, cup_names, robot_cfg, ee_frame_name)
    valid = placement_matrix(*tensors, **placement, require_stable=True)
    return best_assignment_score(valid)
