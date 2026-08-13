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
    first_event_increment,
    grasp_targets_from_fixed_pad,
    pickup_alignment_score,
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
    gate_open: bool,
    open_position_min: float,
    fully_open_position: float,
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
    if gate_open:
        score = pickup_alignment_score(
            positions,
            target,
            gripper,
            position_scale=position_scale,
            open_position_min=open_position_min,
            fully_open_position=fully_open_position,
        )
    else:
        score = target_alignment_score(
            positions, target, position_scale=position_scale
        )
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
    """Pay episode-best open-jaw alignment independently for each unplaced box."""

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
        open_position_min: float = 0.9,
        fully_open_position: float = 1.2,
        box_names: tuple[str, str, str] = BOX_NAMES,
        cup_names: tuple[str, str, str] = CUP_NAMES,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
        ee_frame_name: str = "ee_frame",
        fixed_sensor_cfg: SceneEntityCfg = SceneEntityCfg("fixed_jaw_contact"),
    ) -> torch.Tensor:
        score, _, eligible = _pickup_alignment(
            env,
            half_extents,
            pad_thickness,
            clearance,
            position_scale,
            placement,
            gate_open=True,
            open_position_min=open_position_min,
            fully_open_position=fully_open_position,
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
    """Pay aligned physical closure with a separate one-point budget per box."""

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
        box_names: tuple[str, str, str] = BOX_NAMES,
        cup_names: tuple[str, str, str] = CUP_NAMES,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
        ee_frame_name: str = "ee_frame",
        fixed_sensor_cfg: SceneEntityCfg = SceneEntityCfg("fixed_jaw_contact"),
    ) -> torch.Tensor:
        alignment, current, eligible = _pickup_alignment(
            env,
            half_extents,
            pad_thickness,
            clearance,
            position_scale,
            placement,
            gate_open=False,
            open_position_min=0.0,
            fully_open_position=1.0,
            box_names=box_names,
            cup_names=cup_names,
            robot_cfg=robot_cfg,
            ee_frame_name=ee_frame_name,
            fixed_sensor_cfg=fixed_sensor_cfg,
        )
        increment, self._credited, self._initialized = bounded_closure_increment(
            self._previous,
            current,
            alignment,
            self._credited,
            eligible,
            self._initialized,
            closure_range=closure_range,
        )
        self._previous.copy_(current)
        return increment.sum(dim=-1) / env.step_dt


class grasp_acquired(ManagerTermBase):
    """Pay once per unplaced box for synchronous bilateral same-box contact."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._credited = torch.zeros(
            (env.num_envs, len(BOX_NAMES)), dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self._credited[env_ids] = False

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
        increment, self._credited = first_event_increment(
            contact, self._credited, eligible
        )
        return increment.sum(dim=-1) / env.step_dt


class lift_progress(ManagerTermBase):
    """Pay per-box best height only under synchronous bilateral contact."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._best = torch.zeros((env.num_envs, len(BOX_NAMES)), device=env.device)
        self._initialized = torch.zeros_like(self._best, dtype=torch.bool)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self._best[env_ids] = 0.0
        self._initialized[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        lift_height: float,
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
        score = torch.clamp(
            (positions[..., 2] - object_rest_height) / lift_height, 0.0, 1.0
        )
        contact = _bilateral_contact(
            env, force_threshold, fixed_sensor_cfg, moving_sensor_cfg
        )
        increment, self._best, self._initialized = episode_best_increment(
            score, self._best, self._initialized, contact & eligible
        )
        return increment.sum(dim=-1) / env.step_dt


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
