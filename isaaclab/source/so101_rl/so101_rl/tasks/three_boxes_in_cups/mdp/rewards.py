"""Permutation-invariant dense rewards for placing three boxes in three cups."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .geometry import (
    best_assignment_score,
    matched_entities,
    placement_matrix,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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


def reach_unplaced_box(
    env: ManagerBasedRLEnv,
    std: float,
    placement: dict,
    box_names: tuple[str, str, str] = BOX_NAMES,
    cup_names: tuple[str, str, str] = CUP_NAMES,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
    ee_frame_name: str = "ee_frame",
) -> torch.Tensor:
    tensors = _task_tensors(env, box_names, cup_names, robot_cfg, ee_frame_name)
    box_positions, _, _, _, end_effector, _ = tensors
    matched_boxes, _ = matched_entities(_placed_pairs(tensors, placement))
    distance = torch.linalg.vector_norm(
        box_positions - end_effector[:, None, :], dim=-1
    )
    scores = (1.0 - torch.tanh(distance / std)).masked_fill(matched_boxes, 0.0)
    return scores.max(dim=-1).values


def lift_unplaced_box(
    env: ManagerBasedRLEnv,
    lift_height: float,
    object_rest_height: float,
    placement: dict,
    box_names: tuple[str, str, str] = BOX_NAMES,
    cup_names: tuple[str, str, str] = CUP_NAMES,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
    ee_frame_name: str = "ee_frame",
) -> torch.Tensor:
    """Reward picking up the best unplaced box: height gained above rest.

    Goal-based term: it only cares that a box has been raised, not how. The
    baseline is the box rest height (not the cup base), so this is zero while a
    box sits on the table. Already-placed boxes are masked out and the best
    unplaced box is taken.
    """
    tensors = _task_tensors(env, box_names, cup_names, robot_cfg, ee_frame_name)
    box_positions = tensors[0]
    matched_boxes, _ = matched_entities(_placed_pairs(tensors, placement))
    height_gain = torch.clamp(
        (box_positions[..., 2] - object_rest_height) / lift_height, 0.0, 1.0
    )
    scores = height_gain.masked_fill(matched_boxes, 0.0)
    return scores.max(dim=-1).values


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
