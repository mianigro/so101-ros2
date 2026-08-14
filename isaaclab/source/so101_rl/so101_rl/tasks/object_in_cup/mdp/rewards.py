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
    grasp_targets_from_fixed_pad,
    object_between_jaws,
    retryable_event_increment,
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
    """Pay jaw closure after the cube edge enters between the pads."""

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
        fixed_pad_length: float = 0.025,
        minimum_insertion: float = 0.001,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
        fixed_sensor_cfg: SceneEntityCfg = SceneEntityCfg("fixed_jaw_contact"),
        moving_sensor_cfg: SceneEntityCfg = SceneEntityCfg("moving_jaw_contact"),
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
        object_position = env.scene[object_cfg.name].data.root_pos_w.torch
        fixed_pad = env.scene[fixed_sensor_cfg.name].data
        moving_pad = env.scene[moving_sensor_cfg.name].data
        between = object_between_jaws(
            object_position,
            env.scene[object_cfg.name].data.root_quat_w.torch,
            fixed_pad.pos_w.torch[:, 0, :],
            fixed_pad.quat_w.torch[:, 0, :],
            moving_pad.pos_w.torch[:, 0, :],
            half_extents,
            fixed_pad_length=fixed_pad_length,
            minimum_insertion=minimum_insertion,
        ).unsqueeze(-1)
        increment, self._credited, self._initialized = bounded_closure_increment(
            self._previous,
            current,
            alignment,
            self._credited,
            between,
            self._initialized,
            closure_range=closure_range,
        )
        self._previous.copy_(current)
        return increment.sum(dim=-1) / env.step_dt


class grasp_acquired(ManagerTermBase):
    """Pay once per grasp attempt when both pads contact the cube.

    The impulse re-arms whenever bilateral contact is lost, so a fresh pinch
    after a drop earns the reward again.  Within one continuous contact
    session it still fires at most once (rising edge), so holding the grasp
    cannot farm it.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._in_contact = torch.zeros(
            (env.num_envs, 1), dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self._in_contact[env_ids] = False

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
        increment, self._in_contact = retryable_event_increment(
            contact,
            self._in_contact,
            torch.ones_like(contact, dtype=torch.bool),
        )
        return increment.sum(dim=-1) / env.step_dt


def grasp_held(
    env: ManagerBasedRLEnv,
    half_extents: tuple[float, float, float],
    *,
    closed_threshold: float = 0.15,
    fixed_pad_length: float = 0.025,
    minimum_insertion: float = 0.001,
    object_rest_height: float = 0.0125,
    height_scale: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
    fixed_sensor_cfg: SceneEntityCfg = SceneEntityCfg("fixed_jaw_contact"),
    moving_sensor_cfg: SceneEntityCfg = SceneEntityCfg("moving_jaw_contact"),
) -> torch.Tensor:
    """Dense reward for holding the cube clamped between the jaws.

    Uses grasp *geometry* (cube between the pads with the gripper closed)
    rather than contact force, so the policy gets a continuous gradient into
    and through the pinch even when bilateral contact is marginal.  This
    bridges ``closure_progress`` (closing motion) and ``lift_progress``
    (contact-validated lift), which otherwise share no dense signal, and is
    the primary lever for escaping the reach-and-fake-close stall.

    The binary ``between & closed`` mask is scaled by a height-progress term so
    that clamping the cube on the table pays a fraction (``hold_floor``) of the
    credit while a genuine lift toward ``object_rest_height + height_scale``
    earns the full amount.  This keeps the pinch-to-lift gradient but removes
    the incentive to sit on the table indefinitely.
    """
    object_asset = env.scene[object_cfg.name]
    fixed_pad = env.scene[fixed_sensor_cfg.name].data
    moving_pad = env.scene[moving_sensor_cfg.name].data
    between = object_between_jaws(
        object_asset.data.root_pos_w.torch,
        object_asset.data.root_quat_w.torch,
        fixed_pad.pos_w.torch[:, 0, :],
        fixed_pad.quat_w.torch[:, 0, :],
        moving_pad.pos_w.torch[:, 0, :],
        half_extents,
        fixed_pad_length=fixed_pad_length,
        minimum_insertion=minimum_insertion,
    )
    gripper_pos = _gripper_position(env, robot_cfg)
    closed = gripper_pos <= closed_threshold
    held = (between & closed).to(dtype=gripper_pos.dtype)
    # 0.3 on the table, ramping to 1.0 over height_scale above the rest height.
    height_progress = torch.clamp(
        (object_asset.data.root_pos_w.torch[:, 2] - object_rest_height) / height_scale,
        0.0,
        1.0,
    )
    hold_floor = 0.3
    return held * (hold_floor + (1.0 - hold_floor) * height_progress)


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


class transport_object(ManagerTermBase):
    """Reward moving the lifted cube toward the cup, decaying while held aloft.

    The reward is ``base * (1 - tanh(r_xy/std))`` where ``base`` starts at 1.0
    and decays the longer the cube has been continuously above ``minimum_height``,
    so a policy that simply hovers near the cup cannot farm it indefinitely.

    A grace window of ``grace_steps`` pays full credit (genuine transport right
    after the lift).  Beyond that, ``base`` linearly ramps toward ``decay_floor``
    over the next ``decay_steps``.  The per-env "consecutive steps aloft" counter
    resets to zero the moment the cube drops below ``minimum_height``, so a fresh
    re-transport after a genuine drop re-arms the full reward.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._aloft_steps = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self._aloft_steps[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        minimum_height: float,
        grace_steps: int = 30,
        decay_steps: int = 120,
        decay_floor: float = 0.2,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        cup_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    ) -> torch.Tensor:
        relative = (
            env.scene[object_cfg.name].data.root_pos_w.torch
            - env.scene[cup_cfg.name].data.root_pos_w.torch
        )
        radial_distance = torch.linalg.vector_norm(relative[:, :2], dim=-1)
        lifted = relative[:, 2] >= minimum_height

        # Advance the per-env consecutive-aloft counter and reset it on a drop.
        self._aloft_steps = torch.where(
            lifted, self._aloft_steps + 1, torch.zeros_like(self._aloft_steps)
        )

        # Full credit during the grace window, then a linear ramp to the floor.
        ramp_start = torch.tensor(float(grace_steps), device=env.device)
        ramp_end = ramp_start + float(decay_steps)
        progress = torch.clamp(
            (self._aloft_steps - ramp_start) / (ramp_end - ramp_start), 0.0, 1.0
        )
        base = 1.0 - (1.0 - decay_floor) * progress
        return (1.0 - torch.tanh(radial_distance / std)) * lifted.float() * base


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
