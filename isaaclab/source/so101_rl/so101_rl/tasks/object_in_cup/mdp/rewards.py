"""Rewards for the object-in-cup task.

Every term uses only object/cup/gripper state -- there is no contact sensing
and no prescribed grasp geometry, so any method the policy finds earns its
reward.  The outcome terms pay for task *outcomes* (the cube is picked up,
moved to the cup, and placed inside) and cannot be farmed because earning them
requires the outcome itself.  The single shaping term that opens the ladder,
``approach_progress``, pays only the episode-best improvement of a bounded
gripper-to-cube proximity score: it says "bring the gripper's grasp region to
the cube" -- nothing about approach direction, orientation, or jaw geometry --
and its episode-best bookkeeping makes it impossible to farm by hovering or
re-approaching.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from so101_rl.tasks.common.mdp.pickup import (
    episode_best_increment,
    target_alignment_score,
)

from .geometry import placement_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


class approach_progress(ManagerTermBase):
    """Grasp-agnostic reaching reward: episode-best gripper-to-cube proximity.

    Scores ``1 - tanh(dist / position_scale)`` between the cube centre and the
    nominal grasp point (the ``ee_frame`` ``grasp_frame`` offset, i.e. where a
    25 mm cube centre sits against the fixed pad) and pays only the improvement
    over the episode's best score, so hovering at the cube or backing off and
    re-approaching cannot farm it.  Any approach direction, orientation, or
    grasp strategy earns it identically -- the term prescribes nothing about
    *how* the cube is grabbed, it only bridges the gap between flailing in the
    void and the first lucky grasp that ``lift_progress`` can reward.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._best = torch.zeros(env.num_envs, device=env.device)
        self._initialized = torch.ones_like(self._best, dtype=torch.bool)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self._best[env_ids] = 0.0
        self._initialized[env_ids] = True

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        *,
        position_scale: float = 0.08,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        ee_frame_name: str = "ee_frame",
    ) -> torch.Tensor:
        grasp_point = env.scene[ee_frame_name].data.target_pos_w.torch[:, 0, :]
        cube = env.scene[object_cfg.name].data.root_pos_w.torch
        score = target_alignment_score(
            cube, grasp_point, position_scale=position_scale
        )
        increment, self._best, self._initialized = episode_best_increment(
            score,
            self._best,
            self._initialized,
            torch.ones_like(self._best, dtype=torch.bool),
        )
        return increment / env.step_dt


def lift_progress(
    env: ManagerBasedRLEnv,
    *,
    object_rest_height: float = 0.0125,
    height_scale: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Outcome reward: the cube is picked up, however achieved.

    Pays a smooth ramp on the cube's height above its resting height,
    saturating at ``height_scale`` above rest.  Any method that raises the cube
    earns this -- the first upward nudge pays a sliver and a genuine hold pays
    the full amount -- and it cannot be farmed, because earning it requires the
    cube to actually be up.
    """
    object_z = env.scene[object_cfg.name].data.root_pos_w.torch[:, 2]
    return torch.clamp((object_z - object_rest_height) / height_scale, 0.0, 1.0)


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
