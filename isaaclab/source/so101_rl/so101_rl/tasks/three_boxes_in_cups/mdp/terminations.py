"""Termination conditions for the three-box scenario."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .geometry import complete_assignment, placement_matrix, update_settle_counter
from .rewards import BOX_NAMES, CUP_NAMES, _task_tensors

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import TerminationTermCfg


def any_box_dropped(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    box_names: tuple[str, str, str] = BOX_NAMES,
) -> torch.Tensor:
    heights = torch.stack(
        [env.scene[name].data.root_pos_w.torch[:, 2] for name in box_names], dim=-1
    )
    return (heights < minimum_height).any(dim=-1)


def invalid_state(
    env: ManagerBasedRLEnv,
    box_names: tuple[str, str, str] = BOX_NAMES,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    box_states = torch.cat(
        [env.scene[name].data.root_state_w.torch for name in box_names], dim=-1
    )
    robot = env.scene[robot_cfg.name]
    robot_state = torch.cat(
        (robot.data.joint_pos.torch, robot.data.joint_vel.torch), dim=-1
    )
    return ~(torch.isfinite(box_states).all(dim=-1) & torch.isfinite(robot_state).all(dim=-1))


class all_boxes_stably_placed(ManagerTermBase):
    """Terminate after all boxes occupy distinct cups for a settling window."""

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
        placement: dict,
        box_names: tuple[str, str, str] = BOX_NAMES,
        cup_names: tuple[str, str, str] = CUP_NAMES,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
        ee_frame_name: str = "ee_frame",
    ) -> torch.Tensor:
        tensors = _task_tensors(env, box_names, cup_names, robot_cfg, ee_frame_name)
        valid_pairs = placement_matrix(
            *tensors, **placement, require_stable=True
        )
        complete = complete_assignment(valid_pairs)
        self._consecutive_steps = update_settle_counter(
            self._consecutive_steps, complete
        )
        return self._consecutive_steps >= required_steps
