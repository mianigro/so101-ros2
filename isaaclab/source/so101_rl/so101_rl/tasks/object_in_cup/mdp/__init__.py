"""Task-specific manager terms and the small Isaac Lab MDP surface they use."""

# Keep this list explicit. A wildcard import materializes every lazy Isaac Lab MDP
# export, including USD-dependent helpers, before SimulationApp has launched.
from isaaclab.envs.mdp import (
    action_rate_l2,
    joint_pos,
    joint_vel_l2,
    randomize_actuator_gains,
    randomize_rigid_body_mass,
    randomize_rigid_body_material,
    reset_joints_by_offset,
    time_out,
)

from .critic_observations import (
    CRITIC_STATE_COMPONENTS,
    CRITIC_STATE_DIM,
    critic_task_state,
)
from .events import reset_task_layout
from .geometry import placement_mask, update_settle_counter
from .rewards import (
    grasp_object,
    insert_object,
    lift_object,
    reach_object,
    release_object,
    stable_placement_reward,
    transport_object,
)
from .terminations import invalid_state, object_dropped, stable_placement
from .vision_actions import DelayedRelativeJointPositionActionCfg
from .vision_events import (
    randomize_camera_calibration,
    randomize_preview_material,
    randomize_scene_lighting,
)
from .vision_observations import camera_rgb

__all__ = [
    "CRITIC_STATE_COMPONENTS",
    "CRITIC_STATE_DIM",
    "action_rate_l2",
    "camera_rgb",
    "critic_task_state",
    "DelayedRelativeJointPositionActionCfg",
    "grasp_object",
    "insert_object",
    "invalid_state",
    "joint_pos",
    "joint_vel_l2",
    "lift_object",
    "object_dropped",
    "placement_mask",
    "reach_object",
    "randomize_actuator_gains",
    "randomize_rigid_body_mass",
    "randomize_rigid_body_material",
    "randomize_camera_calibration",
    "randomize_preview_material",
    "randomize_scene_lighting",
    "release_object",
    "reset_task_layout",
    "reset_joints_by_offset",
    "stable_placement",
    "stable_placement_reward",
    "transport_object",
    "time_out",
    "update_settle_counter",
]
