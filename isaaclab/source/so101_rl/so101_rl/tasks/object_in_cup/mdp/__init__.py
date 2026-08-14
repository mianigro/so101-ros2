"""Manager terms that define the object-in-cup scenario semantics."""

from .critic_observations import (
    CRITIC_STATE_COMPONENTS,
    CRITIC_STATE_DIM,
    critic_task_state,
)
from .events import reset_task_layout
from .geometry import placement_mask, update_settle_counter
from .rewards import (
    approach_progress,
    insert_object,
    lift_progress,
    release_object,
    stable_placement_reward,
    transport_object,
)
from .terminations import invalid_state, object_dropped, stable_placement

__all__ = [
    "CRITIC_STATE_COMPONENTS",
    "CRITIC_STATE_DIM",
    "approach_progress",
    "critic_task_state",
    "insert_object",
    "invalid_state",
    "lift_progress",
    "object_dropped",
    "placement_mask",
    "release_object",
    "reset_task_layout",
    "stable_placement",
    "stable_placement_reward",
    "transport_object",
    "update_settle_counter",
]
