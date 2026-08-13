"""Manager terms defining the three-box/three-cup scenario."""

from .critic_observations import (
    THREE_BOX_CRITIC_STATE_COMPONENTS,
    THREE_BOX_CRITIC_STATE_DIM,
    critic_task_state,
)
from .events import reset_three_box_layout, sample_separated_positions
from .geometry import (
    best_assignment_score,
    best_assignment_values,
    complete_assignment,
    matched_entities,
    placement_matrix,
    update_settle_counter,
)
from .rewards import (
    approach_progress,
    closure_progress,
    grasp_acquired,
    insertion_progress,
    lift_progress,
    released_placement_progress,
    stable_placement_progress,
    transport_to_empty_cup,
)
from .terminations import all_boxes_stably_placed, any_box_dropped, invalid_state

__all__ = [
    "THREE_BOX_CRITIC_STATE_COMPONENTS",
    "THREE_BOX_CRITIC_STATE_DIM",
    "all_boxes_stably_placed",
    "approach_progress",
    "any_box_dropped",
    "best_assignment_score",
    "best_assignment_values",
    "complete_assignment",
    "closure_progress",
    "critic_task_state",
    "grasp_acquired",
    "insertion_progress",
    "invalid_state",
    "lift_progress",
    "matched_entities",
    "placement_matrix",
    "released_placement_progress",
    "reset_three_box_layout",
    "sample_separated_positions",
    "stable_placement_progress",
    "transport_to_empty_cup",
    "update_settle_counter",
]
