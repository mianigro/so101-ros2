"""Stable SO-101 actor and action constants shared without Isaac Lab imports."""

SO101_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
SO101_ARM_JOINT_NAMES = SO101_JOINT_NAMES[:-1]

SO101_ACTOR_OBSERVATION_GROUPS = (
    "joint_state",
    "wrist",
    "overhead_1",
    "overhead_2",
)

SO101_ARM_ACTION_PATTERN = "shoulder_.*|elbow_flex|wrist_.*"
SO101_ARM_DELTA_RAD = 1.0 / 30.0
SO101_GRIPPER_DELTA_RAD = 0.10
SO101_ACTION_DELTA_SCALES_RAD = (
    SO101_ARM_DELTA_RAD,
    SO101_ARM_DELTA_RAD,
    SO101_ARM_DELTA_RAD,
    SO101_ARM_DELTA_RAD,
    SO101_ARM_DELTA_RAD,
    SO101_GRIPPER_DELTA_RAD,
)
SO101_NORMALIZED_ACTION_CLIP = (-1.0, 1.0)

__all__ = [
    "SO101_ACTION_DELTA_SCALES_RAD",
    "SO101_ACTOR_OBSERVATION_GROUPS",
    "SO101_ARM_ACTION_PATTERN",
    "SO101_ARM_DELTA_RAD",
    "SO101_ARM_JOINT_NAMES",
    "SO101_GRIPPER_DELTA_RAD",
    "SO101_JOINT_NAMES",
    "SO101_NORMALIZED_ACTION_CLIP",
]
