"""Headless launch: EpisodeRecorder lifecycle node auto-configure + auto-activate."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.events import matches_action
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from launch_ros.substitutions import FindPackageShare
import lifecycle_msgs.msg


def generate_launch_description():

    default_params = PathJoinSubstitution(
        [FindPackageShare("episode_recorder"), "config", "recorder.yaml"]
    )
    default_root_dir = PathJoinSubstitution(
        [
            EnvironmentVariable(
                "ROS_HOME",
                default_value=PathJoinSubstitution(
                    [EnvironmentVariable("HOME"), ".ros"]
                ),
            ),
            "so101_episodes",
        ]
    )

    # --- Launch arguments ---
    params_file = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="YAML with episode_recorder storage and timing parameters",
    )

    setup = DeclareLaunchArgument(
        "setup",
        description=(
            "Required setup: monomanual, monomanual_dual_overhead, or bimanual"
        ),
        choices=["monomanual", "monomanual_dual_overhead", "bimanual"],
    )

    setups_dir = DeclareLaunchArgument(
        "setups_dir",
        default_value=PathJoinSubstitution(
            [
                FindPackageShare("so101_bringup"),
                "config",
                "setups",
            ]
        ),
        description=(
            "Directory of setup YAMLs the recorder derives its topic "
            "set from (so101_bringup/config/setups by default)"
        ),
    )

    setup_config_file = DeclareLaunchArgument(
        "setup_config_file",
        default_value="",
        description=(
            "Optional absolute path to an edited external copy of the setup "
            "YAML; overrides setups_dir/<setup>.yaml"
        ),
    )

    joint_states_topic = DeclareLaunchArgument(
        "joint_states_topic",
        default_value="",
        description=(
            "Optional override of the primary follower joint-state topic "
            "(use /follower_sim/joint_states when Isaac Sim is the follower)."
        ),
    )

    root_dir = DeclareLaunchArgument(
        "root_dir",
        default_value=default_root_dir,
        description="Root directory for episodes",
    )

    experiment_name = DeclareLaunchArgument(
        "experiment_name",
        default_value="",
        description="Optional subfolder under root_dir (handled by the recorder node)",
    )

    task = DeclareLaunchArgument(
        "task",
        default_value="",
        description="Task label to store in rosbag2 metadata custom_data (required).",
    )

    recorder_ns = DeclareLaunchArgument(
        "recorder_ns",
        default_value="",
        description="Namespace for recorder node (optional)",
    )

    recorder = LifecycleNode(
        package="episode_recorder",
        executable="episode_recorder_node",
        name="episode_recorder",
        namespace=LaunchConfiguration("recorder_ns"),
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {
                "setup": LaunchConfiguration("setup"),
                "setups_dir": LaunchConfiguration("setups_dir"),
                "setup_config_file": LaunchConfiguration("setup_config_file"),
                "joint_states_topic": LaunchConfiguration("joint_states_topic"),
                "root_dir": LaunchConfiguration("root_dir"),
                "experiment_name": LaunchConfiguration("experiment_name"),
                "task": LaunchConfiguration("task"),
            },
        ],
    )

    # Configure once the process starts (avoids race)
    configure_on_start = RegisterEventHandler(
        OnProcessStart(
            target_action=recorder,
            on_start=[
                EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=matches_action(recorder),
                        transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
                    )
                )
            ],
        )
    )

    # Activate after it reaches INACTIVE
    activate_after_configure = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=recorder,
            goal_state="inactive",
            entities=[
                EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=matches_action(recorder),
                        transition_id=lifecycle_msgs.msg.Transition.TRANSITION_ACTIVATE,
                    )
                )
            ],
        )
    )

    return LaunchDescription(
        [
            params_file,
            setup,
            setups_dir,
            setup_config_file,
            joint_states_topic,
            root_dir,
            experiment_name,
            task,
            recorder_ns,
            recorder,
            configure_on_start,
            activate_after_configure,
        ]
    )
