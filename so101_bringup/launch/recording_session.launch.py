"""Recording session: setup-driven arms + cameras + headless episode recorder."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.substitutions import FindPackageShare
from so101_bringup.arms_launch import (
    declare_use_follower_argument,
    include_layout_tf,
    teleop_arms_action,
)
from so101_bringup.camera_launch import (
    declare_setup_arguments,
    include_cameras,
    spawn_sim_camera_pipeline,
)
from so101_bringup.rerun_launch import declare_rerun_arguments, rerun_bridge_actions


def generate_launch_description():

    # --- Launch arguments ---
    hardware_type = LaunchConfiguration("hardware_type")  # real|mock|mujoco
    use_follower = LaunchConfiguration("use_follower")

    teleop_params_file = LaunchConfiguration("teleop_params_file")
    teleop_delay_s = LaunchConfiguration("teleop_delay_s")

    root_dir = LaunchConfiguration("root_dir")
    experiment_name = LaunchConfiguration("experiment_name")
    task = LaunchConfiguration("task")

    # The recorder needs the joint-state topic of whichever follower is active:
    # the physical stack publishes /follower/joint_states, the Isaac Sim
    # follower publishes /follower_sim/joint_states (single-pair setups only).
    joint_states_topic = PythonExpression(
        [
            "'/follower/joint_states' if '",
            use_follower,
            "'.lower() in ('true', '1') else '/follower_sim/joint_states'",
        ]
    )

    # --- Setup-driven arms (leaders, followers, one teleop relay per pair) ---
    arms = teleop_arms_action()

    # --- Include cameras launch ---
    cameras_launch = include_cameras()

    # --- Include Headless Recorder launch ---
    recorder_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("episode_recorder"), "launch", "recorder.launch.py"]
            )
        ),
        launch_arguments={
            "setup": LaunchConfiguration("setup"),
            "setup_config_file": LaunchConfiguration("setup_config_file"),
            "joint_states_topic": joint_states_topic,
            "root_dir": root_dir,
            "experiment_name": experiment_name,
            "task": task,
        }.items(),
    )

    # --- Launch Rerun (validated 2D/3D bridges) ---
    rerun_start = rerun_bridge_actions(LaunchConfiguration("setup"))

    # --- Defaults for files ---
    default_teleop_params = PathJoinSubstitution(
        [FindPackageShare("so101_teleop"), "config", "teleop.yaml"]
    )
    default_root_dir = PathJoinSubstitution(
        [
            EnvironmentVariable(
                "ROS_HOME",
                default_value=PathJoinSubstitution([EnvironmentVariable("HOME"), ".ros"]),
            ),
            "so101_episodes",
        ]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("hardware_type", default_value="real"),
            *declare_use_follower_argument(),
            *declare_setup_arguments(),
            DeclareLaunchArgument(
                "use_sim_cameras",
                default_value="false",
                description="Use Isaac raw cameras and republish them as JPEG",
            ),
            DeclareLaunchArgument(
                "teleop_params_file", default_value=default_teleop_params
            ),
            DeclareLaunchArgument("teleop_delay_s", default_value="2.0"),
            DeclareLaunchArgument("root_dir", default_value=default_root_dir),
            DeclareLaunchArgument("experiment_name", default_value="pick_and_place"),
            DeclareLaunchArgument("task", default_value=""),
            *declare_rerun_arguments(rerun_delay_default=teleop_delay_s),
            OpaqueFunction(function=spawn_sim_camera_pipeline),
            arms,
            include_layout_tf(),
            cameras_launch,
            recorder_launch,
            rerun_start,
        ]
    )
