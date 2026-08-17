"""Recording session: arms + cameras + headless episode recorder."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.substitutions import FindPackageShare
from so101_bringup.camera_launch import (
    declare_camera_arguments,
    include_cameras,
    spawn_sim_camera_pipeline,
)
from so101_bringup.rerun_launch import declare_rerun_arguments, rerun_bridge_actions


def generate_launch_description():

    # --- Launch arguments ---
    hardware_type = LaunchConfiguration("hardware_type")  # real|mock|mujoco
    leader_ns = LaunchConfiguration("leader_namespace")
    follower_ns = LaunchConfiguration("follower_namespace")
    leader_frame_prefix = LaunchConfiguration("leader_frame_prefix")
    follower_frame_prefix = LaunchConfiguration("follower_frame_prefix")

    leader_usb = LaunchConfiguration("leader_usb_port")
    follower_usb = LaunchConfiguration("follower_usb_port")

    leader_ctrl_cfg = LaunchConfiguration("leader_controller_config_file")
    follower_ctrl_cfg = LaunchConfiguration("follower_controller_config_file")

    teleop_params_file = LaunchConfiguration("teleop_params_file")
    teleop_delay_s = LaunchConfiguration("teleop_delay_s")

    root_dir = LaunchConfiguration("root_dir")
    experiment_name = LaunchConfiguration("experiment_name")
    task = LaunchConfiguration("task")

    use_follower = LaunchConfiguration("use_follower")

    # --- Include leader bringup ---
    leader_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("so101_bringup"), "launch", "leader.launch.py"]
            )
        ),
        launch_arguments={
            "namespace": leader_ns,
            "hardware_type": hardware_type,
            "usb_port": leader_usb,
            "frame_prefix": leader_frame_prefix,
            "controller_config_file": leader_ctrl_cfg,
            "use_rviz": "false",
        }.items(),
    )

    # --- Include follower bringup ---
    follower_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("so101_bringup"), "launch", "follower.launch.py"]
            )
        ),
        launch_arguments={
            "namespace": follower_ns,
            "hardware_type": hardware_type,
            "usb_port": follower_usb,
            "frame_prefix": follower_frame_prefix,
            "controller_config_file": follower_ctrl_cfg,
            "use_rviz": "false",
        }.items(),
        condition=IfCondition(use_follower),
    )

    # --- Include cameras launch ---
    cameras_launch = include_cameras(follower_ns, follower_frame_prefix)

    # --- Include teleop launch ---
    teleop_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("so101_teleop"), "launch", "teleop.launch.py"]
            )
        ),
        launch_arguments={
            "leader_namespace": leader_ns,
            "follower_namespace": follower_ns,
            "params_file": teleop_params_file,
        }.items(),
    )

    teleop_start = TimerAction(
        period=teleop_delay_s,
        actions=[teleop_include],
    )

    # --- Include Headless Recorder launch ---
    recorder_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("episode_recorder"), "launch", "recorder.launch.py"]
            )
        ),
        launch_arguments={
            "camera_profile": LaunchConfiguration("camera_profile"),
            "root_dir": root_dir,
            "experiment_name": experiment_name,
            "task": task,
        }.items(),
    )

    # --- Launch Rerun (validated 2D/3D bridges) ---
    rerun_start = rerun_bridge_actions(LaunchConfiguration("camera_profile"))

    # --- Defaults for files ---
    default_leader_ctrl_cfg = PathJoinSubstitution(
        [
            FindPackageShare("so101_bringup"),
            "config",
            "ros2_control",
            "leader_controllers.yaml",
        ]
    )
    default_follower_ctrl_cfg = PathJoinSubstitution(
        [
            FindPackageShare("so101_bringup"),
            "config",
            "ros2_control",
            "follower_controllers.yaml",
        ]
    )
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
            DeclareLaunchArgument(
                "use_follower",
                default_value="true",
                description="Start the physical/mock follower controller manager",
            ),
            DeclareLaunchArgument("leader_namespace", default_value="leader"),
            DeclareLaunchArgument("follower_namespace", default_value="follower"),
            DeclareLaunchArgument("leader_frame_prefix", default_value="leader/"),
            DeclareLaunchArgument("follower_frame_prefix", default_value="follower/"),
            DeclareLaunchArgument("leader_usb_port", default_value="/dev/so101_leader"),
            DeclareLaunchArgument(
                "follower_usb_port", default_value="/dev/so101_follower"
            ),
            DeclareLaunchArgument(
                "leader_controller_config_file", default_value=default_leader_ctrl_cfg
            ),
            DeclareLaunchArgument(
                "follower_controller_config_file",
                default_value=default_follower_ctrl_cfg,
            ),
            *declare_camera_arguments(),
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
            leader_launch,
            follower_launch,
            cameras_launch,
            recorder_launch,
            rerun_start,
            teleop_start,
        ]
    )
