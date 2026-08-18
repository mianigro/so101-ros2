"""Teleop: setup-driven arms + cameras + layout TF + RViz + Rerun bridges."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from so101_bringup.arms_launch import (
    active_setup,
    declare_use_follower_argument,
    include_layout_tf,
    teleop_arms_action,
)
from so101_bringup.camera_launch import declare_setup_arguments, include_cameras
from so101_bringup.rerun_launch import declare_rerun_arguments, rerun_bridge_actions

_TRUTHY = {"1", "true", "yes", "on"}


def _rviz_action(context):
    if LaunchConfiguration("use_teleop_rviz").perform(context).strip().lower() not in _TRUTHY:
        return []
    setup = active_setup(context)
    config = "teleop_bimanual.rviz" if setup.name == "bimanual" else "teleop.rviz"
    return [
        Node(
            package="rviz2",
            executable="rviz2",
            name="teleop_rviz",
            arguments=["-d", PathJoinSubstitution([
                FindPackageShare("so101_bringup"), "rviz", config
            ])],
            output="screen",
        )
    ]


def generate_launch_description():

    # --- Launch arguments ---
    hardware_type = LaunchConfiguration("hardware_type")  # real|mock|mujoco
    teleop_params_file = LaunchConfiguration("teleop_params_file")
    teleop_delay_s = LaunchConfiguration("teleop_delay_s")
    leader_rviz = LaunchConfiguration("leader_rviz")
    follower_rviz = LaunchConfiguration("follower_rviz")

    # --- Setup-driven arms (leaders, followers, one teleop relay per pair) ---
    arms = teleop_arms_action()

    # --- Include cameras launch ---
    cameras_launch = include_cameras()

    # --- Include layout launch tf ---
    layout_tf_launch = include_layout_tf()

    # --- Rviz Node (per-setup config) ---
    rviz_node = OpaqueFunction(function=_rviz_action)

    # --- Launch Rerun (validated 2D/3D bridges) ---
    rerun_start = rerun_bridge_actions(LaunchConfiguration("setup"))

    # --- Defaults for files ---
    default_teleop_params = PathJoinSubstitution(
        [FindPackageShare("so101_teleop"), "config", "teleop.yaml"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("hardware_type", default_value="real"),
            *declare_use_follower_argument(),
            DeclareLaunchArgument("teleop_params_file", default_value=default_teleop_params),
            DeclareLaunchArgument("teleop_delay_s", default_value="2.0"),
            *declare_setup_arguments(),
            DeclareLaunchArgument("use_teleop_rviz", default_value="true"),
            *declare_rerun_arguments(rerun_delay_default=teleop_delay_s),
            arms,
            layout_tf_launch,
            cameras_launch,
            rviz_node,
            rerun_start,
        ]
    )
