import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from so101_bringup.camera_launch import declare_camera_arguments, include_cameras


def _launch_setup(context):
    hardware_type = LaunchConfiguration("hardware_type").perform(context)
    namespace = LaunchConfiguration("namespace").perform(context)
    use_rviz = LaunchConfiguration("use_rviz").perform(context)

    use_sim_time = "true" if hardware_type == "mujoco" else "false"

    # 1) Bringup (ros2_control + rsp + spawners) - your existing follower.launch.py
    follower_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("so101_bringup"),
                "launch",
                "follower_split.launch.py",
            )
        ),
        launch_arguments={
            "namespace": namespace,
            "hardware_type": hardware_type,
            "use_rviz": "false",  # MoveIt RViz is launched separately below
        }.items(),
    )

    cameras_launch = include_cameras(namespace, "")

    # 2) Move group (pure MoveIt)
    move_group = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("so101_moveit_config"),
                "launch",
                "move_group.launch.py",
            )
        ),
        launch_arguments={
            "namespace": namespace,
            "variant": "follower",
            "use_sim_time": use_sim_time,
        }.items(),
    )

    # 3) MoveIt RViz (optional)
    moveit_rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("so101_moveit_config"),
                "launch",
                "moveit_rviz.launch.py",
            )
        ),
        launch_arguments={
            "namespace": namespace,
            "variant": "follower",
        }.items(),
        condition=IfCondition(use_rviz),
    )

    return [follower_bringup, cameras_launch, move_group, moveit_rviz]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "hardware_type", default_value="real"
            ),  # real|mock|mujoco
            DeclareLaunchArgument("namespace", default_value="follower"),
            *declare_camera_arguments(use_cameras_default="false"),
            DeclareLaunchArgument(
                "use_rviz",
                default_value="true",
                description="Launch MoveIt RViz (set false when using Rerun)",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
