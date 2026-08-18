"""Static world placement for every arm defined by the active setup."""

import math

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from so101_bringup.arms_launch import active_setup
from so101_bringup.setup_config import layout_transforms


def _spawn_layout(context):
    setup = active_setup(context)
    world_frame = LaunchConfiguration("world_frame").perform(context)
    nodes = []
    for x, y, z, yaw, child_frame in layout_transforms(setup):
        arm = child_frame.removesuffix("/base_link")
        nodes.append(
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name=f"world_to_{arm}_base",
                arguments=[
                    str(x), str(y), str(z),
                    str(yaw), "0.0", "0.0",
                    world_frame,
                    child_frame,
                ],
            )
        )
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("world_frame", default_value="world"),
            DeclareLaunchArgument("setup"),
            DeclareLaunchArgument("setup_config_file", default_value=""),
            OpaqueFunction(function=_spawn_layout),
        ]
    )

# static_transform_publisher arguments: x y z yaw pitch roll parent_frame child_frame
