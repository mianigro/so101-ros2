"""Setup-driven follower arms + cameras.

The overhead cameras publish images only and are intentionally not part of the
TF tree; see docs/hardware.md for the rig layout.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from so101_bringup.arms_launch import follower_arms_action
from so101_bringup.camera_launch import declare_setup_arguments, include_cameras


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("hardware_type", default_value="real"),
            *declare_setup_arguments(),
            DeclareLaunchArgument("use_rviz", default_value="false"),
            follower_arms_action(),
            include_cameras(),
        ]
    )
