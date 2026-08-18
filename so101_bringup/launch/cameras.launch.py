"""Launch the validated camera subsystem for one canonical setup."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction

from so101_bringup.camera_launch import spawn_cameras


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("setup"),
        DeclareLaunchArgument("setup_config_file", default_value=""),
        DeclareLaunchArgument("camera_startup_timeout_s", default_value="10.0"),
        DeclareLaunchArgument("camera_stale_timeout_s", default_value="1.0"),
        OpaqueFunction(function=spawn_cameras),
    ])
