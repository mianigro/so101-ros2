"""Launch one profile camera without requiring pre-existing calibration/extrinsics."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction

from so101_bringup.camera_launch import spawn_cameras


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("camera_profile"),
        DeclareLaunchArgument("camera_id"),
        DeclareLaunchArgument("camera_rig_config_file"),
        DeclareLaunchArgument("follower_namespace", default_value="follower"),
        DeclareLaunchArgument("follower_frame_prefix", default_value="follower/"),
        DeclareLaunchArgument("camera_startup_timeout_s", default_value="10.0"),
        DeclareLaunchArgument("camera_stale_timeout_s", default_value="1.0"),
        OpaqueFunction(
            function=spawn_cameras,
            kwargs={
                "calibration_mode": True,
                "camera_id_argument": "camera_id",
            },
        ),
    ])
