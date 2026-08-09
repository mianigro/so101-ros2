"""Shared launch-description pieces for the strict camera subsystem."""

import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from so101_bringup.camera_config import CameraConfigError, load_camera_setup


def _shutdown_when_process_exits(action, label: str) -> RegisterEventHandler:
    return RegisterEventHandler(
        OnProcessExit(
            target_action=action,
            on_exit=[
                EmitEvent(
                    event=Shutdown(reason=f"required camera process exited: {label}")
                )
            ],
        )
    )


def spawn_cameras(
    context,
    *,
    calibration_mode: bool = False,
    camera_id_argument: str = "",
):
    """Create validated driver, TF, and supervisor actions for one profile."""
    profile_name = LaunchConfiguration("camera_profile").perform(context).strip()
    rig_path = LaunchConfiguration("camera_rig_config_file").perform(context).strip()
    follower_namespace = LaunchConfiguration("follower_namespace").perform(context)
    frame_prefix = LaunchConfiguration("follower_frame_prefix").perform(context)
    calibration_camera_id = (
        LaunchConfiguration(camera_id_argument).perform(context).strip()
        if camera_id_argument
        else ""
    )
    startup_timeout = float(
        LaunchConfiguration("camera_startup_timeout_s").perform(context)
    )
    stale_timeout = float(
        LaunchConfiguration("camera_stale_timeout_s").perform(context)
    )
    if startup_timeout <= 0 or stale_timeout <= 0:
        raise CameraConfigError("camera startup/stale timeouts must be positive")

    profiles_dir = os.path.join(
        get_package_share_directory("so101_bringup"),
        "config",
        "cameras",
        "profiles",
    )
    cameras = load_camera_setup(
        profile_name,
        profiles_dir,
        rig_path,
        follower_namespace,
        frame_prefix,
        calibration_mode=calibration_mode,
        calibration_camera_id=calibration_camera_id,
        validate_devices=True,
    )

    actions = []
    for camera in cameras:
        driver = Node(
            package=camera.driver_package,
            executable=camera.driver_executable,
            name=camera.node_name,
            namespace=camera.namespace,
            parameters=[camera.parameters],
            remappings=list(camera.remappings),
            output="screen",
        )
        actions.extend(
            [driver, _shutdown_when_process_exits(driver, camera.camera_id)]
        )

        if camera.transform is not None:
            transform = camera.transform
            translation = transform["translation"]
            rotation = transform["rotation_xyzw"]
            static_tf = Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name=f"{camera.camera_id}_static_tf",
                arguments=[
                    "--x",
                    str(translation[0]),
                    "--y",
                    str(translation[1]),
                    "--z",
                    str(translation[2]),
                    "--qx",
                    str(rotation[0]),
                    "--qy",
                    str(rotation[1]),
                    "--qz",
                    str(rotation[2]),
                    "--qw",
                    str(rotation[3]),
                    "--frame-id",
                    transform["parent_frame"],
                    "--child-frame-id",
                    camera.frame_id,
                ],
                output="screen",
            )
            actions.extend(
                [
                    static_tf,
                    _shutdown_when_process_exits(
                        static_tf, f"{camera.camera_id} static TF"
                    ),
                ]
            )

    supervisor = Node(
        package="so101_bringup",
        executable="camera_supervisor",
        name="camera_supervisor",
        parameters=[
            {
                "camera_names": [camera.camera_id for camera in cameras],
                "camera_topics": [camera.image_topic for camera in cameras],
                "camera_info_topics": [
                    camera.camera_info_topic for camera in cameras
                ],
                "require_camera_info": not calibration_mode,
                "startup_timeout_s": startup_timeout,
                "stale_timeout_s": stale_timeout,
            }
        ],
        output="screen",
    )
    actions.extend(
        [supervisor, _shutdown_when_process_exits(supervisor, "camera supervisor")]
    )
    return actions


def declare_camera_arguments(*, use_cameras_default: str = "true"):
    return [
        DeclareLaunchArgument("use_cameras", default_value=use_cameras_default),
        DeclareLaunchArgument(
            "camera_profile",
            default_value="",
            description="Required when use_cameras=true: single_overhead or dual_overhead",
        ),
        DeclareLaunchArgument(
            "camera_rig_config_file",
            default_value="",
            description="Required absolute path to the external physical camera rig YAML",
        ),
        DeclareLaunchArgument("camera_startup_timeout_s", default_value="10.0"),
        DeclareLaunchArgument("camera_stale_timeout_s", default_value="1.0"),
    ]


def include_cameras(follower_namespace, follower_frame_prefix):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("so101_bringup"), "launch", "cameras.launch.py"
        ])),
        condition=IfCondition(LaunchConfiguration("use_cameras")),
        launch_arguments={
            "camera_profile": LaunchConfiguration("camera_profile"),
            "camera_rig_config_file": LaunchConfiguration("camera_rig_config_file"),
            "follower_namespace": follower_namespace,
            "follower_frame_prefix": follower_frame_prefix,
            "camera_startup_timeout_s": LaunchConfiguration("camera_startup_timeout_s"),
            "camera_stale_timeout_s": LaunchConfiguration("camera_stale_timeout_s"),
        }.items(),
    )
