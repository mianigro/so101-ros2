"""Shared launch-description pieces for the strict camera subsystem."""

import json

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
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    TextSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from so101_bringup.setup_config import (
    SetupConfigError,
    default_setups_dir,
    load_camera_setup,
    load_setup,
)


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


def _camera_timeouts(context):
    startup_timeout = float(
        LaunchConfiguration("camera_startup_timeout_s").perform(context)
    )
    stale_timeout = float(
        LaunchConfiguration("camera_stale_timeout_s").perform(context)
    )
    if startup_timeout <= 0 or stale_timeout <= 0:
        raise SetupConfigError("camera startup/stale timeouts must be positive")
    return startup_timeout, stale_timeout


def spawn_cameras(context):
    """Create validated driver and supervisor actions for one setup."""
    setup_name = LaunchConfiguration("setup").perform(context).strip()
    setup_config_file = LaunchConfiguration("setup_config_file").perform(context).strip()
    startup_timeout, stale_timeout = _camera_timeouts(context)

    cameras = load_camera_setup(
        setup_name,
        default_setups_dir(),
        path=setup_config_file or None,
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

    supervisor = Node(
        package="so101_bringup",
        executable="camera_supervisor",
        name="camera_supervisor",
        parameters=[
            {
                "camera_names": ParameterValue(
                    TextSubstitution(
                        text=json.dumps([camera.camera_id for camera in cameras])
                    ),
                    value_type=list[str],
                ),
                "camera_topics": ParameterValue(
                    TextSubstitution(
                        text=json.dumps([camera.image_topic for camera in cameras])
                    ),
                    value_type=list[str],
                ),
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


def spawn_sim_camera_pipeline(context):
    """Republish Isaac's raw streams as JPEG and supervise the raw sources."""
    use_sim_cameras = IfCondition(
        LaunchConfiguration("use_sim_cameras")
    ).evaluate(context)
    if not use_sim_cameras:
        return []
    if IfCondition(LaunchConfiguration("use_cameras")).evaluate(context):
        raise SetupConfigError(
            "use_cameras and use_sim_cameras cannot both be true; "
            "select exactly one camera source"
        )

    setup_name = LaunchConfiguration("setup").perform(context).strip()
    setup_config_file = LaunchConfiguration("setup_config_file").perform(context).strip()
    startup_timeout, stale_timeout = _camera_timeouts(context)

    setup = load_setup(setup_name, default_setups_dir(), path=setup_config_file or None)
    if setup.sim is None:
        raise SetupConfigError(
            f"setup '{setup.name}' does not define a sim section; Isaac Sim "
            f"cameras are only supported for setups with sim geometry"
        )
    sim_topics = {
        camera["image_topic"]
        for camera in _sim_cameras(setup)
    }
    profile_topics = {camera["image_topic"] for camera in setup.cameras}
    if sim_topics != profile_topics:
        raise SetupConfigError(
            f"setup '{setup.name}' sim camera topics must match cameras.profile "
            f"topics exactly"
        )

    actions = []
    camera_names: list[str] = []
    raw_topics: list[str] = []
    for camera in setup.cameras:
        raw_topic = camera["image_topic"]

        republisher = Node(
            package="image_transport",
            executable="republish",
            name=f"sim_{camera['id']}_jpeg_republisher",
            parameters=[
                {
                    "in_transport": "raw",
                    "out_transport": "compressed",
                }
            ],
            remappings=[
                ("in", raw_topic),
                ("out/compressed", f"{raw_topic}/compressed"),
            ],
            output="screen",
        )
        actions.extend(
            [
                republisher,
                _shutdown_when_process_exits(
                    republisher, f"{camera['id']} JPEG republisher"
                ),
            ]
        )
        camera_names.append(camera["id"])
        raw_topics.append(raw_topic)

    supervisor = Node(
        package="so101_bringup",
        executable="camera_supervisor",
        name="sim_camera_supervisor",
        parameters=[
            {
                "camera_names": ParameterValue(
                    TextSubstitution(text=json.dumps(camera_names)),
                    value_type=list[str],
                ),
                "camera_topics": ParameterValue(
                    TextSubstitution(text=json.dumps(raw_topics)),
                    value_type=list[str],
                ),
                "startup_timeout_s": startup_timeout,
                "stale_timeout_s": stale_timeout,
            }
        ],
        output="screen",
    )
    actions.extend(
        [supervisor, _shutdown_when_process_exits(supervisor, "sim camera supervisor")]
    )
    return actions


def _sim_cameras(setup):
    sim_cameras = setup.sim.get("cameras") if setup.sim else None
    if not isinstance(sim_cameras, dict):
        raise SetupConfigError("sim section must define a cameras mapping")
    return list(sim_cameras.values())


def declare_setup_arguments(*, use_cameras_default: str = "true"):
    return [
        DeclareLaunchArgument("use_cameras", default_value=use_cameras_default),
        DeclareLaunchArgument(
            "setup",
            default_value="",
            description=(
                "Required when physical or simulated cameras are enabled: "
                "monomanual, monomanual_dual_overhead, or bimanual"
            ),
        ),
        DeclareLaunchArgument(
            "setup_config_file",
            default_value="",
            description=(
                "Optional absolute path to an edited external copy of a setup "
                "YAML (defaults to so101_bringup/config/setups/<setup>.yaml)"
            ),
        ),
        DeclareLaunchArgument("camera_startup_timeout_s", default_value="10.0"),
        DeclareLaunchArgument("camera_stale_timeout_s", default_value="1.0"),
    ]


def include_cameras():
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("so101_bringup"), "launch", "cameras.launch.py"
        ])),
        condition=IfCondition(LaunchConfiguration("use_cameras")),
        launch_arguments={
            "setup": LaunchConfiguration("setup"),
            "setup_config_file": LaunchConfiguration("setup_config_file"),
            "camera_startup_timeout_s": LaunchConfiguration("camera_startup_timeout_s"),
            "camera_stale_timeout_s": LaunchConfiguration("camera_stale_timeout_s"),
        }.items(),
    )
